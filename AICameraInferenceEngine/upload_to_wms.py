import time
import json
import redis
import hashlib
import requests
import threading
from loguru import logger
from repository.BackendRepository import BackendRepository
from utils.readConfig import set_config, save_config, get_config, config as CONFIG, reload_config
from utils.publisher import publisher, PublishEvent
from settings import settings
from utils.log import truncate_long_strings

POOL = redis.ConnectionPool(host=settings.redis_host,
                            port=settings.redis_port, decode_responses=True)
REDIS = redis.Redis(connection_pool=POOL)

# 用於防止重複註冊的鎖和狀態
_register_lock = threading.Lock()
_registering = False

# Keys that must survive power loss — persisted to config.ini alongside Redis.
# Add new keys here; redis_set/redis_get handle the rest automatically.
_PERSISTENT_REDIS_KEYS = {
    'accessTokenForOTA': ('WMS', 'access_token'),
    'cid':               ('WMS', 'cid'),
    'deviceName':        ('WMS', 'device_name'),
}

def _sync_persistent_keys():
    """On startup, bidirectional sync between Redis and config.ini.
    - Redis has value, config doesn't → write to config (existing devices)
    - Redis empty, config has value   → restore to Redis (power-loss recovery)
    """
    need_save = False
    for redis_key, (section, ini_key) in _PERSISTENT_REDIS_KEYS.items():
        redis_val = REDIS.get(redis_key) or ''
        ini_val = get_config(section, ini_key, '')
        if len(redis_val) > 0 and len(ini_val) == 0:
            set_config(section, ini_key, redis_val)
            need_save = True
            logger.info(f'Persisted Redis key [{redis_key}] to config.ini (existing device migration)')
        elif len(redis_val) == 0 and len(ini_val) > 0:
            REDIS.set(redis_key, ini_val)
            logger.info(f'Restored Redis key [{redis_key}] from config.ini')
    if need_save:
        save_config()

def redis_set(key, value):
    """Write to Redis. If the key is in _PERSISTENT_REDIS_KEYS, also persist
    to config.ini so the value survives power loss."""
    REDIS.set(key, value)
    if key in _PERSISTENT_REDIS_KEYS and value:
        section, ini_key = _PERSISTENT_REDIS_KEYS[key]
        set_config(section, ini_key, str(value))

def redis_get(key, default=''):
    """Read from Redis. If empty and the key is in _PERSISTENT_REDIS_KEYS,
    fall back to config.ini."""
    val = REDIS.get(key) or ''
    if len(val) == 0 and key in _PERSISTENT_REDIS_KEYS:
        section, ini_key = _PERSISTENT_REDIS_KEYS[key]
        val = get_config(section, ini_key, '')
        if len(val) > 0:
            REDIS.set(key, val)
            logger.info(f'Restored Redis key [{key}] from config.ini on read')
    return val if len(val) > 0 else default

_sync_persistent_keys()

def init_camera(data = None):
    global REDIS, _register_lock, _registering
    
    
    # 使用鎖防止並發註冊
    with _register_lock:
        # 再次檢查，防止在等待鎖期間其他線程已完成註冊
        existing_token = redis_get("accessTokenForOTA")
        if len(existing_token) > 0:
            logger.info("accessTokenForOTA already present (Redis or config.ini fallback), skipping re-registration")
            return "OK"
            
        if _registering:
            logger.info("Registration already in progress, waiting...")
            # 等待正在進行的註冊完成
            while _registering:
                time.sleep(0.1)
            # 檢查註冊結果
            final_token = redis_get("accessTokenForOTA")
            if len(final_token) > 0:
                return "OK"
            else:
                return "Registration failed"
        
        _registering = True
        try:
            backendRepository = BackendRepository()

            registerURL = f"{get_config('API', 'wms_base', 'https://dscms.family.com.tw/cms')}/camera/register"
            camName = redis_get("deviceName")
            deviceId = get_config('SYSTEM', 'device_id')
            
            if deviceId is None or deviceId == '':
                raise Exception('No device ID found')
            
            data = { "camID": get_config('SYSTEM', 'device_id'), "camName": camName }
            timestamp = int(time.time())
            try:
                if camName is None or camName == '':
                    raise Exception('Register was triggered, but [device name] is un-set, please set [device name] before registering')
                register_signature = str(backendRepository.get_register_signature()).replace('"', '')
                sig = get_sig(json.dumps(data, ensure_ascii=False), timestamp, register_signature)
                logger.info(f'Registered by signature: {register_signature}')
            except Exception as e:
                logger.error(f'Unable to get register signature from backend: {e}')
                return
                # sig = get_sig(json.dumps(data), timestamp, '2GHE3')
            data["signature"] = sig
            data["timestamp"] = timestamp
            headers = {
                'Content-type': 'application/json',
                "Accept-Encoding": "gzip",
                'User-Agent': 'vac-client'
            }
            data = json.dumps(data)
            try:
                ret = requests.post(registerURL, headers=headers, data=data, verify=False)
                response = ret.json()

                if (ret.status_code != 200):
                    logger.error(f"[init_camera] status_code {ret.status_code} text: {ret.text}")

                logger.info(f'Register data: {ret.text}')
                if response.get("success") is False:
                    return f"Error {response}"
            except Exception as e:
                logger.error(
                    f"[init_camera] {e}")
                return f"Error {e}"

            # update token
            token = response.get("accessToken") or ""
            if len(token) == 0:
                return "token error"
            cid = response.get('cid', '')
            redis_set('cid', cid)
            domain = response.get('domain')
            logger.info(f"Setting WMS domain: {domain}")
            if domain:
                need_restart_daemon = False
                if get_config('API', 'wms_clientlog') != domain.get('clientLog') or \
                   get_config('API', 'wms_mqtt_host') != domain.get('mqttHost') or \
                   get_config('API', 'wms_device_log') != domain.get('deviceLog'):
                    need_restart_daemon = True
                set_config('API', 'wms_clientlog', domain.get('clientLog'))
                logger.info(f"Setting Client Log host: {domain.get('clientLog')}")
                set_config('API', 'wms_mqtt_host', domain.get('mqttHost'))
                logger.info(f"Setting MQTT host: {domain.get('mqttHost')}")
                set_config('API', 'wms_device_log', domain.get('deviceLog'))
                logger.info(f"Setting Device Log host: {domain.get('deviceLog')}")
                save_config()
                if need_restart_daemon:
                    backendRepository.restart_daemon()

            redis_set("accessTokenForOTA", token)

            domain = response.get('domain')
            if domain:
                REDIS.set('loggerServer', domain.get('deviceLog'))
            save_config()
            REDIS.save()
            return "OK"
        finally:
            _registering = False


def string_to_md5(data):
    m = hashlib.md5()
    m.update(data.encode("utf-8"))

    return m.hexdigest().upper()


def get_sig(data, timestamp, sig: str = None):
    s = f"{data.replace(' ','')}.{sig if sig is not None else get_config('API', 'wms_sig', '407D43682715C516B3C807CC695326E4C76540DD')}.{timestamp}"
    md5 = string_to_md5(s)
    return md5


def upload(rawDataList):
    global REDIS

    if len(rawDataList) == 0:
        return {"status": "No data"}

    deviceName = redis_get("deviceName")

    if len(deviceName) == 0:
        deviceName = "default"
        redis_set("deviceName", deviceName)

    token = redis_get("accessTokenForOTA")
    if len(token) == 0:
        ret = init_camera()
        if ret != "OK":
            raise Exception('Failed to register camera')

    records = []
    for raw in rawDataList:
        rawData = json.loads(raw)
        d = {
            "camId": get_config('SYSTEM', 'device_id'),
            "cid": redis_get('cid'),
            **rawData
        }
        records.append(d)

    recordURL = f"{get_config('API', 'wms_clientlog')}/api/peopleCounter/record/add"
    headers = {
        'Content-type': 'application/json',
        "Accept-Encoding": "gzip",
        'User-Agent': 'vac-client'
    }

    timestamp = int(time.time())
    dataToSig = json.dumps({"records": records}).replace(" ", "")
    signature = get_sig(dataToSig, timestamp)
    data = {
        "records": records,
        "signature": signature,
        "timestamp": timestamp
    }

    dataToPost = json.dumps(data)
    logger.info(f"[upload] dataToPost: {truncate_long_strings(dataToPost)}")
    ret = requests.post(recordURL, headers=headers,
                        data=dataToPost, verify=False, timeout=5)
    ret.raise_for_status()
    response = ret.json()
    if response.get("message") == "success":
        logger.info(f"[upload] status_code {ret.status_code} text: {ret.content}")
        return response


    logger.error(
        f"[upload] status_code {ret.status_code} text: {ret.text}")
    raise Exception('Failed to sync people record')

publisher.subscribe(PublishEvent.UPDATED_SIGNATURE, init_camera)