import redis
from collections import deque
from classes.SqliteQueue import SqliteQueue
class RedisQueue:
    def __init__(self, main_queue_name, temp_queue_name, host='localhost', port=6379, maxlen= 5000, sqliteQueue: SqliteQueue = None):
        POOL = redis.ConnectionPool(host=host,
                            port=port, decode_responses=True)
        self.sqliteQueue = sqliteQueue
        self.redis_client = redis.Redis(connection_pool=POOL)
        self.main_queue_name = main_queue_name
        self.temp_queue_name = temp_queue_name
        self.fallback_queue = deque(maxlen=maxlen)
        self.maxlen = maxlen
        self.redis_connected = False
        self.lock_temp = False

    def check_redis_connection(self):
        try:
            self.redis_connected = self.redis_client.ping()
        except (redis.exceptions.ConnectionError, redis.exceptions.TimeoutError):
            self.redis_connected = False

    def push_to_queue(self, data):
        if self.redis_connected:
            self.fill_redis_from_sqlite()
            self.redis_client.lpush(self.main_queue_name, data)
            # 檢查主要隊列是否超過 maxlen，若超過則將資料寫入 SQLite Queue
            self.trim_redis_data()
        else:
            self.fallback_queue.append(data)

    def pop_from_queue(self, batch_size):
        payload = []
        if self.redis_connected:
            # 首先嘗試從臨時隊列中提取資料
            temp_length = self.redis_client.llen(self.temp_queue_name)
            main_length = self.redis_client.llen(self.main_queue_name)
            
            for _ in range(min(batch_size, temp_length)):
                data = self.redis_client.rpop(self.temp_queue_name)
                if data:
                    payload.append(data)
                else:
                    break

            # 如果 主要隊列 尚未達到最大數量，並且 SQLite 隊列有資料時，從 SQLite 填充資料到 主要隊列
            self.fill_redis_from_sqlite()
            
            # 如果臨時隊列中資料不足，從主隊列補充
            if len(payload) < batch_size:
                remaining_length = batch_size - len(payload)
                for _ in range(min(remaining_length, main_length)):
                    data = self.redis_client.rpop(self.main_queue_name)
                    if data:
                        payload.append(data)
                    else:
                        break
            # 再次從 SQlite 填充資料
            self.fill_redis_from_sqlite()
        else:
            while self.fallback_queue and len(payload) < batch_size:
                payload.append(self.fallback_queue.popleft())
        return payload

    def clear_temp_queue(self, success, payload):
        if self.redis_connected and success:
            pass
            # for _ in payload:
            #     self.redis_client.rpop(self.temp_queue_name)
        elif not success:
            # If request failed, push back to fallback queue or temp queue
            if self.redis_connected:
                for item in payload:
                    self.redis_client.lpush(self.temp_queue_name, item)
            else:
                self.fallback_queue.extendleft(payload)

    def trim_redis_data(self):
        # 檢查是否有設定最大數量
        if self.maxlen:
            current_length = self.redis_client.llen(self.main_queue_name)
            # 如果目前的隊列長度超過最大值，則將資料取出並寫入 SQLite
            if current_length > self.maxlen:
                excess_data = current_length - self.maxlen
                print(f'Transfer {excess_data} to sqlite')
                for _ in range(excess_data):
                    self.move_data_to_sqlite()

    def move_data_to_sqlite(self):
        data = self.redis_client.rpop(self.main_queue_name)
        if data is not None:
            self.sqliteQueue.lpush(self.main_queue_name, data)

    def fill_redis_from_sqlite(self):
        if self.maxlen:
            current_length = self.redis_client.llen(self.main_queue_name)
            remaining_length = self.maxlen - current_length
            # 檢查 主要隊列 是否能夠填充
            if remaining_length > 0 and self.sqliteQueue.llen(self.main_queue_name) > 0:
                print(f'Transfer {remaining_length} from sqlite to redis')
                data_to_fill = self.sqliteQueue.multipleRpop(self.main_queue_name, remaining_length)
                for item in data_to_fill:
                    self.redis_client.lpush(self.main_queue_name, item)
