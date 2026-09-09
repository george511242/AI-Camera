# AICamera Inference Engine

VAC AI 攝影機的人物偵測、追蹤與 AI 推論服務。執行於 ARM64（Rockchip RK3566）
裝置上，利用 RKNN NPU 加速；同時也保留 ONNX / Torch 路徑作為 fallback。
資料事件透過 Redis queue 持久化後上傳到 WMS 後台。

Docker image：`aicamera_person_tracking:latest`（arm64）。在 docker-compose 內
listen `0.0.0.0:5555`，且需要 `privileged: true` + `/dev:/dev`（V4L2 相機）。

---

## 目錄

- [Tech Stack](#tech-stack)
- [專案結構](#專案結構)
- [整體資料流](#整體資料流)
- [推論模式](#推論模式)
  - [模式 1：離開關注區（leave-zone）](#模式-1離開關注區leave-zone)
  - [模式 2：離開關注區 + 年齡/性別](#模式-2離開關注區--年齡性別)
  - [模式 3：視線追蹤（eye-tracking）](#模式-3視線追蹤eye-tracking)
  - [模式 4：手機滑動偵測（phone-scrolling）](#模式-4手機滑動偵測phone-scrolling)
- [HTTP API](#http-api)
- [WebSocket 事件](#websocket-事件)
- [WMS 上傳流程](#wms-上傳流程)
- [模型清單](#模型清單)
- [Camera Adapter](#camera-adapter)
- [音訊](#音訊)
- [設定來源](#設定來源)
- [Docker / Obfuscation](#docker--obfuscation)
- [本機開發](#本機開發)

---

## Tech Stack

| 類別        | 套件                                              |
|-------------|---------------------------------------------------|
| Runtime     | Python `^3.10` (Poetry)                            |
| Web         | FastAPI `^0.104` + Uvicorn `^0.23` + Gunicorn `^21.2` |
| Inference   | RKNN Toolkit Lite 2（arm64 / RK3566 NPU）          |
|             | onnxruntime `1.22.0`（fallback / x86 dev）          |
|             | torch `2.3.1`（fallback）                          |
| CV          | OpenCV `4.11` headless                             |
| Frame I/O   | `m01-frame-processor`（自家 Rust wheel，含 MJPEG / V4L2 retry） |
| HTTP        | requests `2.28`                                   |
| WS          | websockets `15.0`                                  |
| Queue / IPC | redis `4.3`                                        |
| Logging     | Loguru `0.7`                                       |

---

## 專案結構

```
.
├── main.py                       ← 入口：啟動 inference / WMS sync thread + FastAPI lifespan
├── settings.py                   ← Pydantic settings（HOST / PORT / REDIS / BACKEND…）
├── config.json                   ← 本地推論設定（focusRegion、mode、thresholds）
├── pyproject.toml                ← Poetry
├── Dockerfile                    ← 含 obfuscated 階段（PyArmor 8.5）
├── Dockerfile.x86                ← 開發機用（無 RKNN）
├── entrypoint.sh                 ← 啟動：複製預設 wav、跑 RK3566 CPU scaling、啟 server
├── scaling_frequency.sh          ← RK3566 SoC CPU 頻率設定
├── asound.state                  ← 預設 ALSA mixer 狀態
├── no-phone.wav                  ← 預設「請勿滑手機」警告音
│
├── server/                       ← Gunicorn / Uvicorn launcher、lifespan、logging
├── routers/
│   ├── AppRouter.py              ← 推論控制 / 拍照 / 串流錄影 / 音檔上傳 / health
│   └── DeviceRouter.py           ← USB 音訊 / webcam 偵測 / 切換 source
│
├── classes/                      ← 核心：InferenceManager、InferenceConfig、Inference 工作流、
│                                   RedisQueue、SqliteQueue、ConnectionManager
├── factory/                      ← InferenceFactory（依 mode 載入對應 Inference）、
│                                   WebSocket / Result payload factory、draw util
├── dataModel/                    ← Pydantic schema（InferenceMode、S4MRecordSchema、
│                                   StreamRecordRequestSchema）
│
├── inference/                    ← ML pipeline：
│   ├── model/                    ← Model loader（RKNN / ONNX / Torch） + Workflow
│   ├── tracking/                 ← SORT / MaxIouFilter / Tracker state
│   ├── logger/                   ← RegionLogger / PhoneScrollingLogger /
│   │                              InteractiveLogger / PoseLogger
│   └── utils/                    ← 影像處理 / 座標轉換
│
├── camera_adapter.py             ← Abstract + LocalCameraAdapter（V4L2 via OpenCV）
├── frame_process.py              ← FrameProcessorWithThread：m01 wrapper + retry
├── yolo_part.py                  ← 舊版 darknet YOLO（多半已棄用）
├── upload_to_wms.py              ← WMS 註冊 / 批次上傳 / token 持久化
│
├── websocket/                    ← ConnectionManager（多 client 廣播）、Message model
├── repository/                   ← ConfigRepository（INI）、BackendRepository（呼叫 aicamera_web）
├── utils/                        ← ALSA / 設定讀取 / 事件 publisher / log helper
├── storage/                      ← debug 圖片 + temp SQLite queue
├── audio/                        ← runtime 音檔（runtime volume，預設來自根目錄）
└── model/                        ← runtime model volume（從外部 mount）
```

---

## 整體資料流

```
                          ┌───────────────────────────┐
        V4L2 /dev/videoN  │  FrameProcessorWithThread │
       ────────────────►  │  (m01-frame-processor)    │
                          └────────────┬──────────────┘
                                       ▼
                          ┌───────────────────────────┐
                          │   InferenceFactory        │
                          │   → 依 config 模式選擇    │
                          │     InferenceManager      │
                          └────────────┬──────────────┘
                                       ▼
                ┌───────────────────────┴────────────────────┐
                ▼                       ▼                    ▼
         RKNN models             SORT tracking         Region / Gaze /
         (RK3566 NPU)            (person ID)           PhoneScrolling Logger
                ▲                       │                    │
                │                       ▼                    ▼
                │              ┌──────────────────┐  ┌──────────────────┐
                │              │ ConnectionManager│  │ RedisQueue       │
                │              │ (WebSocket fan-  │  │ (sync_to_wms)    │
                │              │  out: inference, │  │  ↓               │
                │              │  result)         │  │ SqliteQueue      │
                │              └──────────────────┘  │ (fallback)       │
                │                                    └────────┬─────────┘
        ┌───────┴───────┐                                     ▼
        │  /model       │                              upload_to_wms
        │  (volume)     │                              （MD5 sig + token）
        └───────────────┘                                     │
                                                              ▼
                                                       WMS Cloud
```

---

## 推論模式

模式由 `InferenceConfig` 從 `aicamera_web` 後端 `/get_box_info` 拉取，再由
`InferenceFactory.load()` 實例化對應的 Inference workflow（**不需要重啟容器
即可切換模式**）。

### 追蹤共通原則

- 每 frame 偵測到的物件框會與上一 frame 的結果配對為同一人，距離不超過
  `maxBBoxDistance`；超過則視為新物件。
- 連續消失次數超過 `maxDisappearCount` 才視為真正離開，避免單 frame 抖動。
- ID 由 SORT 演算法管理。

### 模式 1：離開關注區（`leave-zone`）

- 觸發：物件離開設定的關注區。
- Log：

  ```json
  {
    "regionId": "<關注區 ID>",
    "personId": "<person UUID>",
    "enterDate": 1715900000.0,
    "leaveDate": 1715900012.5,
    "eventId": "<event UUID>",
    "mode": "leave-zone"
  }
  ```

### 模式 2：離開關注區 + 年齡/性別

- 在模式 1 之上追加年齡 / 性別模型（FPS 略低）。
- Log：

  ```json
  {
    "regionId": "...",
    "personId": "...",
    "age": 34.2,
    "gender": 1,         // 1 = 男, 0 = 女
    "enterDate": 1715900000.0,
    "leaveDate": 1715900012.5,
    "eventId": "...",
    "mode": "leave-zone-with-age-gender"
  }
  ```

> 另有 `leave-zone-with-facial-features` 模式（合併臉部特徵），仍在實驗
> 階段；由 `config.json` 內 `mode` 切換。

### 模式 3：視線追蹤（`eye-tracking`）

- 全畫面追蹤；視線模型最重，FPS 最低。
- 「看鏡頭」定義：偵測到的視線向量長度 < `shortGazeThreshold`。
- 連續 `minGazeCount` 次看鏡頭 → 進入 watching；再連續 `minGazeCount` 次未看 → 離開。
- 觸發：watching → not-watching 的狀態切換。
- Log：

  ```json
  {
    "directionId": 1,        // 1=左上、2=上、3=右上、…
    "personId": "...",
    "startDate": 1715900020.0,
    "endDate": 1715900025.3,
    "eventId": "...",
    "mode": "eye-tracking"
  }
  ```

### 模式 4：手機滑動偵測（`phone-scrolling`）

- 透過 pose keypoints（手腕、手肘角度、頭部低垂角度）推估「正在低頭看手機」。
- 偵測到時：擷取畫面並對人臉打馬賽克後上傳 WMS。
- 同時可觸發音檔播放（`audio/no-phone.wav`，由 `useDeviceVolume` 控制 USB
  speaker 音量）。
- Logger：`inference/logger/PhoneScrollingLogger.py`。

---

## HTTP API

### `AppRouter`

| Method  | Path                          | 用途                                |
|---------|-------------------------------|-------------------------------------|
| GET     | `/health`                     | Frame processor 健康檢查            |
| GET     | `/updateConfig`               | 觸發從後端 `/get_box_info` 拉設定   |
| GET     | `/stream/capture`             | 取最新 JPEG frame                   |
| POST    | `/fetch-settings`             | reload `config.ini`                 |
| POST    | `/fetch-signature`            | 重抓 WMS 簽章                       |
| POST    | `/stream/recording`           | 開始錄影 + 串流上傳                 |
| DELETE  | `/stream/recording`           | 停止錄影                            |
| GET     | `/stream/recording/status`    | 錄影中？                            |
| POST    | `/audio/upload`               | 上傳自訂 `.wav`（覆蓋預設 no-phone）|

### `DeviceRouter`（prefix `/api/v1/device`）

| Method  | Path                              | 用途                            |
|---------|-----------------------------------|---------------------------------|
| GET     | `/audio`                          | 列 USB 音訊裝置                 |
| GET     | `/audio/{index}`                  | 查特定裝置音量                  |
| PUT     | `/audio/{index}/{volume}`         | 設定音量 + 持久化到 ALSA        |
| GET     | `/webcams`                        | 列 `/dev/video*`                |
| PUT     | `/stream`                         | 切換攝影機來源                  |

---

## WebSocket 事件

`websocket/ConnectionManager.py` 支援多 client，每個事件 fan-out 到所有連線。

| Event type    | Payload                                                                 |
|---------------|-------------------------------------------------------------------------|
| `inference`   | 該 frame 的原始 detection / pose / gaze 結果，含 FPS、bbox、keypoints。 |
| `result`      | 已聚合的 logger 事件（leave-zone / eye-tracking / phone-scrolling…）。 |

前端 LiveView / Developer 頁面用這條 WS 即時顯示影像疊圖與事件。

---

## WMS 上傳流程

`upload_to_wms.py` 是獨立 thread：

1. **註冊**：第一次啟動 POST `{wms_base}/camera/register`（帶 device_id + MD5
   簽章），拿到 `accessToken` / `refreshToken`，存到 Redis 與 `config.ini`
   `[WMS]` section（雙寫，斷電可恢復）。
2. **持續上傳**：
   - Inference 結果先寫入 `RedisQueue('sync_to_wms')`。
   - 此 thread 每 `periodTimeSec`（預設 20s）或滿 `batch_size`（預設 30）
     批次 POST `{wms_base}/api/peopleCounter/record/add`。
3. **Token 過期**：401 → 用 refresh token 換新對。
4. **離線**：Redis 不可用時 fallback 寫 `storage/temp_queue.sqlite`，上限
   約 50k 筆 / N 天。
5. **簽章演算法**：`md5(payload + signature_secret + timestamp)`。

---

## 模型清單

放在 `/model` volume（host 目錄由 docker-compose 提供）：

| 檔名                                | 用途              | Backend      |
|-------------------------------------|-------------------|--------------|
| `yolo11n.rknn`                      | 人物偵測          | RKNN (NPU)   |
| `yolov8_pose.rknn`                  | Pose keypoints    | RKNN         |
| `yunet_n_640_640.rknn`              | 臉部偵測          | RKNN         |
| `age_model_1015-pa.rknn`            | 年齡              | RKNN         |
| `gender_model_0925-pa.rknn`         | 性別              | RKNN         |
| `pose-0701-quantize-100.rknn`       | 視線方向          | RKNN         |
| `mobilenet_v2.rknn`                 | 一般分類 fallback | RKNN         |

x86 開發環境會自動 fallback 到 ONNX / Torch 路徑（`inference/model/`）。

---

## Camera Adapter

- 抽象：`CameraAdapter`（`camera_adapter.py`）。
- 實作：`LocalCameraAdapter` 包 OpenCV `cv2.VideoCapture(/dev/videoN)`。
- 偏好設定：MJPEG codec，可旋轉，解析度可變（預設 640×480）。
- 斷線重試：每 2 秒一次，無限次。
- m01 wrapper：`FrameProcessorWithThread` 在背景跑 native MJPEG 抓圖
  （port 8080，container 內部），main thread 透過 buffer 取最新 frame。

> **必要 docker 設定**：`privileged: true` + `volumes: [- "/dev:/dev"]`。

---

## 音訊

- USB 音訊操作走 ALSA（`amixer` / `alsactl`），實作在 `utils/audio.py`。
- `asound.state` 在 image 內，container 啟動時 `alsactl restore` 回復 mixer 設定。
- 預設音檔 `no-phone.wav` 從 image copy 到 `/app/audio/`；可被 `/audio/upload`
  覆寫。
- 音量持久化：透過 `PUT /api/v1/device/audio/{idx}/{vol}` 設定，會寫
  `asound.state`。

---

## 設定來源

優先順序：env var > `config.ini`（持久化） > `config.json`（預設）。

- `settings.py`（Pydantic）—— 服務本身：`HOST`、`PORT`、`REDIS_HOST`、
  `BACKEND_HOST`、`MAX_FPS`、`LOG_LEVEL`、`RKNPU_PLATFORM`、
  `AGE_MODEL_VERSION`…
- `config.json`（repo 內）—— 推論預設：`focusRegion`、`mode`、
  `maxBBoxDistance`、`maxDisappearCount`、`minGazeCount`、`shortGazeThreshold`…
- `config.ini`（runtime）—— 持久化：
  - `[SYSTEM]` `device_id`
  - `[CAMERA]` `width` / `height` / `rotate`
  - `[API]` `wms_base` / `clientlog` / `mqtt_host`
  - `[WMS]` `accessToken` / `refreshToken`（雙寫 Redis）

`InferenceConfig` 會定期向 `aicamera_web` 拉 `/get_box_info` 覆寫 `mode`、
`focusRegion` 等熱配置。

Age 預設使用 `AGE_MODEL_VERSION=v2`。經人工選擇候選模型時可設定
`AGE_MODEL_VERSION=v4`；v4 使用 112x112 RGB preprocessing，並依
`RKNPU_PLATFORM=rk3566|rk3588` 載入各自的 RKNN artifact。未知版本或平台
會在模型載入前直接失敗。

---

## Docker / Obfuscation

Dockerfile 含一個 `obfuscated` stage，使用 **PyArmor 8.5.11** 對部分敏感
模組加密。`release-build.sh` 對此 image 強制 `--no-cache-filter=obfuscated`，
避免 buildx cache 重複用前次的混淆結果。

`Dockerfile.x86` 是純開發用 image，不裝 RKNN runtime（無 `librknnrt.so`）。

---

## 本機開發

```bash
# 安裝
poetry install --only main

# 在沒有 NPU 的機器：用 x86 Dockerfile
docker build -f Dockerfile.x86 -t aicamera-inference-x86 .

# 直接跑
poetry run python main.py
# 預設 port 5555
```

需要連線：

- Redis（`REDIS_HOST=localhost`）
- AICameraBackend（`BACKEND_HOST=http://localhost:5000`）

若沒有實體相機，可指向 RTSP / 影片檔做測試（修改 `LocalCameraAdapter`）。
