# AI Camera v1 距離準確度測試 SOP

這套測試直接在 Orange Pi NPU 上載入四個 RKNN 模型，不需要啟動 VAC inference HTTP 服務，也不使用 `m01_frame_processor`。

## 0. 測試目標

- 模型版本：`model/versions/v1_baseline`
- RK3588 實機：使用 `rknn/rk3588`
- RK3566 模型只能在 RK3566 實機驗證；RK3588 不能模擬 RK3566 NPU 執行結果
- Orange Pi：`orangepi@100.94.109.12`
- SSH key：`~/.ssh/orangepi_ed25519`
- 遠端專案：`/home/orangepi/AICameraInferenceEngine`

執行前記錄程式版本與模型雜湊：

```bash
ssh -i ~/.ssh/orangepi_ed25519 orangepi@100.94.109.12 \
  'cd /home/orangepi/AICameraInferenceEngine && git rev-parse HEAD && \
   sha256sum model/versions/v1_baseline/rknn/rk3588/*.rknn'
```

## 1. 準備資料集

下載並解壓縮：

- SCFace：根目錄需包含 `mugshot_frontal_cropped.txt` 與 `mugshot_frontal_cropped_all/`
- UTKFace：指定到 `crop_part1/`
- AFLW2000-3D：目錄內需有同名的 `.jpg` 與 `.mat`

在 WSL 執行：

```bash
cd /home/g2004/projects/VAC-AI-Camera-Source/vac-camera/vac-camera/Tools/distance_accuracy

python prep_dataset.py \
  --scface-dir "$HOME/datasets/SCface_database" \
  --utk-dir "$HOME/datasets/utkface-new/crop_part1" \
  --aflw-dir "$HOME/datasets/AFLW2000" \
  --out "$HOME/datasets/_distance_test_degraded" \
  --utk-sample 1000 \
  --heights '80:1.0,40:2.0,27:3.0'
```

衍生資料只產生一次。比較不同模型或平台時必須共用相同的圖片與 `manifest.jsonl`。

## 2. 上傳資料與工具

```bash
tar -C "$HOME/datasets" -czf /tmp/distance_test_degraded.tgz _distance_test_degraded

scp -i ~/.ssh/orangepi_ed25519 /tmp/distance_test_degraded.tgz \
  orangepi@100.94.109.12:/home/orangepi/

ssh -i ~/.ssh/orangepi_ed25519 orangepi@100.94.109.12 \
  'cd /home/orangepi && tar xzf distance_test_degraded.tgz'

scp -i ~/.ssh/orangepi_ed25519 run_eval_on_device.py \
  orangepi@100.94.109.12:/home/orangepi/run_eval_on_device.py
```

`--data-root` 會在 Orange Pi 上重新定位影像，因此不需要修改 manifest 內的 WSL 絕對路徑。

## 3. 執行 v1 RK3588 測試

先確認完整 VAC 程式與 v1 模型已部署到 `/home/orangepi/AICameraInferenceEngine`。測試期間不要同時啟動 inference server，以免競爭 NPU 或記憶體。

```bash
ssh -i ~/.ssh/orangepi_ed25519 orangepi@100.94.109.12
cd /home/orangepi/AICameraInferenceEngine
source ~/miniforge3/etc/profile.d/conda.sh
conda activate inference

PYTHONPATH=. python /home/orangepi/run_eval_on_device.py \
  --manifest /home/orangepi/_distance_test_degraded/manifest.jsonl \
  --data-root /home/orangepi/_distance_test_degraded \
  --out /home/orangepi/eval_v1_rk3588.jsonl \
  --version v1-rk3588 \
  --model-root model/versions/v1_baseline \
  --platform rk3588 \
  --age-margin .45 \
  --gender-margin .45 \
  --pose-margin .6
```

啟動時會列出平台、模型目錄和四個模型的完整路徑；任何模型不存在都會在推論前停止。

## 4. 取回與分析

```bash
cd /home/g2004/projects/VAC-AI-Camera-Source/vac-camera/vac-camera/Tools/distance_accuracy

scp -i ~/.ssh/orangepi_ed25519 \
  orangepi@100.94.109.12:/home/orangepi/eval_v1_rk3588.jsonl .

python analyze_results.py \
  --result eval_v1_rk3588.jsonl \
  --gender-positive male
```

建立不受 YuNet 影響的 Age/Gender/Head Pose model-only baseline：

```bash
cd /home/orangepi/vac-v1-distance-test

PYTHONPATH=. /home/orangepi/vac-gender-test/.venv/bin/python \
  run_eval_on_device.py \
  --manifest v1_distance_test/manifest.jsonl \
  --data-root v1_distance_test \
  --out eval_v1_rk3588_gt_crop.jsonl \
  --version v1-rk3588-gt-crop \
  --crop-source gt \
  --model-root model/versions/v1_baseline \
  --platform rk3588 \
  --age-margin .45 \
  --gender-margin .45 \
  --pose-margin .6
```

GT-crop 模式直接使用 manifest bbox，不載入 YuNet，也不評估 SCFace。

若要比較兩個版本：

```bash
python analyze_results.py \
  --before eval_before.jsonl \
  --after eval_after.jsonl \
  --gender-positive male

python plot_results.py \
  --before eval_before.jsonl \
  --after eval_after.jsonl \
  --gender-positive male \
  --out distance_accuracy_before_after.png
```

## 5. 計分規則

| 模型 | 資料集 | 主要指標 |
|---|---|---|
| YuNet | SCFace | IoU >= 0.5 且 score >= 0.75 的偵測成功率 |
| Age | UTKFace | 絕對誤差 <= 5 歲的端到端命中率 |
| Gender | UTKFace | Balanced Accuracy；UTKFace `0=male, 1=female`，預設模型輸出 `raw>=0.5 -> male` |
| Head Pose | AFLW2000-3D | Pitch/Yaw/Roll 環狀誤差都 <= 15 度的端到端命中率 |

YuNet 漏偵、IoU 不足、score 不足或執行錯誤時，下游 Age、Gender、Head Pose 一律算失敗並保留在分母。Gender mapping 必須由模型定義或獨立校準資料預先決定，不可使用測試集答案自動選擇方向。

## 6. RK3566 注意事項

將 `--platform` 改成 `rk3566` 只會選到 RK3566 模型，不能讓 RK3588 硬體變成 RK3566。RK3566 的最終效能與相容性仍需在 RK3566 NPU 實機執行；RKNN-Toolkit2 simulator 可用於轉換後的輸出比較，但不能取代實機效能測試。
