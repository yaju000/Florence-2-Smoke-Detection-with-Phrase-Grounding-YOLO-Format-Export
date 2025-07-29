# Florence-2 Smoke Detection with Phrase Grounding & YOLO Format Export

本專案使用 [Microsoft Florence-2-large](https://huggingface.co/microsoft/Florence-2-large) 模型進行影片逐格分析，結合 Phrase Grounding 技術偵測指定物體（如煙霧 smoke），並以 YOLO 標註格式儲存結果，同時輸出可視化影片與對應的標註影像資料。

## 🔍 功能說明

* 使用 Florence-2 模型進行 Open Vocabulary 物件偵測（Phrase Grounding）
* 支援影片逐幀處理，顯示偵測結果與 bounding boxes
* 輸出：

  * 標註後影片
  * 對應幀圖像（JPEG）
  * YOLO 格式的 `labels` 標註檔案（TXT）

## 📦 環境需求

請確保你使用的 GPU 支援 FP16 並已安裝以下套件：

```bash
pip install torch torchvision transformers opencv-python pillow ultralytics
```

此外需確保：

* 已安裝 CUDA 並可使用 `torch.cuda.is_available() == True`
* 影片輸入與輸出目錄存在或可建立

## 🚀 使用方式

### 1. 下載 Florence-2 模型

程式會自動從 HuggingFace 下載 `microsoft/Florence-2-large` 模型與處理器。

### 2. 修改影片路徑與目標文字

請於主程式中設定輸入影片路徑與欲偵測的文字：

```python
video_path = "/workspace/video/video_test.mp4"
text_input = "smoke"
```

你也可以替換為其他如 `"car"`、`"fire"`、`"person"` 等開放詞彙。

### 3. 執行主程式

直接執行 `.py` 檔案即可：

```bash
python florence_phrase_grounding.py
```

## 📁 輸出結構

成功執行後，將在以下路徑產生對應結果：

```
/workspace/project/
├── labels/                         # YOLO 格式標註 TXT 檔
│   └── test_video_frame_XXX.txt
├── images/                         # 每幀影像 JPEG
│   └── test_video_frame_XXX.jpg
└── microsoft/Florence-2-large_test_video_output.mp4  # 可視化輸出影片
```

YOLO 標註格式為：

```
<class_id> <x_center> <y_center> <width> <height>
```

所有值皆為相對座標（0\~1），可直接用於 YOLOv5/v8 訓練。

## ✨ 技術亮點

* 結合 HuggingFace Florence-2 與 Open Vocabulary Detection 概念
* 支援 GPU FP16 推論以加快速度
* 自動轉換並匯出 YOLO 格式標註，利於後續模型訓練與強化學習

## 📌 注意事項

* 若無偵測到指定文字相關物件（如 smoke），則該幀不會輸出標註框
* 若 GPU 記憶體不足，請調降影片解析度或改用更小模型
* 本模型尚不支援直接影像分割，僅提供 bounding box 偵測

## 📚 參考資源

* [Florence-2 Model Card](https://huggingface.co/microsoft/Florence-2-large)
* [Ultralytics YOLO Format Docs](https://docs.ultralytics.com/datasets/format/#txt)

