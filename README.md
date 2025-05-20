# IMDB BERT Project

BERT-based sentiment analysis on IMDB Movie Review Dataset

## 專案結構

```
imdb_bert_project/
├── configs/               # 配置文件
│   ├── data_config.yaml   # 資料處理配置
│   ├── smoke_test.yaml    # 快速測試配置
│   └── train.yaml         # 訓練配置
├── data/                  # 資料目錄
│   ├── cache/             # 處理緩存
│   ├── processed/         # 處理後資料
│   ├── raw/               # 原始資料
│   └── visualizations/    # 資料視覺化
├── models/                # 模型目錄
├── notebooks/             # Jupyter筆記本
├── src/                   # 源代碼
│   ├── data/              # 資料處理模組
│   ├── models/            # 模型定義
│   └── utils/             # 工具函數
└── tests/                 # 測試代碼
```

## 專案依賴管理

本專案使用 `pyproject.toml` 作為依賴管理的唯一來源，不再使用 requirements.txt。

### （可選）如果想要自行安裝依賴

- 安裝基本依賴：
  ```bash
  pip install .
  ```

- 開發環境安裝（包含所有依賴）：
  ```bash
  pip install -e '.[all]'
  ```

- 安裝特定類型的依賴：
  ```bash
  # 僅開發工具
  pip install -e '.[dev]'
  
  # 僅 Jupyter Notebook 
  pip install -e '.[notebook]'
  
  # 僅實驗追蹤工具
  pip install -e '.[tracking]'
  ```

### （推薦）使用專案自動化工具Makefile

您可以使用 Makefile 中的 `make setup` 指令建立虛擬環境，並安裝本專案需要的所有依賴：

```bash
make setup
```

## 資料處理

下載並準備 IMDB 資料集：

```bash
make data
```

## 模型訓練與評估

訓練模型：

```bash
make train
```

評估模型：

```bash
make evaluate
```

進行預測：

```bash
make predict
```

互動式預測：

```bash
make predict_interactive
```

## 快速測試

執行煙霧測試（快速訓練與評估）：

```bash
make test_cycle_smoke
```

## 開發工具

程式碼格式化：

```bash
make format
```

Linting：

```bash
make lint
```

執行測試：

```bash
make test
```

執行所有檢查：

```bash
make test_all
```
