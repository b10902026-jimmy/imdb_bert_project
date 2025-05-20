# IMDb影評情感分析基於BERT模型 - 課堂報告

## 一、專案概述
- **任務目標**：IMDb影評二分類情感分析（正面/負面）
- **模型架構**：BERT-base-uncased + 自定義分類層
- **技術亮點**：
  - Attention權重可視化（`src/utils/visualization.py`）
  - 基於SHAP的解釋性分析模組（`src/utils/explanation_manager.py`）
  - 動態長度處理（`src/data/processor.py`）

## 二、資料分析
### 資料集特性
- 50,000條平衡樣本（25k正面/25k負面）
- 平均長度：234 tokens（經BERT tokenizer處理後）
- 預處理流程：
  ```python
  # 來自 src/data/processor.py
  def clean_text(text):
      text = re.sub(r'<[^>]+>', '', text)  # 移除HTML標籤
      text = re.sub(r'[\W_]+', ' ', text)  # 保留字母數字
      return text.lower().strip()
  ```

## 三、模型實作
### 關鍵組件
- **BERT架構調整**（`src/models/bert_classifier.py`）：
  ```python
  class BertClassifier(nn.Module):
      def __init__(self, bert_model_name, num_classes=2, dropout=0.1):
          super().__init__()
          self.bert = BertModel.from_pretrained(bert_model_name)
          self.dropout = nn.Dropout(dropout)
          self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
  ```
- **訓練配置**（`configs/train.yaml`）：
  ```yaml
  training:
    batch_size: 16
    learning_rate: 2e-5
    max_seq_length: 256
    num_epochs: 3
  ```

## 四、實驗流程
### 訓練監控
- 早停機制（patience=2）
- 學習率線性warmup（10%訓練步數）
- 混合精度訓練（FP16）

### 評估指標
- 加權F1-score實現（`src/utils/metrics.py`）：
  ```python
  def weighted_f1(y_true, y_pred):
      return f1_score(y_true, y_pred, average='weighted')
  ```

## 五、結果分析
### 測試集表現
| Metric       | Score  |
|--------------|--------|
| Accuracy     | 0.923  |
| Weighted F1  | 0.923  |
| AUC          | 0.972  |

### Attention可視化案例
![Attention Weights](models/results/visualizations/test_attention_weights_0.png)
*圖：模型對關鍵情感詞彙的注意力分布*

## 六、解釋性分析
### 樣本級預測解讀
```python
# 來自 predictions/interactive_explanations/explanation_report.md
解釋樣本："This movie is terribly wonderful!"
- 正面預測概率：72%
- 矛盾詞彙分析：
  - "wonderful" (正向貢獻 +0.38)
  - "terribly" (負向貢獻 -0.29)
```

## 七、結論與未來工作
- **實務價值**：可整合至影評聚合平台
- **改進方向**：
  - 領域適應預訓練（Domain-Adaptive Pretraining）
  - 集成對抗訓練（Adversarial Training）
  - 部署優化（ONNX轉換）

## 附錄
- 完整程式碼：https://github.com/[your_repo]
- 互動式演示：`python src/predict.py --interactive`
