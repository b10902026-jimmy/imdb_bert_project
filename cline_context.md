# 專案進度與變更記錄

## 已完成的工作

### 依賴管理更新 (2025/5/20)
1. 移除 requirements.txt：
   - 完全轉移至 pyproject.toml 作為依賴管理的唯一來源
   - 確保所有先前在 requirements.txt 的依賴都納入 pyproject.toml
   - 添加加入 "accelerate" 和 "plotly" 等缺少的依賴
   - 新增 `[all]` 選項依賴組，包含開發、筆記本和追蹤工具

2. 更新 Makefile：
   - 將 `pip install -r requirements.txt` 更改為 `pip install -e '.[all]'`
   - 所有依賴現在通過 pyproject.toml 管理和安裝
   
3. 更新 README.md：
   - 添加了新的依賴管理說明章節
   - 提供了不同安裝選項的指令
   - 新增了專案結構說明

### Makefile 更新 (2025/5/20)
1. 修改 `setup` 指令：
   - 新增虛擬環境激活說明訊息
   - 在虛擬環境已存在時自動更新依賴項
   - 明確說明不會自動激活虛擬環境
   - 新增技術限制說明（子shell執行、環境變數不持續等）

2. 新增指令：
   - `predict_interactive`: 互動式預測模式
   - `test_all`: 執行所有測試和程式碼檢查
   - `train_smoke`: 快速訓練測試
   - `evaluate_smoke`: 評估快速測試模型
   - `test_cycle_smoke`: 完整快速測試週期
   - `test_data`: 資料處理流程測試
   - `test_config`: 配置檔案驗證測試
   - `test_quick`: 快速模型初始化和指標計算測試

### README.md 更新 (2025/5/20)
1. 新增「Quick Start Guide」表格
2. 重新組織安裝說明：
   - 使用項目符號列出 setup 功能
   - 強調需手動激活虛擬環境
   - 提醒編輯 .env 文件

3. 擴充使用說明：
   - 詳細說明每個主要操作
   - 提供 Makefile 指令和對應 Python 指令
   - 新增快速測試說明

## 待辦事項

1. 測試跨平台相容性
2. 考慮新增 CI/CD 流程
3. 評估是否需要更多測試案例

## 變更記錄

| 日期       | 檔案               | 變更描述                          |
|------------|-------------------|-----------------------------------|
| 2025/5/20 | pyproject.toml    | 更新依賴管理，新增缺失依賴，添加all組 |
| 2025/5/20 | requirements.txt  | 已移除，依賴管理轉移至pyproject.toml |
| 2025/5/20 | Makefile          | 更新setup指令使用pyproject.toml |
| 2025/5/20 | README.md         | 新增專案結構說明和依賴管理指南 |
| 2025/5/20 | tests/test_training.py | 新增資料、配置和模型快速測試功能  |
| 2025/5/20 | Makefile          | 更新 setup 指令，新增多個測試指令 |
| 2025/5/20 | README.md         | 重組內容，新增快速指南表格        |
