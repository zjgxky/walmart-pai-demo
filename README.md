# walmart-pai-demo
Walmart sales forecasting demo using Alibaba Cloud PAI

# Walmart销量预测 - 阿里云PAI MLOps Demo

基于阿里云PAI平台的端到端机器学习运维流程示例。

## 📊 项目概述

使用Walmart销量数据，展示从数据处理到模型部署的完整MLOps流程。

### 核心组件
- **MaxCompute**: 数据存储和处理
- **DataWorks**: 数据开发和任务调度  
- **DSW**: 模型训练和开发
- **EAS**: 模型部署和在线服务

### 数据流程
原始数据 → 数据预处理(DataWorks) → 特征工程(DataWorks) → 模型训练(DSW) → 批量预测(DataWorks) → 模型部署(EAS) → 监控反馈(EAS)

## 📁 项目结构
walmart-pai-demo/
├── README.md                           # 项目说明
├── requirements.txt                    # Python依赖
├── config.yaml                         # 配置文件
│
├── data/
│   └── Walmart.csv                     # 原始数据文件
│
├── sql/
│   └── create_tables.sql               # 建表SQL脚本
│
├── notebooks/                          # 📍新功能：版本可追踪的训练
│   ├── upload_data.ipynb               # 数据上传笔记本
│   └── Walmart_Training.ipynb          # 🆕 增强版训练（集成Git版本管理）
│
├── dataworks/                          # 📍新功能：自动化调度节点
│   ├── data_eda.py                     # 数据处理节点
│   ├── feature_engineering.py          # 特征工程节点
│   ├── batch_prediction.py             # 批量预测节点
│   ├── deploy_to_eas.py                # EAS部署节点
│   ├── monitor_performance.py          # 监控节点
│   └── automated_training_trigger.py   # 🆕 自动化训练触发器
│
├── version_tracking/                   # 🆕 版本管理功能
│   ├── model_lineage.md               # 模型血缘关系文档
│   └── reproduction_guide.md          # 复现指南
│
└── automation/                        # 🆕 自动化功能
    ├── dsw_integration.py             # DSW集成脚本
    └── pipeline_config.yaml           # 自动化流程配置

## 🚀 快速开始

### 1. 克隆仓库
```bash
git clone https://github.com/你的用户名/walmart-pai-demo.git
cd walmart-pai-demo
pip install -r requirements.txt
2. 配置环境
编辑 config.yaml 文件，填入你的实际配置信息。
3. 运行流程

数据处理: 在DataWorks中运行数据处理节点
模型训练: 在DSW中运行训练笔记本
模型部署: 运行部署和监控节点

📈 模型性能

Linear Regression: R² = 0.9431
Elastic Net: R² = 0.9421
Random Forest: R² = 0.8354

