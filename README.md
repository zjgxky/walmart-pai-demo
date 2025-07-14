# Walmart销量预测 - 阿里云PAI MLOps Demo

基于阿里云PAI平台的端到端机器学习运维流程示例，展示从Databricks迁移到PAI的最佳实践。

## 项目概述

本项目演示了在阿里云PAI平台上构建完整MLOps流程，涵盖数据处理、模型训练、版本管理、自动部署和智能监控的全链路解决方案。

### 核心组件

- **MaxCompute**: 数据存储和大数据处理
- **DataWorks**: 数据开发和任务调度  
- **DSW**: 模型开发和训练
- **EAS**: 模型部署和在线服务
- **PAI Model Registry**: 模型版本管理

### 技术特点

基于6个核心技术问题的解决方案：

**1. Python脚本迁移**
- Databricks PySpark脚本 → MaxCompute PyODPS
- 保持Python开发习惯，API层面平滑迁移
- DataWorks节点替代Databricks Notebook调度

**2. 模型版本管理**  
- 自动Git版本追踪和模型血缘关系
- 一键复现任意历史版本训练
- PAI Model Registry深度集成

**3. 代码版本控制**
- Git版本信息自动记录到模型元数据
- 训练-预测代码血缘完整追踪
- 敏感配置文件安全隔离管理

**4. DSW脚本自动化**
- DataWorks调度DSW训练任务
- 端到端自动化：代码同步→训练→部署→监控
- 智能触发机制（数据更新/性能下降）

**5. 环境隔离**
- 开发/生产环境完全隔离
- MaxCompute项目级数据隔离
- EAS服务环境标识和权限控制

**6. 智能监控**
- 7x24小时服务健康和模型准确性监控
- 基于性能阈值的自动重训练决策
- 完整监控数据存储和趋势分析

## 项目结构

```
walmart-pai-demo/
├── README.md                           # 项目说明
├── requirements.txt                    # Python依赖
├── config.yaml                         # 配置文件模板
├── .gitignore                         # Git忽略文件
│
├── data/
│   └── Walmart.csv                     # 原始数据文件
│
├── sql/
│   └── create_tables.sql               # MaxCompute建表脚本
│
├── notebooks/                          # 模型训练
│   ├── upload_data.ipynb               # 数据上传
│   └── Walmart_Training.ipynb          # 版本追踪训练（Git集成）
│
├── dataworks/                          # DataWorks调度节点
│   ├── data_eda.py                     # 数据预处理
│   ├── feature_engineering.py          # 特征工程（VIF选择）
│   ├── batch_prediction.py             # 批量预测（血缘追踪）
│   ├── deploy_to_eas.py                # EAS自动部署
│   ├── monitor_performance.py          # 智能监控与重训练
│   └── automated_training_trigger.py   # 自动化训练触发
│
├── version_tracking/                   # 版本管理功能
│   ├── model_lineage.py               # 模型血缘追踪
│   ├── code_version_utils.py          # Git版本工具
│   └── reproduction_guide.md          # 训练复现指南
│
├── automation/                        # 自动化配置
│   ├── pipeline_config.yaml           # 端到端流程配置
│   └── dsw_integration.py             # DSW集成脚本
│
└── docs/                              # 文档
    ├── setup_guide.md                 # 环境配置指南
    ├── version_management.md          # 版本管理说明
    └── automation_guide.md            # 自动化使用指南
```

## 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone https://github.com/your-username/walmart-pai-demo.git
cd walmart-pai-demo

# 安装依赖
pip install -r requirements.txt

# 配置密钥（创建本地私有配置）
cp config.yaml config_local.yaml
# 编辑 config_local.yaml，填入真实的AccessKey信息
```

### 2. 执行流程

**数据处理**（DataWorks）
```sql
-- 1. 创建MaxCompute表
source sql/create_tables.sql

-- 2. 运行数据处理节点
walmart_data_eda -> walmart_feature_engineering
```

**模型训练**（DSW）
```python
# 3. 在DSW中运行版本追踪训练
jupyter notebook notebooks/Walmart_Training.ipynb
```

**自动化部署**（DataWorks）
```yaml
# 4. 运行自动化流程
batch_prediction -> deploy_to_eas -> monitor_performance
```

### 3. 验证结果

- **MaxCompute表**: 检查处理后的数据表
- **PAI Model Registry**: 查看注册的模型版本
- **EAS服务**: 测试在线预测API
- **监控面板**: 查看服务和模型性能

## 模型性能

| 模型 | 训练R² | 验证R² | 特征数 | 状态 |
|------|--------|--------|--------|------|
| Linear Regression | 0.9545 | **0.9431** | 58 | 已部署 |
| Elastic Net | 0.9524 | 0.9421 | 58 | 待选 |
| Random Forest | 0.8599 | 0.8354 | 58 | 待选 |

**服务性能**: 响应时间 < 100ms, 可用性 > 99.9%, 吞吐量 20+ QPS

## Databricks迁移对比

| 功能域 | Databricks | 阿里云PAI | 迁移复杂度 | 备注 |
|--------|------------|-----------|------------|------|
| 数据处理 | PySpark | PyODPS | 中等 | API适配 |
| 模型训练 | MLflow | DSW + Registry | 低 | 逻辑复用 |
| 版本管理 | MLflow Tracking | Git + 元数据 | 中等 | 更强追溯 |
| 自动化 | Jobs | DataWorks | 中等 | 语法调整 |
| 环境隔离 | Workspace | 项目隔离 | 低 | 权限重配 |
| 监控告警 | UI监控 | 智能监控 | 低 | 自动化提升 |

**总体评估**: 中等迁移成本，显著功能增强

## 版本管理特性

### Git版本追踪
```python
# 每次训练自动记录
{
  "git_commit_id": "abc123...",
  "git_branch": "main",
  "can_reproduce": true,
  "reproduction_steps": [...]
}
```

### 模型血缘追踪
```
训练数据版本 -> 训练代码版本 -> 模型版本 -> 预测代码版本 -> 预测结果
```

### 一键复现
```bash
# 自动生成复现脚本
bash reproduce_training.sh
```

## 自动化流程

### 完整Pipeline
```
代码同步 -> 数据处理 -> 特征工程 -> 模型训练 -> 批量预测 -> EAS部署 -> 智能监控
```

### 触发条件
- **数据更新**: 新数据到达24小时内
- **性能下降**: 模型R²下降超过5%
- **定时触发**: 每日/每周定时执行

### 智能决策
- 自动检测模型性能变化
- 基于阈值智能决策重训练
- 无人值守运维

## 安全配置

### 配置文件分层
```
config_local.yaml (本地私有) > 环境变量 > config.yaml (公开模板)
```

### Git保护
```gitignore
# 敏感信息保护
config_local.yaml
*.key
.env
```

## API使用

### 预测服务调用
```bash
curl -X POST "http://walmart-sales-prediction-api.pai-eas.aliyuncs.com/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "store": 1,
    "temperature": 75.5,
    "fuel_price": 3.2,
    "holiday_flag": 0,
    "unemployment": 8.1,
    "cpi": 210.3
  }'
```

### 返回结果
```json
{
  "predicted_weekly_sales": 52634.78,
  "model_version": "v_20250710_073527",
  "prediction_time": "2025-01-30T10:30:00Z"
}
```

## 文档

- [环境配置指南](docs/setup_guide.md) - 详细环境搭建步骤
- [版本管理说明](docs/version_management.md) - Git版本追踪使用
- [自动化指南](docs/automation_guide.md) - 端到端自动化配置

## 技术支持

- **PAI平台文档**: https://help.aliyun.com/product/30347.html
- **技术咨询**: 阿里云PAI技术支持
- **代码问题**: GitHub Issues

## 许可证

MIT License