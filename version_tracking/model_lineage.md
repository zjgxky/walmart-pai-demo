# 模型血缘关系追踪

## 📊 当前生产模型

### linear_regression_v_20250713_154521 ✅ 
- **Git Commit**: `a1b2c3d4` (2025-07-13 15:45:21)
- **训练数据**: walmart_train_vif (58 features)
- **性能指标**: 训练R²=0.9545, 验证R²=0.9431
- **部署状态**: 已部署到EAS
- **可复现**: ✅ 是
- **复现命令**: `git checkout a1b2c3d4`

## 📈 历史模型版本

### elastic_net_v_20250713_154530
- **Git Commit**: `a1b2c3d4` (2025-07-13 15:45:30)
- **性能指标**: 训练R²=0.9524, 验证R²=0.9421
- **状态**: 候选模型
- **可复现**: ✅ 是

### random_forest_v_20250713_154543  
- **Git Commit**: `a1b2c3d4` (2025-07-13 15:45:43)
- **性能指标**: 训练R²=0.8599, 验证R²=0.8354
- **状态**: 已归档
- **可复现**: ✅ 是

### 历史版本（已废弃）
- `v_20250712_093215` - 性能不达标
- `v_20250711_140832` - 特征工程错误
- `v_20250710_073527` - 初始版本

## 🔄 数据血缘关系

```
walmart_sales_raw (原始数据)
    ↓ [walmart_data_eda.py]
walmart_processed_data (清洗后)
    ↓ [walmart_feature_engineering.py] 
walmart_train_vif (训练集) + walmart_test_vif (测试集)
    ↓ [Walmart_Training.ipynb @ a1b2c3d4]
模型文件 (linear_regression_v_20250713_154521)
    ↓ [deploy_to_eas.py]
EAS服务 (walmart-sales-prediction-api)
```

## 📋 依赖追踪

### 代码依赖
- **训练脚本**: `notebooks/Walmart_Training.ipynb`
- **数据处理**: `dataworks/walmart_data_eda.py`
- **特征工程**: `dataworks/walmart_feature_engineering.py`
- **配置文件**: `config.yaml`
- **依赖包**: `requirements.txt`

### 数据依赖
- **原始数据源**: MaxCompute表 `walmart_sales_raw`
- **特征数据**: MaxCompute表 `walmart_train_vif`
- **数据版本**: 基于表的last_modified_time追踪

### 环境依赖
- **Python版本**: 3.8+
- **核心库**: scikit-learn>=1.0.0, pandas>=1.3.0
- **计算资源**: DSW 2核4GB实例
- **存储**: MaxCompute + OSS

## 🎯 性能基线

| 模型类型 | 训练R² | 验证R² | 训练时间 | 模型大小 | 状态 |
|---------|--------|--------|----------|----------|------|
| Linear Regression | 0.9545 | 0.9431 | 2分钟 | 15KB | ✅ 生产 |
| Elastic Net | 0.9524 | 0.9421 | 5分钟 | 12KB | 待定 |
| Random Forest | 0.8599 | 0.8354 | 15分钟 | 2MB | 已归档 |

## 🚨 性能监控阈值

- **重训练触发**: 验证R² < 0.90 或性能下降 > 5%
- **服务告警**: 响应时间 > 200ms 或错误率 > 1%
- **数据漂移**: 特征分布变化 > 10%

## 📝 变更日志

### 2025-07-13
- ✅ 集成Git版本管理功能
- ✅ 增强模型元数据追踪
- ✅ 实现自动化训练触发

### 2025-07-12  
- 🔧 优化特征工程流程
- 📊 增加VIF特征选择

### 2025-07-11
- 🎯 初始模型训练完成
- 🚀 首次部署到EAS

## 🔄 复现指南

### 完整复现当前生产模型
```bash
# 1. 克隆仓库
git clone https://github.com/你的用户名/walmart-pai-demo.git
cd walmart-pai-demo

# 2. 切换到生产版本
git checkout a1b2c3d4

# 3. 安装依赖
pip install -r requirements.txt

# 4. 运行训练
jupyter notebook notebooks/Walmart_Training.ipynb
```

### 验证模型一致性
```python
# 加载生产模型
import joblib
model = joblib.load('models/linear_regression_v_20250713_154521/model.pkl')

# 检查模型参数
print(f"模型系数数量: {len(model.coef_)}")
print(f"模型截距: {model.intercept_}")
```

## 📊 审计信息

- **最后更新**: 2025-07-13 15:45:21
- **更新人**: 自动化系统 (Git Commit: a1b2c3d4)
- **审核状态**: 已通过
- **合规检查**: ✅ 通过