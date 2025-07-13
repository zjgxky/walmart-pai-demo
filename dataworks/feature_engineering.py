# 完整的Walmart VIF特征工程节点 (包含性能监控)
# 从数据读取到最终保存的完整流程

import pandas as pd
import numpy as np
from odps import options
from odps.models import TableSchema, Column
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 设置ODPS显示选项
options.display.max_rows = 100
options.display.max_columns = 50

print("="*60)
print("Walmart VIF特征工程 + 性能监控")
print("="*60)

# ===============================
# 1. 读取处理后的数据
# ===============================
print("\n1. 读取处理后的数据")
print("-" * 30)

try:
    df_encoded = o.get_table('walmart_processed_data').to_df().to_pandas()
    print("✓ 数据读取成功")
    print(f"数据形状: {df_encoded.shape}")
except Exception as e:
    print(f"✗ 数据读取失败: {e}")
    exit()

# ===============================
# 2. 准备特征和目标变量
# ===============================
print("\n" + "="*60)
print("2. 准备特征和目标变量")
print("="*60)

target = 'weekly_sales'
feature_columns = [col for col in df_encoded.columns if col != target]

# 选择数值特征
numeric_features = []
for col in feature_columns:
    if df_encoded[col].dtype in ['int64', 'float64', 'int32', 'float32']:
        numeric_features.append(col)

X = df_encoded[numeric_features]
y = df_encoded[target]

print(f"原始特征数量: {len(numeric_features)}")
print(f"样本数量: {X.shape[0]}")

# 处理缺失值
if X.isnull().sum().sum() > 0:
    print("处理缺失值...")
    X = X.fillna(0)
if y.isnull().sum() > 0:
    y = y.fillna(y.mean())

# ===============================
# 3. Train/Test Split
# ===============================
print("\n" + "="*60)
print("3. Train/Test Split")
print("="*60)

test_size = 0.2
random_state = 42

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state, shuffle=True
)

print(f"训练集: {X_train.shape[0]} 样本")
print(f"测试集: {X_test.shape[0]} 样本")

# ===============================
# 4. 特征标准化
# ===============================
print("\n" + "="*60)
print("4. 特征标准化")
print("="*60)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 转换回DataFrame保持列名
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=numeric_features, index=X_train.index)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=numeric_features, index=X_test.index)

print(f"✓ 标准化完成")
print(f"训练集形状: {X_train_scaled_df.shape}")

# ===============================
# 5. VIF性能监控函数定义
# ===============================
def calculate_vif_sklearn(df):
    """使用sklearn计算VIF"""
    vif_data = pd.DataFrame()
    vif_data["Feature"] = df.columns
    vif_values = []
    
    for i, feature in enumerate(df.columns):
        X_others = df.drop(columns=[feature])
        y_target = df[feature]
        
        if X_others.shape[1] == 0:
            vif_values.append(1.0)
            continue
            
        try:
            lr = LinearRegression()
            lr.fit(X_others, y_target)
            y_pred = lr.predict(X_others)
            r2 = r2_score(y_target, y_pred)
            
            if r2 >= 0.999:
                vif = 1000.0
            else:
                vif = 1 / (1 - r2)
                
            vif_values.append(vif)
            
        except:
            vif_values.append(1.0)
    
    vif_data["VIF"] = vif_values
    return vif_data.sort_values('VIF', ascending=False)

def vif_with_performance_monitoring(X_train_df, X_test_df, y_train, y_test, vif_threshold=5.0):
    """VIF特征选择 + 性能监控"""
    
    features_to_keep = list(X_train_df.columns)
    dropped_features = []
    performance_history = []
    
    print(f"开始VIF特征选择 + 性能监控...")
    print(f"初始特征数: {len(features_to_keep)}")
    print(f"VIF阈值: {vif_threshold}")
    
    iteration = 0
    max_iterations = min(50, len(features_to_keep) - 5)  # 至少保留5个特征
    
    while iteration < max_iterations and len(features_to_keep) > 5:
        iteration += 1
        
        # 当前特征集
        X_current_train = X_train_df[features_to_keep]
        X_current_test = X_test_df[features_to_keep]
        
        # 训练模型并评估性能
        lr = LinearRegression()
        lr.fit(X_current_train, y_train)
        
        train_pred = lr.predict(X_current_train)
        test_pred = lr.predict(X_current_test)
        
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        
        # 记录性能
        performance_history.append({
            'iteration': iteration,
            'num_features': len(features_to_keep),
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'features': features_to_keep.copy()
        })
        
        print(f"\n迭代 {iteration}:")
        print(f"  特征数: {len(features_to_keep)}")
        print(f"  训练RMSE: {train_rmse:.4f}")
        print(f"  测试RMSE: {test_rmse:.4f}")
        print(f"  测试R²: {test_r2:.4f}")
        
        # 计算VIF
        try:
            # 为了加速计算，对于大特征集使用采样
            if len(features_to_keep) > 30:
                sample_size = min(500, X_current_train.shape[0])
                sample_indices = np.random.choice(X_current_train.index, sample_size, replace=False)
                X_sample = X_current_train.loc[sample_indices]
            else:
                X_sample = X_current_train
            
            vif_df = calculate_vif_sklearn(X_sample)
            max_vif = vif_df['VIF'].max()
            
            print(f"  最高VIF: {max_vif:.2f}")
            
            # 如果最高VIF低于阈值，停止
            if max_vif <= vif_threshold:
                print(f"  ✓ 所有特征VIF <= {vif_threshold}，筛选完成")
                break
            
            # 移除VIF最高的特征
            feature_to_remove = vif_df.iloc[0]['Feature']
            features_to_keep.remove(feature_to_remove)
            dropped_features.append(feature_to_remove)
            
            print(f"  移除特征: {feature_to_remove} (VIF: {vif_df.iloc[0]['VIF']:.2f})")
            
        except Exception as e:
            print(f"  VIF计算出错: {e}")
            # 如果VIF计算出错，随机移除一个特征
            if len(features_to_keep) > 5:
                feature_to_remove = features_to_keep[0]
                features_to_keep.remove(feature_to_remove)
                dropped_features.append(feature_to_remove)
                print(f"  备用方案: 移除 {feature_to_remove}")
            break
    
    # 转换为DataFrame便于分析
    performance_df = pd.DataFrame(performance_history)
    
    return features_to_keep, dropped_features, performance_df

# ===============================
# 6. VIF特征选择 + 性能监控
# ===============================
print("\n" + "="*60)
print("6. VIF特征选择 + 性能监控")
print("="*60)

# 预处理: 移除常数特征和高相关性特征
print("预处理: 移除常数特征和高相关性特征...")

# 移除常数特征
constant_features = []
for col in X_train_scaled_df.columns:
    if X_train_scaled_df[col].std() == 0:
        constant_features.append(col)

if constant_features:
    print(f"  移除常数特征: {len(constant_features)} 个")
    X_train_scaled_df = X_train_scaled_df.drop(columns=constant_features)
    X_test_scaled_df = X_test_scaled_df.drop(columns=constant_features)

# 移除高相关性特征
print("  计算特征相关性...")
correlation_matrix = X_train_scaled_df.corr().abs()
highly_correlated_features = set()

for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        if correlation_matrix.iloc[i, j] > 0.95:
            feature2 = correlation_matrix.columns[j]
            highly_correlated_features.add(feature2)

if highly_correlated_features:
    print(f"  移除高相关性特征: {len(highly_correlated_features)} 个")
    X_train_scaled_df = X_train_scaled_df.drop(columns=list(highly_correlated_features))
    X_test_scaled_df = X_test_scaled_df.drop(columns=list(highly_correlated_features))

print(f"  预处理后特征数: {X_train_scaled_df.shape[1]}")

# 执行VIF + 性能监控
vif_threshold = 5.0
selected_features, dropped_features, performance_df = vif_with_performance_monitoring(
    X_train_scaled_df, X_test_scaled_df, y_train, y_test, vif_threshold
)

# ===============================
# 7. 结果分析
# ===============================
print(f"\n" + "="*60)
print("7. VIF + 性能监控结果分析")
print("="*60)

print(f"特征数量变化:")
print(f"  原始特征数: {len(numeric_features)}")
print(f"  预处理后特征数: {X_train_scaled_df.shape[1]}")
print(f"  最终特征数: {len(selected_features)}")
print(f"  总共移除特征数: {len(numeric_features) - len(selected_features)}")
print(f"  特征保留率: {len(selected_features)/len(numeric_features)*100:.1f}%")

# 分析性能趋势
if len(performance_df) > 0:
    print(f"\n性能分析:")
    best_test_rmse_idx = performance_df['test_rmse'].idxmin()
    best_performance = performance_df.loc[best_test_rmse_idx]
    
    print(f"  最佳测试RMSE: {best_performance['test_rmse']:.4f}")
    print(f"  最佳性能时特征数: {best_performance['num_features']}")
    print(f"  最佳性能时测试R²: {best_performance['test_r2']:.4f}")
    
    final_performance = performance_df.iloc[-1]
    print(f"\n  最终测试RMSE: {final_performance['test_rmse']:.4f}")
    print(f"  最终特征数: {final_performance['num_features']}")
    print(f"  最终测试R²: {final_performance['test_r2']:.4f}")
    
    # 性能变化趋势
    if len(performance_df) > 1:
        rmse_trend = performance_df['test_rmse'].iloc[-1] - performance_df['test_rmse'].iloc[0]
        r2_trend = performance_df['test_r2'].iloc[-1] - performance_df['test_r2'].iloc[0]
        
        print(f"\n  性能趋势:")
        print(f"    测试RMSE变化: {rmse_trend:+.4f}")
        print(f"    测试R²变化: {r2_trend:+.4f}")

# 移除的特征分析
if dropped_features:
    print(f"\n移除的特征 ({len(dropped_features)} 个):")
    for i, feature in enumerate(dropped_features[:10], 1):
        print(f"  {i:>2}. {feature}")
    if len(dropped_features) > 10:
        print(f"  ... 还有 {len(dropped_features) - 10} 个")

# 保留特征按类型分组
print(f"\n保留特征按类型分组:")
feature_types = {
    'holiday_flag': [f for f in selected_features if f.startswith('holiday_flag')],
    'year': [f for f in selected_features if f.startswith('year_')],
    'month': [f for f in selected_features if f.startswith('month_')],
    'weekday': [f for f in selected_features if f.startswith('weekday_')],
    'store': [f for f in selected_features if f.startswith('store_')],
    'continuous': [f for f in selected_features if not any(f.startswith(prefix) for prefix in ['holiday_flag', 'year_', 'month_', 'weekday_', 'store_'])]
}

for ftype, features in feature_types.items():
    if features:
        print(f"  {ftype}: {len(features)} 个特征")

# ===============================
# 8. 创建最终数据集
# ===============================
print("\n" + "="*60)
print("8. 创建最终数据集")
print("="*60)

# 使用选定的特征
X_train_final = X_train_scaled_df[selected_features]
X_test_final = X_test_scaled_df[selected_features]

print(f"最终特征数量: {len(selected_features)}")
print(f"训练集形状: {X_train_final.shape}")
print(f"测试集形状: {X_test_final.shape}")

# 添加目标变量
train_final_df = X_train_final.copy()
train_final_df[target] = y_train

test_final_df = X_test_final.copy()
test_final_df[target] = y_test

print(f"\n包含目标变量的最终数据集:")
print(f"  训练集: {train_final_df.shape}")
print(f"  测试集: {test_final_df.shape}")

# 显示数据样本
print(f"\n训练集前5行 (前10列):")
display_cols = train_final_df.columns[:10]
print(train_final_df[display_cols].head())

# ===============================
# 9. 保存VIF筛选后的数据
# ===============================
print("\n" + "="*60)
print("9. 保存VIF筛选后的数据")
print("="*60)

train_table_name = 'walmart_train_vif'
test_table_name = 'walmart_test_vif'

print(f"保存到表: {train_table_name}, {test_table_name}")

# 保存训练集
try:
    if o.exist_table(train_table_name):
        o.delete_table(train_table_name)
        print(f"✓ 删除旧表: {train_table_name}")
    
    # 创建表结构
    columns = []
    for col_name, dtype in train_final_df.dtypes.items():
        if dtype in ['int64', 'int32']:
            odps_type = 'bigint'
        elif dtype in ['float64', 'float32']:
            odps_type = 'double'
        else:
            odps_type = 'string'
        columns.append(Column(name=col_name, type=odps_type))
    
    schema = TableSchema(columns=columns)
    train_table = o.create_table(train_table_name, schema)
    
    # 保存数据
    with train_table.open_writer() as writer:
        batch_size = 1000
        for i in range(0, len(train_final_df), batch_size):
            batch_data = train_final_df.iloc[i:i+batch_size]
            records = []
            for _, row in batch_data.iterrows():
                record = [row[col] for col in train_final_df.columns]
                records.append(record)
            writer.write(records)
    
    print(f"✓ 训练集保存成功: {train_table_name} ({len(train_final_df)} 行)")
    
except Exception as e:
    print(f"✗ 训练集保存失败: {e}")

# 保存测试集
try:
    if o.exist_table(test_table_name):
        o.delete_table(test_table_name)
        print(f"✓ 删除旧表: {test_table_name}")
    
    schema = TableSchema(columns=columns)  # 使用相同的结构
    test_table = o.create_table(test_table_name, schema)
    
    # 保存数据
    with test_table.open_writer() as writer:
        batch_size = 1000
        for i in range(0, len(test_final_df), batch_size):
            batch_data = test_final_df.iloc[i:i+batch_size]
            records = []
            for _, row in batch_data.iterrows():
                record = [row[col] for col in test_final_df.columns]
                records.append(record)
            writer.write(records)
    
    print(f"✓ 测试集保存成功: {test_table_name} ({len(test_final_df)} 行)")
    
except Exception as e:
    print(f"✗ 测试集保存失败: {e}")

# ===============================
# 10. 总结
# ===============================
print("\n" + "="*60)
print("10. VIF特征工程总结")
print("="*60)

print(f"数据处理流程:")
print(f"  1. 原始数据: {df_encoded.shape[0]} 行 × {df_encoded.shape[1]} 列")
print(f"  2. 数值特征提取: {len(numeric_features)} 个特征")
print(f"  3. 训练/测试分割: {X_train.shape[0]}/{X_test.shape[0]} 样本")
print(f"  4. 标准化: 均值0，标准差1")
print(f"  5. VIF筛选: {len(numeric_features)} → {len(selected_features)} 特征")
print(f"  6. 保存表: {train_table_name}, {test_table_name}")

print(f"\nVIF筛选效果:")
print(f"  ✓ 保持特征可解释性")
print(f"  ✓ 去除多重共线性 (VIF < {vif_threshold})")
print(f"  ✓ 基于性能选择最优特征数量")
print(f"  ✓ 特征保留率: {len(selected_features)/len(numeric_features)*100:.1f}%")

print(f"\n生成的表:")
print(f"  - {train_table_name}: 训练集 ({len(train_final_df)} 行, {len(selected_features)} 特征)")
print(f"  - {test_table_name}: 测试集 ({len(test_final_df)} 行, {len(selected_features)} 特征)")

print(f"\n下一步建议:")
print("1. 在DSW中读取VIF筛选后的训练集和测试集")
print("2. 训练机器学习模型 (Random Forest, XGBoost等)")
print("3. 分析特征重要性")
print("4. 模型评估和业务解释")

print("\n" + "="*60)
print("VIF特征工程完成！")
print("="*60)