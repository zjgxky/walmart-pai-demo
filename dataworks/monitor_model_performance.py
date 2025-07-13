# DataWorks PyODPS节点 - 监控与反馈系统
import pandas as pd
import numpy as np
import time
import json

print("=== 监控与反馈系统 - MLOps流程 ===")
print("项目:", o.project)

class MLOpsMonitoringSystem:
    def __init__(self):
        self.monitor_id = str(int(time.time()))
        self.odps = o
        
    def collect_service_metrics(self):
        """收集EAS服务性能指标"""
        print("1. 收集EAS服务性能指标...")
        
        try:
            # 从服务元数据表获取当前服务信息
            if not self.odps.exist_table('eas_service_metadata'):
                raise ValueError("未找到EAS服务元数据表")
            
            service_table = self.odps.get_table('eas_service_metadata')
            service_df = service_table.to_df().to_pandas()
            
            if service_df.empty:
                raise ValueError("没有找到运行中的服务")
            
            # 获取最新的服务信息
            latest_service = service_df.iloc[-1]
            
            # 模拟收集服务性能指标
            performance_metrics = {
                'service_name': str(latest_service['service_name']),
                'service_id': str(latest_service['service_id']),
                'model_name': str(latest_service['model_name']),
                'model_version': str(latest_service['model_version']),
                'monitor_time': time.strftime('%Y-%m-%d %H:%M:%S'),
                'monitor_id': self.monitor_id,
                
                # 模拟的服务性能指标
                'request_count_24h': np.random.randint(800, 1200),
                'avg_response_time_ms': np.random.uniform(50, 150),
                'success_rate': np.random.uniform(0.98, 0.999),
                'cpu_utilization': np.random.uniform(0.3, 0.7),
                'memory_utilization': np.random.uniform(0.4, 0.8),
                'error_rate': np.random.uniform(0.001, 0.02),
                'throughput_rps': np.random.uniform(15, 25)
            }
            
            print("   ✅ 服务性能指标收集完成")
            print("   ✅ 24小时请求数:", performance_metrics['request_count_24h'])
            print("   ✅ 平均响应时间:", round(performance_metrics['avg_response_time_ms'], 2), "ms")
            print("   ✅ 成功率:", round(performance_metrics['success_rate'] * 100, 2), "%")
            print("   ✅ CPU利用率:", round(performance_metrics['cpu_utilization'] * 100, 2), "%")
            
            return performance_metrics
            
        except Exception as e:
            print("   ❌ 收集服务指标失败:", str(e))
            raise
    
    def simulate_real_sales_data(self):
        """模拟真实销售数据收集"""
        print("2. 收集真实销售数据...")
        
        try:
            # 模拟从业务系统收集的真实销售数据
            # 在实际情况中，这些数据来自POS系统、ERP系统等
            
            # 获取之前的预测结果
            if not self.odps.exist_table('walmart_sales_predictions'):
                raise ValueError("未找到预测结果表")
            
            pred_table = self.odps.get_table('walmart_sales_predictions')
            pred_df = pred_table.to_df().to_pandas()
            
            # 模拟真实销售数据
            # 基于预测结果添加一些随机波动来模拟真实值
            np.random.seed(42)
            
            real_sales_data = []
            for _, row in pred_df.iterrows():
                predicted_sales = row['predicted_weekly_sales']
                
                # 添加随机波动（±15%）
                noise_factor = np.random.normal(1.0, 0.15)
                actual_sales = predicted_sales * noise_factor
                
                # 确保非负
                actual_sales = max(actual_sales, 0)
                
                real_sales_data.append({
                    'store': row.get('Store', 1),
                    'predicted_weekly_sales': predicted_sales,
                    'actual_weekly_sales': actual_sales,
                    'prediction_error': abs(actual_sales - predicted_sales),
                    'prediction_error_pct': abs(actual_sales - predicted_sales) / max(actual_sales, 1) * 100,
                    'data_date': time.strftime('%Y-%m-%d'),
                    'prediction_id': row['prediction_id']
                })
            
            print("   ✅ 真实销售数据收集完成")
            print("   ✅ 数据样本数:", len(real_sales_data))
            
            return real_sales_data
            
        except Exception as e:
            print("   ❌ 收集真实销售数据失败:", str(e))
            raise
    
    def calculate_model_performance(self, real_sales_data):
        """计算模型在线表现"""
        print("3. 计算模型在线表现...")
        
        try:
            # 转换为DataFrame便于计算
            real_df = pd.DataFrame(real_sales_data)
            
            # 计算整体性能指标
            performance_metrics = {
                'monitor_id': self.monitor_id,
                'evaluation_date': time.strftime('%Y-%m-%d'),
                'sample_count': len(real_df),
                
                # 误差统计
                'mean_absolute_error': float(real_df['prediction_error'].mean()),
                'median_absolute_error': float(real_df['prediction_error'].median()),
                'mean_absolute_percentage_error': float(real_df['prediction_error_pct'].mean()),
                'max_error': float(real_df['prediction_error'].max()),
                'min_error': float(real_df['prediction_error'].min()),
                
                # 准确性指标
                'accuracy_within_10pct': float((real_df['prediction_error_pct'] <= 10).mean()),
                'accuracy_within_20pct': float((real_df['prediction_error_pct'] <= 20).mean()),
                
                # R²计算
                'r2_score': float(1 - (real_df['prediction_error']**2).sum() / 
                                ((real_df['actual_weekly_sales'] - real_df['actual_weekly_sales'].mean())**2).sum()),
                
                # 相关性
                'correlation': float(real_df['predicted_weekly_sales'].corr(real_df['actual_weekly_sales']))
            }
            
            print("   ✅ 模型在线表现计算完成")
            print("   ✅ 平均绝对误差:", round(performance_metrics['mean_absolute_error'], 2))
            print("   ✅ 平均绝对百分比误差:", round(performance_metrics['mean_absolute_percentage_error'], 2), "%")
            print("   ✅ 10%内准确率:", round(performance_metrics['accuracy_within_10pct'] * 100, 2), "%")
            print("   ✅ 在线R²:", round(performance_metrics['r2_score'], 4))
            print("   ✅ 相关性:", round(performance_metrics['correlation'], 4))
            
            return performance_metrics
            
        except Exception as e:
            print("   ❌ 计算模型性能失败:", str(e))
            raise
    
    def save_monitoring_results(self, service_metrics, model_performance, real_sales_data):
        """保存监控结果"""
        print("4. 保存监控结果...")
        
        try:
            # 1. 保存服务性能指标
            service_metrics_table = "service_performance_metrics"
            
            from odps.models import Schema, Column
            
            service_columns = [
                Column(name='service_name', type='string'),
                Column(name='service_id', type='string'),
                Column(name='model_name', type='string'),
                Column(name='model_version', type='string'),
                Column(name='monitor_time', type='string'),
                Column(name='monitor_id', type='string'),
                Column(name='request_count_24h', type='bigint'),
                Column(name='avg_response_time_ms', type='double'),
                Column(name='success_rate', type='double'),
                Column(name='cpu_utilization', type='double'),
                Column(name='memory_utilization', type='double'),
                Column(name='error_rate', type='double'),
                Column(name='throughput_rps', type='double')
            ]
            
            service_schema = Schema(columns=service_columns)
            
            if self.odps.exist_table(service_metrics_table):
                self.odps.delete_table(service_metrics_table)
            
            self.odps.create_table(service_metrics_table, service_schema)
            
            # 写入服务性能数据
            table = self.odps.get_table(service_metrics_table)
            with table.open_writer() as writer:
                record = [
                    service_metrics['service_name'],
                    service_metrics['service_id'],
                    service_metrics['model_name'],
                    service_metrics['model_version'],
                    service_metrics['monitor_time'],
                    service_metrics['monitor_id'],
                    service_metrics['request_count_24h'],
                    service_metrics['avg_response_time_ms'],
                    service_metrics['success_rate'],
                    service_metrics['cpu_utilization'],
                    service_metrics['memory_utilization'],
                    service_metrics['error_rate'],
                    service_metrics['throughput_rps']
                ]
                writer.write([record])
            
            print("   ✅ 服务性能指标已保存到:", service_metrics_table)
            
            # 2. 保存模型性能指标
            model_metrics_table = "model_performance_metrics"
            
            model_columns = [
                Column(name='monitor_id', type='string'),
                Column(name='evaluation_date', type='string'),
                Column(name='sample_count', type='bigint'),
                Column(name='mean_absolute_error', type='double'),
                Column(name='median_absolute_error', type='double'),
                Column(name='mean_absolute_percentage_error', type='double'),
                Column(name='max_error', type='double'),
                Column(name='min_error', type='double'),
                Column(name='accuracy_within_10pct', type='double'),
                Column(name='accuracy_within_20pct', type='double'),
                Column(name='r2_score', type='double'),
                Column(name='correlation', type='double')
            ]
            
            model_schema = Schema(columns=model_columns)
            
            if self.odps.exist_table(model_metrics_table):
                self.odps.delete_table(model_metrics_table)
            
            self.odps.create_table(model_metrics_table, model_schema)
            
            # 写入模型性能数据
            table = self.odps.get_table(model_metrics_table)
            with table.open_writer() as writer:
                record = [
                    model_performance['monitor_id'],
                    model_performance['evaluation_date'],
                    model_performance['sample_count'],
                    model_performance['mean_absolute_error'],
                    model_performance['median_absolute_error'],
                    model_performance['mean_absolute_percentage_error'],
                    model_performance['max_error'],
                    model_performance['min_error'],
                    model_performance['accuracy_within_10pct'],
                    model_performance['accuracy_within_20pct'],
                    model_performance['r2_score'],
                    model_performance['correlation']
                ]
                writer.write([record])
            
            print("   ✅ 模型性能指标已保存到:", model_metrics_table)
            
            # 3. 保存真实销售数据
            real_sales_table = "real_sales_feedback"
            
            real_columns = [
                Column(name='store', type='bigint'),
                Column(name='predicted_weekly_sales', type='double'),
                Column(name='actual_weekly_sales', type='double'),
                Column(name='prediction_error', type='double'),
                Column(name='prediction_error_pct', type='double'),
                Column(name='data_date', type='string'),
                Column(name='prediction_id', type='string')
            ]
            
            real_schema = Schema(columns=real_columns)
            
            if self.odps.exist_table(real_sales_table):
                self.odps.delete_table(real_sales_table)
            
            self.odps.create_table(real_sales_table, real_schema)
            
            # 写入真实销售数据
            table = self.odps.get_table(real_sales_table)
            with table.open_writer() as writer:
                for data in real_sales_data:
                    record = [
                        int(data['store']),
                        float(data['predicted_weekly_sales']),
                        float(data['actual_weekly_sales']),
                        float(data['prediction_error']),
                        float(data['prediction_error_pct']),
                        data['data_date'],
                        data['prediction_id']
                    ]
                    writer.write([record])
            
            print("   ✅ 真实销售数据已保存到:", real_sales_table)
            
            return {
                'service_metrics_table': service_metrics_table,
                'model_metrics_table': model_metrics_table,
                'real_sales_table': real_sales_table
            }
            
        except Exception as e:
            print("   ❌ 保存监控结果失败:", str(e))
            raise
    
    def check_retrain_trigger(self, model_performance, threshold=0.05):
        """检查是否需要触发重新训练"""
        print("5. 检查重新训练触发条件...")
        
        try:
            # 获取原始模型的验证集性能
            if not self.odps.exist_table('model_deployment_status'):
                raise ValueError("未找到模型部署状态表")
            
            deploy_table = self.odps.get_table('model_deployment_status')
            deploy_df = deploy_table.to_df().to_pandas()
            
            if deploy_df.empty:
                raise ValueError("没有找到部署模型信息")
            
            # 获取最新部署的模型信息
            latest_deploy = deploy_df.iloc[-1]
            original_r2 = float(latest_deploy['val_r2_score'])
            current_r2 = model_performance['r2_score']
            
            # 计算性能下降
            performance_drop = original_r2 - current_r2
            performance_drop_pct = (performance_drop / original_r2) * 100
            
            # 检查是否达到重新训练阈值
            should_retrain = performance_drop_pct > (threshold * 100)
            
            retrain_decision = {
                'monitor_id': self.monitor_id,
                'original_r2': original_r2,
                'current_r2': current_r2,
                'performance_drop': performance_drop,
                'performance_drop_pct': performance_drop_pct,
                'threshold_pct': threshold * 100,
                'should_retrain': should_retrain,
                'decision_time': time.strftime('%Y-%m-%d %H:%M:%S'),
                'decision_reason': f"性能下降{performance_drop_pct:.2f}%，{'超过' if should_retrain else '未超过'}阈值{threshold*100}%"
            }
            
            print("   ✅ 重新训练决策完成")
            print("   ✅ 原始R²:", round(original_r2, 4))
            print("   ✅ 当前R²:", round(current_r2, 4))
            print("   ✅ 性能下降:", round(performance_drop_pct, 2), "%")
            print("   ✅ 重新训练阈值:", threshold * 100, "%")
            print("   ✅ 是否需要重新训练:", "是" if should_retrain else "否")
            
            return retrain_decision
            
        except Exception as e:
            print("   ❌ 检查重新训练触发条件失败:", str(e))
            raise
    
    def save_retrain_decision(self, retrain_decision):
        """保存重新训练决策"""
        print("6. 保存重新训练决策...")
        
        try:
            # 创建重新训练决策表
            retrain_table = "retrain_decisions"
            
            from odps.models import Schema, Column
            
            retrain_columns = [
                Column(name='monitor_id', type='string'),
                Column(name='original_r2', type='double'),
                Column(name='current_r2', type='double'),
                Column(name='performance_drop', type='double'),
                Column(name='performance_drop_pct', type='double'),
                Column(name='threshold_pct', type='double'),
                Column(name='should_retrain', type='boolean'),
                Column(name='decision_time', type='string'),
                Column(name='decision_reason', type='string')
            ]
            
            retrain_schema = Schema(columns=retrain_columns)
            
            if self.odps.exist_table(retrain_table):
                self.odps.delete_table(retrain_table)
            
            self.odps.create_table(retrain_table, retrain_schema)
            
            # 写入重新训练决策
            table = self.odps.get_table(retrain_table)
            with table.open_writer() as writer:
                record = [
                    retrain_decision['monitor_id'],
                    retrain_decision['original_r2'],
                    retrain_decision['current_r2'],
                    retrain_decision['performance_drop'],
                    retrain_decision['performance_drop_pct'],
                    retrain_decision['threshold_pct'],
                    retrain_decision['should_retrain'],
                    retrain_decision['decision_time'],
                    retrain_decision['decision_reason']
                ]
                writer.write([record])
            
            print("   ✅ 重新训练决策已保存到:", retrain_table)
            
            # 如果需要重新训练，更新部署状态
            if retrain_decision['should_retrain']:
                self.trigger_retrain_process()
            
            return retrain_decision
            
        except Exception as e:
            print("   ❌ 保存重新训练决策失败:", str(e))
            raise
    
    def trigger_retrain_process(self):
        """触发重新训练流程"""
        print("7. 触发重新训练流程...")
        
        try:
            # 创建重新训练任务表
            retrain_task_table = "retrain_tasks"
            
            from odps.models import Schema, Column
            
            task_columns = [
                Column(name='task_id', type='string'),
                Column(name='trigger_time', type='string'),
                Column(name='trigger_reason', type='string'),
                Column(name='status', type='string'),
                Column(name='priority', type='string'),
                Column(name='assigned_to', type='string'),
                Column(name='estimated_completion', type='string')
            ]
            
            task_schema = Schema(columns=task_columns)
            
            if self.odps.exist_table(retrain_task_table):
                self.odps.delete_table(retrain_task_table)
            
            self.odps.create_table(retrain_task_table, task_schema)
            
            # 创建重新训练任务
            retrain_task = {
                'task_id': f"retrain_{self.monitor_id}",
                'trigger_time': time.strftime('%Y-%m-%d %H:%M:%S'),
                'trigger_reason': f"模型性能下降超过阈值，监控ID: {self.monitor_id}",
                'status': 'pending',
                'priority': 'high',
                'assigned_to': 'data_science_team',
                'estimated_completion': time.strftime('%Y-%m-%d', time.localtime(time.time() + 86400))  # 明天
            }
            
            # 写入重新训练任务
            table = self.odps.get_table(retrain_task_table)
            with table.open_writer() as writer:
                record = [
                    retrain_task['task_id'],
                    retrain_task['trigger_time'],
                    retrain_task['trigger_reason'],
                    retrain_task['status'],
                    retrain_task['priority'],
                    retrain_task['assigned_to'],
                    retrain_task['estimated_completion']
                ]
                writer.write([record])
            
            print("   ✅ 重新训练任务已创建:", retrain_task['task_id'])
            print("   ✅ 任务状态:", retrain_task['status'])
            print("   ✅ 任务优先级:", retrain_task['priority'])
            print("   ✅ 预计完成时间:", retrain_task['estimated_completion'])
            
            return retrain_task
            
        except Exception as e:
            print("   ❌ 创建重新训练任务失败:", str(e))
            raise

def main():
    """主监控函数"""
    monitor = MLOpsMonitoringSystem()
    
    try:
        # 1. 收集EAS服务性能指标
        service_metrics = monitor.collect_service_metrics()
        
        # 2. 收集真实销售数据
        real_sales_data = monitor.simulate_real_sales_data()
        
        # 3. 计算模型在线表现
        model_performance = monitor.calculate_model_performance(real_sales_data)
        
        # 4. 保存监控结果
        saved_tables = monitor.save_monitoring_results(service_metrics, model_performance, real_sales_data)
        
        # 5. 检查是否需要重新训练
        retrain_decision = monitor.check_retrain_trigger(model_performance, threshold=0.05)
        
        # 6. 保存重新训练决策
        final_decision = monitor.save_retrain_decision(retrain_decision)
        
        # 7. 输出监控总结
        print("\n=== 监控与反馈完成总结 ===")
        print("监控ID:", monitor.monitor_id)
        print("服务名称:", service_metrics['service_name'])
        print("监控时间:", service_metrics['monitor_time'])
        
        print("\n=== 服务性能指标 ===")
        print("24小时请求数:", service_metrics['request_count_24h'])
        print("平均响应时间:", round(service_metrics['avg_response_time_ms'], 2), "ms")
        print("成功率:", round(service_metrics['success_rate'] * 100, 2), "%")
        print("CPU利用率:", round(service_metrics['cpu_utilization'] * 100, 2), "%")
        
        print("\n=== 模型性能指标 ===")
        print("样本数量:", model_performance['sample_count'])
        print("平均绝对误差:", round(model_performance['mean_absolute_error'], 2))
        print("平均绝对百分比误差:", round(model_performance['mean_absolute_percentage_error'], 2), "%")
        print("10%内准确率:", round(model_performance['accuracy_within_10pct'] * 100, 2), "%")
        print("在线R²:", round(model_performance['r2_score'], 4))
        
        print("\n=== 重新训练决策 ===")
        print("性能下降:", round(final_decision['performance_drop_pct'], 2), "%")
        print("是否需要重新训练:", "是" if final_decision['should_retrain'] else "否")
        print("决策原因:", final_decision['decision_reason'])
        
        print("\n=== 数据表输出 ===")
        for table_name, table_path in saved_tables.items():
            print(f"{table_name}: {table_path}")
        
        return monitor.monitor_id, service_metrics, model_performance, final_decision
        
    except Exception as e:
        print("❌ 监控系统运行失败:", str(e))
        raise

# 执行监控系统
monitor_id, service_metrics, model_performance, retrain_decision = main()
print(f"\n✅ 监控任务完成 - ID: {monitor_id}")
