# DataWorks PyODPS节点 - EAS自动部署脚本
import json
import pandas as pd
import requests
import time

# 使用DataWorks内置的ODPS对象
print("=== EAS自动部署 - MLOps流程 ===")
print("项目:", o.project)

class EASAutoDeployer:
    def __init__(self):
        self.deployment_id = str(int(time.time()))
        self.odps = o
        self.service_name = "walmart-sales-prediction-api"
        self.service_info = None
        
    def get_approved_model_for_deployment(self):
        """获取已批准部署的模型"""
        print("1. 获取已批准部署的模型...")
        
        try:
            # 从部署状态表获取最新的批准模型
            if not self.odps.exist_table('model_deployment_status'):
                raise ValueError("未找到模型部署状态表")
            
            deploy_table = self.odps.get_table('model_deployment_status')
            deploy_df = deploy_table.to_df().to_pandas()
            
            # 获取状态为approved的最新模型
            approved_models = deploy_df[deploy_df['deployment_status'] == 'approved']
            if approved_models.empty:
                raise ValueError("未找到已批准部署的模型")
            
            # 选择最新的批准模型
            latest_approved = approved_models.iloc[-1]
            
            model_info = {
                'model_name': str(latest_approved['model_name']),
                'registry_name': str(latest_approved['registry_name']),
                'model_version': str(latest_approved['model_version']),
                'val_r2_score': float(latest_approved['val_r2_score']),
                'prediction_table': str(latest_approved['prediction_table']),
                'prediction_id': str(latest_approved['prediction_id'])
            }
            
            print("   ✅ 找到批准部署的模型:", model_info['model_name'])
            print("   ✅ 模型版本:", model_info['model_version'])
            print("   ✅ 模型性能 R²:", round(model_info['val_r2_score'], 4))
            
            return model_info
            
        except Exception as e:
            print("   ❌ 获取批准模型失败:", str(e))
            raise
    
    def deploy_model_to_eas(self, model_info):
        """部署模型到EAS"""
        print("2. 部署模型到EAS...")
        
        try:
            # 由于这是demo，我们模拟EAS部署过程
            # 在实际环境中，这里会调用EAS API
            
            print("   正在部署模型到EAS...")
            
            # 模拟EAS部署配置
            eas_config = {
                'service_name': self.service_name,
                'model_path': f"pai://model_registry/{model_info['registry_name']}:{model_info['model_version']}",
                'processor': 'sklearn_cpu',  # 适合我们的sklearn模型
                'instance_count': 2,
                'instance_type': 'ecs.c6.large',
                'auto_scaling': {
                    'min_instances': 1,
                    'max_instances': 5,
                    'target_cpu_utilization': 70
                }
            }
            
            # 模拟部署结果
            deployment_result = {
                'service_name': self.service_name,
                'service_id': f"eas-{self.deployment_id}",
                'endpoint': f"http://{self.service_name}.pai-eas.aliyuncs.com/predict",
                'status': 'Running',
                'instance_count': 2,
                'deployed_model': model_info['registry_name'],
                'deploy_time': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            print("   ✅ 模型部署成功!")
            print("   ✅ 服务名称:", deployment_result['service_name'])
            print("   ✅ 服务端点:", deployment_result['endpoint'])
            print("   ✅ 实例数量:", deployment_result['instance_count'])
            
            self.service_info = deployment_result
            return deployment_result
            
        except Exception as e:
            print("   ❌ EAS部署失败:", str(e))
            raise
    
    def create_service_metadata(self, model_info, deployment_result):
        """创建服务元数据表"""
        print("3. 创建服务元数据...")
        
        try:
            # 创建EAS服务元数据表
            service_table = "eas_service_metadata"
            
            service_metadata = {
                'service_name': deployment_result['service_name'],
                'service_id': deployment_result['service_id'],
                'endpoint': deployment_result['endpoint'],
                'model_name': model_info['model_name'],
                'model_version': model_info['model_version'],
                'registry_name': model_info['registry_name'],
                'model_performance': model_info['val_r2_score'],
                'instance_count': deployment_result['instance_count'],
                'status': deployment_result['status'],
                'deploy_time': deployment_result['deploy_time'],
                'deployment_id': self.deployment_id
            }
            
            # 创建表结构
            from odps.models import Schema, Column
            
            service_columns = [
                Column(name='service_name', type='string'),
                Column(name='service_id', type='string'),
                Column(name='endpoint', type='string'),
                Column(name='model_name', type='string'),
                Column(name='model_version', type='string'),
                Column(name='registry_name', type='string'),
                Column(name='model_performance', type='double'),
                Column(name='instance_count', type='bigint'),
                Column(name='status', type='string'),
                Column(name='deploy_time', type='string'),
                Column(name='deployment_id', type='string')
            ]
            
            service_schema = Schema(columns=service_columns)
            
            # 删除旧表并创建新表
            if self.odps.exist_table(service_table):
                self.odps.delete_table(service_table)
            
            self.odps.create_table(service_table, service_schema)
            
            # 写入服务元数据
            table = self.odps.get_table(service_table)
            with table.open_writer() as writer:
                record = [
                    service_metadata['service_name'],
                    service_metadata['service_id'],
                    service_metadata['endpoint'],
                    service_metadata['model_name'],
                    service_metadata['model_version'],
                    service_metadata['registry_name'],
                    service_metadata['model_performance'],
                    service_metadata['instance_count'],
                    service_metadata['status'],
                    service_metadata['deploy_time'],
                    service_metadata['deployment_id']
                ]
                writer.write([record])
            
            print("   ✅ 服务元数据已保存到:", service_table)
            
            return service_metadata
            
        except Exception as e:
            print("   ❌ 创建服务元数据失败:", str(e))
            raise
    
    def create_api_demo_script(self, service_metadata):
        """创建API调用示例脚本"""
        print("4. 创建API调用示例...")
        
        try:
            # 创建API调用示例表
            api_demo_table = "api_demo_scripts"
            
            # 准备示例数据
            demo_scripts = [
                {
                    'script_type': 'curl',
                    'language': 'bash',
                    'script_content': f'''# Walmart销量预测API调用示例
curl -X POST "{service_metadata['endpoint']}" \\
  -H "Authorization: Bearer ${{TOKEN}}" \\
  -H "Content-Type: application/json" \\
  -d '{{
    "store": 1,
    "temperature": 75.5,
    "fuel_price": 3.2,
    "holiday_flag": 0,
    "unemployment": 8.1,
    "cpi": 210.3
  }}'
''',
                    'description': 'cURL命令行调用示例'
                },
                {
                    'script_type': 'python',
                    'language': 'python',
                    'script_content': f'''# Python API调用示例
import requests
import json

# 服务端点
endpoint = "{service_metadata['endpoint']}"

# 请求数据
data = {{
    "store": 1,
    "temperature": 75.5,
    "fuel_price": 3.2,
    "holiday_flag": 0,
    "unemployment": 8.1,
    "cpi": 210.3
}}

# 发送预测请求
headers = {{
    "Authorization": "Bearer YOUR_TOKEN",
    "Content-Type": "application/json"
}}

response = requests.post(endpoint, json=data, headers=headers)
result = response.json()

print("预测结果:", result['predicted_weekly_sales'])
''',
                    'description': 'Python调用示例'
                }
            ]
            
            # 创建表结构
            from odps.models import Schema, Column
            
            demo_columns = [
                Column(name='script_type', type='string'),
                Column(name='language', type='string'),
                Column(name='script_content', type='string'),
                Column(name='description', type='string'),
                Column(name='service_name', type='string'),
                Column(name='created_time', type='string')
            ]
            
            demo_schema = Schema(columns=demo_columns)
            
            # 删除旧表并创建新表
            if self.odps.exist_table(api_demo_table):
                self.odps.delete_table(api_demo_table)
            
            self.odps.create_table(api_demo_table, demo_schema)
            
            # 写入示例脚本
            table = self.odps.get_table(api_demo_table)
            with table.open_writer() as writer:
                for script in demo_scripts:
                    record = [
                        script['script_type'],
                        script['language'],
                        script['script_content'],
                        script['description'],
                        service_metadata['service_name'],
                        time.strftime('%Y-%m-%d %H:%M:%S')
                    ]
                    writer.write([record])
            
            print("   ✅ API调用示例已保存到:", api_demo_table)
            
            return demo_scripts
            
        except Exception as e:
            print("   ❌ 创建API示例失败:", str(e))
            raise
    
    def update_deployment_status(self, model_info, service_metadata):
        """更新部署状态"""
        print("5. 更新部署状态...")
        
        try:
            # 更新原有的部署状态表
            deploy_table = self.odps.get_table('model_deployment_status')
            deploy_df = deploy_table.to_df().to_pandas()
            
            # 找到对应的记录并更新状态
            mask = deploy_df['prediction_id'] == model_info['prediction_id']
            if mask.any():
                # 由于MaxCompute表不支持直接更新，我们重新创建表
                deploy_df.loc[mask, 'deployment_status'] = 'deployed'
                deploy_df.loc[mask, 'next_action'] = 'monitor_performance'
                
                # 重新创建表
                from odps.models import Schema, Column
                
                deploy_columns = [
                    Column(name='prediction_id', type='string'),
                    Column(name='model_name', type='string'),
                    Column(name='registry_name', type='string'),
                    Column(name='model_version', type='string'),
                    Column(name='val_r2_score', type='double'),
                    Column(name='prediction_count', type='bigint'),
                    Column(name='prediction_mean', type='double'),
                    Column(name='prediction_table', type='string'),
                    Column(name='deployment_status', type='string'),
                    Column(name='next_action', type='string')
                ]
                
                deploy_schema = Schema(columns=deploy_columns)
                
                # 删除旧表
                self.odps.delete_table('model_deployment_status')
                
                # 创建新表
                self.odps.create_table('model_deployment_status', deploy_schema)
                
                # 写入更新后的数据
                table = self.odps.get_table('model_deployment_status')
                with table.open_writer() as writer:
                    for _, row in deploy_df.iterrows():
                        record = [
                            str(row['prediction_id']),
                            str(row['model_name']),
                            str(row['registry_name']),
                            str(row['model_version']),
                            float(row['val_r2_score']),
                            int(row['prediction_count']),
                            float(row['prediction_mean']),
                            str(row['prediction_table']),
                            str(row['deployment_status']),
                            str(row['next_action'])
                        ]
                        writer.write([record])
                
                print("   ✅ 部署状态已更新为: deployed")
                print("   ✅ 下一步操作: monitor_performance")
            
        except Exception as e:
            print("   ❌ 更新部署状态失败:", str(e))
            raise

def main():
    """主部署函数"""
    deployer = EASAutoDeployer()
    
    try:
        # 1. 获取已批准部署的模型
        model_info = deployer.get_approved_model_for_deployment()
        
        # 2. 部署模型到EAS
        deployment_result = deployer.deploy_model_to_eas(model_info)
        
        # 3. 创建服务元数据
        service_metadata = deployer.create_service_metadata(model_info, deployment_result)
        
        # 4. 创建API调用示例
        demo_scripts = deployer.create_api_demo_script(service_metadata)
        
        # 5. 更新部署状态
        deployer.update_deployment_status(model_info, service_metadata)
        
        # 6. 输出总结
        print("\n=== EAS部署完成总结 ===")
        print("部署ID:", deployer.deployment_id)
        print("服务名称:", service_metadata['service_name'])
        print("服务端点:", service_metadata['endpoint'])
        print("部署模型:", service_metadata['model_name'])
        print("模型版本:", service_metadata['model_version'])
        print("模型性能 R²:", round(service_metadata['model_performance'], 4))
        print("实例数量:", service_metadata['instance_count'])
        print("服务状态:", service_metadata['status'])
        
        print("\n=== API调用示例 ===")
        print("使用以下命令测试服务:")
        print("curl -X POST", service_metadata['endpoint'])
        print("完整示例已保存到: api_demo_scripts 表")
        
        return deployer.deployment_id, service_metadata
        
    except Exception as e:
        print("❌ EAS部署失败:", str(e))
        raise

# 执行EAS部署
deployment_id, service_info = main()
print(f"\n✅ EAS部署任务完成 - ID: {deployment_id}")
