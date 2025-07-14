"""
模型血缘追踪工具 - 追踪从数据到模型的完整链路
"""
import json
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any

class ModelLineageTracker:
    def __init__(self, odps_client):
        self.odps = odps_client
        
    def record_training_lineage(self, training_info: Dict[str, Any]):
        """记录训练血缘信息"""
        lineage_record = {
            'lineage_id': f"training_{training_info['training_id']}",
            'lineage_type': 'model_training',
            'timestamp': datetime.now().isoformat(),
            
            # 输入数据
            'input_tables': ['walmart_train_vif'],
            'input_data_version': training_info.get('data_version'),
            
            # 代码版本
            'git_commit_id': training_info.get('git_commit_id'),
            'git_branch': training_info.get('git_branch'),
            'training_script': training_info.get('training_script'),
            
            # 输出模型
            'output_models': training_info.get('trained_models', []),
            'best_model': training_info.get('best_model'),
            
            # 性能指标
            'model_metrics': training_info.get('model_metrics', {}),
            
            # 环境信息
            'execution_environment': {
                'platform': 'DSW',
                'instance_type': training_info.get('instance_type'),
                'training_duration': training_info.get('duration')
            }
        }
        
        self._save_lineage_record(lineage_record)
        
    def record_prediction_lineage(self, prediction_info: Dict[str, Any]):
        """记录预测血缘信息"""
        lineage_record = {
            'lineage_id': f"prediction_{prediction_info['prediction_id']}",
            'lineage_type': 'batch_prediction',
            'timestamp': datetime.now().isoformat(),
            
            # 输入数据和模型
            'input_tables': ['walmart_test_vif'],
            'input_model': prediction_info.get('model_name'),
            'model_version': prediction_info.get('model_version'),
            
            # 代码版本
            'prediction_git_commit': prediction_info.get('prediction_git_commit'),
            'training_git_commit': prediction_info.get('training_git_commit'),
            
            # 输出结果
            'output_table': prediction_info.get('output_table'),
            'prediction_count': prediction_info.get('prediction_count'),
            
            # 执行环境
            'execution_environment': {
                'platform': 'DataWorks',
                'node_type': 'PyODPS'
            }
        }
        
        self._save_lineage_record(lineage_record)
    
    def _save_lineage_record(self, lineage_record: Dict[str, Any]):
        """保存血缘记录到MaxCompute"""
        try:
            # 创建血缘表（如果不存在）
            lineage_table = "model_lineage_tracking"
            
            if not self.odps.exist_table(lineage_table):
                self._create_lineage_table(lineage_table)
            
            # 插入血缘记录
            from odps.models import Record
            table = self.odps.get_table(lineage_table)
            
            with table.open_writer() as writer:
                record = [
                    lineage_record['lineage_id'],
                    lineage_record['lineage_type'],
                    lineage_record['timestamp'],
                    json.dumps(lineage_record.get('input_tables', [])),
                    json.dumps(lineage_record.get('output_models', [])),
                    lineage_record.get('git_commit_id', ''),
                    json.dumps(lineage_record)  # 完整记录作为JSON
                ]
                writer.write([record])
                
            print(f"✅ 血缘记录已保存: {lineage_record['lineage_id']}")
            
        except Exception as e:
            print(f"❌ 保存血缘记录失败: {e}")
    
    def _create_lineage_table(self, table_name: str):
        """创建血缘追踪表"""
        from odps.models import Schema, Column
        
        columns = [
            Column(name='lineage_id', type='string'),
            Column(name='lineage_type', type='string'),
            Column(name='timestamp', type='string'),
            Column(name='input_tables', type='string'),
            Column(name='output_models', type='string'),
            Column(name='git_commit_id', type='string'),
            Column(name='full_record', type='string')
        ]
        
        schema = Schema(columns=columns)
        self.odps.create_table(table_name, schema)
        print(f"✅ 血缘追踪表已创建: {table_name}")
    
    def query_model_lineage(self, model_name: str) -> List[Dict[str, Any]]:
        """查询模型血缘信息"""
        try:
            sql = f"""
            SELECT * FROM model_lineage_tracking 
            WHERE full_record LIKE '%{model_name}%' 
            ORDER BY timestamp DESC
            """
            
            result = self.odps.execute_sql(sql).open_reader()
            lineage_records = []
            
            for record in result:
                lineage_data = json.loads(record[6])  # full_record列
                lineage_records.append(lineage_data)
            
            return lineage_records
            
        except Exception as e:
            print(f"❌ 查询血缘失败: {e}")
            return []