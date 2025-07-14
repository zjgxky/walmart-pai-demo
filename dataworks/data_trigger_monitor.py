# dataworks/data_trigger_monitor.py
# 简化版数据触发监控（避免与monitor_performance.py重复）

import time
from datetime import datetime, timedelta

print("=== 数据驱动的训练触发监控 ===")
print("项目:", o.project)

class DataTriggerMonitor:
    def __init__(self):
        self.monitor_id = str(int(time.time()))
        self.odps = o
        
    def check_data_freshness_only(self):
        """仅检查数据新鲜度，不重复性能检查"""
        print("📊 检查数据新鲜度...")
        
        try:
            if not self.odps.exist_table('walmart_train_vif'):
                return False, "训练数据表不存在"
            
            train_table = self.odps.get_table('walmart_train_vif')
            last_modified = train_table.last_modified_time
            
            current_time = datetime.now()
            data_age_hours = (current_time - last_modified).total_seconds() / 3600
            
            print(f"   数据最后更新: {last_modified}")
            print(f"   数据年龄: {data_age_hours:.1f} 小时")
            
            # 数据更新后24小时内需要重训练
            needs_retrain = data_age_hours <= 24
            reason = f"数据年龄{data_age_hours:.1f}小时，{'需要' if needs_retrain else '不需要'}重训练"
            
            return needs_retrain, reason
            
        except Exception as e:
            return False, f"检查数据新鲜度失败: {e}"
    
    def create_data_trigger_record(self, reason):
        """创建数据触发记录（不创建重训练任务，避免重复）"""
        trigger_record = {
            'trigger_id': f"data_trigger_{self.monitor_id}",
            'trigger_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'trigger_type': 'data_freshness',
            'trigger_reason': reason,
            'status': 'triggered',
            'next_action': 'check_performance_monitor'  # 指向性能监控
        }
        
        try:
            trigger_table = "data_trigger_logs"
            
            from odps.models import Schema, Column
            
            if not self.odps.exist_table(trigger_table):
                trigger_columns = [
                    Column(name='trigger_id', type='string'),
                    Column(name='trigger_time', type='string'),
                    Column(name='trigger_type', type='string'),
                    Column(name='trigger_reason', type='string'),
                    Column(name='status', type='string'),
                    Column(name='next_action', type='string')
                ]
                trigger_schema = Schema(columns=trigger_columns)
                self.odps.create_table(trigger_table, trigger_schema)
            
            table = self.odps.get_table(trigger_table)
            with table.open_writer() as writer:
                record = [
                    trigger_record['trigger_id'],
                    trigger_record['trigger_time'],
                    trigger_record['trigger_type'],
                    trigger_record['trigger_reason'],
                    trigger_record['status'],
                    trigger_record['next_action']
                ]
                writer.write([record])
            
            print(f"✅ 数据触发记录已创建: {trigger_record['trigger_id']}")
            print(f"   下一步: {trigger_record['next_action']}")
            
            return trigger_record
            
        except Exception as e:
            print(f"❌ 创建数据触发记录失败: {e}")
            raise

def main():
    """数据触发监控主函数"""
    monitor = DataTriggerMonitor()
    
    try:
        # 仅检查数据新鲜度
        needs_retrain, reason = monitor.check_data_freshness_only()
        
        if needs_retrain:
            print("🎯 检测到数据更新触发条件")
            trigger_record = monitor.create_data_trigger_record(reason)
            print("✅ 数据触发完成，请查看性能监控节点进行后续处理")
        else:
            print("✅ 数据检查完成，暂无触发条件")
            print(f"   检查结果: {reason}")
        
        return monitor.monitor_id
        
    except Exception as e:
        print(f"❌ 数据触发监控失败: {e}")
        raise

# 执行数据触发监控
trigger_id = main()
print(f"\n✅ 数据触发监控完成 - ID: {trigger_id}")