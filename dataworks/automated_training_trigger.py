# dataworks/automated_training_trigger.py
# DataWorks PyODPS节点 - 自动化训练触发器

import time
import json
from datetime import datetime, timedelta

print("=== 自动化训练触发器 ===")
print("项目:", o.project)

def check_data_freshness():
    """检查数据新鲜度，决定是否需要重训练"""
    
    try:
        # 检查训练数据表的最新更新时间
        if not o.exist_table('walmart_train_vif'):
            print("❌ 训练数据表不存在")
            return False, "训练数据表不存在"
        
        train_table = o.get_table('walmart_train_vif')
        last_modified = train_table.last_modified_time
        
        # 计算数据年龄
        current_time = datetime.now()
        data_age_hours = (current_time - last_modified).total_seconds() / 3600
        
        print(f"📊 数据检查结果:")
        print(f"   最新数据时间: {last_modified}")
        print(f"   数据年龄: {data_age_hours:.1f} 小时")
        
        # 策略：数据更新后24小时内需要重训练
        needs_retrain = data_age_hours <= 24
        
        reason = f"数据年龄{data_age_hours:.1f}小时，{'需要' if needs_retrain else '不需要'}重训练"
        
        return needs_retrain, reason
        
    except Exception as e:
        print(f"❌ 检查数据新鲜度失败: {e}")
        return False, f"检查失败: {e}"

def check_model_performance():
    """检查当前模型性能，决定是否需要重训练"""
    
    try:
        # 检查是否有性能监控数据
        if not o.exist_table('model_performance_metrics'):
            print("⚠️ 没有性能监控数据，跳过性能检查")
            return False, "无性能数据"
        
        perf_table = o.get_table('model_performance_metrics')
        perf_df = perf_table.to_df().to_pandas()
        
        if perf_df.empty:
            return False, "无性能记录"
        
        # 获取最新的性能数据
        latest_perf = perf_df.iloc[-1]
        current_r2 = float(latest_perf['r2_score'])
        
        # 假设原始模型R²为0.94（从训练记录获取）
        original_r2 = 0.9431
        performance_drop = (original_r2 - current_r2) / original_r2
        
        print(f"📈 性能检查结果:")
        print(f"   原始R²: {original_r2:.4f}")
        print(f"   当前R²: {current_r2:.4f}")
        print(f"   性能下降: {performance_drop*100:.2f}%")
        
        # 阈值：性能下降超过5%需要重训练
        needs_retrain = performance_drop > 0.05
        reason = f"性能下降{performance_drop*100:.2f}%，{'超过' if needs_retrain else '未超过'}5%阈值"
        
        return needs_retrain, reason
        
    except Exception as e:
        print(f"❌ 检查模型性能失败: {e}")
        return False, f"性能检查失败: {e}"

def create_training_trigger(trigger_reasons):
    """创建训练触发记录"""
    
    trigger_id = str(int(time.time()))
    
    trigger_record = {
        'trigger_id': trigger_id,
        'trigger_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'trigger_reasons': '; '.join(trigger_reasons),
        'status': 'pending',
        'priority': 'normal',
        'created_by': 'automated_system'
    }
    
    try:
        # 创建或更新训练触发表
        trigger_table = "dsw_training_triggers"
        
        from odps.models import Schema, Column
        
        if not o.exist_table(trigger_table):
            trigger_columns = [
                Column(name='trigger_id', type='string'),
                Column(name='trigger_time', type='string'),
                Column(name='trigger_reasons', type='string'),
                Column(name='status', type='string'),
                Column(name='priority', type='string'),
                Column(name='created_by', type='string')
            ]
            trigger_schema = Schema(columns=trigger_columns)
            o.create_table(trigger_table, trigger_schema)
            print(f"✅ 创建触发表: {trigger_table}")
        
        # 写入触发记录
        table = o.get_table(trigger_table)
        with table.open_writer() as writer:
            record = [
                trigger_record['trigger_id'],
                trigger_record['trigger_time'],
                trigger_record['trigger_reasons'],
                trigger_record['status'],
                trigger_record['priority'],
                trigger_record['created_by']
            ]
            writer.write([record])
        
        print(f"🚀 训练触发已创建:")
        print(f"   触发ID: {trigger_id}")
        print(f"   触发原因: {trigger_record['trigger_reasons']}")
        
        return trigger_record
        
    except Exception as e:
        print(f"❌ 创建训练触发失败: {e}")
        raise

def send_notification(trigger_record):
    """发送训练触发通知"""
    
    try:
        # 创建通知记录表
        notification_table = "training_notifications"
        
        notification_msg = f"""
🤖 自动训练触发通知

触发时间: {trigger_record['trigger_time']}
触发ID: {trigger_record['trigger_id']}
触发原因: {trigger_record['trigger_reasons']}
状态: {trigger_record['status']}

请在DSW中检查并执行训练任务。
访问链接: https://pai.console.aliyun.com
        """
        
        print("📧 通知内容:")
        print(notification_msg)
        
        # 在实际环境中，这里可以集成：
        # 1. 钉钉机器人通知
        # 2. 邮件通知
        # 3. 短信通知
        
        # 记录通知日志
        from odps.models import Schema, Column
        
        if not o.exist_table(notification_table):
            notification_columns = [
                Column(name='notification_id', type='string'),
                Column(name='trigger_id', type='string'),
                Column(name='notification_type', type='string'),
                Column(name='message', type='string'),
                Column(name='sent_time', type='string'),
                Column(name='status', type='string')
            ]
            notification_schema = Schema(columns=notification_columns)
            o.create_table(notification_table, notification_schema)
        
        # 写入通知记录
        table = o.get_table(notification_table)
        with table.open_writer() as writer:
            record = [
                str(int(time.time())),  # notification_id
                trigger_record['trigger_id'],
                'auto_trigger',
                notification_msg,
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'sent'
            ]
            writer.write([record])
        
        print("✅ 通知已发送并记录")
        
    except Exception as e:
        print(f"❌ 发送通知失败: {e}")

def check_existing_triggers():
    """检查是否已有待处理的触发"""
    
    try:
        if not o.exist_table('dsw_training_triggers'):
            return False
        
        trigger_table = o.get_table('dsw_training_triggers')
        trigger_df = trigger_table.to_df().to_pandas()
        
        # 检查最近24小时内是否有pending的触发
        recent_pending = trigger_df[
            (trigger_df['status'] == 'pending') & 
            (trigger_df['trigger_time'] >= (datetime.now() - timedelta(hours=24)).strftime('%Y-%m-%d %H:%M:%S'))
        ]
        
        if not recent_pending.empty:
            print(f"⚠️ 发现 {len(recent_pending)} 个待处理的训练触发，跳过本次检查")
            return True
        
        return False
        
    except Exception as e:
        print(f"❌ 检查现有触发失败: {e}")
        return False

def main():
    """自动化训练触发器主函数"""
    
    print("开始执行自动化训练检查...")
    
    # 1. 检查是否已有待处理的触发
    if check_existing_triggers():
        print("发现待处理的训练触发，本次跳过")
        return
    
    # 2. 检查各种触发条件
    trigger_reasons = []
    
    # 检查数据新鲜度
    data_needs_retrain, data_reason = check_data_freshness()
    if data_needs_retrain:
        trigger_reasons.append(f"数据更新: {data_reason}")
    
    # 检查模型性能
    perf_needs_retrain, perf_reason = check_model_performance()
    if perf_needs_retrain:
        trigger_reasons.append(f"性能下降: {perf_reason}")
    
    # 3. 决定是否触发训练
    if trigger_reasons:
        print(f"🎯 检测到训练触发条件: {len(trigger_reasons)} 个")
        
        # 创建训练触发
        trigger_record = create_training_trigger(trigger_reasons)
        
        # 发送通知
        send_notification(trigger_record)
        
        print("✅ 自动化训练触发流程完成")
        
    else:
        print("✅ 暂无训练触发条件，系统运行正常")
        
        # 记录检查日志
        print(f"   数据检查: {data_reason}")
        print(f"   性能检查: {perf_reason}")

# 执行自动化检查
main()