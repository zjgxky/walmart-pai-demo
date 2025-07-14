#!/usr/bin/env python3
"""
DSW集成脚本 - 自动化DSW训练任务提交和监控
"""
import subprocess
import time
import json
from datetime import datetime

class DSWIntegrationManager:
    def __init__(self, config):
        self.config = config
        self.dsw_instance_name = None
        
    def submit_training_job(self, git_commit_id=None):
        """提交DSW训练任务"""
        print("🚀 提交DSW训练任务...")
        
        # 生成唯一的实例名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.dsw_instance_name = f"walmart-training-{timestamp}"
        
        # 构建训练命令
        training_command = f"""
        cd /root/code/walmart-pai-demo && \
        git pull origin main && \
        export TRAINING_COMMIT_ID='{git_commit_id or 'latest'}' && \
        jupyter nbconvert --execute notebooks/Walmart_Training.ipynb --to notebook
        """
        
        # DSW CLI命令
        dsw_command = [
            "dsw", "run",
            "--instance-name", self.dsw_instance_name,
            "--image", "registry.cn-hangzhou.aliyuncs.com/pai-dlc/pytorch-training:1.12.0-gpu-py38",
            "--command", training_command,
            "--instance-type", "ecs.gn6i-c4g1.xlarge",
            "--timeout", "3600"
        ]
        
        try:
            result = subprocess.run(dsw_command, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ DSW训练任务已提交: {self.dsw_instance_name}")
                return True
            else:
                print(f"❌ DSW任务提交失败: {result.stderr}")
                return False
        except Exception as e:
            print(f"❌ DSW集成失败: {e}")
            return False
    
    def wait_for_completion(self, max_wait_minutes=60):
        """等待训练完成"""
        print(f"⏳ 等待训练完成，最长等待{max_wait_minutes}分钟...")
        
        start_time = time.time()
        max_wait_seconds = max_wait_minutes * 60
        
        while time.time() - start_time < max_wait_seconds:
            # 检查DSW实例状态
            status = self._check_dsw_status()
            
            if status == "Completed":
                print("✅ DSW训练任务已完成")
                return True
            elif status == "Failed":
                print("❌ DSW训练任务失败")
                return False
            elif status == "Running":
                print("🔄 训练进行中...")
                time.sleep(60)  # 等待1分钟
            else:
                print(f"🤔 未知状态: {status}")
                time.sleep(30)
        
        print("⏰ 训练超时")
        return False
    
    def _check_dsw_status(self):
        """检查DSW实例状态"""
        try:
            # 模拟状态检查（实际环境中调用DSW API）
            # 这里可以通过查询MaxCompute训练记录表来判断
            return "Running"  # 简化返回
        except Exception as e:
            print(f"⚠️ 状态检查失败: {e}")
            return "Unknown"

if __name__ == "__main__":
    # 示例用法
    config = {}
    dsw_manager = DSWIntegrationManager(config)
    
    if dsw_manager.submit_training_job():
        dsw_manager.wait_for_completion()