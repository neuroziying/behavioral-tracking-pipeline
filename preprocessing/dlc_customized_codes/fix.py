import deeplabcut as dlc
import os

print("="*60)
print("DLC 2.3.11 恢复训练")
print("="*60)

# 重要：config文件路径（根据你的实际位置调整）
config_path = 'C:/Users/Administrator/Desktop/DLC_Projects/MouseBehavior-Ziying_Wang-2025-12-12/config.yaml'

# 验证文件存在
if not os.path.exists(config_path):
    print(f"❌ 找不到配置文件: {config_path}")
    # 尝试其他可能位置
    config_path = 'config.yaml'  # 当前目录
    if not os.path.exists(config_path):
        print("❌ 当前目录也没有config.yaml")
        exit()

print(f"✅ 使用配置文件: {config_path}")

# 检查点目录
checkpoint_dir = r'C:\Users\Administrator\Desktop\DLC_Projects\MouseBehavior-Ziying_Wang-2025-12-12\dlc-models\iteration-0\MouseBehaviorDec12-trainset95shuffle1\train'

if os.path.exists(checkpoint_dir):
    files = os.listdir(checkpoint_dir)
    snapshot_files = [f for f in files if f.startswith('snapshot-')]
    print(f"检查点目录: {checkpoint_dir}")
    print(f"找到 {len(snapshot_files)} 个检查点文件:")
    for f in sorted(snapshot_files):
        print(f"  - {f}")
else:
    print(f"❌ 检查点目录不存在: {checkpoint_dir}")

# 确认
response = input("\n是否开始恢复训练？(y/n): ")
if response.lower() != 'y':
    print("恢复取消")
    exit()

print("\n开始训练...")

# DLC 2.3.11的正确调用方式
try:
    # 方法1：使用位置参数
    dlc.train_network(
        config_path,      # 配置文件
        1,                # shuffle (必须和之前一样！)
        0,                # trainingsetindex
        displayiters=1000,
        saveiters=20000,   # 设为20000，更频繁保存
        maxiters=200000,
        max_snapshots_to_keep=5
    )
except Exception as e:
    print(f"错误: {e}")
    print("\n尝试另一种调用方式...")
    
    # 方法2：使用**kwargs
    params = {
        'shuffle': 1,
        'trainingsetindex': 0,
        'displayiters': 1000,
        'saveiters': 20000,
        'maxiters': 200000,
        'max_snapshots_to_keep': 5
    }
    dlc.train_network(config_path, **params)