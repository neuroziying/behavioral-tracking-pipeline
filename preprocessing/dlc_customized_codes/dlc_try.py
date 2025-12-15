import deeplabcut as dlc
import os

# 使用你验证通过的绝对路径
video_path = r'c:/Users/Administrator/Desktop/1011.mp4'
# 为项目选择一个干净的英文工作目录
working_dir = r'C:/Users/Administrator/Desktop/DLC_Projects'

config_path = dlc.create_new_project(
    project='MouseBehavior',  # 新项目名，避免和旧项目混淆
    experimenter='Ziying_Wang',
    videos=[video_path],
    copy_videos=True,               # 重要：复制视频到项目文件夹
    working_directory=working_dir   # 项目将创建在这个目录下
)

print(f"✅ 项目创建成功！")
print(f"配置文件: {config_path}")

# 第二步：自动提取帧
print("\n正在从视频中提取关键帧...")
dlc.extract_frames(config_path, 'automatic', 'kmeans')
print("✅ 帧提取完成！")

# 第三步：创建训练数据集（初始化标注文件）
print("\n正在创建训练数据集...")
dlc.create_training_dataset(config_path)
print("✅ 训练数据集已创建。")

# 第四步：打开标注工具
print("\n正在启动标注工具...")
dlc.label_frames(config_path)
print("标注工具已启动。请在新窗口中开始标注。")