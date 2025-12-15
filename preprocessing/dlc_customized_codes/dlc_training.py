import deeplabcut as dlc
import os
import time

# config.yaml 路径
config_path = r"C:/Users/Administrator/Desktop/DLC_Projects/MouseBehavior-Ziying_Wang-2025-12-12/config.yaml"
video_path = r'C:/Users/Administrator/Desktop/DLC_Projects/MouseBehavior-Ziying_Wang-2025-12-12/videos/1011.mp4'

# 训练网络
print("="*50)
print("开始训练神经网络 (max_iters=5000)")
print("="*50)
start_time = time.time()
try:
    dlc.train_network(
    config_path,
    shuffle=1,
    trainingsetindex=0,
    maxiters=200000,   
    displayiters=1000,
    saveiters=20000, 
)
    training_time = time.time() - start_time
    print(f"训练完成，耗时: {training_time/60:.2f} 分钟")
except Exception as e:
    print(f"训练错误: {e}")
    exit() 

time.sleep(2)

# 分析视频
print("\n" + "="*50)
print("分析视频")
print("="*50)
try:
    dlc.analyze_videos(config_path, [video_path], videotype='.mp4')
    print("视频分析完成")
    dlc.extract_outlier_frames(config_path, [video_path], automatic=True)
    print("困难帧提取完成")
except Exception as e:
    print(f"视频分析失败: {e}")

# 标记视频
print("\n" + "="*50)
print("创建带标记的视频")
print("="*50)
try:
    dlc.create_labeled_video(config_path, [video_path], videotype='.mp4')
    labeled_video_dir = os.path.join(os.path.dirname(config_path), 'videos')  #保存路径
    print(f"标记视频创建完成: {labeled_video_dir}")
except Exception as e:
    print(f"创建标记视频失败: {e}")

# 性能评估
print("\n" + "="*50)
print("性能评估")
print("="*50)
try:
    dlc.evaluate_network(config_path, plotting=True)
    print("评估完成")
except Exception as e:
    print(f"评估失败: {e}")

print("\n" + "="*50)
print("成功; 结束")
print("="*50)