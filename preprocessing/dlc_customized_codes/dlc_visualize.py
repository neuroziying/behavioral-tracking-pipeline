import pandas as pd
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

# 路径设置
data_file = r'C:/Users/Administrator/Desktop/DLC_Projects/MouseBehavior-Ziying_Wang-2025-12-12/videos/1011DLC_resnet50_MouseBehaviorDec12shuffle1_5000.h5'
video_path = r'C:/Users/Administrator/Desktop/DLC_Projects/MouseBehavior-Ziying_Wang-2025-12-12/videos/1011.mp4'

print("加载数据...")
df = pd.read_hdf(data_file)
cap = cv2.VideoCapture(video_path)

# 选择要查看的几个关键帧（覆盖视频的不同时间段）
frame_indices = [0, 500, 1500, 3000, 6000, 10000]

# 创建一个大画布
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

print("生成静态快照...")
for i, (idx, ax) in enumerate(zip(frame_indices, axes)):
    # 跳转到指定帧
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    if not ret:
        ax.text(0.5, 0.5, f'Frame {idx} not found', ha='center')
        ax.axis('off')
        continue
    
    # BGR转RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    ax.imshow(frame_rgb)
    
    # 绘制所有身体部位
    for bodypart in df.columns.get_level_values('bodyparts').unique():
        try:
            x = df.iloc[idx][(slice(None), bodypart, 'x')].values[0]
            y = df.iloc[idx][(slice(None), bodypart, 'y')].values[0]
            
            if not np.isnan(x) and not np.isnan(y):
                # 画点
                ax.plot(x, y, 'o', markersize=10, markeredgewidth=2, 
                       markeredgecolor='white', alpha=0.8)
                # 写标签（只写主要部位，避免杂乱）
                if bodypart in ['nose', 'left_ear', 'right_ear', 'center_back', 'tail_base']:
                    ax.text(x+15, y-10, bodypart, fontsize=9, color='white',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.5))
        except Exception as e:
            pass
    
    ax.set_title(f'Frame {idx} (时间: {idx/30:.1f}s)', fontsize=12)
    ax.axis('off')

plt.tight_layout()
output_path = r'C:/Users/Administrator/Desktop/sample_frames_with_points.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n✅ 静态快照已保存！")
print(f"文件位置: {output_path}")
print("请立即打开查看。如果标记点大致落在小鼠身体上，说明训练基本成功！")

plt.show()
cap.release()