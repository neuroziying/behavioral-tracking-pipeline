import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

data_file = r'C:/Users/Administrator/Desktop/DLC_Projects/MouseBehavior-Ziying_Wang-2025-12-12/videos/1011DLC_resnet50_MouseBehaviorDec12shuffle1_5000.h5'

# 读取数据
print("="*60)
print("读取dlc文件")
print("="*60)

df = pd.read_hdf(data_file)
print(f"文件: {os.path.basename(data_file)}")
print(f"总帧数: {df.shape[0]}, 总列数: {df.shape[1]}")
print(f"身体部位数: {len(df.columns.get_level_values('bodyparts').unique())}")
print()



print("加载数据...")
df = pd.read_hdf(data_file)

print("="*50)
print("数据列结构分析")
print("="*50)

# 查看实际的列层级结构
print(f"列索引层级数: {df.columns.nlevels}")
print(f"列索引名称: {df.columns.names}")
print(f"\n前5个列名示例:")
for i, col in enumerate(df.columns[:5]):
    print(f"  {i}: {col}")

print("\n" + "="*50)
print("部位混淆分析 - 简化版")
print("="*50)

# 方法1：尝试按层级提取
if df.columns.nlevels == 3:
    # 如果是3层结构
    bodyparts = df.columns.get_level_values('bodyparts').unique()
    print(f"检测到3层结构，身体部位: {list(bodyparts)}")
    
    # 分析每个部位
    results = []
    for bp in bodyparts:
        try:
            # 提取该部位的所有列
            bp_cols = [col for col in df.columns if col[1] == bp]
            bp_df = df[bp_cols]
            
            # 提取x, y列
            x_cols = [col for col in bp_cols if col[2] == 'x']
            y_cols = [col for col in bp_cols if col[2] == 'y']
            
            if x_cols and y_cols:
                x_vals = df[x_cols[0]].dropna().values
                y_vals = df[y_cols[0]].dropna().values
                
                if len(x_vals) > 0:
                    results.append({
                        'bodypart': bp,
                        'mean_x': np.mean(x_vals),
                        'mean_y': np.mean(y_vals),
                        'points': len(x_vals),
                        'std_x': np.std(x_vals),
                        'std_y': np.std(y_vals)
                    })
        except:
            pass
            
elif df.columns.nlevels == 2:
    # 如果是2层结构
    print("检测到2层结构，尝试解析列名...")
    
    # 从列名中提取部位名称
    bodyparts_set = set()
    for col in df.columns:
        # 假设列名格式类似 "DLC_resnet50_..._nose_x"
        parts = str(col).split('_')
        # 寻找可能的部位名称（排除模型名和坐标类型）
        for part in parts:
            if part in ['x', 'y', 'likelihood', 'DLC', 'resnet50', 'MouseBehaviorDec12shuffle1', '5000']:
                continue
            if len(part) > 2:  # 假设部位名称长度>2
                bodyparts_set.add(part)
    
    bodyparts = list(bodyparts_set)
    print(f"从列名解析出的可能部位: {bodyparts}")
    
    results = []
    for bp in bodyparts:
        # 查找包含该部位名的列
        x_cols = [col for col in df.columns if bp in str(col) and '_x' in str(col)]
        y_cols = [col for col in df.columns if bp in str(col) and '_y' in str(col)]
        
        if x_cols and y_cols:
            x_vals = df[x_cols[0]].dropna().values
            y_vals = df[y_cols[0]].dropna().values
            
            if len(x_vals) > 0:
                results.append({
                    'bodypart': bp,
                    'mean_x': np.mean(x_vals),
                    'mean_y': np.mean(y_vals),
                    'points': len(x_vals),
                    'std_x': np.std(x_vals),
                    'std_y': np.std(y_vals)
                })

# 显示结果
if results:
    print(f"\n{'部位':<15} {'平均X':>10} {'平均Y':>10} {'标准差X':>10} {'标准差Y':>10} {'数据点':>10}")
    print("-" * 75)
    
    for r in sorted(results, key=lambda x: x['mean_x']):
        print(f"{r['bodypart']:<15} {r['mean_x']:>10.1f} {r['mean_y']:>10.1f} "
              f"{r['std_x']:>10.1f} {r['std_y']:>10.1f} {r['points']:>10}")
    
    # 可视化
    plt.figure(figsize=(12, 8))
    
    # 散点图：各部位平均位置
    plt.subplot(1, 2, 1)
    for r in results:
        plt.scatter(r['mean_x'], r['mean_y'], s=150, alpha=0.7, label=r['bodypart'])
        plt.errorbar(r['mean_x'], r['mean_y'], 
                    xerr=r['std_x'], yerr=r['std_y'],
                    alpha=0.3, capsize=5)
        plt.text(r['mean_x']+15, r['mean_y']+15, r['bodypart'], fontsize=9)
    
    plt.xlabel('X(pix)')
    plt.ylabel('Y(pix)')
    plt.title('average(std)')
    plt.grid(True, alpha=0.3)
    
    # 热力图：位置重叠度分析
    plt.subplot(1, 2, 2)
    from scipy.spatial.distance import pdist, squareform
    
    positions = np.array([[r['mean_x'], r['mean_y']] for r in results])
    labels = [r['bodypart'] for r in results]
    
    # 计算各部位间的距离矩阵
    dist_matrix = squareform(pdist(positions))
    
    # 绘制热力图
    im = plt.imshow(dist_matrix, cmap='viridis_r')
    plt.colorbar(im, label='dist_bodyparts (pix)')
    plt.xticks(range(len(labels)), labels, rotation=45, ha='right')
    plt.yticks(range(len(labels)), labels)
    plt.title('average dist')
    
    # 在热力图上显示数值
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i != j:
                plt.text(j, i, f'{dist_matrix[i, j]:.0f}', 
                        ha='center', va='center', color='white', fontsize=8)
    
    plt.tight_layout()
    output_path = r'C:/Users\Administrator\Desktop\bodypart_analysis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ 分析图已保存: {output_path}")
    plt.show()
    
    # 找出可能混淆的部位（距离过近的）
    print("\n" + "="*50)
    print("潜在混淆部位分析（距离<50像素）")
    print("="*50)
    
    for i in range(len(results)):
        for j in range(i+1, len(results)):
            dist = np.sqrt((results[i]['mean_x'] - results[j]['mean_x'])**2 + 
                          (results[i]['mean_y'] - results[j]['mean_y'])**2)
            if dist < 50:  # 阈值设为50像素
                print(f"⚠️  {results[i]['bodypart']} 和 {results[j]['bodypart']} 平均距离仅 {dist:.1f} 像素，可能混淆！")
    
else:
    print("未能提取到部位数据，请检查数据文件结构。")
    print("\n尝试直接打印数据框头部:")
    print(df.head())
# 4. 基础统计
print("基础统计分析:")
print("-"*40)
# 提取所有x坐标和y坐标
all_x = df.xs('x', axis=1, level=2)
all_y = df.xs('y', axis=1, level=2)

print(f"   X坐标范围: [{all_x.min().min():.1f}, {all_x.max().max():.1f}] 像素")
print(f"   Y坐标范围: [{all_y.min().min():.1f}, {all_y.max().max():.1f}] 像素")
print(f"   X坐标均值: {all_x.mean().mean():.1f} ± {all_x.std().mean():.1f} 像素")
print(f"   Y坐标均值: {all_y.mean().mean():.1f} ± {all_y.std().mean():.1f} 像素")
print()

# 5. 置信度分析
print("预测置信度分析:")
print("-"*40)
likelihoods = df.xs('likelihood', axis=1, level=2)
mean_confidence = likelihoods.mean().mean()
low_confidence_frames = (likelihoods < 0.6).any(axis=1).sum()  # 置信度低于0.6的帧

print(f"   平均置信度: {mean_confidence:.3f} (范围0-1, 越高越好)")
print(f"   存在低置信度点(<0.6)的帧数: {low_confidence_frames}/{len(df)} ({(low_confidence_frames/len(df)*100):.1f}%)")
print("="*60)

# 6. 导出为扁平化CSV（用于其他软件如Excel, Prism）
print("\n💾 导出数据为通用格式...")
output_csv = data_file.replace('.h5', '_flat.csv')
# 扁平化列名：将多级索引合并为单级
df_flat = df.copy()
df_flat.columns = ['_'.join(col).strip() for col in df_flat.columns.values]
df_flat.to_csv(output_csv)
print(f"✅ 已导出扁平化CSV文件: {os.path.basename(output_csv)}")
print(f"   路径: {output_csv}")
print("   此文件可直接用Excel、GraphPad Prism、MATLAB等软件打开分析。")