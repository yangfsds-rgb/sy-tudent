import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import rcParams
import seaborn as sns
import numpy as np

# 配置中文显示
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False

# 数值映射
emotion_map = {
    '悲伤': 0,
    '愤怒': 0.1,
    '惊讶': 0.2,
    '愉快': 0.3,
    '中性': 0.4
}

behavior_map = {
    '阅读': 0.2,
    '举手': 0.1,
    '写作': 0.3
}

# 创建示例数据
def create_data():
    frames = range(1, 19)
    behavior = ['阅读']*18
    emotion = ['悲伤','悲伤','中性','悲伤','悲伤','悲伤','悲伤','悲伤',
               '悲伤','悲伤','悲伤','悲伤','悲伤','悲伤','悲伤','悲伤','悲伤','悲伤']

    df = pd.DataFrame({
        '帧序': frames,
        '行为': behavior,
        '情绪': emotion
    })

    return df

# 情绪与行为的折线图与阶梯图
def plot_behavior_emotion(df):
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # 情绪折线（带标记）
    ax1.plot(df['帧序'], df['情绪'].map(emotion_map),
             marker='o', linestyle='--', color='#e74c3c', label='情绪状态')
    ax1.set_ylabel('情绪状态', fontsize=12)
    ax1.set_yticks([0, 0.1, 0.2, 0.3, 0.4])
    ax1.set_yticklabels(['悲伤', '愤怒', '惊讶', '愉快', '中性'])

    # 行为阶梯图（设置虚线）
    ax2 = ax1.twinx()  # 创建第二个y轴
    ax2.step(df['帧序'], df['行为'].map(behavior_map),
             where='post', linestyle='--', color='#3498db', linewidth=3, label='行为持续')
    ax2.set_ylabel('行为类型', fontsize=12)
    ax2.set_yticks([0.1, 0.2, 0.3])
    ax2.set_yticklabels(['举手', '阅读', '写作'])

    # 增强可视化
    ax1.set_xlabel('连续帧序列', fontsize=12)
    ax1.set_xticks(range(1, 19))
    ax1.set_title('学生行为-情绪时序变化 (ID:1)', pad=20, fontsize=14)

    # 在每一帧上添加标注
    for i, row in df.iterrows():
        ax1.text(row['帧序'], emotion_map[row['情绪']], row['情绪'], ha='center', va='bottom', fontsize=10, color='red')
        ax2.text(row['帧序'], behavior_map[row['行为']], row['行为'], ha='center', va='top', fontsize=10, color='blue')

    # 添加网格
    ax1.grid(axis='y', alpha=0.3)
    fig.tight_layout()

    # 添加图例
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    plt.show()

# 行为与情绪关联矩阵的热图
def plot_crosstab(df):
    cross_tab = pd.crosstab(df['行为'], df['情绪'], normalize='all') * 100

    plt.figure(figsize=(8, 4))
    sns.heatmap(cross_tab, annot=True, fmt=".1f", cmap="YlGnBu",
                linewidths=.5, cbar_kws={'label': '关联强度 (%)'})

    plt.title('行为-情绪关联矩阵', pad=15)
    plt.xlabel('情绪状态')
    plt.ylabel('行为类型')
    plt.xticks(rotation=0)
    plt.show()

# 行为-情绪联合分布比例的堆积柱状图
def plot_joint_distribution(df):
    joint_dist = pd.crosstab(df['行为'], df['情绪'], normalize='index') * 100

    plt.figure(figsize=(10, 6))
    colors = ['#e74c3c', '#f1c40f', '#2ecc71', '#3498db', '#9b59b6']
    joint_dist.plot(kind='bar', stacked=True, color=colors, edgecolor='black')

    plt.ylabel('比例分布 (%)', fontsize=12, labelpad=10)
    plt.xlabel('行为类型', fontsize=12, labelpad=10)
    plt.xticks(rotation=0, fontsize=10)
    plt.yticks(fontsize=10)
    plt.title('行为-情绪联合分布比例', pad=15, fontsize=14, fontweight='bold')
    plt.legend(title='情绪状态', bbox_to_anchor=(1.02, 1), loc='upper left')

    # 添加数值标注
    for bars in plt.gca().containers:
        plt.bar_label(bars, fmt='%.1f%%', label_type='center',
                      color='white', fontsize=8, fontweight='bold')

    plt.tight_layout()
    plt.show()

# 情绪状态时序分布极坐标图
def plot_polar(df):
    categories = list(emotion_map.keys())
    values = df['情绪'].value_counts().reindex(categories, fill_value=0).values
    values = np.append(values, values[0])  # 闭合曲线
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)
    ax.plot(angles, values, color='#3498db', linewidth=2, linestyle='-', marker='o')
    ax.fill(angles, values, color='#3498db', alpha=0.25)

    # 极坐标轴设置
    ax.set_theta_offset(np.pi/2)
    ax.set_theta_direction(-1)
    plt.xticks(angles[:-1], categories, fontsize=10)
    ax.tick_params(axis='y', labelsize=8, grid_alpha=0.3)
    ax.set_rlabel_position(0)
    plt.title('情绪状态时序分布极坐标图', y=1.15, fontsize=14, fontweight='bold')

    # 添加统计注释
    plt.figtext(0.5, 0.95, f"总样本量: {len(df)} | 主导情绪: {df['情绪'].mode()[0]}",
                ha='center', fontsize=10, style='italic')
    plt.show()

# 主函数，调用其他绘图函数
def main():
    df = create_data()
    plot_behavior_emotion(df)
    plot_crosstab(df)
    plot_joint_distribution(df)
    plot_polar(df)

# 执行主函数
if __name__ == "__main__":
    main()
