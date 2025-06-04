import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib import rcParams
import seaborn as sns

# 配置中文显示
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False

# 数值映射
emotion_map = {
    'Sad': 0,
    'Angry': 0.1,
    'Surprised': 0.2,
    'Happy': 0.3,
    'Neutral': 0.4
}

behavior_map = {
    'Reading': 0.2,
    'Raising Hand': 0.1,
    'Writing': 0.3
}

# 创建示例数据
def create_data():
    frames = range(1, 19)
    behavior = ['Reading']*18
    emotion = ['Sad','Sad','Neutral','Sad','Sad','Sad','Sad','Sad',
               'Sad','Sad','Sad','Sad','Sad','Sad','Sad','Sad','Sad','Sad']
    df = pd.DataFrame({
        'Frame': frames,
        'Behavior': behavior,
        'Emotion': emotion
    })
    return df

# 情绪与行为的折线图与阶梯图
def plot_behavior_emotion(df):
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Emotion line plot (with markers)
    ax1.plot(df['Frame'], df['Emotion'].map(emotion_map),
             marker='o', linestyle='--', color='#e74c3c', label='Emotion State')
    ax1.set_ylabel('Emotion State', fontsize=12)
    ax1.set_yticks([0, 0.1, 0.2, 0.3, 0.4])
    ax1.set_yticklabels(['Sad', 'Anger', 'Surprise', 'Happy', 'Neutral'])

    # Behavior step plot (with dashed lines)
    ax2 = ax1.twinx()  # Create second y-axis
    ax2.step(df['Frame'], df['Behavior'].map(behavior_map),
             where='post', linestyle='--', color='#3498db', linewidth=3, label='Behavior Duration')
    ax2.set_ylabel('Behavior Type', fontsize=12)
    ax2.set_yticks([0.1, 0.2, 0.3])
    ax2.set_yticklabels(['Raise Hand', 'Reading', 'Writing'])

    # Enhance visualization
    ax1.set_xlabel('Sequential Frame Number', fontsize=12)
    ax1.set_xticks(range(1, 19))
    ax1.set_title('Student Behavior-Emotion Time Series (ID:1)', pad=20, fontsize=14)

    # Add annotations for each frame
    for i, row in df.iterrows():
        ax1.text(row['Frame'], emotion_map[row['Emotion']], row['Emotion'], ha='center', va='bottom', fontsize=10, color='red')
        ax2.text(row['Frame'], behavior_map[row['Behavior']], row['Behavior'], ha='center', va='top', fontsize=10, color='blue')

    # Add grid
    ax1.grid(axis='y', alpha=0.3)
    fig.tight_layout()

    # Add legend
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    plt.show()


# 行为与情绪关联矩阵的热图
def plot_crosstab(df):
    cross_tab = pd.crosstab(df['Behavior'], df['Emotion'], normalize='all') * 100

    plt.figure(figsize=(8, 4))
    sns.heatmap(cross_tab, annot=True, fmt=".1f", cmap="YlGnBu",
                linewidths=.5, cbar_kws={'label': 'Association Strength (%)'})

    plt.title('Behavior-Emotion Association Matrix', pad=15)
    plt.xlabel('Emotion State')
    plt.ylabel('Behavior Type')
    plt.xticks(rotation=0)
    plt.show()


# 行为-情绪联合分布比例的堆积柱状图
def plot_joint_distribution(df):
    joint_dist = pd.crosstab(df['Behavior'], df['Emotion'], normalize='index') * 100

    plt.figure(figsize=(10, 6))
    colors = ['#e74c3c', '#f1c40f', '#2ecc71', '#3498db', '#9b59b6']
    joint_dist.plot(kind='bar', stacked=True, color=colors, edgecolor='black')

    plt.ylabel('Proportion Distribution (%)', fontsize=12, labelpad=10)
    plt.xlabel('Behavior Type', fontsize=12, labelpad=10)
    plt.xticks(rotation=0, fontsize=10)
    plt.yticks(fontsize=10)
    plt.title('Behavior-Emotion Joint Distribution Proportions', pad=15, fontsize=14, fontweight='bold')
    plt.legend(title='Emotion State', bbox_to_anchor=(1.02, 1), loc='upper left')

    # Add value annotations
    for bars in plt.gca().containers:
        plt.bar_label(bars, fmt='%.1f%%', label_type='center',
                      color='white', fontsize=8, fontweight='bold')

    plt.tight_layout()
    plt.show()


# 情绪状态时序分布极坐标图
# Emotion state time sequence distribution polar plot
def plot_polar(df):
    categories = list(emotion_map.keys())
    values = df['Emotion'].value_counts().reindex(categories, fill_value=0).values
    values = np.append(values, values[0])  # Close the curve
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)
    ax.plot(angles, values, color='#3498db', linewidth=2, linestyle='-', marker='o')
    ax.fill(angles, values, color='#3498db', alpha=0.25)

    # Polar axis settings
    ax.set_theta_offset(np.pi/2)
    ax.set_theta_direction(-1)
    plt.xticks(angles[:-1], categories, fontsize=10)
    ax.tick_params(axis='y', labelsize=8, grid_alpha=0.3)
    ax.set_rlabel_position(0)
    plt.title('Emotion State Time Sequence Distribution Polar Plot', y=1.15, fontsize=14, fontweight='bold')

    # Add statistical annotation
    plt.figtext(0.5, 0.95, f"Total Samples: {len(df)} | Dominant Emotion: {df['Emotion'].mode()[0]}",
                ha='center', fontsize=10, style='italic')
    plt.show()

# 行为、情绪和专注度的条形图
def plot_behavior_emotion_focus(data, title='Behavior, Emotion, and Focus Information'):
    ids = [item["ID"] for item in data]
    behaviors = [item["Behavior"] for item in data]
    emotions = [item["Emotion"] for item in data]
    confidences = [item["Confidence"] for item in data]
    emotion_confidences = [item["Emotion Confidence"] for item in data]
    focus_scores = [item["Focus Score"] for item in data]

    behavior_color = "blue"
    emotion_color = "green"
    focus_color = "cyan"

    fig, ax = plt.subplots(figsize=(12, 8))

    bar_width = 0.2
    index = np.arange(len(data))

    bars_behavior = ax.barh(index, confidences, height=bar_width, color=behavior_color, label="Behavior Confidence")
    bars_emotion_confidence = ax.barh(index + bar_width, emotion_confidences, height=bar_width, color=emotion_color, label="Emotion Confidence")
    bars_focus = ax.barh(index + 2 * bar_width, focus_scores, height=bar_width, color=focus_color, label="Focus Score")

    ax.set_yticks(index + bar_width)
    ax.set_yticklabels([f"Person {id}" for id in ids])

    ax.set_xlabel('Value')
    ax.set_title(title)

    ax.legend()

    for i, bar in enumerate(bars_behavior):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2, f"{confidences[i]:.2f}", va="center", fontsize=10)
    for i, bar in enumerate(bars_emotion_confidence):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2, f"{emotion_confidences[i]:.2f}", va="center", fontsize=10)
    for i, bar in enumerate(bars_focus):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2, f"{focus_scores[i]:.2f}", va="center", fontsize=10)

    plt.tight_layout()
    plt.show()

# 示例数据
data = [
    {"ID": 1, "Behavior": "write", "Confidence": 0.41, "Emotion": "neutral", "Emotion Confidence": 0.69, "Focus Score": 0.20},
    {"ID": 2, "Behavior": "read", "Confidence": 0.43, "Emotion": "sad", "Emotion Confidence": 1.00, "Focus Score": 0.03},
    {"ID": 3, "Behavior": "rise hand", "Confidence": 0.48, "Emotion": "fear", "Emotion Confidence": 0.68, "Focus Score": 0.01},
    {"ID": 4, "Behavior": "read", "Confidence": 0.60, "Emotion": "neutral", "Emotion Confidence": 1.00, "Focus Score": 0.17},
    {"ID": 5, "Behavior": "read", "Confidence": 0.37, "Emotion": "happy", "Emotion Confidence": 0.97, "Focus Score": 0.35}
]

# 调用图形化函数
df = create_data()

plot_behavior_emotion(df)
plot_crosstab(df)
plot_joint_distribution(df)
plot_polar(df)
plot_behavior_emotion_focus(data)
