import matplotlib.pyplot as plt
import numpy as np

def plot_behavior_emotion_focus(data, title='Behavior, Emotion, and Focus Information'):
    """
    绘制行为、情绪和专注度的条形图。

    Parameters:
    data (list of dict): 每个数据点的字典，包含"ID"、"Behavior"、"Confidence"、"Emotion"、"Emotion Confidence"、"Focus Score"等字段。
    title (str): 图表的标题，默认为'Behavior, Emotion, and Focus Information'。
    """
    # 处理数据
    ids = [item["ID"] for item in data]
    behaviors = [item["Behavior"] for item in data]
    emotions = [item["Emotion"] for item in data]
    confidences = [item["Confidence"] for item in data]
    emotion_confidences = [item["Emotion Confidence"] for item in data]
    focus_scores = [item["Focus Score"] for item in data]

    # 设置颜色
    behavior_color = "blue"
    emotion_color = "green"
    focus_color = "cyan"

    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 8))

    # 绘制每个人的数据
    bar_width = 0.2
    index = np.arange(len(data))  # 每个人的位置

    # 绘制行为条形图
    bars_behavior = ax.barh(index, confidences, height=bar_width, color=behavior_color, label="Behavior Confidence")

    # 绘制情绪信心条形图
    bars_emotion_confidence = ax.barh(index + bar_width, emotion_confidences, height=bar_width, color=emotion_color, label="Emotion Confidence")

    # 绘制专注度条形图
    bars_focus = ax.barh(index + 2 * bar_width, focus_scores, height=bar_width, color=focus_color, label="Focus Score")

    # 设置标签
    ax.set_yticks(index + bar_width)
    ax.set_yticklabels([f"Person {id}" for id in ids])

    ax.set_xlabel('Value')
    ax.set_title(title)

    # 显示图例
    ax.legend()

    # 标注每个条形图的值
    for i, bar in enumerate(bars_behavior):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2, f"{confidences[i]:.2f}", va="center", fontsize=10)
    for i, bar in enumerate(bars_emotion_confidence):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2, f"{emotion_confidences[i]:.2f}", va="center", fontsize=10)
    for i, bar in enumerate(bars_focus):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2, f"{focus_scores[i]:.2f}", va="center", fontsize=10)

    # 显示图形
    plt.tight_layout()
    plt.show()

# 示例数据
data = [
    {"ID": 1, "Behavior": "write", "Confidence": 0.41, "Emotion": "neutral", "Emotion Confidence": 0.69, "Focus Score": 0.20},
    {"ID": 2, "Behavior": "read", "Confidence": 0.43, "Emotion": "sad", "Emotion Confidence": 1.00, "Focus Score": 0.03},
    {"ID": 3, "Behavior": "rise hand", "Confidence": 0.48, "Emotion": "fear", "Emotion Confidence": 0.68, "Focus Score": 0.01},
    {"ID": 4, "Behavior": "read", "Confidence": 0.60, "Emotion": "neutral", "Emotion Confidence": 1.00, "Focus Score": 0.17},
    {"ID": 5, "Behavior": "read", "Confidence": 0.81, "Emotion": "neutral", "Emotion Confidence": 0.98, "Focus Score": 0.22},
    {"ID": 6, "Behavior": "read", "Confidence": 0.82, "Emotion": "angry", "Emotion Confidence": 1.00, "Focus Score": 0.18},
    {"ID": 7, "Behavior": "read", "Confidence": 0.86, "Emotion": "sad", "Emotion Confidence": 0.54, "Focus Score": 0.15},
    {"ID": 8, "Behavior": "read", "Confidence": 0.96, "Emotion": "fear", "Emotion Confidence": 0.77, "Focus Score": 0.14}
]

# 调用封装好的函数
plot_behavior_emotion_focus(data)
