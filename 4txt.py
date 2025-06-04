import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib import rcParams
import seaborn as sns
from tkinter import simpledialog

# 配置中文显示
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False

class FocusVisualizer:
    def __init__(self):
        # 数值映射
        self.emotion_map = {
            '悲伤': 0,
            '愤怒': 0.1,
            '惊讶': 0.2,
            '愉快': 0.3,
            '中性': 0.4,
            'sad': 0,
            'angry': 0.1,
            'surprise': 0.2,
            'happy': 0.3,
            'neutral': 0.4
        }

        self.behavior_map = {
            '阅读': 0.2,
            '举手': 0.1,
            '写作': 0.3,
            'read': 0.2,
            'rise_hand': 0.1,
            'write': 0.3
        }

        # 颜色配置
        self.colors = {
            'behavior': '#3498db',
            'emotion': '#e74c3c',
            'focus': '#2ecc71'
        }

    def select_student_id(self, df):
        """弹出对话框选择要分析的学生ID"""
        unique_ids = df['ID'].unique()
        if len(unique_ids) == 0:
            return None

        # 在控制台打印可选ID
        print("可用的学生ID:", ", ".join(unique_ids))

        # 弹出输入对话框
        selected_id = simpledialog.askstring("选择学生", f"请输入要分析的学生ID（可选ID: {', '.join(unique_ids)}）：")

        if selected_id and selected_id in unique_ids:
            return selected_id
        return None

    def plot_student_analysis(self, df, student_id):
        """针对特定学生的综合分析"""
        student_data = df[df['ID'] == student_id]
        if student_data.empty:
            print(f"未找到ID为 {student_id} 的学生数据")
            return

        plt.figure(figsize=(15, 10))

        # 1. 行为-情绪趋势图
        plt.subplot(2, 2, 1)
        self._plot_student_trend(student_data, student_id)

        # 2. 专注度变化曲线
        plt.subplot(2, 2, 2)
        self._plot_focus_trend(student_data, student_id)

        # 3. 行为分布饼图
        plt.subplot(2, 2, 3)
        self._plot_behavior_distribution(student_data, student_id)

        # 4. 情绪分布饼图
        plt.subplot(2, 2, 4)
        self._plot_emotion_distribution(student_data, student_id)

        plt.tight_layout()
        plt.suptitle(f"学生ID {student_id} 综合分析", y=1.02, fontsize=16)
        plt.show()

    def _plot_student_trend(self, df, student_id):
        """绘制学生行为情绪趋势图（子图）"""
        fig, ax1 = plt.subplots(figsize=(8, 4))

        # 情绪折线图
        ax1.plot(df['帧序'], df['情绪'].map(self.emotion_map),
                 marker='o', linestyle='--', color=self.colors['emotion'],
                 label='情绪状态')
        ax1.set_ylabel('情绪状态')
        ax1.set_yticks(list(self.emotion_map.values())[:5])
        ax1.set_yticklabels(list(self.emotion_map.keys())[:5])

        # 行为阶梯图
        ax2 = ax1.twinx()
        ax2.step(df['帧序'], df['行为'].map(self.behavior_map),
                 where='post', linestyle='--', color=self.colors['behavior'],
                 linewidth=2, label='行为类型')
        ax2.set_ylabel('行为类型')
        ax2.set_yticks(list(self.behavior_map.values())[:3])
        ax2.set_yticklabels(list(self.behavior_map.keys())[:3])

        plt.title(f'行为-情绪趋势 (ID:{student_id})')
        plt.grid(True, alpha=0.3)

    def _plot_focus_trend(self, df, student_id):
        """绘制专注度变化曲线（子图）"""
        plt.plot(df['帧序'], df['专注度'], marker='o', color=self.colors['focus'])
        plt.xlabel('帧序列')
        plt.ylabel('专注度')
        plt.title(f'专注度变化 (ID:{student_id})')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)

    def _plot_behavior_distribution(self, df, student_id):
        """绘制行为分布饼图（子图）"""
        behavior_counts = df['行为'].value_counts()
        behavior_counts.plot.pie(autopct='%1.1f%%',
                                 colors=[self.colors['behavior']]*len(behavior_counts),
                                 startangle=90)
        plt.title(f'行为分布 (ID:{student_id})')
        plt.ylabel('')

    def _plot_emotion_distribution(self, df, student_id):
        """绘制情绪分布饼图（子图）"""
        emotion_counts = df['情绪'].value_counts()
        emotion_counts.plot.pie(autopct='%1.1f%%',
                                colors=[self.colors['emotion']]*len(emotion_counts),
                                startangle=90)
        plt.title(f'情绪分布 (ID:{student_id})')
        plt.ylabel('')

    def read_data_from_txt(self, file_path):
        """从文本文件读取分析结果数据"""
        data = []
        frame_data = []

        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()

                if line.startswith("Processing image:"):
                    if frame_data:
                        data.append(frame_data)
                        frame_data = []
                elif line.startswith("Person ID:"):
                    try:
                        parts = line.split(', ')
                        person_data = {}

                        for part in parts:
                            if ': ' in part:
                                key, value = part.split(': ')
                                person_data[key] = value

                        # 转换数据类型
                        person_data["Confidence"] = float(person_data.get("Confidence", 0))
                        person_data["Emotion Confidence"] = float(person_data.get("Emotion Confidence", 0))
                        person_data["Focus Score"] = float(person_data.get("Focus Score", 0))

                        frame_data.append(person_data)
                    except Exception as e:
                        print(f"处理行时出错: {line}, 错误: {e}")
                        continue

        if frame_data:
            data.append(frame_data)

        return data

    def create_dataframe(self, data):
        """将原始数据转换为DataFrame格式"""
        records = []
        for frame_id, frame in enumerate(data, 1):
            for person in frame:
                records.append({
                    '帧序': frame_id,
                    'ID': person.get('ID', ''),
                    '行为': person.get('Behavior', ''),
                    '行为置信度': person.get('Confidence', 0),
                    '情绪': person.get('Emotion', ''),
                    '情绪置信度': person.get('Emotion Confidence', 0),
                    '专注度': person.get('Focus Score', 0)
                })
        return pd.DataFrame(records)

    def visualize_all(self, file_path):
        """执行全套可视化分析"""
        try:
            raw_data = self.read_data_from_txt(file_path)
            if not raw_data:
                print("警告: 没有读取到有效数据")
                return

            df = self.create_dataframe(raw_data)

            if df.empty:
                print("警告: 数据框为空")
                return

            # 选择特定学生分析
            selected_id = self.select_student_id(df)
            if selected_id:
                self.plot_student_analysis(df, selected_id)
            else:
                print("未选择有效ID，将展示全体数据")
                # 这里可以添加全体数据分析的代码

        except Exception as e:
            print(f"分析过程中出错: {str(e)}")


if __name__ == "__main__":
    visualizer = FocusVisualizer()
    visualizer.visualize_all('results-id.txt')