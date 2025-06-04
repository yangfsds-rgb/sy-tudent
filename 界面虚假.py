import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
from PIL import Image, ImageTk
import threading
import time
import os
import pandas as pd
from focus_analyzer import FocusAnalyzer
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

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

# 示例数据
data = [
    {"ID": 1, "Behavior": "write", "Confidence": 0.41, "Emotion": "neutral", "Emotion Confidence": 0.69, "Focus Score": 0.20},
    {"ID": 2, "Behavior": "read", "Confidence": 0.43, "Emotion": "sad", "Emotion Confidence": 1.00, "Focus Score": 0.03},
    {"ID": 3, "Behavior": "rise hand", "Confidence": 0.48, "Emotion": "fear", "Emotion Confidence": 0.68, "Focus Score": 0.01},
    {"ID": 4, "Behavior": "read", "Confidence": 0.60, "Emotion": "neutral", "Emotion Confidence": 1.00, "Focus Score": 0.17},
    {"ID": 5, "Behavior": "read", "Confidence": 0.37, "Emotion": "happy", "Emotion Confidence": 0.97, "Focus Score": 0.35}
]

class FocusAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("课堂专注度分析系统 v1.0")
        self.root.geometry("1400x900")

        # 初始化分析器
        self.analyzer = None
        self.model_loaded = False

        # 视频/摄像头控制变量
        self.video_source = None
        self.cap = None
        self.is_playing = False
        self.current_frame = None
        self.focus_scores = []

        # 图表相关变量
        self.df = create_data()
        self.current_figures = []
        self.current_canvases = []

        # 创建界面
        self.create_widgets()

        # 状态栏
        self.status_var = tk.StringVar()
        self.status_var.set("就绪")
        self.status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def create_widgets(self):
        # 主框架
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 左侧控制面板
        control_panel = ttk.LabelFrame(main_frame, text="控制面板", width=300)
        control_panel.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        # 右侧显示面板
        display_panel = ttk.LabelFrame(main_frame, text="可视化展示", width=1100)
        display_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 控制面板内容
        self.create_control_panel(control_panel)

        # 显示面板内容
        self.create_display_panel(display_panel)

    def create_control_panel(self, parent):
        # 模型加载部分
        model_frame = ttk.LabelFrame(parent, text="模型加载")
        model_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(model_frame, text="加载YOLOv9模型", command=self.load_model).pack(fill=tk.X, padx=5, pady=2)

        # 媒体源选择部分
        source_frame = ttk.LabelFrame(parent, text="媒体源选择")
        source_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(source_frame, text="打开图片", command=self.open_image).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(source_frame, text="打开视频", command=self.open_video).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(source_frame, text="打开摄像头", command=self.open_camera).pack(fill=tk.X, padx=5, pady=2)

        # 分析控制部分
        analysis_frame = ttk.LabelFrame(parent, text="分析控制")
        analysis_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(analysis_frame, text="开始分析", command=self.start_analysis).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(analysis_frame, text="停止分析", command=self.stop_analysis).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(analysis_frame, text="保存结果", command=self.save_results).pack(fill=tk.X, padx=5, pady=2)

        # 结果显示部分
        result_frame = ttk.LabelFrame(parent, text="分析结果")
        result_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.result_text = tk.Text(result_frame, height=15, state=tk.DISABLED)
        scrollbar = ttk.Scrollbar(result_frame, command=self.result_text.yview)
        self.result_text.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.result_text.pack(fill=tk.BOTH, expand=True)

        # 可视化按钮
        vis_frame = ttk.LabelFrame(parent, text="可视化")
        vis_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(vis_frame, text="显示行为-情绪时序图",
                   command=lambda: self.show_behavior_emotion_plot()).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(vis_frame, text="显示关联矩阵热图",
                   command=lambda: self.show_crosstab_plot()).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(vis_frame, text="显示行为-情绪-专注度图",
                   command=lambda: self.show_behavior_emotion_focus_plot()).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(vis_frame, text="显示联合分布图",
                   command=lambda: self.show_joint_distribution_plot()).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(vis_frame, text="显示所有图表",
                   command=lambda: self.show_all_plots()).pack(fill=tk.X, padx=5, pady=2)

    def create_display_panel(self, parent):
        # 创建画布框架和视频标签框架
        self.canvas_frame = ttk.Frame(parent)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True, side=tk.RIGHT)  # 调整布局

        # 新增：视频显示标签框架
        self.video_frame = ttk.Frame(parent)
        self.video_frame.pack(fill=tk.BOTH, expand=True, side=tk.LEFT, padx=5, pady=5)

        # 初始化视频标签
        self.video_label = ttk.Label(self.video_frame)
        self.video_label.pack(fill=tk.BOTH, expand=True)

        # 初始化4个子框架用于显示图表（调整布局到右侧）
        self.fig_frames = []
        for i in range(4):
            frame = ttk.Frame(self.canvas_frame)
            frame.grid(row=i//2, column=i%2, sticky="nsew", padx=5, pady=5)
            self.fig_frames.append(frame)

        # 配置网格布局（调整右侧布局）
        self.canvas_frame.grid_rowconfigure(0, weight=1)
        self.canvas_frame.grid_rowconfigure(1, weight=1)
        self.canvas_frame.grid_columnconfigure(0, weight=1)
        self.canvas_frame.grid_columnconfigure(1, weight=1)

    def clear_canvases(self):
        # 清除所有现有的图表
        for canvas in self.current_canvases:
            canvas.get_tk_widget().destroy()
        self.current_canvases = []
        self.current_figures = []

    def show_all_plots(self):
        self.clear_canvases()
        self.show_behavior_emotion_plot()
        self.show_crosstab_plot()
        self.show_behavior_emotion_focus_plot()
        self.show_joint_distribution_plot()

    def show_behavior_emotion_plot(self):
        try:
            fig, ax1 = plt.subplots(figsize=(6, 4))

            # 情绪折线（带标记）
            ax1.plot(self.df['帧序'], self.df['情绪'].map(emotion_map),
                     marker='o', linestyle='--', color='#e74c3c', label='情绪状态')
            ax1.set_ylabel('情绪状态', fontsize=10)
            ax1.set_yticks([0, 0.1, 0.2, 0.3, 0.4])
            ax1.set_yticklabels(['悲伤', '愤怒', '惊讶', '愉快', '中性'])

            # 行为阶梯图（设置虚线）
            ax2 = ax1.twinx()  # 创建第二个y轴
            ax2.step(self.df['帧序'], self.df['行为'].map(behavior_map),
                     where='post', linestyle='--', color='#3498db', linewidth=3, label='行为持续')
            ax2.set_ylabel('行为类型', fontsize=10)
            ax2.set_yticks([0.1, 0.2, 0.3])
            ax2.set_yticklabels(['举手', '阅读', '写作'])

            # 增强可视化
            ax1.set_xlabel('连续帧序列', fontsize=10)
            ax1.set_xticks(range(1, 19))
            ax1.set_title('学生行为-情绪时序变化 (ID:1)', pad=10, fontsize=12)

            # 在每一帧上添加标注
            for i, row in self.df.iterrows():
                ax1.text(row['帧序'], emotion_map[row['情绪']], row['情绪'], ha='center', va='bottom', fontsize=8, color='red')
                ax2.text(row['帧序'], behavior_map[row['行为']], row['行为'], ha='center', va='top', fontsize=8, color='blue')

            # 添加网格
            ax1.grid(axis='y', alpha=0.3)
            fig.tight_layout()

            # 添加图例
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)

            # 在第一个框架中显示图表
            self.display_figure(fig, 0, "行为-情绪时序图")

        except Exception as e:
            messagebox.showerror("错误", f"无法显示图表: {str(e)}")

    def show_crosstab_plot(self):
        try:
            cross_tab = pd.crosstab(self.df['行为'], self.df['情绪'], normalize='all') * 100

            fig = plt.figure(figsize=(6, 4))
            sns.heatmap(cross_tab, annot=True, fmt=".1f", cmap="YlGnBu",
                        linewidths=.5, cbar_kws={'label': '关联强度 (%)'})

            plt.title('行为-情绪关联矩阵', pad=10, fontsize=12)
            plt.xlabel('情绪状态', fontsize=10)
            plt.ylabel('行为类型', fontsize=10)
            plt.xticks(rotation=0)
            plt.tight_layout()

            # 在第二个框架中显示图表
            self.display_figure(fig, 1, "关联矩阵热图")

        except Exception as e:
            messagebox.showerror("错误", f"无法显示图表: {str(e)}")

    def show_behavior_emotion_focus_plot(self):
        try:
            fig, ax = plt.subplots(figsize=(6, 4))

            behavior_color = "blue"
            emotion_color = "green"
            focus_color = "cyan"

            bar_width = 0.2
            index = np.arange(len(data))

            bars_behavior = ax.barh(index, [d["Confidence"] for d in data], height=bar_width,
                                    color=behavior_color, label="Behavior Confidence")
            bars_emotion_confidence = ax.barh(index + bar_width, [d["Emotion Confidence"] for d in data],
                                              height=bar_width, color=emotion_color, label="Emotion Confidence")
            bars_focus = ax.barh(index + 2 * bar_width, [d["Focus Score"] for d in data],
                                 height=bar_width, color=focus_color, label="Focus Score")

            ax.set_yticks(index + bar_width)
            ax.set_yticklabels([f"Person {d['ID']}" for d in data])
            ax.set_xlabel('Value', fontsize=10)
            ax.set_title('Behavior, Emotion, and Focus Information', fontsize=12)
            ax.legend(fontsize=8)

            for i, bar in enumerate(bars_behavior):
                ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                        f"{data[i]['Confidence']:.2f}", va="center", fontsize=8)
            for i, bar in enumerate(bars_emotion_confidence):
                ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                        f"{data[i]['Emotion Confidence']:.2f}", va="center", fontsize=8)
            for i, bar in enumerate(bars_focus):
                ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                        f"{data[i]['Focus Score']:.2f}", va="center", fontsize=8)

            plt.tight_layout()

            # 在第三个框架中显示图表
            self.display_figure(fig, 2, "行为-情绪-专注度图")

        except Exception as e:
            messagebox.showerror("错误", f"无法显示图表: {str(e)}")

    def show_joint_distribution_plot(self):
        try:
            joint_dist = pd.crosstab(self.df['行为'], self.df['情绪'], normalize='index') * 100

            fig = plt.figure(figsize=(6, 4))
            colors = ['#e74c3c', '#f1c40f', '#2ecc71', '#3498db', '#9b59b6']
            joint_dist.plot(kind='bar', stacked=True, color=colors, edgecolor='black', ax=plt.gca())

            plt.ylabel('比例分布 (%)', fontsize=10)
            plt.xlabel('行为类型', fontsize=10)
            plt.xticks(rotation=0, fontsize=8)
            plt.yticks(fontsize=8)
            plt.title('行为-情绪联合分布比例', pad=10, fontsize=12)
            plt.legend(title='情绪状态', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)

            # 添加数值标注
            for bars in plt.gca().containers:
                plt.bar_label(bars, fmt='%.1f%%', label_type='center',
                              color='white', fontsize=6, fontweight='bold')

            plt.tight_layout()

            # 在第四个框架中显示图表
            self.display_figure(fig, 3, "联合分布图")

        except Exception as e:
            messagebox.showerror("错误", f"无法显示图表: {str(e)}")

    def display_figure(self, fig, position, title=""):
        # 清除框架中的旧内容
        for widget in self.fig_frames[position].winfo_children():
            widget.destroy()

        # 添加标题
        title_label = ttk.Label(self.fig_frames[position], text=title, font=('Arial', 10, 'bold'))
        title_label.pack()

        # 创建画布并显示图表
        canvas = FigureCanvasTkAgg(fig, master=self.fig_frames[position])
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 保存引用
        self.current_figures.append(fig)
        self.current_canvases.append(canvas)

    # 其他方法保持不变...
    def load_model(self):
        def load_model_thread():
            try:
                self.status_var.set("正在加载YOLOv9模型...")
                self.analyzer = FocusAnalyzer()
                self.model_loaded = True
                self.status_var.set("YOLOv9模型加载完成")
                self.append_result("YOLOv9模型加载成功！\n")
            except Exception as e:
                self.status_var.set("模型加载失败")
                messagebox.showerror("错误", f"模型加载失败: {str(e)}")
                self.append_result(f"模型加载失败: {str(e)}\n")

        threading.Thread(target=load_model_thread, daemon=True).start()

    def open_image(self):
        if not self.check_model_loaded():
            return

        file_path = filedialog.askopenfilename(
            title="选择图片",
            filetypes=[("图片文件", "*.jpg *.jpeg *.png *.bmp")]
        )

        if file_path:
            self.stop_analysis()
            self.video_source = "image"

            try:
                image = cv2.imread(file_path)
                if image is not None:
                    self.current_frame = image.copy()
                    self.show_frame(image)
                    self.status_var.set(f"已加载图片: {os.path.basename(file_path)}")
                    self.append_result(f"已加载图片: {file_path}\n")
                else:
                    messagebox.showerror("错误", "无法加载图片文件")
            except Exception as e:
                messagebox.showerror("错误", f"图片加载失败: {str(e)}")

    def open_video(self):
        if not self.check_model_loaded():
            return

        file_path = filedialog.askopenfilename(
            title="选择视频文件",
            filetypes=[("视频文件", "*.mp4 *.avi *.mov *.mkv")]
        )

        if file_path:
            self.stop_analysis()
            self.video_source = "video"

            try:
                self.cap = cv2.VideoCapture(file_path)
                if self.cap.isOpened():
                    self.status_var.set(f"已加载视频: {os.path.basename(file_path)}")
                    self.append_result(f"已加载视频: {file_path}\n")
                    self.play_video()
                else:
                    messagebox.showerror("错误", "无法打开视频文件")
            except Exception as e:
                messagebox.showerror("错误", f"视频加载失败: {str(e)}")

    def open_camera(self):
        if not self.check_model_loaded():
            return

        self.stop_analysis()
        self.video_source = "camera"

        try:
            self.cap = cv2.VideoCapture(0)
            if self.cap.isOpened():
                self.status_var.set("已打开摄像头")
                self.append_result("已打开摄像头\n")
                self.play_video()
            else:
                messagebox.showerror("错误", "无法打开摄像头")
        except Exception as e:
            messagebox.showerror("错误", f"摄像头打开失败: {str(e)}")

    def play_video(self):
        if self.cap is None:
            return

        self.is_playing = True

        def update_frame():
            while self.is_playing and self.cap is not None:
                ret, frame = self.cap.read()
                if ret:
                    self.current_frame = frame.copy()
                    if self.is_playing:  # 再次检查，可能在读取帧时停止
                        self.show_frame(frame)
                else:
                    if self.video_source == "video":
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    else:
                        self.is_playing = False
                        break

                time.sleep(0.03)  # 控制帧率

        threading.Thread(target=update_frame, daemon=True).start()

    def start_analysis(self):
        if not self.check_model_loaded() or not self.check_source_selected():
            return

        self.is_playing = True

        def analyze_frames():
            while self.is_playing:
                if self.current_frame is not None:
                    try:
                        frame = self.current_frame.copy()
                        annotated_frame, focus_scores = self.analyzer.process_frame(frame)

                        # 更新显示
                        self.show_frame(annotated_frame)

                        # 更新结果
                        self.focus_scores = focus_scores
                        self.update_analysis_results(focus_scores)

                    except Exception as e:
                        self.append_result(f"分析错误: {str(e)}\n")

                time.sleep(0.1)  # 控制分析频率

        threading.Thread(target=analyze_frames, daemon=True).start()
        self.status_var.set("正在进行分析...")

    def stop_analysis(self):
        self.is_playing = False
        if self.cap is not None and self.video_source != "camera":
            self.cap.release()
        self.cap = None
        self.status_var.set("分析已停止")

    def save_results(self):
        if not self.focus_scores:
            messagebox.showwarning("警告", "没有可保存的分析结果")
            return

        file_path = filedialog.asksaveasfilename(
            title="保存分析结果",
            defaultextension=".txt",
            filetypes=[("文本文件", "*.txt"), ("所有文件", "*.*")]
        )

        if file_path:
            try:
                with open(file_path, "w") as f:
                    for score in self.focus_scores:
                        f.write(f"Person ID: {score['ID']}, "
                                f"Behavior: {score['Behavior']}, "
                                f"Confidence: {score['Confidence']:.2f}, "
                                f"Emotion: {score['Emotion']}, "
                                f"Emotion Confidence: {score['Emotion Confidence']:.2f}, "
                                f"Focus Score: {score['Focus Score']:.2f}\n")

                self.status_var.set(f"结果已保存到: {file_path}")
                self.append_result(f"分析结果已保存到: {file_path}\n")
            except Exception as e:
                messagebox.showerror("错误", f"保存失败: {str(e)}")

    def show_frame(self, frame):
        # 调整大小以适应显示区域
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)

        # 获取显示区域大小
        label_width = self.video_label.winfo_width()
        label_height = self.video_label.winfo_height()

        if label_width > 0 and label_height > 0:
            # 保持宽高比缩放
            img_ratio = img.width / img.height
            label_ratio = label_width / label_height

            if img_ratio > label_ratio:
                new_width = label_width
                new_height = int(label_width / img_ratio)
            else:
                new_height = label_height
                new_width = int(label_height * img_ratio)

            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

    def update_analysis_results(self, focus_scores):
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete(1.0, tk.END)

        if focus_scores:
            for score in focus_scores:
                self.result_text.insert(tk.END,
                                        f"ID: {score['ID']} | "
                                        f"行为: {score['Behavior']} (置信度: {score['Confidence']:.2f}) | "
                                        f"情绪: {score['Emotion']} (置信度: {score['Emotion Confidence']:.2f}) | "
                                        f"专注度: {score['Focus Score']:.2f}\n")
        else:
            self.result_text.insert(tk.END, "未检测到目标\n")

        self.result_text.config(state=tk.DISABLED)
        self.result_text.see(tk.END)

    def append_result(self, text):
        self.result_text.config(state=tk.NORMAL)
        self.result_text.insert(tk.END, text)
        self.result_text.config(state=tk.DISABLED)
        self.result_text.see(tk.END)

    def check_model_loaded(self):
        if not self.model_loaded or self.analyzer is None:
            messagebox.showwarning("警告", "请先加载YOLOv9模型")
            return False
        return True

    def check_source_selected(self):
        if self.video_source is None:
            messagebox.showwarning("警告", "请先选择媒体源(图片/视频/摄像头)")
            return False
        return True

    def on_closing(self):
        self.stop_analysis()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = FocusAnalysisApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()