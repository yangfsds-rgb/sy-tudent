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

# Configure for Chinese display
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False

# Value mappings
emotion_map = {
    'sad': 0,
    'angry': 0.1,
    'fear': 0.2,
    'happy': 0.3,
    'neutral': 0.4
}

behavior_map = {
    'rise_hand': 0.1,
    'read': 0.2,
    'write': 0.3
}

class FocusAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Classroom Focus Analysis System v1.0")
        self.root.geometry("1400x900")

        # Initialize analyzer
        self.analyzer = None
        self.model_loaded = False

        # Video/camera control variables
        self.video_source = None
        self.cap = None
        self.is_playing = False
        self.current_frame = None
        self.focus_scores = []
        self.frame_count = 0

        # Chart-related variables
        self.current_figures = []
        self.current_canvases = []

        # Create the interface
        self.create_widgets()

        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        self.status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left control panel
        control_panel = ttk.LabelFrame(main_frame, text="Control Panel", width=300)
        control_panel.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        # Right display panel
        display_panel = ttk.LabelFrame(main_frame, text="Visualization", width=1100)
        display_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Control panel content
        self.create_control_panel(control_panel)

        # Display panel content
        self.create_display_panel(display_panel)

    def create_control_panel(self, parent):
        # Model loading section
        model_frame = ttk.LabelFrame(parent, text="Model Loading")
        model_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(model_frame, text="Load YOLOv9 Model", command=self.load_model).pack(fill=tk.X, padx=5, pady=2)

        # Media source selection section
        source_frame = ttk.LabelFrame(parent, text="Media Source")
        source_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(source_frame, text="Open Image", command=self.open_image).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(source_frame, text="Open Video", command=self.open_video).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(source_frame, text="Open Camera", command=self.open_camera).pack(fill=tk.X, padx=5, pady=2)

        # Analysis control section
        analysis_frame = ttk.LabelFrame(parent, text="Analysis Control")
        analysis_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(analysis_frame, text="Start Analysis", command=self.start_analysis).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(analysis_frame, text="Stop Analysis", command=self.stop_analysis).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(analysis_frame, text="Save Results", command=self.save_results).pack(fill=tk.X, padx=5, pady=2)

        # Result display section
        result_frame = ttk.LabelFrame(parent, text="Analysis Results")
        result_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.result_text = tk.Text(result_frame, height=15, state=tk.DISABLED)
        scrollbar = ttk.Scrollbar(result_frame, command=self.result_text.yview)
        self.result_text.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.result_text.pack(fill=tk.BOTH, expand=True)

        # Visualization buttons
        vis_frame = ttk.LabelFrame(parent, text="Visualization")
        vis_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(vis_frame, text="Show Behavior-Emotion Time Series Plot",
                   command=lambda: self.show_behavior_emotion_plot()).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(vis_frame, text="Show Crosstab Heatmap",
                   command=lambda: self.show_crosstab_plot()).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(vis_frame, text="Show Behavior-Emotion-Focus Plot",
                   command=lambda: self.show_behavior_emotion_focus_plot()).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(vis_frame, text="Show Joint Distribution Plot",
                   command=lambda: self.show_joint_distribution_plot()).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(vis_frame, text="Show All Plots",
                   command=lambda: self.show_all_plots()).pack(fill=tk.X, padx=5, pady=2)

    def create_display_panel(self, parent):
        # Create canvas frame and video label frame
        self.canvas_frame = ttk.Frame(parent)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True, side=tk.RIGHT)  # Adjust layout

        # New: Video display label frame
        self.video_frame = ttk.Frame(parent)
        self.video_frame.pack(fill=tk.BOTH, expand=True, side=tk.LEFT, padx=5, pady=5)

        # Initialize video label
        self.video_label = ttk.Label(self.video_frame)
        self.video_label.pack(fill=tk.BOTH, expand=True)

        # Initialize 4 subframes for displaying charts (adjust layout to the right)
        self.fig_frames = []
        for i in range(4):
            frame = ttk.Frame(self.canvas_frame)
            frame.grid(row=i//2, column=i%2, sticky="nsew", padx=5, pady=5)
            self.fig_frames.append(frame)

        # Configure grid layout (adjust right side layout)
        self.canvas_frame.grid_rowconfigure(0, weight=1)
        self.canvas_frame.grid_rowconfigure(1, weight=1)
        self.canvas_frame.grid_columnconfigure(0, weight=1)
        self.canvas_frame.grid_columnconfigure(1, weight=1)

    def clear_canvases(self):
        # Clear all existing charts
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

    def create_dataframe(self):
        if not self.focus_scores:
            return None

        data = []
        for i, score in enumerate(self.focus_scores):
            data.append({
                'Frame': i+1,
                'Behavior': score['Behavior'],
                'Emotion': score['Emotion'],
                'Focus Score': score['Focus Score']
            })

        return pd.DataFrame(data)

    def show_behavior_emotion_plot(self):
        try:
            df = self.create_dataframe()
            if df is None or df.empty:
                messagebox.showwarning("Warning", "No available analysis data")
                return

            fig, ax1 = plt.subplots(figsize=(6, 4))

            # Emotion line (with markers)
            ax1.plot(df['Frame'], df['Emotion'].map(emotion_map),
                     marker='o', linestyle='--', color='#e74c3c', label='Emotion State')
            ax1.set_ylabel('Emotion State', fontsize=10)
            ax1.set_yticks([0, 0.1, 0.2, 0.3, 0.4])
            ax1.set_yticklabels(['Sad', 'Angry', 'Fear', 'Happy', 'Neutral'])

            # Behavior step plot (set dashed line)
            ax2 = ax1.twinx()  # Create second y-axis
            ax2.step(df['Frame'], df['Behavior'].map(behavior_map),
                     where='post', linestyle='--', color='#3498db', linewidth=3, label='Behavior Duration')
            ax2.set_ylabel('Behavior Type', fontsize=10)
            ax2.set_yticks([0.1, 0.2, 0.3])
            ax2.set_yticklabels(['Raise Hand', 'Reading', 'Writing'])

            # Enhance visualization
            ax1.set_xlabel('Consecutive Frame Sequence', fontsize=10)
            ax1.set_xticks(range(1, len(df)+1))
            ax1.set_title('Student Behavior-Emotion Time Series', pad=10, fontsize=12)

            # Add annotations for each frame
            for i, row in df.iterrows():
                ax1.text(row['Frame'], emotion_map[row['Emotion']], row['Emotion'], ha='center', va='bottom', fontsize=8, color='red')
                ax2.text(row['Frame'], behavior_map[row['Behavior']], row['Behavior'], ha='center', va='top', fontsize=8, color='blue')

            # Add grid
            ax1.grid(axis='y', alpha=0.3)
            fig.tight_layout()

            # Add legend
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)

            # Display the chart in the first frame
            self.display_figure(fig, 0, "Behavior-Emotion Time Series Plot")

        except Exception as e:
            messagebox.showerror("Error", f"Unable to display the plot: {str(e)}")

    def show_crosstab_plot(self):
        try:
            df = self.create_dataframe()
            if df is None or df.empty:
                messagebox.showwarning("Warning", "No available analysis data")
                return

            cross_tab = pd.crosstab(df['Behavior'], df['Emotion'], normalize='all') * 100

            fig = plt.figure(figsize=(6, 4))
            sns.heatmap(cross_tab, annot=True, fmt=".1f", cmap="YlGnBu",
                        linewidths=.5, cbar_kws={'label': 'Correlation Strength (%)'})

            plt.title('Behavior-Emotion Correlation Matrix', pad=10, fontsize=12)
            plt.xlabel('Emotion State', fontsize=10)
            plt.ylabel('Behavior Type', fontsize=10)
            plt.xticks(rotation=0)
            plt.tight_layout()

            # Display the chart in the second frame
            self.display_figure(fig, 1, "Correlation Matrix Heatmap")

        except Exception as e:
            messagebox.showerror("Error", f"Unable to display the plot: {str(e)}")

    def show_behavior_emotion_focus_plot(self):
        try:
            if not self.focus_scores:
                messagebox.showwarning("Warning", "No available analysis data")
                return

            fig, ax = plt.subplots(figsize=(6, 4))

            behavior_color = "blue"
            emotion_color = "green"
            focus_color = "cyan"

            bar_width = 0.2
            index = np.arange(len(self.focus_scores))

            bars_behavior = ax.barh(index, [d["Confidence"] for d in self.focus_scores], height=bar_width,
                                    color=behavior_color, label="Behavior Confidence")
            bars_emotion_confidence = ax.barh(index + bar_width, [d["Emotion Confidence"] for d in self.focus_scores],
                                              height=bar_width, color=emotion_color, label="Emotion Confidence")
            bars_focus = ax.barh(index + 2 * bar_width, [d["Focus Score"] for d in self.focus_scores],
                                 height=bar_width, color=focus_color, label="Focus Score")

            ax.set_yticks(index + bar_width)
            ax.set_yticklabels([f"Person {d['ID']}" for d in self.focus_scores])
            ax.set_xlabel('Value', fontsize=10)
            ax.set_title('Behavior, Emotion, and Focus Information', fontsize=12)
            ax.legend(fontsize=8)

            for i, bar in enumerate(bars_behavior):
                ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                        f"{self.focus_scores[i]['Confidence']:.2f}", va="center", fontsize=8)
            for i, bar in enumerate(bars_emotion_confidence):
                ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                        f"{self.focus_scores[i]['Emotion Confidence']:.2f}", va="center", fontsize=8)
            for i, bar in enumerate(bars_focus):
                ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                        f"{self.focus_scores[i]['Focus Score']:.2f}", va="center", fontsize=8)

            plt.tight_layout()

            # Display the chart in the third frame
            self.display_figure(fig, 2, "Behavior-Emotion-Focus Plot")

        except Exception as e:
            messagebox.showerror("Error", f"Unable to display the plot: {str(e)}")

    def show_joint_distribution_plot(self):
        try:
            df = self.create_dataframe()
            if df is None or df.empty:
                messagebox.showwarning("Warning", "No available analysis data")
                return

            joint_dist = pd.crosstab(df['Behavior'], df['Emotion'], normalize='index') * 100

            fig = plt.figure(figsize=(6, 4))
            colors = ['#e74c3c', '#f1c40f', '#2ecc71', '#3498db', '#9b59b6']
            joint_dist.plot(kind='bar', stacked=True, color=colors, edgecolor='black', ax=plt.gca())

            plt.ylabel('Proportional Distribution (%)', fontsize=10)
            plt.xlabel('Behavior Type', fontsize=10)
            plt.xticks(rotation=0, fontsize=8)
            plt.yticks(fontsize=8)
            plt.title('Behavior-Emotion Joint Distribution Proportion', pad=10, fontsize=12)
            plt.legend(title='Emotion State', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)

            # Add value annotations
            for bars in plt.gca().containers:
                plt.bar_label(bars, fmt='%.1f%%', label_type='center',
                              color='white', fontsize=6, fontweight='bold')

            plt.tight_layout()

            # Display the chart in the fourth frame
            self.display_figure(fig, 3, "Joint Distribution Plot")

        except Exception as e:
            messagebox.showerror("Error", f"Unable to display the plot: {str(e)}")

    def display_figure(self, fig, position, title=""):
        # Clear old content in the frame
        for widget in self.fig_frames[position].winfo_children():
            widget.destroy()

        # Add title
        title_label = ttk.Label(self.fig_frames[position], text=title, font=('Arial', 10, 'bold'))
        title_label.pack()

        # Create canvas and display the chart
        canvas = FigureCanvasTkAgg(fig, master=self.fig_frames[position])
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Save references
        self.current_figures.append(fig)
        self.current_canvases.append(canvas)

    def load_model(self):
        def load_model_thread():
            try:
                self.status_var.set("Loading YOLOv9 model...")
                self.analyzer = FocusAnalyzer()
                self.model_loaded = True
                self.status_var.set("YOLOv9 model loaded successfully")
                self.append_result("YOLOv9 model loaded successfully!\n")
            except Exception as e:
                self.status_var.set("Model loading failed")
                messagebox.showerror("Error", f"Model loading failed: {str(e)}")
                self.append_result(f"Model loading failed: {str(e)}\n")

        threading.Thread(target=load_model_thread, daemon=True).start()

    def open_image(self):
        if not self.check_model_loaded():
            return

        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )

        if file_path:
            self.stop_analysis()
            self.video_source = "image"

            try:
                image = cv2.imread(file_path)
                if image is not None:
                    self.current_frame = image.copy()
                    self.show_frame(image)
                    self.status_var.set(f"Loaded image: {os.path.basename(file_path)}")
                    self.append_result(f"Loaded image: {file_path}\n")
                else:
                    messagebox.showerror("Error", "Unable to load image file")
            except Exception as e:
                messagebox.showerror("Error", f"Image loading failed: {str(e)}")

    def open_video(self):
        if not self.check_model_loaded():
            return

        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")]
        )

        if file_path:
            self.stop_analysis()
            self.video_source = "video"

            try:
                self.cap = cv2.VideoCapture(file_path)
                if self.cap.isOpened():
                    self.status_var.set(f"Loaded video: {os.path.basename(file_path)}")
                    self.append_result(f"Loaded video: {file_path}\n")
                    self.play_video()
                else:
                    messagebox.showerror("Error", "Unable to open video file")
            except Exception as e:
                messagebox.showerror("Error", f"Video loading failed: {str(e)}")

    def open_camera(self):
        if not self.check_model_loaded():
            return

        self.stop_analysis()
        self.video_source = "camera"

        try:
            self.cap = cv2.VideoCapture(0)
            if self.cap.isOpened():
                self.status_var.set("Camera opened")
                self.append_result("Camera opened\n")
                self.play_video()
            else:
                messagebox.showerror("Error", "Unable to open camera")
        except Exception as e:
            messagebox.showerror("Error", f"Camera opening failed: {str(e)}")

    def play_video(self):
        if self.cap is None:
            return

        self.is_playing = True

        def update_frame():
            while self.is_playing and self.cap is not None:
                ret, frame = self.cap.read()
                if ret:
                    self.current_frame = frame.copy()
                    if self.is_playing:  # Check again, might stop during frame reading
                        self.show_frame(frame)
                else:
                    if self.video_source == "video":
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    else:
                        self.is_playing = False
                        break

                time.sleep(0.03)  # Control frame rate

        threading.Thread(target=update_frame, daemon=True).start()

    def start_analysis(self):
        if not self.check_model_loaded() or not self.check_source_selected():
            return

        self.is_playing = True
        self.frame_count = 0
        self.focus_scores = []

        def analyze_frames():
            # Image processing logic
            if self.video_source == "image":
                try:
                    if self.current_frame is None:
                        return

                    # Single process
                    frame = self.current_frame.copy()
                    annotated_frame, focus_scores = self.analyzer.process_frame(frame)

                    # Update display
                    self.show_frame(annotated_frame)

                    # Update results
                    self.frame_count += 1
                    for score in focus_scores:
                        score['Frame'] = self.frame_count
                    self.focus_scores.extend(focus_scores)
                    self.update_analysis_results(focus_scores)

                    # Automatically pop up save dialog
                    #self.save_results()

                except Exception as e:
                    self.append_result(f"Image analysis error: {str(e)}\n")
                finally:
                    self.stop_analysis()

            # Video/camera processing logic
            else:
                while self.is_playing:
                    if self.current_frame is not None:
                        try:
                            frame = self.current_frame.copy()
                            annotated_frame, focus_scores = self.analyzer.process_frame(frame)

                            self.show_frame(annotated_frame)

                            # Update results
                            self.frame_count += 1
                            for score in focus_scores:
                                score['Frame'] = self.frame_count
                            self.focus_scores.extend(focus_scores)
                            self.update_analysis_results(focus_scores)

                        except Exception as e:
                            self.append_result(f"Analysis error: {str(e)}\n")

                    time.sleep(0.1)  # Control analysis frequency

        threading.Thread(target=analyze_frames, daemon=True).start()
        self.status_var.set("Analysis in progress...")

    def stop_analysis(self):
        self.is_playing = False
        if self.cap is not None and self.video_source != "camera":
            self.cap.release()
        self.cap = None
        self.status_var.set("Analysis stopped")

    def save_results(self):
        if not self.focus_scores:
            messagebox.showwarning("Warning", "No analysis results to save")
            return

        # Automatically generate default file name (only for image)
        default_name = ""
        if self.video_source == "image" and hasattr(self, 'current_image_path'):
            base_name = os.path.basename(self.current_image_path)
            default_name = os.path.splitext(base_name)[0] + "_result.csv"

        file_path = filedialog.asksaveasfilename(
            title="Save Analysis Results",
            defaultextension=".csv",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")],
            initialfile=default_name  # Add default file name
        )

        if file_path:
            try:
                df = self.create_dataframe()
                df.to_csv(file_path, index=False, encoding='utf_8_sig')
                self.status_var.set(f"Results saved to: {file_path}")
                self.append_result(f"Analysis results saved to: {file_path}\n")
            except Exception as e:
                messagebox.showerror("Error", f"Save failed: {str(e)}")
    # Added file path recording in open_image method
    def open_image(self):
        if not self.check_model_loaded():
            return

        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )

        if file_path:
            self.stop_analysis()
            self.video_source = "image"
            self.current_image_path = file_path  # New path recording

            try:
                image = cv2.imread(file_path)
                if image is not None:
                    self.current_frame = image.copy()
                    self.show_frame(image)
                    self.status_var.set(f"Loaded image: {os.path.basename(file_path)}")
                    self.append_result(f"Loaded image: {file_path}\n")
                else:
                    messagebox.showerror("Error", "Unable to load image file")
            except Exception as e:
                messagebox.showerror("Error", f"Image loading failed: {str(e)}")

    def show_frame(self, frame):
        # Adjust size to fit the display area
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)

        # Get display area size
        label_width = self.video_label.winfo_width()
        label_height = self.video_label.winfo_height()

        if label_width > 0 and label_height > 0:
            # Maintain aspect ratio scaling
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
                                        f"Behavior: {score['Behavior']} (Confidence: {score['Confidence']:.2f}) | "
                                        f"Emotion: {score['Emotion']} (Confidence: {score['Emotion Confidence']:.2f}) | "
                                        f"Focus Score: {score['Focus Score']:.2f}\n")
        else:
            self.result_text.insert(tk.END, "No targets detected\n")

        self.result_text.config(state=tk.DISABLED)
        self.result_text.see(tk.END)

    def append_result(self, text):
        self.result_text.config(state=tk.NORMAL)
        self.result_text.insert(tk.END, text)
        self.result_text.config(state=tk.DISABLED)
        self.result_text.see(tk.END)

    def check_model_loaded(self):
        if not self.model_loaded or self.analyzer is None:
            messagebox.showwarning("Warning", "Please load the YOLOv9 model first")
            return False
        return True

    def check_source_selected(self):
        if self.video_source is None:
            messagebox.showwarning("Warning", "Please select a media source (Image/Video/Camera) first")
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
