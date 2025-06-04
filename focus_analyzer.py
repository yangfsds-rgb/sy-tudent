import cv2
import torch
import numpy as np
from collections import deque
from deepface import DeepFace
from utils.general import non_max_suppression, scale_coords
from models.common import DetectMultiBackend
from utils.datasets import letterbox

class FocusAnalyzer:
    def __init__(self, yolo_weights="weights/yolov9s+CBAM.pt"):
        # 硬件配置
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 初始化YOLOv9
        self.yolo = self._init_yolo(yolo_weights)
        self.class_names = ['rise_hand', 'read', 'write']  # 修改为三个类别
        self.conf_thres = 0.2  # 修改为0.2的置信度阈值

        # 状态跟踪缓存
        self.behavior_window = deque(maxlen=30)  # 30帧行为记录
        self.emotion_history = deque(maxlen=60)  # 60次情绪记录

        # 行为权重设置，三种行为的权重和为1
        self.behavior_weights = {'rise_hand': 0.1, 'read': 0.2, 'write': 0.1}

        # 情绪权重设置，五种情绪的权重和为1
        self.emotion_weights = {
            'neutral': 0.4,
            'happy': 0.2,
            'surprise': 0,
            'angry': 0,
            'sad': 0
        }

        # 保存第一次出现的面部特征
        self.face_embeddings = {}
        self.person_counter = 1  # 用来分配新的ID

    def _init_yolo(self, weights_path):
        """直接加载 YOLOv9 模型"""
        device = torch.device(self.device)  # 将字符串转换为 torch.device 对象
        model = DetectMultiBackend(weights_path, device=device)  # 加载权重
        return model


    def _detect_behavior(self, frame):
        """使用 YOLOv9 进行行为检测"""
        img = letterbox(frame, new_shape=640)[0]  # 使用letterbox进行图像预处理
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x640x640
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device).float() / 255.0

        # 确保输入张量的形状为 (1, 3, H, W)
        if img.ndimension() == 3:
            img = img.unsqueeze(0)  # 添加 batch 维度

        # 推理
        with torch.no_grad():
            pred = self.yolo(img)[0]  # 获取预测结果

        # 使用 NMS 进行非极大抑制
        pred = non_max_suppression(pred, self.conf_thres, 0.45)

        # 初始化结果
        detected_boxes = []

        # 遍历检测结果
        for det in pred:
            if det is not None and len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()  # 缩放坐标
                for *xyxy, conf, cls in reversed(det):
                    label = self.class_names[int(cls)]
                    confidence = conf.item()

                    # 置信度过滤
                    if confidence < self.conf_thres:
                        continue

                    detected_boxes.append((*xyxy, label, confidence))  # 保存检测到的框及其标签和置信度

        return detected_boxes

    def _analyze_emotion(self, frame):
        """面部表情分析"""
        try:
            analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            emotion = analysis[0]['dominant_emotion']
            conf = analysis[0]['emotion'][emotion]
            return emotion, conf
        except:
            return 'neutral', 0.1
    def _get_face_embedding(self, frame):
        """提取面部特征"""
        try:
            # 提取面部特征
            embedding = DeepFace.represent(frame, model_name='Facenet', enforce_detection=False)
            return embedding[0]['embedding']
        except Exception as e:
            print(f"Error in face embedding extraction: {e}")
            return None

    def _match_face_embedding(self, current_embedding):
        """与已保存的面部特征进行匹配"""
        max_similarity = 0
        person_id = None
        for stored_person_id, stored_embedding in self.face_embeddings.items():
            # 使用cosine相似度来匹配面部特征
            similarity = np.dot(stored_embedding, current_embedding) / (np.linalg.norm(stored_embedding) * np.linalg.norm(current_embedding))
            if similarity > max_similarity:
                max_similarity = similarity
                person_id = stored_person_id

        # 如果相似度小于0.8，则认为是新的人脸
        if max_similarity < 0.9:
            person_id = None

        return person_id

    def process_frame(self, frame, result_txt="results.txt"):
        """处理图片并计算专注度"""
        # 并行执行检测
        detected_boxes = self._detect_behavior(frame.copy())

        # 初始化关注度
        focus_scores = []

        # 打开文件进行写入
        with open(result_txt, "a") as f:
            f.write(f"Processing image: \n")

            # 计算每个行为的关注度并进行特征融合
            for box in detected_boxes:
                x1, y1, x2, y2, label, confidence = box
                detected_region = frame[int(y1):int(y2), int(x1):int(x2)]  # 提取对应区域

                # 获取情绪分析结果
                emotion, emotion_confidence = self._analyze_emotion(detected_region)

                # 提取当前帧的面部特征
                current_embedding = self._get_face_embedding(detected_region)

                if current_embedding is not None:
                    # 检查是否为已知人脸
                    person_id = self._match_face_embedding(current_embedding)

                    if person_id is None:
                        # 新的面部特征，分配一个新的ID
                        person_id = self.person_counter
                        self.face_embeddings[person_id] = current_embedding
                        self.person_counter += 1
                        print(f"New person detected, assigned ID: {person_id}")

                    # 计算 Behavior Score（行为分数）
                    behavior_score = self.behavior_weights[label] * confidence
                    # 将情绪置信度从百分比转换为0到1之间
                    emotion_confidence /= 100

                    # 计算 Emotion Score（情绪分数）
                    emotion_score = self.emotion_weights.get(emotion, 0) * emotion_confidence

                    # 计算 Focus Score（专注度）
                    focus_score = (0.3 * behavior_score) + (0.7 * emotion_score)

                    # 确保专注度在0到1之间
                    focus_score = max(0, min(1, focus_score))

                    # 生成数据并添加到列表中
                    data = {
                        "ID": person_id,
                        "Behavior": label,
                        "Confidence": confidence,
                        "Emotion": emotion,
                        "Emotion Confidence": emotion_confidence,
                        "Focus Score": focus_score
                    }
                    focus_scores.append(data)

                    # 打印行为类别，置信度，情绪类别，情绪置信度和专注度
                  #  print(f"Person ID: {person_id}, Behavior: {label}, Confidence: {confidence:.2f}, Emotion: {emotion}, "
                   #       f"Emotion Confidence: {emotion_confidence:.2f}, Focus Score: {focus_score:.2f}")

                    # 将结果写入文件
                    f.write(f"Person ID: {person_id}, Behavior: {label}, Confidence: {confidence:.2f}, Emotion: {emotion}, "
                            f"Emotion Confidence: {emotion_confidence:.2f}, Focus Score: {focus_score:.2f}\n")

                    # 在图像中标注 Person ID、行为类别和专注度
                    cv2.putText(frame, f"ID: {person_id}", (int(x1), int(y1) - 10),  # 在框上方标注 Person ID
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                    # 标注行为类别和专注度得分
                    cv2.putText(frame, f" Focus Score: {focus_score:.2f}",
                                (int(x1), int(y1) + 30),  # 显示专注度 + 专注度得分
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                    # 绘制矩形框
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)  # 绘制矩形框

        # 返回处理后的图像和关注度数据
        return frame, focus_scores


    def analyze_image(self, image_path, output_txt="results.txt", output_image=None):
        """分析单张图片并保存结果到文本文件和图片文件"""
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Error: Unable to load image at {image_path}")
            return

        # 使用同步方法处理帧并获取结果
        annotated_frame, focus_scores = self.process_frame(frame, result_txt=output_txt)

        # 如果指定了输出图片路径，则保存标注后的图像
        if output_image:
            cv2.imwrite(output_image, annotated_frame)

        cv2.imshow("Image Analysis", annotated_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# 使用示例
if __name__ == "__main__":
    analyzer = FocusAnalyzer()

    # 分析图片并保存结果
    analyzer.analyze_image("spark.png", output_txt="results-id.txt", output_image="output_image.jpg")
