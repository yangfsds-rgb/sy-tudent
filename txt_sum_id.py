def analyze_txt_file(file_path, target_id):
    person_data = {}  # 存储每个ID的连续次数和行为情绪信息
    last_frame_persons = {}  # 上一帧的Person ID及其行为和情绪信息

    with open(file_path, 'r') as file:
        for line in file:
            if "Processing frame:" in line:
                # 当前处理的帧
                current_frame_persons = {}  # 当前帧的Person ID及其行为和情绪信息

            if "Person ID" in line:
                # 解析每行数据
                parts = line.split(',')
                person_id = int(parts[0].split(':')[1].strip())
                behavior = parts[1].split(':')[1].strip()
                emotion = parts[3].split(':')[1].strip()  # 直接提取Emotion

                # 只处理目标ID
                if person_id == target_id:
                    # 更新每个 Person ID 的连续出现次数
                    if person_id not in person_data:
                        person_data[person_id] = {
                            "max_consecutive_count": 1,
                            "consecutive_count": 1,
                            "behavior_changes": [behavior],
                            "emotion_changes": [emotion]  # 确保是Emotion而不是Emotion Confidence
                        }
                    else:
                        # 检查是否与上一帧连续
                        if person_id in last_frame_persons:
                            prev_behavior, prev_emotion = last_frame_persons[person_id]
                            if prev_behavior == behavior and prev_emotion == emotion:
                                person_data[person_id]["consecutive_count"] += 1
                            else:
                                person_data[person_id]["consecutive_count"] = 1
                        else:
                            person_data[person_id]["consecutive_count"] = 1

                        # 更新最大连续次数
                        person_data[person_id]["max_consecutive_count"] = max(person_data[person_id]["max_consecutive_count"], person_data[person_id]["consecutive_count"])

                    # 记录该帧的行为与情绪
                    person_data[person_id]["behavior_changes"].append(behavior)
                    person_data[person_id]["emotion_changes"].append(emotion)

                # 记录当前帧的 Person ID 信息
                last_frame_persons[person_id] = (behavior, emotion)

        # 打印结果
        if target_id in person_data:
            data = person_data[target_id]
            print(f"Person ID: {target_id}")
            print(f"  Max Consecutive Frames: {data['max_consecutive_count']}")
            print(f"  Behavior in Consecutive Frames: {data['behavior_changes']}")
            print(f"  Emotion in Consecutive Frames: {data['emotion_changes']}")
        else:
            print(f"Person ID {target_id} not found in the frames.")

# 使用例子
file_path = "results.txt"  # 输入你的txt文件路径
target_id = 1  # 目标Person ID
analyze_txt_file(file_path, target_id)
