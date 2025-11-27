import cv2
import time
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO


# ------------ Визначення пози (по скелету) ----------------
def determine_pose(keypoints):
    try:
        shoulder_y = (keypoints[5][1] + keypoints[6][1]) / 2
        hip_y = (keypoints[11][1] + keypoints[12][1]) / 2
        knee_y = (keypoints[13][1] + keypoints[14][1]) / 2
        ankle_y = (keypoints[15][1] + keypoints[16][1]) / 2
    except Exception:
        return "Unknown"

    if abs(shoulder_y - ankle_y) < 40:
        return "Lying"
    if knee_y < hip_y - 40:
        return "Sitting"
    if shoulder_y > hip_y + 30:
        return "Bent"
    return "Standing"


# ------------ Підготовка гендер-моделі (ONNX) -------------
def init_gender_model(onnx_path="gender_googlenet.onnx"):
    """
    GoogleNet age/gender модель з onnx/models:
    - вхід: 1x3x224x224, BGR
    - треба відняти mean [104,117,123]
    """
    try:
        sess = ort.InferenceSession(
            onnx_path,
            providers=["CPUExecutionProvider"]
        )
        input_name = sess.get_inputs()[0].name
        output_name = sess.get_outputs()[0].name
        print("[INFO] Gender model loaded:", onnx_path)
        return sess, input_name, output_name
    except Exception as e:
        print("[WARN] Cannot load gender model:", e)
        return None, None, None


def softmax(x):
    x = x.astype(np.float32)
    x = x - np.max(x)
    exp = np.exp(x)
    return exp / np.sum(exp)


def predict_gender(face_bgr, sess, input_name, output_name):
    """
    face_bgr: зріз обличчя (BGR)
    Повертає (gender_str, confidence_percent)
    """
    if sess is None or face_bgr is None or face_bgr.size == 0:
        return "Unknown", 0.0

    try:
        img = cv2.resize(face_bgr, (224, 224))
        img = img.astype(np.float32)

        # BGR + mean subtraction (104,117,123)
        mean = np.array([104.0, 117.0, 123.0], dtype=np.float32)
        img = img - mean

        # HWC -> CHW -> NCHW
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0)

        outputs = sess.run([output_name], {input_name: img})[0]  # shape (1,2)
        probs = softmax(outputs[0])  # [p0, p1]

        # За прикладом з документації: [p_male, p_female]
        p_male = float(probs[0])
        p_female = float(probs[1])

        if p_male >= p_female:
            gender = "Male"
            conf = p_male
        else:
            gender = "Female"
            conf = p_female

        return gender, round(conf * 100.0, 1)

    except Exception:
        return "Unknown", 0.0


# ------------ Головна функція ---------------
def main():
    # Модель поз (YOLOv8 pose)
    pose_model = YOLO("yolov8n-pose.pt")   # або "yolov8n-pose", якщо .pt нема

    # Модель визначення статі (ONNX)
    gender_sess, gender_in, gender_out = init_gender_model("gender_googlenet.onnx")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    next_id = 0
    tracked = {}  # person_id -> (x_center, y_center)

    print("[INFO] Running... Press Q to exit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # для YOLO робимо квадрат 640x640
        small = cv2.resize(frame, (640, 640))
        results = pose_model(small, verbose=False)

        new_positions = []

        for r in results:
            if r.keypoints is None:
                continue

            for kp in r.keypoints:
                xy = kp.xy.cpu().numpy()[0]  # (17, 2)

                xs = xy[:, 0]
                ys = xy[:, 1]

                x1_s, y1_s = int(xs.min()), int(ys.min())
                x2_s, y2_s = int(xs.max()), int(ys.max())

                # масштаб назад в оригінальний кадр
                scale_x = frame.shape[1] / 640.0
                scale_y = frame.shape[0] / 640.0

                x1 = int(x1_s * scale_x)
                y1 = int(y1_s * scale_y)
                x2 = int(x2_s * scale_x)
                y2 = int(y2_s * scale_y)

                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                new_positions.append((cx, cy, xy, x1, y1, x2, y2))

        # відстеження ID по близькості центрів
        for (cx, cy, xy, x1, y1, x2, y2) in new_positions:
            min_dist = 999999
            best_id = None

            for pid, (px, py) in tracked.items():
                d = abs(px - cx) + abs(py - cy)
                if d < min_dist and d < 120:
                    min_dist = d
                    best_id = pid

            if best_id is None:
                best_id = next_id
                next_id += 1

            tracked[best_id] = (cx, cy)

            pose = determine_pose(xy)

            # ---------- FACE ROI для моделі статі ----------
            h = y2 - y1
            if h <= 0:
                face_roi = None
            else:
                fy1 = y1
                fy2 = y1 + int(0.55 * h)  # верхня половина тіла
                fy1 = max(0, min(fy1, frame.shape[0] - 1))
                fy2 = max(0, min(fy2, frame.shape[0] - 1))
                fx1 = max(0, min(x1, frame.shape[1] - 1))
                fx2 = max(0, min(x2, frame.shape[1] - 1))

                if fy2 > fy1 and fx2 > fx1:
                    face_roi = frame[fy1:fy2, fx1:fx2]
                else:
                    face_roi = None

            gender, conf = predict_gender(face_roi, gender_sess, gender_in, gender_out)

            # ---------- колір рамки по позі ----------
            if pose in ["Standing", "Sitting"]:
                color = (0, 255, 0)       # зелений
            elif pose == "Bent":
                color = (0, 255, 255)     # жовтий
            elif pose == "Lying":
                color = (0, 0, 255)       # червоний
            else:
                color = (255, 0, 0)       # синій / невідомо

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label = f"ID {best_id} | {gender} {conf:.1f}%"
            cv2.putText(frame, label,
                        (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        color,
                        2)

        cv2.imshow("Pose + Gender (ONNX) + ID", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
