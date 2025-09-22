# ===========================
# inference.py
# ===========================
import cv2, torch, numpy as np, time
import mediapipe as mp
from torch import nn

# --- Model (must match training) ---
class SignLSTM(nn.Module):
    def __init__(self, num_classes, input_size=63):
        super().__init__()
        self.lstm = nn.LSTM(input_size,128,batch_first=True)
        self.fc = nn.Linear(128,num_classes)
    def forward(self,x):
        _,(h,_) = self.lstm(x)
        return self.fc(h[-1])

# --- Load trained model ---
ckpt = torch.load(r"D:\WORK\Python\Project\vsl_mediapipe\sign_model.pth", map_location="cpu")
classes = ckpt["classes"]
model = SignLSTM(num_classes=len(classes), input_size=63*2)  # 2 hands → 126
model.load_state_dict(ckpt["state_dict"])
model.eval()

# --- Mediapipe setup ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
seq, SEQ_LENGTH = [], 30
p_time = 0

while True:
    ret, frame = cap.read()
    if not ret: break
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)
    landmarks = []

    if res.multi_hand_landmarks:
        for hand in res.multi_hand_landmarks:
            for lm in hand.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

    if landmarks:
        seq.append(landmarks)
        if len(seq) > SEQ_LENGTH: seq.pop(0)

        if len(seq) == SEQ_LENGTH:
            x = torch.tensor([seq], dtype=torch.float32)
            with torch.no_grad():
                out = model(x)
                pred_id = torch.argmax(out, dim=1).item()
                label = classes[pred_id]
            cv2.putText(frame, f"{label}", (30,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2)

    else:
        cv2.putText(frame, "No hand", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 2)

    c_time = time.time()
    fps = 1/(c_time-p_time) if c_time-p_time>0 else 0
    p_time = c_time
    cv2.putText(frame, f"FPS: {int(fps)}", (30, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)

    cv2.imshow("Hand Sign Realtime", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"): break

cap.release(); cv2.destroyAllWindows()

