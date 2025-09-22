import os, glob
import numpy as np
import torch
import torch.nn as nn
import mediapipe as mp
import cv2

# -------------------------
# Feature extractor (Mediapipe)
# -------------------------
def extract_seq(video, seq_len=30):
    mp_hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    cap = cv2.VideoCapture(video)
    seq = []

    while len(seq) < seq_len and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        res = mp_hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # default: no hand (zero padding)
        left_hand = [0] * 63
        right_hand = [0] * 63

        if res.multi_hand_landmarks and res.multi_handedness:
            for hand_landmarks, handedness in zip(res.multi_hand_landmarks, res.multi_handedness):
                pts = []
                for lm in hand_landmarks.landmark:
                    pts.extend([lm.x, lm.y, lm.z])
                label = handedness.classification[0].label
                if label == "Left":
                    left_hand = pts
                else:
                    right_hand = pts

        seq.append(np.array(left_hand + right_hand))

    cap.release()
    mp_hands.close()

    # pad if video shorter
    while len(seq) < seq_len:
        seq.append(np.zeros(126))

    return np.array(seq)


# -------------------------
# Model
# -------------------------
class SignLSTM(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(126, 128, batch_first=True)
        self.fc = nn.Linear(128, num_classes)
    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])


# -------------------------
# Training function
# -------------------------
def train_model(data_root, classes, seq_len=30, epochs=30, lr=1e-3):
    train_videos, labels = [], []
    for idx, cname in enumerate(classes):
        vids = glob.glob(os.path.join(data_root, cname, "*.mp4"))
        train_videos.extend(vids)
        labels.extend([idx] * len(vids))

    if not train_videos:
        raise RuntimeError("⚠️ No videos found in dataset. Check your folder structure!")

    # convert to tensors
    X = [extract_seq(v, seq_len) for v in train_videos]
    X = torch.tensor(np.stack(X), dtype=torch.float32)  # [N, seq_len, 126]
    y = torch.tensor(labels)

    model = SignLSTM(num_classes=len(classes))
    opt = torch.optim.Adam(model.parameters(), lr)
    loss_fn = nn.CrossEntropyLoss()

    # training loop
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        out = model(X)
        loss = loss_fn(out, y)
        loss.backward()
        opt.step()
        acc = (out.argmax(1) == y).float().mean().item()
        print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f} | Acc: {acc*100:.2f}%")

    return model


# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    data_root = r"D:\WORK\Python\Project\vsl_mediapipe\vsl_video"
    
    # Only 2 classes
    classes = ["cam_on", "toi"]

    model = train_model(data_root, classes, epochs=30)
    torch.save(model.state_dict(), "vsl_model_v2.pth")
    print("✅ Model saved with only 2 classes: cam_on, toi")
