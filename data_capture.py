import cv2
import mediapipe as mp
import numpy as np
import os

# Mediapipe modules
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
face = mp_face.FaceMesh(max_num_faces=1, min_detection_confidence=0.7)

# Save directory
DATA_DIR = "vsl_sequences_face"
SIGNS = ["xin_chao", "cam_on"]
SEQUENCE_LENGTH = 30  # number of frames per sequence

# Make directories
for sign in SIGNS:
    os.makedirs(os.path.join(DATA_DIR, sign), exist_ok=True)

cap = cv2.VideoCapture(0)

for sign in SIGNS:
    print(f"Collecting data for {sign}")
    for sample in range(20):  # 20 sequences per sign
        sequence = []
        while len(sequence) < SEQUENCE_LENGTH:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hand_result = hands.process(rgb)
            face_result = face.process(rgb)

            # Default vectors (63 for hand, 1404 for face)
            hand_landmarks = np.zeros(63)
            face_landmarks = np.zeros(1404)

            # Extract hand landmarks
            if hand_result.multi_hand_landmarks:
                hand = hand_result.multi_hand_landmarks[0]
                for idx, lm in enumerate(hand.landmark):
                    hand_landmarks[idx*3:(idx*3)+3] = [lm.x, lm.y, lm.z]
                mp_drawing.draw_landmarks(
                    frame, hand, mp_hands.HAND_CONNECTIONS
                )

            # Extract face landmarks
            if face_result.multi_face_landmarks:
                face_lms = face_result.multi_face_landmarks[0]
                for idx, lm in enumerate(face_lms.landmark):
                    face_landmarks[idx*3:(idx*3)+3] = [lm.x, lm.y, lm.z]
                mp_drawing.draw_landmarks(
                    frame, face_lms, mp_face.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )

            # Combine both
            landmarks = np.concatenate([hand_landmarks, face_landmarks])
            sequence.append(landmarks)

            # Display info
            cv2.putText(frame, f"Sign: {sign} Sample: {sample+1}/20 Frame: {len(sequence)}/{SEQUENCE_LENGTH}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("Collecting", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                exit()

        # Save one sequence
        sequence = np.array(sequence)
        np.save(os.path.join(DATA_DIR, sign, f"{sample}.npy"), sequence)

cap.release()
cv2.destroyAllWindows()
