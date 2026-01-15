import os
import pickle
import cv2
import mediapipe as mp
from pathlib import Path

# Initialize using the solution wrapper
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True, 
    min_detection_confidence=0.5,
    max_num_hands=1
)

DATA_DIR = Path('./data')
data, labels = [], []

if not DATA_DIR.exists():
    print("Error: 'data' folder not found!")
else:
    for dir_ in os.listdir(DATA_DIR):
        class_path = DATA_DIR / dir_
        if not class_path.is_dir(): continue

        print(f"Processing: {dir_}")
        for img_path in os.listdir(class_path):
            img = cv2.imread(str(class_path / img_path))
            if img is None: continue
                
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Collect landmarks
                    coords = [[lm.x, lm.y] for lm in hand_landmarks.landmark]
                    x_vals, y_vals = zip(*coords)
                    
                    # Normalize (Relative to hand's top-left)
                    data_aux = []
                    for x, y in coords:
                        data_aux.append(x - min(x_vals))
                        data_aux.append(y - min(y_vals))

                    data.append(data_aux)
                    labels.append(dir_)

    with open('data.pickle', 'wb') as f:
        pickle.dump({'data': data, 'labels': labels}, f)
    print(f"Success! Processed {len(data)} samples.")

hands.close()