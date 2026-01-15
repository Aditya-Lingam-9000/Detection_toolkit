import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions
import pickle


# Initialize ONCE
options = vision.FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="D:\CV\CV_Projects\face_landmarker.task"),
    num_faces=1,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False
)

face_landmarker = vision.FaceLandmarker.create_from_options(options)
def get_face_landmarks(image, draw=False, static_image_mode=True):
    if image is None:
        return []

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # ✅ THIS is the correct constructor for your version
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = face_landmarker.detect(mp_image)

    if not result.face_landmarks:
        return []

    landmarks = result.face_landmarks[0]

    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    coords -= coords.min(axis=0)   # same normalization as before

    return coords.flatten().tolist()




emotions = ['angry','disgusted','fearful','happy','neutral','sad','suprised']

with open('CV_Projects\model.p', 'rb') as f:
    model = pickle.load(f)


cap = cv2.VideoCapture(2)

ret, frame = cap.read()

while ret:
    ret, frame = cap.read()

    face_landmarks = get_face_landmarks(frame, draw=True, static_image_mode=False)

    output = model.predict([face_landmarks])

    cv2.putText(frame,
                emotions[int(output[0])],
               (10, frame.shape[0] - 1),
               cv2.FONT_HERSHEY_SIMPLEX,
               3,
               (0, 255, 0),
               5)

    cv2.imshow('frame', frame)

    cv2.waitKey(1)


cap.release()
cv2.destroyAllWindows()