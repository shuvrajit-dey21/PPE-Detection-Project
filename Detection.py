import cv2
import numpy as np
from os import listdir
from os.path import isfile, join

# Use the same dataset path as in Dataset.py and Training.py
data_path = 'C:/Users/paula/OneDrive/Desktop/RATNA/dataset'
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]

Training_Data, Labels = [], []

for i, file in enumerate(onlyfiles):
    image_path = join(data_path, file)
    images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if images is None:
        print(f"Warning: Unable to read {image_path}, skipping.")
        continue
    Training_Data.append(np.asarray(images, dtype=np.uint8))
    Labels.append(i)

if len(Training_Data) == 0:
    print("No valid images found for training.")
    exit()

Labels = np.asarray(Labels, dtype=np.int32)

model = cv2.face.LBPHFaceRecognizer_create()
model.train(np.asarray(Training_Data), np.asarray(Labels))

print("Dataset Model Training Completed")

face_classifier = cv2.CascadeClassifier('C:/Users/paula/OneDrive/Desktop/RATNA/haarcascade_frontalface_default.xml')

def face_extractor(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if faces is None or len(faces) == 0:
        return None
    for (x, y, w, h) in faces:
        return img[y:y+h, x:x+w]
    return None

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    face = face_extractor(frame)
    if face is not None:
        face = cv2.resize(face, (200, 200))
        face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        result = model.predict(face_gray)

        confidence = int(100 * (1 - (result[1]) / 300)) if result[1] < 500 else 0

        display_text = "Anima"
        color = (0, 0, 255)
        if confidence > 82:
            display_text = "Recognized"
            color = (255, 255, 255)

        cv2.putText(frame, display_text, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, color, 2)
        cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), color, 2)
    else:
        cv2.putText(frame, "Face Not Found", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow('Face Detector', frame)

    if cv2.waitKey(1) == 13:  # Enter key
        break

cap.release()
cv2.destroyAllWindows()