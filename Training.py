import cv2
import numpy as np
from os import listdir
from os.path import isfile, join

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