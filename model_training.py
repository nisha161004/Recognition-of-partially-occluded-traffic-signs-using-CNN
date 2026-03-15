import os
import cv2
import numpy as np

dataset_path = r"C:\Users\shanxac\Desktop\TrafficSignProject\dataset"

images = []
labels = []

classes = os.listdir(dataset_path)

print("Classes:", classes)

for label, folder in enumerate(classes):

    folder_path = os.path.join(dataset_path, folder)

    subfolders = os.listdir(folder_path)

    for sub in subfolders:

        sub_path = os.path.join(folder_path, sub)

        for img_name in os.listdir(sub_path):

            img_path = os.path.join(sub_path, img_name)

            image = cv2.imread(img_path)

            if image is None:
                continue

            image = cv2.resize(image, (128,128))

            images.append(image)
            labels.append(label)

print("Total images loaded:", len(images))
images = np.array(images)
labels = np.array(labels)

print("Images shape:", images.shape)
print("Labels shape:", labels.shape)
images = images / 255.0
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    images, labels, test_size=0.2, random_state=42)

print("Training images:", X_train.shape)
print("Testing images:", X_test.shape)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential()

model.add(Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dense(len(classes), activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()
model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))
model.save("traffic_sign_model.keras")
