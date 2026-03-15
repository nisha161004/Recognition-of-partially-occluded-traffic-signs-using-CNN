import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

# List of traffic sign classes
classes = [
    'bike lane','blind pedestrians','bus stop','children',
    'end of residential area','give way','left','main road',
    'maximum speed limit 40','no entry','no stopping',
    'parking place','pedestrian crossing','pedestrian crossing red',
    'photo and video recording','residential area','right',
    'speed hump','speed hump red','speed limit 5',
    'stop','stop line','straight','tow truck','turning point','video control'
]

# Load trained model
model = load_model("traffic_sign_model.keras")

# Read input image
input_img_path = "test.jpg"
input_img = cv2.imread(input_img_path)

if input_img is None:
    raise FileNotFoundError(f"Input image not found at {input_img_path}")

# Resize for display
input_display = cv2.resize(input_img, (250, 250))
cv2.rectangle(input_display, (90, 80), (160, 170), (0, 255, 0), 2)

# Prepare image for prediction
img = cv2.resize(input_img, (128, 128))
img = img / 255.0
img = np.reshape(img, (1, 128, 128, 3))

# Predict
prediction = model.predict(img)
class_index = np.argmax(prediction)
predicted_sign = classes[class_index]
confidence = prediction[0][class_index] * 100

print("Predicted Traffic Sign:", predicted_sign)
print("Confidence:", round(confidence, 2), "%")

# Prediction text
text = f"{predicted_sign.upper()} ({round(confidence, 2)}%)"
predicted_display = input_display.copy()
cv2.rectangle(predicted_display, (90, 80), (160, 170), (0, 255, 0), 2)
cv2.putText(predicted_display, text, (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# --- Load reference sign image robustly ---
# Absolute path to your reference folder
reference_folder = r"C:\Users\shanxac\Desktop\TrafficSignProject\sign_reference"

# List all files in folder (case-insensitive matching)
reference_files = os.listdir(reference_folder)
matching_file = None
for f in reference_files:
    if f.lower() == (predicted_sign + ".jpg").lower():
        matching_file = os.path.join(reference_folder, f)
        break

if matching_file and os.path.exists(matching_file):
    clear_img = cv2.imread(matching_file)
    if clear_img is None:
        clear_display = np.zeros((250, 250, 3), dtype=np.uint8)
        cv2.putText(clear_display, "IMAGE ERROR", (40, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    else:
        clear_display = cv2.resize(clear_img, (250, 250))
else:
    print(f"Reference image not found for {predicted_sign} in {reference_folder}")
    clear_display = np.zeros((250, 250, 3), dtype=np.uint8)
    cv2.putText(clear_display, "NO IMAGE", (40,120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

# Create comparison collage
collage = np.hstack((input_display, predicted_display, clear_display))
cv2.putText(collage, "INPUT", (80, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
cv2.putText(collage, "PREDICTION", (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
cv2.putText(collage, "CLEAR SIGN", (540, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

# Show result
cv2.imshow("Traffic Sign Recognition Comparison", collage)
cv2.waitKey(0)
cv2.destroyAllWindows()
