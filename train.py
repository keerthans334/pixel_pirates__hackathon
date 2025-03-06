import os
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from imutils import paths
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.utils import to_categorical

# Constants
IMAGE_SIZE = (224, 224)
NUM_IMAGES_PER_GESTURE = 100
TEST_SIZE = 0.2
RANDOM_STATE = 42
LEARNING_RATE = 1e-4
EPOCHS = 20
BATCH_SIZE = 32

# Initialize MediaPipe Hand Landmarker
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
hand_landmarker = vision.HandLandmarker.create_from_options(options)

def sanitize_gesture_name(gesture_name):
    return gesture_name.replace(",", "").replace("?", "").replace(" ", "_")

def create_folders(gestures):
    os.makedirs("gesture_images", exist_ok=True)
    for gesture in gestures:
        sanitized_gesture = sanitize_gesture_name(gesture)
        os.makedirs(os.path.join("gesture_images", sanitized_gesture), exist_ok=True)

def capture_images(gestures):
    cap = cv2.VideoCapture(0)
    print(f"Press a number (0-{len(gestures) - 1}) to save gesture image, 'q' to quit, 'd' to delete the last image.")
    image_count = {sanitize_gesture_name(gesture): 0 for gesture in gestures}
    current_gesture = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        results = hand_landmarker.detect(mp_image)

        if results.hand_landmarks:
            for hand_landmarks_list in results.hand_landmarks:
                for landmark in hand_landmarks_list:
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

        cv2.imshow("Capture Gestures", frame)
        key = cv2.waitKey(1) & 0xFF

        if key in [ord(str(i)) for i in range(len(gestures))]:
            gesture_idx = int(chr(key))
            gesture_name = gestures[gesture_idx]
            sanitized_gesture = sanitize_gesture_name(gesture_name)
            folder_path = os.path.join("gesture_images", sanitized_gesture)
            current_gesture = sanitized_gesture

            if image_count[sanitized_gesture] >= NUM_IMAGES_PER_GESTURE:
                print(f"Already collected {NUM_IMAGES_PER_GESTURE} images for '{gesture_name}'.")
                continue

            img_path = os.path.join(folder_path, f"{sanitized_gesture}_{image_count[sanitized_gesture]}.jpg")
            cv2.imwrite(img_path, frame)
            image_count[sanitized_gesture] += 1
            print(f"Saved: {img_path}")

            if image_count[sanitized_gesture] == NUM_IMAGES_PER_GESTURE:
                print(f"Collected {NUM_IMAGES_PER_GESTURE} images for '{gesture_name}'.")

        elif key == ord('q'):
            break
        elif key == ord('d'):
            if current_gesture and image_count[current_gesture] > 0:
                image_count[current_gesture] -= 1
                img_path = os.path.join("gesture_images", current_gesture, f"{current_gesture}_{image_count[current_gesture]}.jpg")
                if os.path.exists(img_path):
                    os.remove(img_path)
                    print(f'Deleted {img_path}')
                else:
                    print(f"Warning: Image {img_path} not found.")

    cap.release()
    cv2.destroyAllWindows()

def train_model(gestures):
    # Save the gesture list to a file
    with open("gestures.txt", "w") as f:
        f.write("\n".join(gestures))

    # Rest of the training code...
    image_paths = list(paths.list_images("gesture_images"))
    data, labels = [], []
    label_encoder = LabelEncoder()

    for image_path in image_paths:
        label = image_path.split(os.path.sep)[-2]
        if label not in [sanitize_gesture_name(g) for g in gestures]:
            continue  # Skip images not in the specified gestures
        image = load_img(image_path, target_size=IMAGE_SIZE)
        image = img_to_array(image)
        image = preprocess_input(image)
        data.append(image)
        labels.append(label)

    # Rest of the training code...

    labels = label_encoder.fit_transform(labels)
    labels = to_categorical(labels)
    data = np.array(data)

    base_model = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))
    head_model = base_model.output
    head_model = GlobalAveragePooling2D()(head_model)
    head_model = Dense(128, activation="relu")(head_model)
    head_model = Dense(len(gestures), activation="softmax")(head_model)
    model = Model(inputs=base_model.input, outputs=head_model)

    for layer in base_model.layers:
        layer.trainable = False

    opt = Adam(learning_rate=LEARNING_RATE)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    train_datagen = ImageDataGenerator(rotation_range=20, zoom_range=0.15, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15, horizontal_flip=True, fill_mode="nearest")
    train_datagen.fit(data)

    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=TEST_SIZE, stratify=labels, random_state=RANDOM_STATE)

    model.fit(train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE), steps_per_epoch=len(X_train) // BATCH_SIZE, validation_data=(X_test, y_test), validation_steps=len(X_test) // BATCH_SIZE, epochs=EPOCHS, verbose=1)

    predictions = model.predict(X_test, batch_size=BATCH_SIZE)
    predictions = np.argmax(predictions, axis=1)
    y_test_decoded = np.argmax(y_test, axis=1)

    print("\nTraining Progress:")
    print(classification_report(y_test_decoded, predictions, target_names=[sanitize_gesture_name(gesture) for gesture in gestures]))

    cm = confusion_matrix(y_test_decoded, predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=[sanitize_gesture_name(gesture) for gesture in gestures], yticklabels=[sanitize_gesture_name(gesture) for gesture in gestures])
    plt.title("Confusion Matrix")
    plt.show()

    model.save("gesture_model.keras")
    print("Model saved as gesture_model.keras.")
    print("Training completed.")

if __name__ == "__main__":
    gestures = ["Bye", "Hi"]  # Default gestures (can be overridden by frontend)
    create_folders(gestures)
    capture_images(gestures)
    train_model(gestures)