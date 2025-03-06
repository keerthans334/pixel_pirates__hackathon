import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import tensorflow as tf
import pyttsx3
import time
import threading
import numpy as np
from tensorflow.keras.applications import mobilenet_v2
import queue

# Load gestures from file
def load_gestures():
    with open("gestures.txt", "r") as f:
        gestures = f.read().splitlines()
    return gestures

GESTURES = load_gestures()  # Load gestures dynamically

def display_prediction(frame, gesture, confidence):
    """Display the prediction on the frame."""
    cv2.putText(frame, f"Prediction: {gesture}, Confidence: {confidence:.2f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

def display_error(frame, error):
    """Display an error message on the frame."""
    cv2.putText(frame, f"Error: {error}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

def display_gesture_history(frame, gesture_history):
    """Display the gesture history on the frame."""
    y_offset = 100
    for g in gesture_history:
        cv2.putText(frame, f"Prev: {g}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 25

def display_confidence_threshold(frame, confidence_threshold):
    """Display the confidence threshold on the frame."""
    cv2.putText(frame, f"Conf: {confidence_threshold:.2f}", (450, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

def speech_worker(engine, speech_queue):
    """Worker function to process speech requests from the queue."""
    while True:
        text = speech_queue.get()
        if text is None:  # Sentinel value to stop the thread
            break
        engine.say(text)
        engine.runAndWait()
        speech_queue.task_done()

def main():
    cap = cv2.VideoCapture(0)
    engine = pyttsx3.init()
    engine.setProperty("rate", 180)
    engine.setProperty("volume", 0.9)
    verbose_mode = False
    confidence_threshold = 0.5
    gesture_history = []
    last_gesture = None
    last_gesture_time = 0
    gesture_hold_duration = 2

    # Create a queue for speech requests
    speech_queue = queue.Queue()
    # Start the speech worker thread
    speech_thread = threading.Thread(target=speech_worker, args=(engine, speech_queue))
    speech_thread.start()

    try:
        model = tf.keras.models.load_model("gesture_model.keras")
    except FileNotFoundError:
        print("Model file 'gesture_model.keras' not found. Please train the model first.")
        return

    # Initialize MediaPipe Hand Landmarker
    base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
    hand_landmarker = vision.HandLandmarker.create_from_options(options)

    while True:
        try:
            # Read a frame from the camera
            ret, frame = cap.read()
            if not ret:
                break

            frame_resized = cv2.resize(frame, (224, 224))
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            frame_preprocessed = mobilenet_v2.preprocess_input(frame_rgb)

            frame_np = np.expand_dims(frame_preprocessed, axis=0)

            # Perform gesture recognition
            prediction = model.predict(frame_np)
            gesture_index = np.argmax(prediction)
            gesture = GESTURES[gesture_index]
            confidence = prediction[0][gesture_index]

            if confidence > confidence_threshold:
                current_time = time.time()
                if gesture == last_gesture and current_time - last_gesture_time >= gesture_hold_duration:
                    if last_gesture != "":
                        # Add the gesture text to the speech queue
                        speech_queue.put(gesture)
                        last_gesture = ""
                        gesture_history.insert(0, gesture)
                        if len(gesture_history) > 5:
                            gesture_history.pop()

                elif gesture != last_gesture:
                    last_gesture = gesture
                    last_gesture_time = current_time

            # Display the prediction
            display_prediction(frame, gesture, confidence)

            # Display the gesture history
            display_gesture_history(frame, gesture_history)

            # Display the confidence threshold
            display_confidence_threshold(frame, confidence_threshold)

            # Detect hand landmarks
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            results = hand_landmarker.detect(mp_image)

            if results.hand_landmarks:
                for hand_landmarks_list in results.hand_landmarks:
                    for landmark in hand_landmarks_list:
                        # Draw landmarks on the frame using OpenCV
                        x = int(landmark.x * frame.shape[1])
                        y = int(landmark.y * frame.shape[0])
                        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)  # Draw a circle for each landmark

            if verbose_mode:
                print(f"Prediction: {gesture}, Confidence: {confidence:.2f}")

        except (ValueError, IndexError, TypeError) as e:
            # Display an error message
            display_error(frame, e)

        # Display the frame
        cv2.imshow("Gesture Recognition", frame)

        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('v'):
            verbose_mode = not verbose_mode
            print(f"Verbose mode: {verbose_mode}")
        elif key == ord('c'):
            gesture_history.clear()
        elif key == ord('a'):
            confidence_threshold = min(0.95, confidence_threshold + 0.05)
            print(f"Confidence threshold: {confidence_threshold:.2f}")
        elif key == ord('s'):
            confidence_threshold = max(0.5, confidence_threshold - 0.05)
            print(f"Confidence threshold: {confidence_threshold:.2f}")

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

    # Stop the speech worker thread
    speech_queue.put(None)  # Sentinel value to stop the thread
    speech_thread.join()
    engine.stop()

if __name__ == "__main__":
    main()