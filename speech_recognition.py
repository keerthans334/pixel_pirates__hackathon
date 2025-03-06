import speech_recognition as sr
import threading
import queue
import time

class VoiceToTextConverter:
    def __init__(self, language="en-US"):
        self.recognizer = sr.Recognizer()
        self.language = language
        self.running = False
        self.audio_queue = queue.Queue()
        self.text_queue = queue.Queue()
        self.error_queue = queue.Queue()
        self.thread = None

    def start(self):
        """Starts the voice-to-text conversion process in a separate thread."""
        if self.running:
            return  # Prevent starting if already running
        self.running = True
        self.thread = threading.Thread(target=self._process_audio)
        self.thread.daemon = True  # Allows program to exit even if thread is running
        self.thread.start()

    def stop(self):
        """Stops the voice-to-text conversion process."""
        if not self.running:
            return
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join()  # Wait for the thread to finish

    def _process_audio(self):
        """Internal method to process audio from the queue and convert to text."""
        while self.running:
            try:
                audio = self.audio_queue.get(timeout=1)  # Get audio from queue, timeout after 1 second
                try:
                    text = self.recognizer.recognize_google(audio, language=self.language)
                    self.text_queue.put(text)
                except sr.UnknownValueError:
                    self.error_queue.put("Speech Recognition could not understand audio")
                except sr.RequestError as e:
                    self.error_queue.put(f"Could not request results from Google Speech Recognition service; {e}")
                except Exception as e:
                    self.error_queue.put(f"An unexpected error occurred during speech recognition: {e}")

                self.audio_queue.task_done()
            except queue.Empty:
                pass # Continue if queue is empty (timeout)
            except Exception as e:
                self.error_queue.put(f"An unexpected error occurred in the audio processing thread: {e}")
                self.running = False #stop the thread if major error occured.

    def capture_audio(self, source):
        """Captures audio from the given source and adds it to the queue."""
        try:
            audio = self.recognizer.listen(source, timeout=5) # Listen for a max of 5 seconds.
            self.audio_queue.put(audio)

        except sr.WaitTimeoutError:
            self.error_queue.put("Audio capture timed out.")
        except Exception as e:
            self.error_queue.put(f"An error occurred during audio capture: {e}")

    def get_text(self, block=False, timeout=None):
        """Retrieves text from the queue. Returns None if no text is available."""
        try:
            return self.text_queue.get(block=block, timeout=timeout)
        except queue.Empty:
            return None

    def get_error(self, block=False, timeout=None):
        """Retrieves errors from the queue. Returns None if no errors are available."""
        try:
            return self.error_queue.get(block=block, timeout=timeout)
        except queue.Empty:
            return None

# Example usage:
if __name__ == "__main__":
    converter = VoiceToTextConverter()
    converter.start()

    with sr.Microphone() as source:
        print("Say something!")
        for _ in range(5): # capture 5 audio clips.
            converter.capture_audio(source)
            time.sleep(0.5) # Wait a bit before capturing next audio.

            text = converter.get_text()
            error = converter.get_error()

            if text:
                print(f"You said: {text}")
            if error:
                print(f"Error: {error}")

    converter.stop()
    print("Voice-to-text conversion stopped.")