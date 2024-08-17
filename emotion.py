import cv2
from deepface import DeepFace
import threading
import tkinter as tk
from tkinter import Label, Button, messagebox
from PIL import Image, ImageTk

# Load face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

class EmotionDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Real-time Emotion Detection")
        self.root.geometry("800x600")
        self.root.configure(bg='#2E4053')

        # Create a label to display the video feed
        self.lbl_video = Label(self.root, bg='#2E4053')
        self.lbl_video.pack(pady=20)

        # Create start and stop buttons
        self.btn_start = Button(self.root, text="Start", command=self.start_video, font=('Helvetica', 12), bg='#28B463', fg='white', padx=20, pady=10)
        self.btn_start.pack(side=tk.LEFT, padx=50)

        self.btn_stop = Button(self.root, text="Stop", command=self.stop_video, font=('Helvetica', 12), bg='#E74C3C', fg='white', padx=20, pady=10)
        self.btn_stop.pack(side=tk.RIGHT, padx=50)

        self.running = False
        self.cap = None

    def start_video(self):
        self.cap = cv2.VideoCapture(0)
        self.running = True
        self.thread = threading.Thread(target=self.video_loop)
        self.thread.start()

    def stop_video(self):
        self.running = False
        messagebox.showinfo("Info", "Video capture stopped.")
        
    def video_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)
            faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                face_roi = rgb_frame[y:y + h, x:x + w]
                result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
                emotion = result[0]['dominant_emotion']
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            # Convert the frame to ImageTk format and update the GUI label
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)
            self.lbl_video.imgtk = imgtk
            self.lbl_video.configure(image=imgtk)
            self.lbl_video.update()

        self.cap.release()

# Create the main window
root = tk.Tk()
app = EmotionDetectionApp(root)
root.mainloop()
