import customtkinter as ctk
from tkinter import messagebox
from PIL import Image, ImageTk
import threading
import cv2
import time
from ultralytics import YOLO
from enum import Enum
import torch


# Model types
class ModelType(Enum):
    YOLOv8n = 'yolov8n.pt'
    YOLOv8s = 'yolov8s.pt'
    YOLOv8m = 'yolov8m.pt'


# Scan for cameras
def list_available_cameras(max_tested=5):
    available = []
    for i in range(max_tested):
        cap = cv2.VideoCapture(i)
        if cap.read()[0]:
            available.append(i)
        cap.release()
    return available


# Main App
class YOLOApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("YOLOv8 Live Detection")
        self.geometry("900x700")
        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("green")

        self.cameras = []
        self.camera_index = None
        self.model_type = ModelType.YOLOv8n
        self.model = None
        self.running = False
        self.device = "cpu"

        self.setup_ui()

    def setup_ui(self):
        # Title
        ctk.CTkLabel(self, text="🎯 YOLOv8 Live Detection",
                     font=("Arial Rounded MT Bold", 28)).pack(pady=20)

        # Controls
        control_frame = ctk.CTkFrame(self)
        control_frame.pack(pady=10)

        self.scan_button = ctk.CTkButton(control_frame, text="🔍 Scan Cameras",
                                         command=self.scan_cameras)
        self.scan_button.grid(row=0, column=0, padx=10)

        self.camera_menu = ctk.CTkOptionMenu(control_frame, values=["No Cameras"],
                                             command=self.select_camera)
        self.camera_menu.grid(row=0, column=1, padx=10)

        self.model_menu = ctk.CTkOptionMenu(control_frame,
                                            values=[model.name for model in ModelType],
                                            command=self.select_model)
        self.model_menu.grid(row=0, column=2, padx=10)

        self.start_button = ctk.CTkButton(control_frame, text="🚀 Start Detection",
                                          command=self.start_detection)
        self.start_button.grid(row=0, column=3, padx=10)

        self.stop_button = ctk.CTkButton(control_frame, text="🛑 Stop",
                                         command=self.stop_detection, fg_color="red")
        self.stop_button.grid(row=0, column=4, padx=10)

        # Live preview
        self.preview_label = ctk.CTkLabel(self, text="")
        self.preview_label.pack(pady=20)

        # FPS + Device display
        self.fps_label = ctk.CTkLabel(self, text="FPS: 0.00 | Device: CPU",
                                      font=("Arial", 18))
        self.fps_label.pack()

    def scan_cameras(self):
        self.cameras = list_available_cameras()
        if not self.cameras:
            self.camera_menu.configure(values=["No Cameras"])
            messagebox.showwarning("Warning", "No cameras found!")
        else:
            self.camera_menu.configure(values=[str(i) for i in self.cameras])
            self.camera_menu.set(str(self.cameras[0]))
            self.camera_index = self.cameras[0]

    def select_camera(self, choice):
        self.camera_index = int(choice)

    def select_model(self, choice):
        self.model_type = ModelType[choice]

    def start_detection(self):
        if self.camera_index is None:
            messagebox.showwarning("Warning", "Please scan and select a camera first.")
            return

        self.model = YOLO(self.model_type.value)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Running on {self.device.upper()} 🚀")

        self.running = True
        self.start_button.configure(state="disabled")  # Disable start while running
        threading.Thread(target=self.detection_loop, daemon=True).start()

    def stop_detection(self):
        self.running = False
        self.start_button.configure(state="normal")  # Re-enable start button

    def detection_loop(self):
        cap = cv2.VideoCapture(self.camera_index)

        while self.running:
            ret, frame = cap.read()
            if not ret:
                break

            start_time = time.time()

            # Run YOLOv8 detection
            results = self.model(frame, verbose=False, device=self.device)
            annotated_frame = results[0].plot()

            # Convert for Tkinter
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(annotated_frame)
            img = img.resize((800, 500))
            img_tk = ImageTk.PhotoImage(img)

            self.preview_label.configure(image=img_tk)
            self.preview_label.image = img_tk

            fps = 1 / (time.time() - start_time)
            self.fps_label.configure(text=f"FPS: {fps:.2f} | Device: {self.device.upper()}")

        cap.release()
        self.start_button.configure(state="normal")


if __name__ == "__main__":
    app = YOLOApp()
    app.mainloop()
