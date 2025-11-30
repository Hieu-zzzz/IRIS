import cv2
from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os

class DrinkDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Nhận diện Đồ uống - YOLOv8")
        self.root.geometry("1000x700")
        
        # Load model
        model_path = "runs/detect/drink_detection/weights/best.pt"
        if not os.path.exists(model_path):
            messagebox.showerror("Lỗi", f"Không tìm thấy model tại {model_path}")
            self.root.destroy()
            return
        
        self.model = YOLO(model_path)
        self.current_image = None
        self.result_image = None
        
        # Tạo giao diện
        self.create_widgets()
    
    def create_widgets(self):
        # Frame cho buttons
        button_frame = tk.Frame(self.root, bg="#2c3e50", height=80)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Buttons
        btn_style = {
            'font': ('Arial', 12, 'bold'),
            'bg': '#3498db',
            'fg': 'white',
            'activebackground': '#2980b9',
            'cursor': 'hand2',
            'relief': tk.RAISED,
            'bd': 3
        }
        
        tk.Button(button_frame, text="📁 Chọn ảnh", command=self.load_image, 
                 width=15, **btn_style).pack(side=tk.LEFT, padx=10, pady=10)
        
        tk.Button(button_frame, text="🎥 Webcam", command=self.open_webcam, 
                 width=15, **btn_style).pack(side=tk.LEFT, padx=10, pady=10)
        
        tk.Button(button_frame, text="🔍 Nhận diện", command=self.detect_image, 
                 width=15, bg='#27ae60', fg='white', activebackground='#229954',
                 cursor='hand2', relief=tk.RAISED, bd=3,
                 font=('Arial', 12, 'bold')).pack(side=tk.LEFT, padx=10, pady=10)
        
        tk.Button(button_frame, text="💾 Lưu kết quả", command=self.save_result, 
                 width=15, **btn_style).pack(side=tk.LEFT, padx=10, pady=10)
        
        # Frame cho hiển thị ảnh
        display_frame = tk.Frame(self.root, bg='#ecf0f1')
        display_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Canvas cho ảnh gốc
        left_frame = tk.Frame(display_frame, bg='#ecf0f1')
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        tk.Label(left_frame, text="Ảnh gốc", font=('Arial', 14, 'bold'), 
                bg='#ecf0f1').pack(pady=5)
        self.canvas_original = tk.Canvas(left_frame, bg='white', 
                                        relief=tk.SUNKEN, bd=2)
        self.canvas_original.pack(fill=tk.BOTH, expand=True)
        
        # Canvas cho ảnh kết quả
        right_frame = tk.Frame(display_frame, bg='#ecf0f1')
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        
        tk.Label(right_frame, text="Kết quả nhận diện", 
                font=('Arial', 14, 'bold'), bg='#ecf0f1').pack(pady=5)
        self.canvas_result = tk.Canvas(right_frame, bg='white', 
                                       relief=tk.SUNKEN, bd=2)
        self.canvas_result.pack(fill=tk.BOTH, expand=True)
        
        # Label thông tin
        self.info_label = tk.Label(self.root, text="Sẵn sàng nhận diện!", 
                                   font=('Arial', 12), bg='#34495e', 
                                   fg='white', height=2)
        self.info_label.pack(fill=tk.X, padx=10, pady=5)
    
    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Chọn ảnh",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), 
                      ("All files", "*.*")]
        )
        
        if file_path:
            self.current_image = cv2.imread(file_path)
            self.current_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
            self.display_image(self.current_image, self.canvas_original)
            self.info_label.config(text=f"Đã tải: {os.path.basename(file_path)}")
            # Xóa kết quả cũ
            self.canvas_result.delete("all")
            self.result_image = None
    
    def detect_image(self):
        if self.current_image is None:
            messagebox.showwarning("Cảnh báo", "Vui lòng chọn ảnh trước!")
            return
        
        self.info_label.config(text="Đang nhận diện...")
        self.root.update()
        
        # Chuyển về BGR cho YOLO
        img_bgr = cv2.cvtColor(self.current_image, cv2.COLOR_RGB2BGR)
        
        # Detect
        results = self.model(img_bgr)
        
        # Vẽ kết quả
        annotated = results[0].plot()
        self.result_image = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        
        # Hiển thị
        self.display_image(self.result_image, self.canvas_result)
        
        # Thông tin kết quả
        detections = results[0].boxes
        num_objects = len(detections)
        
        if num_objects > 0:
            detected_items = []
            for box in detections:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                name = self.model.names[cls_id]
                detected_items.append(f"{name} ({conf:.2%})")
            
            info_text = f"Phát hiện {num_objects} đối tượng: " + ", ".join(detected_items)
        else:
            info_text = "Không phát hiện đối tượng nào"
        
        self.info_label.config(text=info_text)
    
    def display_image(self, img, canvas):
        # Resize ảnh để fit canvas
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            canvas_width = 450
            canvas_height = 500
        
        h, w = img.shape[:2]
        scale = min(canvas_width/w, canvas_height/h) * 0.95
        new_w, new_h = int(w*scale), int(h*scale)
        
        img_resized = cv2.resize(img, (new_w, new_h))
        img_pil = Image.fromarray(img_resized)
        img_tk = ImageTk.PhotoImage(img_pil)
        
        canvas.delete("all")
        canvas.create_image(canvas_width//2, canvas_height//2, 
                           image=img_tk, anchor=tk.CENTER)
        canvas.image = img_tk
    
    def save_result(self):
        if self.result_image is None:
            messagebox.showwarning("Cảnh báo", "Chưa có kết quả để lưu!")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png"), ("All files", "*.*")]
        )
        
        if file_path:
            img_bgr = cv2.cvtColor(self.result_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(file_path, img_bgr)
            messagebox.showinfo("Thành công", f"Đã lưu: {file_path}")
    
    def open_webcam(self):
        # Tạo cửa sổ webcam
        webcam_window = tk.Toplevel(self.root)
        webcam_window.title("Nhận diện realtime - Webcam")
        webcam_window.geometry("800x650")
        
        canvas = tk.Canvas(webcam_window, width=780, height=580)
        canvas.pack(pady=10)
        
        btn_frame = tk.Frame(webcam_window)
        btn_frame.pack()
        
        is_running = [True]
        
        def stop_webcam():
            is_running[0] = False
            webcam_window.destroy()
        
        tk.Button(btn_frame, text="⏹ Dừng", command=stop_webcam,
                 font=('Arial', 12, 'bold'), bg='#e74c3c', fg='white',
                 width=15, cursor='hand2').pack()
        
        cap = cv2.VideoCapture(0)
        
        def update_frame():
            if not is_running[0]:
                cap.release()
                return
            
            ret, frame = cap.read()
            if ret:
                # Detect
                results = self.model(frame)
                annotated = results[0].plot()
                
                # Convert to RGB
                frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                
                # Resize
                frame_rgb = cv2.resize(frame_rgb, (780, 580))
                
                # Display
                img_pil = Image.fromarray(frame_rgb)
                img_tk = ImageTk.PhotoImage(img_pil)
                canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
                canvas.image = img_tk
            
            if is_running[0]:
                webcam_window.after(10, update_frame)
        
        update_frame()

if __name__ == "__main__":
    root = tk.Tk()
    app = DrinkDetectionApp(root)
    root.mainloop()
