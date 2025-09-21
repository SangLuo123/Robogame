import cv2
import threading

class Camera:
    def __init__(self, camera_index):
        self.camera_index = camera_index
        self.cap = None
        self.lock = threading.Lock()
        self.is_initialized = False
        
    def initialize(self):
        """初始化相机"""
        if not self.is_initialized:
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                raise RuntimeError("无法打开摄像头")
            
            # 设置相机参数（根据你的相机调整）
            # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            # self.cap.set(cv2.CAP_PROP_FPS, 30)
            # self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
            
            # 让相机预热几帧
            for _ in range(10):
                self.cap.read()
            
            self.is_initialized = True
            print("相机初始化完成")
    
    def get_frame(self):
        """获取当前帧"""
        with self.lock:
            if not self.is_initialized:
                self.initialize()
            
            ret, frame = self.cap.read()
            if not ret:
                raise RuntimeError("无法从摄像头获取图像")
            frame = cv2.rotate(frame, cv2.ROTATE_180)  # 根据需要旋转图像
            return frame
    
    def release(self):
        """释放相机资源"""
        with self.lock:
            if self.cap and self.is_initialized:
                self.cap.release()
                self.is_initialized = False
                print("相机已释放")