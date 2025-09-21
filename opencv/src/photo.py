import os
import cv2
import sys
import time
import numpy as np
from datetime import datetime
import json

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from src.load import load_calib
from src.multicam import _CamWorker, MultiCam


def main():
    save_dir = os.path.join(ROOT, "captures")
    os.makedirs(save_dir, exist_ok=True)
    K, D = load_calib(os.path.join(ROOT, "calib", "calibdown.npz"))
    w = 640
    h = 480
    mapx0, mapy0, newK0 = _CamWorker.build_undistort_maps(K, D, (w, h))
    mc = MultiCam()
    mc.add_camera("cam1", "/dev/cam_down", width=w, height=h, undistort_maps=(mapx0, mapy0, newK0), fourcc="MJPG")
    mc.start()
    
    print("按空格键保存图像，按ESC键退出")
    
    try:
        while True:
            # 获取最新帧
            frame_packet = mc.latest("cam1")
            
            if frame_packet is not None:
                # 显示图像
                cv2.imshow("Undistorted Camera View", frame_packet.image)
                
                # 显示FPS信息
                fps_text = f"FPS: {frame_packet.fps:.1f}"
                cv2.putText(frame_packet.image, fps_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 等待按键
            key = cv2.waitKey(1) & 0xFF
            
            # 按空格键保存图像
            if key == ord(' '):
                if frame_packet is not None:
                    # 生成文件名
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    filename = os.path.join(save_dir, f"capture_{timestamp}.jpg")
                    
                    # 保存图像
                    cv2.imwrite(filename, frame_packet.image)
                    print(f"图像已保存: {filename}")
                else:
                    print("没有可用的帧来保存")
            
            # 按ESC键退出
            elif key == 27:
                break
                
    except KeyboardInterrupt:
        print("程序被用户中断")
    
    finally:
        # 清理资源
        cv2.destroyAllWindows()
        mc.stop()
        print("程序已退出")

if __name__ == "__main__":
    main()
