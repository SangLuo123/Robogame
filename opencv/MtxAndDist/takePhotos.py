import cv2
import os

# 创建保存目录
save_dir = "/home/orangepi/Robogame/opencv/img"
os.makedirs(save_dir, exist_ok=True)

camera = cv2.VideoCapture(0)
if not camera.isOpened():
    print("无法打开摄像头")
    exit()

i = 1
print("按空格键拍摄图像，按 ESC 键退出程序")

while True:
    ret, frame = camera.read()
    if not ret:
        print("无法读取图像")
        continue

    # 显示图像
    cv2.imshow("Live View", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC 键
        break
    elif key == 32:  # 空格键
        img_path = os.path.join(save_dir, f"{i}.png")
        cv2.imwrite(img_path, frame, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        print(f"[{i}] 已保存：{img_path}")
        i += 1

camera.release()
cv2.destroyAllWindows()
