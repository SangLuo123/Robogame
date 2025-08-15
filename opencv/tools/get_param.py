import cv2
import numpy as np
import json
import os
from package import preprocess_frame, UndistortedCapture

def adjust_hsv_with_trackbars(camera_index=0,
                              do_white_balance=True,
                              preview_downscale=1.0,
                              save_path="hsv_range.json",
                              blur_ksize=5):
    def nothing(x): pass
    lower = [0, 0, 0]
    upper = [179, 255, 255]

    cv2.namedWindow("Trackbars")
    cv2.resizeWindow("Trackbars", 420, 320)
    cv2.createTrackbar("H Min", "Trackbars", lower[0], 179, nothing)
    cv2.createTrackbar("S Min", "Trackbars", lower[1], 255, nothing)
    cv2.createTrackbar("V Min", "Trackbars", lower[2], 255, nothing)
    cv2.createTrackbar("H Max", "Trackbars", upper[0], 179, nothing)
    cv2.createTrackbar("S Max", "Trackbars", upper[1], 255, nothing)
    cv2.createTrackbar("V Max", "Trackbars", upper[2], 255, nothing)

    if os.path.exists(save_path):
        try:
            with open(save_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            l = data.get("lower"); u = data.get("upper")
            if l and u:
                lower = [int(l[0]), int(l[1]), int(l[2])]
                upper = [int(u[0]), int(u[1]), int(u[2])]
                cv2.setTrackbarPos("H Min", "Trackbars", lower[0])
                cv2.setTrackbarPos("S Min", "Trackbars", lower[1])
                cv2.setTrackbarPos("V Min", "Trackbars", lower[2])
                cv2.setTrackbarPos("H Max", "Trackbars", upper[0])
                cv2.setTrackbarPos("S Max", "Trackbars", upper[1])
                cv2.setTrackbarPos("V Max", "Trackbars", upper[2])
                print(f"[INFO] 加载已保存的 HSV：{save_path}")
        except Exception as e:
            print(f"[WARN] 读取 {save_path} 失败：{e}")

    cap = UndistortedCapture(camera_index, calib_path="./mtx_dist/calib.npz", alpha=0.0)
    if not cap.isOpened():
        print("[ERR] 无法打开摄像头")
        return lower, upper
    # cap = cv2.VideoCapture(camera_index)
    # if not cap.isOpened():
    #     print("[ERR] 无法打开摄像头")
    #     return lower, upper

    prev_range = None
    print("[INFO] 操作：按 S 保存，按 L 加载，ESC/Q 退出")

    while True:
        ok, _, frame = cap.read()
        if not ok:
            print("[WARN] 读取帧失败，退出")
            break

        frame_p = preprocess_frame(frame,
                                   downscale=preview_downscale,
                                   do_white_balance=do_white_balance,
                                   blur_ksize=blur_ksize)

        hsv = cv2.cvtColor(frame_p, cv2.COLOR_BGR2HSV)

        h_min = cv2.getTrackbarPos("H Min", "Trackbars")
        s_min = cv2.getTrackbarPos("S Min", "Trackbars")
        v_min = cv2.getTrackbarPos("V Min", "Trackbars")
        h_max = cv2.getTrackbarPos("H Max", "Trackbars")
        s_max = cv2.getTrackbarPos("S Max", "Trackbars")
        v_max = cv2.getTrackbarPos("V Max", "Trackbars")

        if h_min > h_max: h_min, h_max = h_max, h_min
        if s_min > s_max: s_min, s_max = s_max, s_min
        if v_min > v_max: v_min, v_max = v_max, v_min

        lower = [h_min, s_min, v_min]
        upper = [h_max, s_max, v_max]

        mask = cv2.inRange(hsv, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8))
        mask = cv2.medianBlur(mask, 5)
        result = cv2.bitwise_and(frame_p, frame_p, mask=mask)

        if (h_min, h_max, s_min, s_max, v_min, v_max) != prev_range:
            print(f"[HSV] Lower:{lower} Upper:{upper}")
            prev_range = (h_min, h_max, s_min, s_max, v_min, v_max)

        # ====== 拼接 2x2 布局 ======
        def resize_img(img, h):
            return cv2.resize(img, (int(img.shape[1] * h / img.shape[0]), h))

        disp_h = 300  # 每个图块高度
        orig_disp = resize_img(frame_p, disp_h)
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        mask_disp = resize_img(mask_bgr, disp_h)
        result_disp = resize_img(result, disp_h)

        # 第四格重复第三格
        fourth_disp = result_disp.copy()

        # 保证宽度一致（用最小宽度对齐）
        min_w = min(orig_disp.shape[1], mask_disp.shape[1], result_disp.shape[1], fourth_disp.shape[1])
        orig_disp = cv2.resize(orig_disp, (min_w, disp_h))
        mask_disp = cv2.resize(mask_disp, (min_w, disp_h))
        result_disp = cv2.resize(result_disp, (min_w, disp_h))
        fourth_disp = cv2.resize(fourth_disp, (min_w, disp_h))

        top_row = np.hstack((orig_disp, mask_disp))
        bottom_row = np.hstack((result_disp, fourth_disp))
        combined = np.vstack((top_row, bottom_row))

        cv2.imshow("HSV Tuning 2x2", combined)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q'), ord('Q')):
            break
        elif key in (ord('s'), ord('S')):
            try:
                with open(save_path, "w", encoding="utf-8") as f:
                    json.dump({"lower": lower, "upper": upper}, f, ensure_ascii=False, indent=2)
                print(f"[OK] 已保存到 {save_path}")
            except Exception as e:
                print(f"[ERR] 保存失败：{e}")
        elif key in (ord('l'), ord('L')):
            try:
                with open(save_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                l = data.get("lower"); u = data.get("upper")
                if l and u:
                    lower = [int(l[0]), int(l[1]), int(l[2])]
                    upper = [int(u[0]), int(u[1]), int(u[2])]
                    cv2.setTrackbarPos("H Min", "Trackbars", lower[0])
                    cv2.setTrackbarPos("S Min", "Trackbars", lower[1])
                    cv2.setTrackbarPos("V Min", "Trackbars", lower[2])
                    cv2.setTrackbarPos("H Max", "Trackbars", upper[0])
                    cv2.setTrackbarPos("S Max", "Trackbars", upper[1])
                    cv2.setTrackbarPos("V Max", "Trackbars", upper[2])
                    print(f"[OK] 已加载 {save_path}")
            except Exception as e:
                print(f"[ERR] 加载失败：{e}")

    cap.release()
    cv2.destroyAllWindows()
    return lower, upper

if __name__ == "__main__":
    adjust_hsv_with_trackbars(0)
