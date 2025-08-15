# findOpt_byHSV.py
import json
import os
import cv2
import numpy as np

# === 按你的项目实际修改导入路径 ===
from package import preprocess_frame
from package import UndistortedCapture

JSON_PATH = "hsv_range.json"                # get_param 生成的文件
CALIB_PATH = "./mtx_dist/calib.npz"       # 相机标定参数
CAM_ID = 0                                  # 摄像头编号
ALPHA = 0.0                                 # 去畸变 alpha（0~1）

def load_hsv_ranges(json_path: str):
    """
    从 json 读取 HSV 范围：
    - 支持 {"lower":[...], "upper":[...]}
    - 也支持 [{"lower":[...], "upper":[...]} , ...]
    - 自动处理 H 跨 0°（例如 170~10°）为两段
    返回：[(lower, upper), ...]，lower/upper 为 np.uint8 向量
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"未找到 {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    def normalize_one(lo, up):
        h1, s1, v1 = map(int, lo)
        h2, s2, v2 = map(int, up)
        # 夹紧到合法区间
        h1, h2 = np.clip([h1, h2], 0, 179)
        s1, s2 = np.clip([s1, s2], 0, 255)
        v1, v2 = np.clip([v1, v2], 0, 255)
        lo = np.array([h1, s1, v1], dtype=np.uint8)
        up = np.array([h2, s2, v2], dtype=np.uint8)

        # 跨 0°（例如红色 170~10），拆两段： [h1,179] 和 [0,h2]
        if h1 > h2:
            a1 = (np.array([h1, s1, v1], np.uint8), np.array([179, s2, v2], np.uint8))
            a2 = (np.array([0,  s1, v1], np.uint8), np.array([h2, s2, v2], np.uint8))
            return [a1, a2]
        else:
            return [(lo, up)]

    ranges = []
    if isinstance(data, dict) and "lower" in data and "upper" in data:
        ranges.extend(normalize_one(data["lower"], data["upper"]))
    elif isinstance(data, list):
        for item in data:
            if "lower" in item and "upper" in item:
                ranges.extend(normalize_one(item["lower"], item["upper"]))
    else:
        raise ValueError("JSON 格式不支持，请提供 {lower,upper} 或 [{lower,upper},...]")

    return ranges

def build_2x2_canvas(img1, img2, img3, tile_h=300):
    """把三张图拼成 2x2（第四格复用第三张）"""
    def to_bgr(x):
        return cv2.cvtColor(x, cv2.COLOR_GRAY2BGR) if x.ndim == 2 else x

    def resize_h(x, h):
        return cv2.resize(x, (int(x.shape[1] * h / x.shape[0]), h))

    a = resize_h(to_bgr(img1), tile_h)
    b = resize_h(to_bgr(img2), tile_h)
    c = resize_h(to_bgr(img3), tile_h)
    d = c.copy()

    # 统一宽度
    w = min(a.shape[1], b.shape[1], c.shape[1], d.shape[1])
    a = cv2.resize(a, (w, tile_h))
    b = cv2.resize(b, (w, tile_h))
    c = cv2.resize(c, (w, tile_h))
    d = cv2.resize(d, (w, tile_h))

    top = np.hstack((a, b))
    bottom = np.hstack((c, d))
    return np.vstack((top, bottom))

def pick_dart_blob(mask_bin):
    """
    输入：0/255 二值掩码
    返回：clean_mask（只保留飞镖那一团），best_cnt（对应轮廓），score
    """
    h, w = mask_bin.shape[:2]

    # 形态学去噪（核可按目标尺寸微调）
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    m = cv2.morphologyEx(mask_bin, cv2.MORPH_OPEN,  k_open)
    m = cv2.morphologyEx(m,        cv2.MORPH_CLOSE, k_close)

    # 连通域
    num, labels, stats, cents = cv2.connectedComponentsWithStats(m, connectivity=8)
    # stats: [label, x, y, w, h, area]

    best = None
    best_score = -1.0

    for lab in range(1, num):
        x, y, ww, hh, area = stats[lab]
        if area < 1000:         # 最小面积阈值：按相机分辨率调整
            continue
        aspect = max(ww, hh) / (min(ww, hh) + 1e-6)  # 细长程度（飞镖一般>1.5）
        # 用轮廓做更精细的度量
        comp = (labels == lab).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        cnt = max(cnts, key=cv2.contourArea)
        area_cnt = cv2.contourArea(cnt)
        peri = cv2.arcLength(cnt, True)
        hull = cv2.convexHull(cnt)
        area_hull = cv2.contourArea(hull)
        if area_cnt < 800:
            continue
        solidity = area_cnt / (area_hull + 1e-6)     # 实心度（0~1，飞镖普遍较高）
        circularity = 4*np.pi*area_cnt / (peri*peri + 1e-6)  # 圆度（飞镖偏低）

        # 经验打分：面积越大越好，细长越好，实心度适中偏高更好，圆度低一点（非圆）
        score = (area_cnt/1000.0) + 1.2*min(aspect, 4.0) + 2.0*solidity - 0.8*circularity

        if score > best_score:
            best_score = score
            best = (comp, cnt)

    clean = np.zeros_like(mask_bin)
    best_cnt = None
    if best is not None:
        comp, best_cnt = best
        clean = comp.copy()

    return clean, best_cnt, best_score


def detect_from_json(json_path=JSON_PATH, camera_index=CAM_ID):
    # 载入 HSV 范围
    hsv_ranges = load_hsv_ranges(json_path)
    print("[INFO] 已加载 HSV 范围：", hsv_ranges)

    # 打开带去畸变的摄像头
    cap = UndistortedCapture(camera_index, calib_path=CALIB_PATH, alpha=ALPHA)
    if not cap.isOpened():
        print("[ERR] 无法打开摄像头")
        return

    print("[INFO] 热键：R 重新加载JSON，S 保存拼接图，ESC/Q 退出")

    while True:
        ok, raw, und = cap.read()
        if not ok:
            print("[WARN] 读取帧失败，退出")
            break

        # 统一预处理（你的 vision_prep 包）
        frame = preprocess_frame(
            und,            # 这里用去畸变后的帧
            downscale=1.0,  # 预览缩放倍率，可按需改
            do_white_balance=True,
            blur_ksize=5
        )

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 合并掩码（多段 + 多区间）
        full_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lo, up in hsv_ranges:
            mask = cv2.inRange(hsv, lo, up)
            full_mask = cv2.bitwise_or(full_mask, mask)

        # 轻量去噪（可按需调整）
        full_mask = cv2.medianBlur(full_mask, 5)

        # result = cv2.bitwise_and(frame, frame, mask=full_mask)
        
        clean_mask, dart_cnt, dart_score = pick_dart_blob(full_mask)
        result = cv2.bitwise_and(frame, frame, mask=clean_mask)

        # 可视化：在 result 上画轮廓和最小外接矩形
        vis = result.copy()
        if dart_cnt is not None:
            box = cv2.boxPoints(cv2.minAreaRect(dart_cnt))  # 返回4点
            box = box.astype(np.int32)
            cv2.drawContours(vis, [box], 0, (0,255,0), 2)
            cv2.drawContours(vis, [dart_cnt], -1, (255,0,0), 2)
            cv2.putText(vis, f"score={dart_score:.1f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        canvas = build_2x2_canvas(frame, full_mask, result, tile_h=300)
        cv2.imshow("HSV Detect 2x2", canvas)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q'), ord('Q')):
            break
        elif key in (ord('r'), ord('R')):
            try:
                hsv_ranges = load_hsv_ranges(json_path)
                print("[OK] 重新加载 HSV 范围：", hsv_ranges)
            except Exception as e:
                print("[ERR] 重新加载失败：", e)
        elif key in (ord('s'), ord('S')):
            cv2.imwrite("hsv_detect_2x2.png", canvas)
            print("[OK] 已保存 hsv_detect_2x2.png")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_from_json()
