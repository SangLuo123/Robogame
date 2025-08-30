# demo_twocam.py
import cv2, time
from multicam import MultiCam

TARGET_WIDTH = 960   # 拼接图目标显示宽度（像素），可再调小

def fit_width(im, width):
    h, w = im.shape[:2]
    scale = float(width) / float(w)
    new_size = (width, max(1, int(h * scale)))
    return cv2.resize(im, new_size, interpolation=cv2.INTER_AREA)

def wait_ready(mc, names, timeout_s=3.0):
    t0 = time.time()
    ready = {n: False for n in names}
    while time.time() - t0 < timeout_s:
        for n in names:
            if not ready[n] and mc.latest(n) is not None:
                print(f"[OK] {n} produced first frame")
                ready[n] = True
        if all(ready.values()):
            return True
        time.sleep(0.02)
    for n, ok in ready.items():
        if not ok:
            print(f"[ERR] {n} no frame within {timeout_s}s")
    return False

def main():
    mc = MultiCam()
    # 先用较低分辨率验证链路更稳；OK 后再回到 1280x720
    backend = cv2.CAP_DSHOW  # Windows使用DirectShow后端 # linux上直接使用默认赋值即可
    mc.add_camera("cam0", 0,   width=640, height=480, fourcc="MJPG", backend=backend)
    mc.add_camera("cam1", 2, width=640, height=480, fourcc="MJPG", backend=backend)
    mc.start()

    # 创建可缩放窗口，并设置初始大小
    win = "cam0 | cam1"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, TARGET_WIDTH, int(TARGET_WIDTH * 9 / 16))

    try:
        # 等待两路都出第一帧（快速定位设备问题）
        wait_ready(mc, ["cam0", "cam1"], timeout_s=3.0)

        last_diag = 0.0
        while True:
            # 调试阶段可以把 max_skew_ms 放宽到 80~120ms，看是否能同步
            pair = mc.get_pair_synced("cam0", "cam1", max_skew_ms=80, timeout_ms=300)
            if pair is None:
                now = time.time()
                if now - last_diag > 0.5:
                    p0, p1 = mc.latest("cam0"), mc.latest("cam1")
                    if p0 is None or p1 is None:
                        print("[WARN] sync timeout (one camera has no frames)")
                    else:
                        skew_ms = abs(p0.ts_ns - p1.ts_ns) / 1e6
                        print(f"[WARN] sync timeout, skew={skew_ms:.1f}ms, fps0={p0.fps:.1f}, fps1={p1.fps:.1f}")
                    last_diag = now
                continue

            p0, p1 = pair
            img0, img1 = p0.image, p1.image

            # 拼接前按目标宽度缩放
            img0s = fit_width(img0, TARGET_WIDTH // 2)
            img1s = fit_width(img1, TARGET_WIDTH // 2)
            vis = cv2.hconcat([img0s, img1s])

            # HUD
            ts_ms = int((max(p0.ts_ns, p1.ts_ns)) / 1e6)
            cv2.putText(vis, f"cam0 fps:{p0.fps:.1f}  cam1 fps:{p1.fps:.1f}  ts:{ts_ms}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)

            cv2.imshow(win, vis)
            k = cv2.waitKey(1) & 0xFF
            if k in (27, ord('q')):
                break

    finally:
        mc.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
