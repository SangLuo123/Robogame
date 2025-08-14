import numpy as np

def load_calib(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    if "mtx" not in data or "dist" not in data:
        raise KeyError("npz 中必须包含 'mtx' 和 'dist'")
    mtx = data["mtx"].astype(np.float32)
    dist = data["dist"].astype(np.float32).reshape(-1)  # 展平为(5,)等
    return mtx, dist