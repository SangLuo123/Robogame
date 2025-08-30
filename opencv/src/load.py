import json
import os
import numpy as np

def load_config(path):
    """
    读取项目配置（若没有就给默认值）
    内容建议包含：串口、tag_size、T_robot_cam、tag_map、HSV、相机ID等
    """
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    # 默认配置（可在 data/config.json 里覆盖）
    return {
        "serial_port": "/dev/ttlUSB0",
        "baud": 115200,
        "tag_size": 120,
        "cameras": [
            {
                "id": "cam1",
                "device": "/dev/cam_down",
                "calib": "data/calib/camdown.npz",
                "role": "tag+dart",
                "hsv": {
                    "lower": [
                        0,
                        75,
                        148
                    ],
                    "upper": [
                        13,
                        235,
                        255
                    ]
                }
            },
            {
                "id": "cam0",
                "device": "/dev/cam_up",
                "calib": "data/calib/camup.npz",
                "role": "tag+led",
                "hsv": {
                    "lower": [
                        45,
                        60,
                        80
                    ],
                    "upper": [
                        85,
                        255,
                        255
                    ]
                }
            }
        ],
        "hsv_range": {
            "lower": [
                0,
                75,
                148
            ],
            "upper": [
                13,
                235,
                255
            ]
        },
        "T_robot_cam_for": {
            "cam0": {
                "R": [
                    [
                        1,
                        0,
                        0
                    ],
                    [
                        0,
                        1,
                        0
                    ],
                    [
                        0,
                        0,
                        1
                    ]
                ],
                "t": [
                    0.0,
                    0.0,
                    0.0
                ]
            },
            "cam1": {
                "R": [
                    [
                        1,
                        0,
                        0
                    ],
                    [
                        0,
                        1,
                        0
                    ],
                    [
                        0,
                        0,
                        1
                    ]
                ],
                "t": [
                    0.0,
                    0.0,
                    0.0
                ]
            }
        },
        "tag_map": {
            "3": {
                "tl": [
                    2032,
                    5000,
                    265.5
                ],
                "tr": [
                    2032,
                    5000,
                    385.5
                ],
                "br": [
                    2152,
                    5000,
                    385.5
                ],
                "bl": [
                    2152,
                    5000,
                    265.5
                ]
            },
            "2": {
                "tl": [
                    0,
                    4250,
                    265.5
                ],
                "tr": [
                    0,
                    4130,
                    265.5
                ],
                "br": [
                    0,
                    4130,
                    385.5
                ],
                "bl": [
                    0,
                    4250,
                    385.5
                ]
            },
            "4": {
                "tl": [
                    4807,
                    3865,
                    315.5
                ],
                "tr": [
                    4807,
                    3985,
                    315.5
                ],
                "br": [
                    4807,
                    3985,
                    435.5
                ],
                "bl": [
                    4807,
                    3865,
                    435.5
                ]
            },
            "5": {
                "tl": [
                    4807,
                    2550,
                    315.5
                ],
                "tr": [
                    4807,
                    2670,
                    315.5
                ],
                "br": [
                    4807,
                    2670,
                    435.5
                ],
                "bl": [
                    4807,
                    2550,
                    435.5
                ]
            },
            "6": {
                "tl": [
                    4807,
                    1093,
                    315.5
                ],
                "tr": [
                    4807,
                    1213,
                    315.5
                ],
                "br": [
                    4807,
                    1213,
                    435.5
                ],
                "bl": [
                    4807,
                    1093,
                    435.5
                ]
            }
        },
        "goal": [
            1.5,
            0.5
        ],
        "reach_tol_m": 0.10,
        "kv": 0.6,
        "ktheta": 1.2,
        "sync": {
            "max_skew_ms": 80,
            "timeout_ms": 300
        },
        "display": {
            "window_width": 960
        }
    }
    
def load_calib(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    if "mtx" not in data or "dist" not in data:
        raise KeyError("npz 中必须包含 'mtx' 和 'dist'")
    mtx = data["mtx"].astype(np.float32)
    dist = data["dist"].astype(np.float32).reshape(-1)  # 展平为(5,)等
    return mtx, dist