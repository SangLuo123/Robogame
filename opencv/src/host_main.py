from comm import SerialLink
import time, math

link = SerialLink("/dev/ttlUSB0", 115200)
link.open()

# 可选回调
link.on_ack = lambda ok, f: print("ACK:", "OK" if ok else f)
link.on_enc = lambda enc: print("ENC:", enc)

try:
    t0 = time.time()
    while True:
        t = time.time() - t0
        v = 0.25
        w = 0.5 * math.sin(2*math.pi*0.3*t)
        link.send_vel(v, w)                    # 真实控制
        if int(t*5) % 5 == 0:
            link.query_encoders()              # 需要时查询编码器
        link.heartbeat(0.2, mode="last")       # 兜底保活（无独立线程）
        time.sleep(0.05)                       # 20Hz
except KeyboardInterrupt:
    pass
finally:
    link.send_stop()
    link.close()
