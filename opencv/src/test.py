# debug_serial.py
# 需要 pip install pyserial
import time
import serial
import serial.tools.list_ports

PORT = "/dev/ttyUSB0"   # <- 改成你在设备管理器看到的 COM 号
BAUD = 115200   # <- 改成你的波特率

def list_ports():
    print("可用串口：")
    for p in serial.tools.list_ports.comports():
        print(f"  {p.device} - {p.description} - {p.hwid}")

def test_send():
    try:
        ser = serial.Serial(PORT, BAUD, timeout=1)
        print("Opened:", ser.name)
        # 立即禁用 DTR（防止复位）
        try:
            ser.setDTR(False)
            ser.setRTS(False)
            print("DTR/RTS set False")
        except Exception as e:
            print("setDTR/RTS failed:", e)

        time.sleep(0.1)

        # 发送一个简单的字节序列（修改为下位机期望的数据）
        data = b'HELLO\n'   # 如果下位机期待二进制，请改为相应字节
        print("Sending:", data)
        ser.write(data)
        ser.flush()
        print("Flushed")

        # 等待并尝试读取回包（如果下位机会回 ACK）
        t0 = time.time()
        resp = b''
        while time.time() - t0 < 15:
            chunk = ser.read(ser.in_waiting or 1)
            if chunk:
                resp += chunk
            else:
                time.sleep(0.05)
        print("Received:", resp)
        ser.close()
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    list_ports()
    test_send()
