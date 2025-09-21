import serial
import serial.tools.list_ports
import threading
import time

class SerialTestTool:
    def __init__(self, port, baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self.running = False

    def start(self):
        try:
            self.ser = serial.Serial(
                self.port, 
                self.baudrate, 
                timeout=1
            )
            print(f"打开端口 {self.port} 成功")
            self.running = True

            # 开线程接收
            threading.Thread(target=self._receive_data, daemon=True).start()
            # 开线程发送
            threading.Thread(target=self._send_data, daemon=True).start()
        except Exception as e:
            print(f"打开端口失败: {e}")

    def _receive_data(self):
        while self.running:
            try:
                if self.ser.in_waiting > 0:
                    data = self.ser.readline().decode("utf-8", errors="ignore").strip()
                    if data:
                        print(f"收到: {data}")
                        # 收到任意消息后回复 $AOK#
                        self.ser.write("$AOK#\n".encode("utf-8"))
                        print("已回复: $AOK#")
            except Exception as e:
                print(f"接收出错: {e}")
            time.sleep(0.1)

    def _send_data(self):
        while self.running:
            try:
                msg = input("请输入要发送的数据 (quit 退出): ")
                if msg.lower() == "quit":
                    self.stop()
                    break
                self.ser.write((msg + "\n").encode("utf-8"))
                print(f"已发送: {msg}")
            except Exception as e:
                print(f"发送出错: {e}")

    def stop(self):
        self.running = False
        if hasattr(self, "ser"):
            self.ser.close()
        print("关闭端口")

def main():
    port = "/dev/pts/4"   # 下位机端口
    tool = SerialTestTool(port)
    tool.start()
    while tool.running:
        time.sleep(0.5)

if __name__ == "__main__":
    main()