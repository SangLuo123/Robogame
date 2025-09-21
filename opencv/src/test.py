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
        
        count = 1  # 发送计数器
        last_send_time = time.time()  # 上次发送时间
        
        # 缓冲区用于累积接收的数据
        receive_buffer = bytearray()
        
        print("开始运行，按 Ctrl+C 停止...")
        
        while True:
            # 检查是否到达发送时间
            current_time = time.time()
            if current_time - last_send_time >= 5:
                # 发送带计数器的消息
                data = f"$HELLO{count}#".encode()
                print(f"Sending ({count}): {data}")
                ser.write(data)
                ser.flush()
                count += 1
                last_send_time = current_time
            
            # 尝试读取接收到的数据
            chunk = ser.read(ser.in_waiting or 1)
            if chunk:
                receive_buffer.extend(chunk)
                
                # 检查缓冲区中是否有完整的消息（以#结尾）
                if b'#' in receive_buffer:
                    # 找到第一个#的位置
                    hash_index = receive_buffer.find(b'#')
                    if hash_index >= 0:
                        # 提取完整消息（包括#）
                        complete_message = receive_buffer[:hash_index + 1]
                        print("Received complete message:", complete_message)
                        
                        # 移除已处理的消息
                        receive_buffer = receive_buffer[hash_index + 1:]
            
            # 短暂休眠以避免CPU占用过高
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print("Error:", e)
    finally:
        try:
            ser.close()
            print("串口已关闭")
        except:
            pass

if __name__ == "__main__":
    list_ports()
    test_send()

    """
    $AOK#
    $AERR#
    """