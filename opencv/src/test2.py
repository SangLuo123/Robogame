# debug_serial_enhanced.py
import time
import serial
import serial.tools.list_ports
import platform

def get_default_port():
    if platform.system() == "Windows":
        return "COM3"  # 修改为你的实际端口
    else:
        return "/dev/ttyACM0"  # Linux/macOS

PORT = get_default_port()
BAUD = 115200

def list_ports():
    print("=== 可用串口列表 ===")
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        print("未找到可用串口！")
        return
    
    for i, p in enumerate(ports):
        print(f"{i+1}. {p.device} - {p.description} - {p.hwid}")

def test_communication():
    try:
        print(f"\n=== 尝试打开端口: {PORT} ===")
        ser = serial.Serial(
            port=PORT,
            baudrate=BAUD,
            timeout=1,
            write_timeout=1,
            rtscts=False,    # 禁用硬件流控
            dsrdtr=False     # 禁用硬件流控
        )
        
        print(f"✓ 端口打开成功: {ser.name}")
        print(f"✓ 波特率: {ser.baudrate}")
        print(f"✓ 超时设置: {ser.timeout}")
        
        # 尝试不同的DTR/RTS组合
        print("\n=== 测试DTR/RTS设置 ===")
        combinations = [
            (False, False),
            (True, False), 
            (False, True),
            (True, True)
        ]
        
        for dtr, rts in combinations:
            try:
                ser.setDTR(dtr)
                ser.setRTS(rts)
                print(f"尝试 DTR={dtr}, RTS={rts}")
                time.sleep(0.5)  # 等待设备响应
                
                # 清空输入缓冲区
                if ser.in_waiting > 0:
                    data = ser.read(ser.in_waiting)
                    print(f"  清空缓冲区: {len(data)} 字节")
                
                # 发送测试命令
                test_cmd = b"AT\r\n"  # 通用测试命令
                print(f"  发送: {test_cmd}")
                ser.write(test_cmd)
                ser.flush()
                
                # 等待并读取响应
                time.sleep(1)
                if ser.in_waiting > 0:
                    response = ser.read(ser.in_waiting)
                    print(f"  ✓ 收到响应: {response}")
                    if response:
                        break  # 收到数据，停止测试
                else:
                    print("  ✗ 无响应")
                    
            except Exception as e:
                print(f"  设置失败: {e}")
        
        # 正式通信测试
        print("\n=== 开始正式通信测试 ===")
        count = 0
        start_time = time.time()
        
        while time.time() - start_time < 30:  # 测试30秒
            # 发送数据
            if count % 10 == 0:  # 每10次循环发送一次
                send_data = f"PING{count}\r\n".encode()
                print(f"发送: {send_data}")
                ser.write(send_data)
                ser.flush()
            
            # 检查接收
            if ser.in_waiting > 0:
                received = ser.read(ser.in_waiting)
                print(f"✓ 收到数据: {received} (长度: {len(received)} 字节)")
            
            # 尝试多种请求方式
            if count == 5:
                print("尝试发送查询命令...")
                queries = [
                    b"\r\n",           # 空行
                    b"?\r\n",          # 查询
                    b"STATUS\r\n",     # 状态查询
                    b"DATA\r\n",       # 数据请求
                ]
                for query in queries:
                    print(f"发送查询: {query}")
                    ser.write(query)
                    ser.flush()
                    time.sleep(0.5)
                    if ser.in_waiting > 0:
                        response = ser.read(ser.in_waiting)
                        print(f"  响应: {response}")
            
            count += 1
            time.sleep(0.1)
            
        print("\n=== 测试完成 ===")
        
    except serial.SerialException as e:
        print(f"✗ 串口错误: {e}")
    except KeyboardInterrupt:
        print("\n用户中断")
    except Exception as e:
        print(f"✗ 错误: {e}")
    finally:
        try:
            ser.close()
            print("端口已关闭")
        except:
            pass

if __name__ == "__main__":
    list_ports()
    print(f"\n使用端口: {PORT}, 波特率: {BAUD}")
    input("按回车键开始测试...")
    test_communication()