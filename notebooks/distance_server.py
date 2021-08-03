import socket               # 导入 socket 模块
import numpy as np

s = socket.socket()         # 创建 socket 对象
host = socket.gethostname() # 获取本地主机名
port = 12345                # 设置端口
s.bind((host, port))        # 绑定端口

s.listen(5)                 # 等待客户端连接
while True:
    c, addr = s.accept()     # 建立客户端连接
    data = c.recv(49152)  #接收数据
    query = np.fromstring(data, dtype=np.float32)
    query = query.reshape(2,-1)
    x1 = query[0]
    x2 = query[1]
    ans = np.linalg.norm(x1-x2).astype(dtype=np.float64)
    c.send(ans.tostring()) #然后再发送数据
    c.close()
