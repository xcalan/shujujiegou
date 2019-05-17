# coding:utf-8
# TCP 客户端

from socket import *

clientSocket = socket(AF_INET, SOCK_STREAM)
clientSocket.connect(("192.168.119.153", 8989)) # 连接TCP服务端的IP和端口

clientSocket.send("haha".encode("gb2312"))

recvData = clientSocket.recv(1024)

print("recvData:%s" % recvData)

clientSocket.close()