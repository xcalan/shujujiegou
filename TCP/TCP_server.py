# coding:utf-8
# TCP 服务端

from socket import *

serverSocket = socket(AF_INET, SOCK_STREAM) # TCP 套接字
serverSocket.bind(("", 8899)) # 服务端本地ip不限，端口号绑定
serverSocket.listen(5) # 将主动套接字变为被动套接字（挂起连接队列的最大长度，提前准备5个挂起的连接）

"""堵塞，等待客户端连接"""
clientSocket, clientInfo = serverSocket.accept()
# clientSocket 表示这个新的客户端
# clientInfo 表示这个新的客户端的ip及port

"""堵塞，等待客户端发送数据"""
recvData = clientSocket.recv(1024) # 从新的客户端一次接收数据的长度

print("%s：%s" %(str(clientInfo), recvData))

clientSocket.close()
serverSocket.close()
