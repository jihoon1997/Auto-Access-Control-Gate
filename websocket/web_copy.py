import asyncio
import websockets
import datetime
import re
import time
import os


async def time(websocket, path):
    while True: 
        f = open('/home/geunwoo/yolov53/yolov531/yolov5/Temp.txt', 'rt')
        Temp = f.read()
        f.close()
        f = open('/home/geunwoo/yolov53/yolov531/yolov5/Mask.txt', 'rt')
        Mask = f.read()
        f.close()
        
        if len(Mask) == 0 :
            Mask = "Not-Detected"
        if len(Temp) == 0 :
            Temp = "36.5"
            
        Data = Mask + "," + Temp

        print(Data)
        print(Data)
        await websocket.send(Data)
        await asyncio.sleep(0.005)

start_server = websockets.serve(time, "192.168.0.69", 5500)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
