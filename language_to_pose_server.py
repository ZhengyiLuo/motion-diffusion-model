#!/usr/bin/env python3
import glob
import os
import sys
import pdb
import os.path as osp
sys.path.append(os.getcwd())

import cv2
import joblib
import numpy as np
import time


import asyncio
import cv2
import numpy as np
import threading
from scipy.spatial.transform import Rotation as sRot

import time
import torch
from collections import deque
from datetime import datetime
from torchvision import transforms as T
import time

from aiohttp import web
import aiohttp
import jinja2
import json
import scipy.interpolate as interpolate
import subprocess
from io import StringIO
from mdm_talker import MDMTalker

STANDING_POSE = np.array([[[-0.1443, -0.9426, -0.2548],
         [-0.2070, -0.8571, -0.2571],
         [-0.0800, -0.8503, -0.2675],
         [-0.1555, -1.0663, -0.3057],
         [-0.2639, -0.5003, -0.2846],
         [-0.0345, -0.4931, -0.3108],
         [-0.1587, -1.2094, -0.2755],
         [-0.2534, -0.1022, -0.3361],
         [-0.0699, -0.1012, -0.3517],
         [-0.1548, -1.2679, -0.2675],
         [-0.2959, -0.0627, -0.2105],
         [-0.0213, -0.0424, -0.2277],
         [-0.1408, -1.4894, -0.2892],
         [-0.2271, -1.3865, -0.2622],
         [-0.0715, -1.3832, -0.2977],
         [-0.1428, -1.5753, -0.2303],
         [-0.3643, -1.3792, -0.2646],
         [ 0.0509, -1.3730, -0.3271],
         [-0.3861, -1.1423, -0.3032],
         [ 0.0634, -1.1300, -0.3714],
         [-0.4086, -0.9130, -0.2000],
         [ 0.1203, -0.8943, -0.3002],
         [-0.4000, -0.8282, -0.1817],
         [ 0.1207, -0.8087, -0.2787]]]).repeat(5, axis = 0)

def fps_20_to_30(mdm_jts):
    jts = []
    N = mdm_jts.shape[0]
    for i in range(24):
        int_x = mdm_jts[:, i, 0]
        int_y = mdm_jts[:, i, 1]
        int_z = mdm_jts[:, i, 2]
        x = np.arange(0, N)
        f_x = interpolate.interp1d(x, int_x)
        f_y = interpolate.interp1d(x, int_y)
        f_z = interpolate.interp1d(x, int_z)
        
        new_x = f_x(np.linspace(0, N-1, int(N * 1.5)))
        new_y = f_y(np.linspace(0, N-1, int(N * 1.5)))
        new_z = f_z(np.linspace(0, N-1, int(N * 1.5)))
        jts.append(np.stack([new_x, new_y, new_z], axis = 1))
    jts = np.stack(jts, axis = 1)
    return jts

 
    
async def websocket_handler(request):
    print('Websocket connection starting')
    global pose_mat, trans, dt, sim_talker, ws_talkers
    sim_talker = aiohttp.web.WebSocketResponse()
    ws_talkers.append(sim_talker)
    await sim_talker.prepare(request)
    print('Websocket connection ready')

    async for msg in sim_talker:
        if msg.type == aiohttp.WSMsgType.TEXT:
            if msg.data == "get_pose":
                await sim_talker.send_json({
                    "pose_mat": pose_mat.tolist(),
                    "trans": trans.tolist(),
                    "dt": dt,
                })

    print('Websocket connection closed')
    return sim_talker 

async def pose_getter(request):
    # query env configurations
    global pose_mat, trans, dt, j3d, tracking_res, ticker, reset_offset, reset_buffer, mdm_motions, cycle_motion
    curr_paths = {}
    
    if reset_offset:
        offset = - offset_height - mdm_motions[0, 0, 1]
        mdm_motions[..., 1] += offset
        reset_offset = False
        
    if reset_buffer:
        if buffer > 0:
            mdm_motions = np.concatenate([np.repeat(mdm_motions[0:1], buffer, axis = 0), mdm_motions])
        else:
            mdm_motions = mdm_motions[-buffer:]
            
        reset_buffer = False
    
    if cycle_motion:
        if ticker > len(mdm_motions) - 1 :
            ticker = 0
            mdm_motions[..., [0, 2]] -= mdm_motions[:1, :1, [0, 2]] -  mdm_motions[-1:, -1:, [0, 2]]
            
            
        j3d_curr = mdm_motions[ticker]
        
        
    else:
        j3d_curr = mdm_motions[min(len(mdm_motions)-1, ticker)]
        
    j3d[0] = j3d_curr
    json_resp = {
        "j3d": j3d.tolist(),
        "dt": dt,
    }
    ticker += 1
        
    return web.json_response(json_resp)

def generate_text(prompts):
    global offset_height, mdm_talker, buffer, mdm_motions, ticker
    
    prompts = prompts.split("\n")
    num_prompt = len(prompts)
    gen_mdm_motions = mdm_talker.generate_motion(prompts)
    mat = sRot.from_euler('xyz', np.array([-np.pi / 2, 0, 0]), degrees=False).as_matrix()
    gen_mdm_motions = np.matmul(gen_mdm_motions, mat.dot(mat))
    
    offset = - offset_height - gen_mdm_motions[ 0:1, 0:1, 1]
    gen_mdm_motions[..., 1] += offset
    gen_mdm_motions[..., [0, 2]] -= gen_mdm_motions[:1, :1, [0, 2]] - mdm_motions[ticker:(ticker+1), :1, [0, 2]]
    
    
    mdm_motions = fps_20_to_30(gen_mdm_motions)
    ticker = 0

async def send_to_clients(post):
    global ws_talkers
    for ws_talker in ws_talkers:
        if not ws_talker is None:
            try:
                print(f"Sending to client: {post}")
                await ws_talker.send_str(post)
            except Exception as e:
                ws_talker.close()
                ws_talkers.remove(ws_talker)
def commandline_input():
    global trans, dt, reset_offset, offset_height, superfast, j3d, j2d, num_ppl, bbox, frame, fps
    
    while True:
        command = input('Type MDM Prompt: ')
        if command == 'exit':
            print('Exiting!')
            raise SystemExit(0)
        elif command == '':
            print('Empty Command!')
        # elif command.startswith(":r"):
        #     splits = command.split(":")
        #     if len(splits) > 1:
        #         offset_height = float(splits[-1])
        #     reset_offset = True
        # elif command.startswith("fps"):
        #     print(fps)
        else:
            generate_text(command)
        

def main(request):
    return {'name': 'Andrew'}

if __name__ == "__main__":
    print("Running PHC Demo")
   
    
    j3d, j2d, trans, dt, ws_talkers, reset_offset, offset_height, sim_talker, num_ppl = np.zeros([5, 24, 3]), None, np.zeros([3]), 1 / 10, [], True, 0.92,  None, 0
    cycle_motion, mdm_motions, ticker, to_metrabs, buffer, reset_buffer = True, np.zeros([120, 24, 3]), 0, sRot.from_euler('xyz', np.array([-np.pi / 2, 0, 0]), degrees=False).as_matrix(), 120, False
    
    mdm_talker = MDMTalker()
    
    j3d = STANDING_POSE.copy()
    mdm_motions[:] = STANDING_POSE[:1].copy()
    
    frame = None
    superfast = True
    app = web.Application(client_max_size=1024**2)
    app.router.add_route('GET', '/ws', websocket_handler)
    app.router.add_route('GET', '/get_pose', pose_getter)
    app.add_routes([web.get('/', main)])
    
    print("=================================================================")
    print("r: reset offset (use r:0.91), s: start recording, e: end recording, w: write video")
    print("=================================================================")
    threading.Thread(target=commandline_input, daemon=True).start()
    web.run_app(app, port=8080)
    
