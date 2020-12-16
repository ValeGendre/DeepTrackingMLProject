import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

bb_centers = []
frame_ids = []
input_path = 'Trackingnew.txt'
with open(input_path, 'r') as f:
    while True:
        ID = f.readline() # ID of the frame
        line = f.readline()
        if line is '':
            break
        cx, cy = line.split()
        frame_ids.append(ID.split()[0])
        bb_centers.append([int(cx), int(cy)])

frames = []

for ids, c in zip(frame_ids, bb_centers):
    frame = plt.imread(ids)
    cx, cy = c
    p1 = (cx - 25, cy - 25)
    p2 = (cx + 25, cy + 25)
    cv.rectangle(frame, p1, p2, (255,0,0), 2, 1)
    frames.append(frame)

out = cv.VideoWriter(f'Labeled_Tracking_Video1.mp4',cv.VideoWriter_fourcc(*'XVID'), 3, (250, 250))
for i in range(len(frames)):
    out.write(frames[i])
out.release()
