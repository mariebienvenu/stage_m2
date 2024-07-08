
import os
import numpy as np
import cv2

from app.video import Video


directory = "C:/Users/Marie Bienvenu/stage_m2/sprite sheets/"
sheet_name = "fireworks"
assert os.path.exists(f"{directory}/{sheet_name}.jpg"), f"Sprite sheet not found : {directory}/{sheet_name}.jpg does not exist."

n_lines = 3
n_frames = 10
size = 144

def load(line:int, frame:int):
    arr = cv2.imread(f"{directory}/{sheet_name}_{line}/{frame}.jpg")
    return arr

def make_video(line:int, target_fps=60, sprite_fps=10):
    copy = target_fps//sprite_fps
    frames = np.zeros((n_frames*copy, size, size, 3))
    for i in range(n_frames):
        arr = load(line, i+1)
        for j in range(copy):
            frames[i*copy+j, :,:] = arr

    video = Video.from_array(frames, f"{directory}/{sheet_name}_{line}.mp4", target_fps, verbose=3)
    return video

for line in range(1, n_lines+1):
    vid = make_video(line)
    #vid.play_frame_by_frame()


def make_correct_spritesheet(line:int):
    result = np.zeros((size, size*n_frames, 3), dtype=np.uint8)
    for i in range(n_frames):
        arr = load(line, i+1)
        result[:, i*size:(i+1)*size,:] = np.copy(arr)
    cv2.imwrite(f"{directory}/{sheet_name}_{line}.jpg", result)
    return result


for line in range(1, n_lines+1):
    im = make_correct_spritesheet(line)
    cv2.imshow("win", im)
    cv2.waitKey(0)