import logging
import time
import cv2
import math
import requests
import urllib.parse
import numpy as np

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

LINE_ACCESS_TOKEN="your_token"
url = "https://notify-api.line.me/api/notify"
message = "Alert! Fall detected!!"

msg = urllib.parse.urlencode({"message":message})
LINE_HEADERS = {'Content-Type':'application/x-www-form-urlencoded',"Authorization":"Bearer "+LINE_ACCESS_TOKEN}

camera = '0'
# camera = 'test/7.mp4'
size_h = 320
size_w = 480
mhi_duration = 1500  # ms
GAUSSIAN_KERNEL = (3, 3)
fps_time = 0
j = 0

cam = cv2.VideoCapture(camera)

rec, frame = cam.read()

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, GAUSSIAN_KERNEL)
fgbg = cv2.createBackgroundSubtractorMOG2(500, 64, False)

fps = int(cam.get(cv2.CAP_PROP_FPS))
interval = int(max(1, math.ceil(fps / 10) if (fps / 10 - math.floor(fps / 10)) >= 0.5 else math.floor(fps / 10)))
ms_per_frame = 1000 / fps  # ms

count = interval

prev_mhi = [np.zeros((size_w, size_h), np.float32) for i in range(interval)]
prev_timestamp = [i * ms_per_frame for i in range(interval)]
prev_frames = [None] * interval

for i in range(interval):
    rec1, frame1 = cam.read()
    frame1 = cv2.resize(frame1, (size_h, size_w), interpolation=cv2.INTER_AREA)
    frame1 = cv2.GaussianBlur(frame1, GAUSSIAN_KERNEL, 0.5)
    prev_frames[i] = frame1.copy()

if __name__ == '__main__':
    model = 'mobilenet_thin'
    resize = '272x192'
    resize_out_ratio = 3.0

    w, h = model_wh(resize)
    if w > 0 and h > 0:
        e = TfPoseEstimator(get_graph_path(model), target_size=(w,h))
    else:
        e = TfPoseEstimator(get_graph_path(model), target_size=(432,368))

    frame_width = int(cam.get(3))
    frame_height = int(cam.get(4))

    ret_val, image = cam.read()

    while True:
        ret_val, image = cam.read()

        count = 0
        y1 = [0, 0]
        frame = 0

        if not ret_val: break

        frame2 = cv2.resize(image, (size_h, size_w), interpolation=cv2.INTER_AREA)

        gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (21, 21), 0.5)

        prev_ind = count % interval
        prev_timestamp[prev_ind] += interval * ms_per_frame
        count += 1

        frame_diff = cv2.absdiff(frame2, prev_frames[prev_ind])
        gray_diff = cv2.cvtColor(frame_diff, cv2.COLOR_BGR2GRAY)
        blur_diff = cv2.GaussianBlur(gray_diff, (21, 21), 0.5)

        fgmask = fgbg.apply(blur_diff)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

        thresh = cv2.adaptiveThreshold(fgmask, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 17, 2)

        _, motion_mask = cv2.threshold(gray_diff, 32,  # THRESHOLD
                                       1, cv2.THRESH_BINARY)

        cv2.motempl.updateMotionHistory(motion_mask, prev_mhi[prev_ind], prev_timestamp[prev_ind], mhi_duration)
        mhi = np.uint8(
            np.clip((prev_mhi[prev_ind] - (prev_timestamp[prev_ind] - mhi_duration)) / mhi_duration, 0, 1) * 255)
        # print(mhi)
        prev_frames[prev_ind] = frame2.copy()

        sum_mhi = np.sum(mhi)
        sum_fgmask = np.sum(fgmask)
        c_motion = (sum_fgmask / sum_mhi) * 100

        humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=resize_out_ratio)
        image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
        no_people = len(humans)

        # print('no. of people: ', no_people)

        if c_motion < 65:
            continue
        elif c_motion >= 65:
            for human in humans:
                for i in range(len(humans)):
                    try:
                        a = human.body_parts[0]
                        x = a.x*image.shape[1]
                        y = a.y*image.shape[0]
                        y1.append(y)
                    except:
                        pass
                    if ((y - y1[-2]) > 25):
                        cv2.putText(image, "Fall detected", (20,50), cv2.FONT_HERSHEY_COMPLEX, 2.5, (0,0,255), 2, 6)
                        print("Total no. of people:", no_people)
                        print("Fall detected.", i + 1, count)
                        session = requests.Session()
                        a = session.post(url, headers=LINE_HEADERS, data=msg)

        cv2.putText(image, "People: %d" % (no_people), (10,50), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,255,0), 2)
        cv2.putText(image, "FPS: %f" % (1.0 / (time.time() - fps_time)), (10,10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,255,0), 2)

        cv2.imshow('Result', image)
        fps_time = time.time()

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    cam.release()
