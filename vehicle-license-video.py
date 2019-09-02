# -*- coding: utf-8 -*-
import sys
import traceback

import cv2
import numpy as np

# import darknet.python.darknet as dn
# from darknet.python.darknet import detect_frame, nparray_to_image
import darknetAB.darknet as dn
from darknetAB.darknet import detect_image
from src.drawing_utils import draw_label, draw_losangle, write2img
from src.keras_ocr_utils import LPR
from src.keras_utils import load_model, detect_lp
from src.label import Label
from src.utils import crop_region

YELLOW = (0, 255, 255)
RED = (0, 0, 255)

if __name__ == '__main__':
    try:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
        # vehicle detection model
        vehicle_threshold = .5

        vehicle_weights = b'darknetAB/yolov3.weights'
        vehicle_netcfg = b'darknetAB/cfg/yolov3.cfg'
        vehicle_dataset = b'darknetAB/cfg/coco.data'

        vehicle_net = dn.load_net_custom(vehicle_netcfg, vehicle_weights, 0, 1)  # batchsize=1
        vehicle_meta = dn.load_meta(vehicle_dataset)

        # license plate detection model
        wpod_net = load_model('data/lp-detector/wpod-net_update1.h5')

        # license plate recognition model
        ocrmodel = LPR("data/ocr-model/ocr_plate_all_gru.h5")

        # Create an image we reuse for each detect
        darknet_image = dn.make_image(dn.network_width(vehicle_net),
                                      dn.network_height(vehicle_net), 3)
        vid = cv2.VideoCapture(input_file)
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        videoWriter = cv2.VideoWriter(output_file, fourcc, 30, (960, 544))
        print('Searching for vehicles and licenses using YOLO and Keras...')
        frame = 1
        while True:
            return_value, arr = vid.read()
            if not return_value:
                break
            frame_rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb,
                                       (dn.network_width(vehicle_net),
                                        dn.network_height(vehicle_net)),
                                       interpolation=cv2.INTER_LINEAR)

            dn.copy_image_from_bytes(darknet_image, frame_resized.tobytes())
            # im = nparray_to_image(arr)
            R = detect_image(vehicle_net, vehicle_meta, darknet_image, thresh=vehicle_threshold)
            R = [r for r in R if r[0].decode('utf-8') in ['car', 'bus', 'truck']]
            if len(R):
                WH = np.array(arr.shape[1::-1], dtype=float)
                Lcars = []
                # Icars = []
                for i, r in enumerate(R):

                    cx, cy, w, h = (np.array(r[2]) / np.concatenate((WH, WH))).tolist()
                    tl = np.array([cx - w / 2., cy - h / 2.])
                    br = np.array([cx + w / 2., cy + h / 2.])
                    label = Label(0, tl, br)
                    Lcars.append(label)
                    Icar = crop_region(arr, label)
                    # Icars.append(Icar)
                    # print('Searching for license plates using WPOD-NET')
                    ratio = float(max(Icar.shape[:2])) / min(Icar.shape[:2])
                    side = int(ratio * 288.)
                    bound_dim = min(side + (side % (2 ** 4)), 608)
                    # print("\t\tBound dim: %d, ratio: %f" % (bound_dim, ratio))
                    Llp, LlpImgs, _ = detect_lp(wpod_net, Icar / 255, bound_dim, 2 ** 4, (240, 80),
                                                0.5)
                    if len(LlpImgs):
                        Ilp = LlpImgs[0]
                        # Ilp = cv2.cvtColor(Ilp, cv2.COLOR_BGR2GRAY)
                        # Ilp = cv2.cvtColor(Ilp, cv2.COLOR_GRAY2BGR)
                        # s = Shape(Llp[0].pts)
                        res, confidence = ocrmodel.recognizeOneframe(Ilp * 255.)

                        pts = Llp[0].pts * label.wh().reshape(2, 1) + label.tl().reshape(2, 1)
                        ptspx = pts * np.array(arr.shape[1::-1], dtype=float).reshape(2, 1)
                        draw_losangle(arr, ptspx, RED, 3)
                        if confidence > 0.5:
                            llp = Label(0, tl=pts.min(1), br=pts.max(1))
                            arr = write2img(arr, llp, res)
                for i, lcar in enumerate(Lcars):
                    draw_label(arr, lcar, color=YELLOW, thickness=3)
            videoWriter.write(arr)
            print('finish writing %d frame!' % frame)
            frame = frame + 1
        # videoWriter.release()
    except:
        traceback.print_exc()
        sys.exit(1)

    sys.exit(0)
