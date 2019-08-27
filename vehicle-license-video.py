# -*- coding: utf-8 -*-
import sys
import cv2
import numpy as np
import traceback

import darknet.python.darknet as dn

from src.label 				import Label, Shape
from os.path 				import splitext, basename, isdir
from os 					import makedirs
from src.utils 				import crop_region,im2single
from src.keras_utils 			import load_model, detect_lp
from src.drawing_utils			import draw_label, draw_losangle, write2img
from darknet.python.darknet import detect_frame, nparray_to_image
from src.keras_ocr_utils import LPR

YELLOW = (  0,255,255)
RED    = (  0,  0,255)


if __name__ == '__main__':
    try:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
        # vehicle detection model
        vehicle_threshold = .5

        vehicle_weights = b'data/vehicle-detector/yolo-voc.weights'
        vehicle_netcfg = b'data/vehicle-detector/yolo-voc.cfg'
        vehicle_dataset = b'data/vehicle-detector/voc.data'

        vehicle_net = dn.load_net(vehicle_netcfg, vehicle_weights, 0)
        vehicle_meta = dn.load_meta(vehicle_dataset)

        # license plate detection model
        wpod_net = load_model('data/lp-detector/wpod-net_update1.h5')

        # license plate recognition model
        ocrmodel = LPR("data/ocr-model/ocr_plate_all_gru.h5")

        vid = cv2.VideoCapture(input_file)
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        videoWriter = cv2.VideoWriter(output_file, fourcc, 25, (1280, 720))
        print('Searching for vehicles and licenses using YOLO and Keras...')
        frame = 1
        while True:
            return_value, arr = vid.read()
            if not return_value:
                break
            im = nparray_to_image(arr)
            R, _ = detect_frame(vehicle_net, vehicle_meta, im, thresh=vehicle_threshold)
            R = [r for r in R if r[0].decode('utf-8') in ['car', 'bus']]
            if len(R):
                WH = np.array(arr.shape[1::-1], dtype=float)
                Lcars = []
                for i,r in enumerate(R):

                    cx,cy,w,h = (np.array(r[2])/np.concatenate( (WH,WH) )).tolist()
                    tl = np.array([cx - w/2., cy - h/2.])
                    br = np.array([cx + w/2., cy + h/2.])
                    label = Label(0,tl,br)
                    Lcars.append(label)
                    Icar = crop_region(arr,label)
                    # print('Searching for license plates using WPOD-NET')
                    ratio = float(max(Icar.shape[:2])) / min(Icar.shape[:2])
                    side = int(ratio * 288.)
                    bound_dim = min(side + (side % (2 ** 4)), 608)
                    # print("\t\tBound dim: %d, ratio: %f" % (bound_dim, ratio))
                    Llp, LlpImgs, _ = detect_lp(wpod_net, Icar/255, bound_dim, 2 ** 4, (240, 80),
                                                0.5)
                    if len(LlpImgs):
                        Ilp = LlpImgs[0]
                        # Ilp = cv2.cvtColor(Ilp, cv2.COLOR_BGR2GRAY)
                        # Ilp = cv2.cvtColor(Ilp, cv2.COLOR_GRAY2BGR)
                        # s = Shape(Llp[0].pts)
                        res, confidence = ocrmodel.recognizeOneframe(Ilp*255.)

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
            frame = frame+1
    except:
        traceback.print_exc()
        sys.exit(1)

    sys.exit(0)
