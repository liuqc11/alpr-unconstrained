# -*- coding: utf-8 -*-
import sys
import cv2
import numpy as np
import traceback

import darknet.python.darknet as dn

from src.label 				import Label, lwrite
from os.path 				import splitext, basename, isdir
from os 					import makedirs
from src.utils 				import crop_region
from src.drawing_utils			import draw_label
from darknet.python.darknet import detect_frame, nparray_to_image
YELLOW = (  0,255,255)


if __name__ == '__main__':
	try:
		input_file = sys.argv[1]
		output_file = sys.argv[2]

		vehicle_threshold = .5

		vehicle_weights = b'data/vehicle-detector/yolo-voc.weights'
		vehicle_netcfg = b'data/vehicle-detector/yolo-voc.cfg'
		vehicle_dataset = b'data/vehicle-detector/voc.data'

		vehicle_net = dn.load_net(vehicle_netcfg, vehicle_weights, 0)
		vehicle_meta = dn.load_meta(vehicle_dataset)
		vid = cv2.VideoCapture(input_file)
		fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', '2')
		videoWriter = cv2.VideoWriter(output_file, fourcc, 25, (1920, 1080))
		print('Searching for vehicles and licenses using YOLO and Keras...')
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
					#Icar = crop_region(arr,label)
					Lcars.append(label)
				for i, lcar in enumerate(Lcars):
					draw_label(arr, lcar, color=YELLOW, thickness=3)
			videoWriter.write(arr)

	except:
		traceback.print_exc()
		sys.exit(1)

	sys.exit(0)
