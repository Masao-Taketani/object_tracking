from centroid_tracker import CentroidTracker
import imutils
from imutils.video import VideoStream
import numpy as np
import argparse
import time
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", default="deploy.prototxt", \
				help="path to Caffe prototxt file")
ap.add_argument("-m", "--model", default="res10_300x300_ssd_iter_140000.caffemodel",\
				 help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,\
				help="detection threshold")
args = vars(ap.parse_args())

cent_tracker = CentroidTracker()
(H, W) = (None, None)

print("loading the model")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

print("start video streaming")
vs = VideoStream(src=0).start()
# to warm up the camera, wait 2.0 sec
time.sleep(2.0)

while True:
	frame = vs.read()
	frame = imutils.resize(frame, width=400)

	# this statement is used for the initializer step
	if W is None or H is None:
		(H, W) = frame.shape[:2]

	# preprocess each frame to input it into DNN model
	# mean is used for mean subtraction
	blob = cv2.dnn.blobFromImage(frame, scalefactor=1.0, size=(H, W),\
								 mean=(104.0, 177.0, 123.0))
	net.setInput(blob)
	detections = net.forward()
	rects = []

	for i in range(detections.shape[2]):
		if detections[0, 0, i, 2] > args["confidence"]:
			box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
			rects.append(box.astype("int"))

			# draw bbox
			(start_x, start_y, end_x, end_y) = box.astype("int")
			cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), \
						 (255, 0, 0), 2)

	objs = cent_tracker.update_centroid(rects)

	for (obj_id, centroid) in objs.items():
		# draw the id and the centroid for each object
		text = "Person {}".format(obj_id)
		# to draw id
		cv2.putText(frame, text, (centroid[0] - 50, centroid[1] - 80),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
		# to draw centroid
		#cv2.circle(frame, (centroid[0], centroid[1]), 4, (255, 0, 0), -1)

	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	if key == ord("q"):
		break

cv2.destroyAllWindows()
vs.stop()