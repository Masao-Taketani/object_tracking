from centroid_tracker import CentroidTracker
from trackable_object import TrackableObject
import imutils
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import time
import dlib
import cv2


ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", default="mobilenet_ssd_model/",\
	help="path to Caffe prototxt file")
ap.add_argument("-m", "--model", default="mobilenet_ssd_model/",\
	help="path to Caffe pre-trained model")
ap.add_argument("-i", "--input", type=str,\
	help="path to optional input video file. If None, it will start\
	 video streaming")
ap.add_argument("-o", "--output", type=str,\
	help="path to optional output video file")
ap.add_argument("-c", "--confidence", type=float, default=0.4,\
	help="threshold to eliminate detections")
ap.add_argument("-s", "--skip-frames", type=int, default=30,
	help="# of skip frames between detections")
# vars: return values in a dict format
args = vars(ap.parse_args())

# MobileNet SSD Classes
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
"sofa", "train", "tvmonitor"]

print("load the model")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

############### I don't get this
if not args.get("input", False):
#####################################
	print("starting video stream")
	vs = VideoStream(src=0).start()
	# webcam warmup time
	time.sleep(2.0)
else:
	print("opening the specified video file")
	vs = cv2.VideoCapture(args["input"])

writer = None
W = None
H = None

cent_tracker = CentroidTracker(max_disappeared=40, max_distance=50)
trackers = []
trackable_objs = {}

total_frames = 0
total_downs = 0
total_ups = 0

fps = FPS().start()

while True:
	frame = vs.read()
	frame = frame[1] if args.get("input", False) else frame

	# if the optinal input is set and the obtained frame is None,
	# it means the frame reached at the end of the video
	if args["input"] is not None and frame is None:
		break

	# resize frame by keeping the ratio
	frame = imutils.resize(frame, width=500)
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

	# this is done for the initializer step
	if W is None or H is None:
		(H, W) = frame.shape[:2]

	# to write output video file
	if args["output"] is not None and writer is None:
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 30,\
			(W, H), True)
