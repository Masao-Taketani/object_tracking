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
ap.add_argument("-p", "--prototxt", default="mobilenet_ssd_model/MobileNetSSD_deploy.prototxt",\
	help="path to Caffe prototxt file")
ap.add_argument("-m", "--model", default="mobilenet_ssd_model/MobileNetSSD_deploy.caffemodel",\
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

# dict.get(key, return value if the specified key doesn't exit)
# which means if "input" is None, it will return False
if not args.get("input", False):
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

cent_tracker = CentroidTracker(max_frames_to_disappear=40, max_distance=50)
trackers = []
trackable_objs_dict = {}

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
		# fourcc: specify the codec of the movie
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 30,\
			(W, H), True)

	status = "Waiting"
	rects = []

	# detection phase
	if total_frames % args["skip_frames"] == 0:
		status = "Detecting"
		trackers = []

		# blobFromImage(img, scalefactor, size, mean)
		blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
		net.setInput(blob)
		detections = net.forward()

		for i in np.arange(0, detections.shape[2]):
			confidence = detections[0, 0, i, 2]

			if confidence > args["confidence"]:
				idx = int(detections[0, 0, i, 1])

				if CLASSES[idx] != "person":
					continue

				bbox = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
				(start_x, start_y, end_x, end_y) = bbox.astype("int")

				rect = dlib.rectangle(start_x, start_y, end_x, end_y)
				tracker = dlib.correlation_tracker()
				tracker.start_track(rgb, rect)

				trackers.append(tracker)
	# tracking phase
	else:
		for tracker in trackers:
			status = "Tracking"

			# update the tracking rect
			tracker.update(rgb)
			pos = tracker.get_position()

			start_x = int(pos.left())
			start_y = int(pos.top())
			end_x = int(pos.right())
			end_y = int(pos.bottom())

			rects.append((start_x, start_y, end_x, end_y))

	cv2.line(frame, (0, H // 2), (W, H // 2), (0, 255, 255), 2)

	objs = cent_tracker.update_centroid(rects)

	for (obj_id, centroid) in objs.items():
		tra_objs = trackable_objs_dict.get(obj_id, None)
		if tra_objs is None:
			tra_objs = TrackableObject(obj_id, centroid)
		else:
			y = [c[1] for c in tra_objs.centroids]
			direction = centroid[1] - np.mean(y)
			tra_objs.centroids.append(centroid)

			if not tra_objs.counted:
				if direction < 0 and centroid[1] < H // 2:
					total_ups += 1
					tra_objs.counted = True
				elif direction > 0 and centroid[1] > H // 2:
					total_downs += 1
					tra_objs.counted = True

		trackable_objs_dict[obj_id] = tra_objs

		text = "ID {}".format(obj_id)
		cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
		cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

	info = [
		("Up", total_ups),
		("Down", total_downs),
		("Status", status)
	]

	for (i, (k, v)) in enumerate(info):
		text = "{}: {}".format(k, v)
		cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
			cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

	if writer is not None:
		writer.write(frame)

	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	if key == ord("q"):
		break

	total_frames += 1
	fps.update()

fps.stop()
print("elapsed time: {:.2f}".format(fps.elapsed()))
print("approx. FPS: {:.2f}".format(fps.fps()))

if writer is not None:
	writer.release()

if not args.get("input", False):
	vs.stop()
else:
	vs.release()

cv2.destroyAllWindows()
