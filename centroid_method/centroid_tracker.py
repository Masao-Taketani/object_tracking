from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np

class CentroidTracker():
	def __init__(self, max_frames_to_disappear=50):
		self.next_obj_id = 0
		self.objs_dict = OrderedDict()
		self.disappeared_frames_dict = OrderedDict()
		self.max_frames_to_disappear = max_frames_to_disappear

	def register_new_obj(self, centroid):
		self.objs_dict[self.next_obj_id] = centroid
		self.disappeared_frames_dict[self.next_obj_id] = 0
		self.next_obj_id += 1

	def deregister_obj(self, obj_id):
		del self.objs_dict[obj_id]
		del self.disappeared_frames_dict[obj_id]

	def update_centroid(self, rects):
		# rects: every set of calculated rects for each frame
		if len(rects) == 0:
			for obj_id in list(self.disappeared_frames_dict.keys()):
				self.disappeared_frames_dict[obj_id] += 1

				if self.disappeared_frames_dict[obj_id] > self.max_frames_to_disappear:
					self.deregister_obj(obj_id)

			return self.objs_dict

		# 2 for col means (centroid_x, centroid_y) 
		input_centroids = np.zeros((len(rects), 2), dtype="int")

		# calculate centroids from the bboxes
		for (i, (start_x, start_y, end_x, end_y)) in enumerate(rects):
			centroid_x = int((start_x + end_x) / 2.0)
			centroid_y = int((start_y + end_y) / 2.0)
			input_centroids[i] = (centroid_x, centroid_y)

		# if there is no registered objects, register found objects
		if len(self.objs_dict) == 0:
			for i in range(len(input_centroids)):
				self.register_new_obj(input_centroids[i])
		else:
			obj_ids = list(self.objs_dict.keys())
			obj_centroids = list(self.objs_dict.values())

			# dist.cdist(XA, XB, metric='euclidean', *args, **kwargs)
			# Compute distance between each pair of the existing objects centroids
			# and new input centroids.
			distance = dist.cdist(np.array(obj_centroids), input_centroids, metric="euclidean")
			rows = distance.min(axis=1).argsort()
			cols = distance.argmin(axis=1)[rows]

			# generate set object by the constructor
			used_rows = set()
			used_cols = set()

			for (row, col) in zip(rows, cols):
				if row in used_rows or col in used_cols:
					continue

				obj_id = obj_ids[row]
				self.objs_dict[obj_id] = input_centroids[col]
				# reset the disapeared counter for the object id
				self.disappeared_frames_dict[obj_id] = 0

				used_rows.add(row)
				used_cols.add(col)

			# to find unused centroids for disappeared & new objects
			unused_rows = set(range(distance.shape[0])).difference(used_rows)
			unused_cols = set(range(distance.shape[1])).difference(used_cols)

			# in case that obj centroids >= input centroids
			if distance.shape[0] >= distance.shape[1]:
				for row in unused_rows:
					obj_id = obj_ids[row]
					self.disappeared_frames_dict[obj_id] += 1

					if self.disappeared_frames_dict[obj_id] > self.max_frames_to_disappear:
						self.deregister_obj(obj_id)
			# else we need to register new objects
			else:
				for col in unused_cols:
					self.register_new_obj(input_centroids[col])

		return self.objs_dict