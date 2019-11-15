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
			for object_id in list(self.disappeared_frames_dict.keys()):
				self.disappeared_frames_dict[objectt_id] += 1

				if self.disappeared_frames_dict[objectt_id] > self.max_frames_to_disappear:
					self.deregister_obj(object_id)

			return self.obj_id