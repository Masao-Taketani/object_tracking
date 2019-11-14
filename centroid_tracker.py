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