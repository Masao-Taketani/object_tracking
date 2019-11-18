class TrackableObject:
	def __init__(self, obj_id, centroid):
		self.obj_id = obj_id
		self.centroids = [centroid]
		self.counted = False