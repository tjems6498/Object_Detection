import numpy as np


bboxes = np.array([[2,0.8, 10,10,20,20],[2, 0.005, 20, 50, 20, 40], [3, 0.00007, 20 ,20, 30, 50]])

retained_idx = [box.tolist() for box in bboxes[1:, :] if (bboxes[0, 0]!= box[0]) or (box[1] >= 0.01)]
print(retained_idx)