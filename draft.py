import numpy as np
import cv2

data_dir = 'X:\segformer-pytorch-master\pred_result\cracktree\crack0175.npz'
data = np.load(data_dir)
img = cv2.cvtColor(data['image'], cv2.COLOR_RGB2BGR)
# pred = cv2.cvtColor(data['pred'], cv2.COLOR_RGB2GRAY)
# label = cv2.cvtColor(data['label'], cv2.COLOR_RGB2GRAY)

# print(np.max(data['pred'], axis=0))
cv2.imshow('image', img)
cv2.imshow('pred', data['pred'].astype(np.float64))
cv2.imshow('label', data['label'].astype(np.float64))
cv2.waitKey(0)
cv2.destroyAllWindows()