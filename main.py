import math
import matplotlib
import cv2
import os
import numpy as np
from dwpose_utils.dwpose_detector import dwpose_detector_aligned
from dwpose_utils.wholebody import Wholebody
from skeleton_extraction import draw_pose
import argparse
from PIL import Image


raw_image = './ref.jpg'
raw_image = cv2.imread(raw_image)

ref_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
ref_pose = dwpose_detector_aligned(ref_image)
# faces = ref_pose['faces']

h, w, _ = ref_image.shape


cavas = draw_pose(ref_pose, h, w)
cavas = cavas.transpose(1, 2, 0)

print(cavas.shape, raw_image.shape)
cavas = np.hstack([cavas, raw_image])
cv2.imshow('Canvas Image', cavas)

cv2.waitKey(0)
cv2.destroyAllWindows()