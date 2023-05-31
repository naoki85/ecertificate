import sys
import os
import numpy as np
sys.path.append(os.pardir)
from common.util import im2col


x1 = np.random.rand(1, 3, 7, 7)
print(x1.shape)
col1 = im2col(x1, 5, 5, stride=1, pad=0)
print(col1.shape)

exit()

x2 = np.random.rand(10, 3, 7, 7)
col2 = im2col(x2, 5, 5, stride=1, pad=0)
print(col2.shape)