import numpy as np
import sys
import os
current_path = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.dirname(current_path)
sys.path.append(parent_dir) # 親ディレクトリも対象にする
from dataset.mnist import load_mnist


# 平均 2 乗和誤差
def mean_squared_error(y, t):
    return 0.5 * np.sum((y-t) ** 2)


# クロスエントロピー誤差
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size_tmp = y.shape[0]
    # return -np.sum(np.log(y[np.arange(batch_size), t])) / batch_size_tmp t にラベル名が直接入る場合
    return -np.sum(t * np.log(y)) / batch_size_tmp


(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]

print(x_batch.shape)
print(t_batch.shape)
