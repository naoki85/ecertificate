---
layout: post
title:  "深層学習でよく使う関数"
date:   2023-07-10 09:03:00 +0900
categories: supervised-learning
---

# シグモイド関数

$$
sigmoid(x) = \frac{1}{1 + exp(-x)}
$$

シグモイド関数の導関数は、 `y(1 - y)` になる。  
[シグモイド関数の微分](https://qiita.com/yosshi4486/items/d111272edeba0984cef2)

# ソフトマックス関数

$$
softmax(z) = \frac{exp(z_i)}{\sum_j exp(z_i)}
$$

ソフトマックス関数は入力の差にのみ依存する。  
この性質を利用して、 z - max(z) のように最大値を引くとオーバーフロー対策ができる。  
  
[ソフトマックス関数の微分](https://qiita.com/hatahataDev/items/4f4c744a534f475ce263)  

```py
class Softmax:
    # ...
    def backward(self, dout):
        dx = self.out * dout
        sumdx = np.sum(dx, axis=1, keepdims=True)
        dx -= self.out * sumdx
        return dx
```

# ソフトマックス関数と交差エントロピー

逆伝播の微分式は、 $$(y -t) / S$$ になる。  
導出過程は、 [こちらの記事](https://www.anarchive-beta.com/entry/2020/08/06/180000) や、ゼロつくの付録を参照。

```py
class SoftmaxWithLoss:
    # ...
    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size

        return dx
```

# アフィンノードの逆伝播

[Python DeepLearningに再挑戦 15 誤差逆伝播法 Affine/Softmaxレイヤの実装](https://pythonskywalker.hatenablog.com/entry/2016/12/25/144926)  
アフィンノードの逆伝播では、

$$
\frac{\partial L}{\partial W} = X^T \frac{\partial L}{\partial Y}
$$

$$
\frac{\partial L}{\partial X} = \frac{\partial L}{\partial Y} W^T
$$

入力 x 側の逆伝播では勾配 * W^T、重み W 側の逆伝播では X^T * 勾配を行う。

```py
def backward(self, dout):
    W, b = self.params
    dx = np.dot(dout, W.T)
    dW = np.dot(self.x.T, dout)
    db = np.sum(dout, axis=0)
    # ...
```
バイアスは、バッチサイズ方向に合計をする。  
バッチサイズで合計すると、このような感じになる。

```py
>>> import numpy as np
>>> dY = np.array([[1, 2, 3], [4, 5, 6]])
>>> dY
array([[1, 2, 3],
       [4, 5, 6]])
>>> dB = np.sum(dY, axis=0)
>>> dB
array([5, 7, 9])
```