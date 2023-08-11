---
layout: post
title:  "深層学習でよく使う関数"
date:   2023-07-10 09:03:00 +0900
categories: supervised-learning
---

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