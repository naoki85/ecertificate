---
layout: post
title:  "確率分布"
date:   2023-07-04 10:47:00 +0900
categories: probability
---

# ベルヌーイ分布

[ベルヌーイ分布](https://ja.wikipedia.org/wiki/ベルヌーイ分布)

ベルヌーイ分布は、2つの結果がある試行において、片方の結果が出る確率を p とした場合の離散型確率分布です。具体的には、成功を1、失敗を0で表し、成功が出る確率を p 、失敗が出る確率を 1-p とします。このとき、ベルヌーイ分布の確率質量関数は以下のように表されます。
ここで、 k は成功の回数です。

$$
f(k;p)=\left\{
\begin{aligned}
p^k(1-p)^{1-k} \quad (k=0,1) \\
0 \quad (otherwise)
\end{aligned}
\right.
$$

## 期待値と分散

ベルヌーイ分布の期待値は、成功が出る確率 **p** です。

```
E[X]=p
```

また、分散は以下のようになります。

```
Var(X)=p(1-p)
```