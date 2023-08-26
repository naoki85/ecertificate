---
layout: post
title:  "確率分布"
date:   2023-07-02 10:47:00 +0900
categories: probability
---

# 離散型確率分布

## 分散

$$
V[x] = E[x^2] - (E[x])^2 
$$

# 正規分布（ガウス分布）

[正規分布](https://bellcurve.jp/statistics/course/7797.html)  

$$
f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} exp( -\frac{(x-\mu)^2}{2\sigma^2} )
$$

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

$$
E[X]=p
$$

また、分散は以下のようになります。

$$
Var(X)=p(1-p)
$$

# 多項分布(マルチヌーイ分布)

[多項分布](https://bellcurve.jp/statistics/course/26597.html){:target="_blank"}  
事象 $$A_i$$ が起きる確率をそれぞれ $$p_i$$ とすると、確率変数 $$X_i$$ が多項分布に従う場合、それぞれの試行が $$x_i$$ 回起こる確率は、

$$
P(X_1 = x_1, X_2 = x_2, \ldots, X_k = x_k) = \frac{n!}{x_1!x_2!\ldots x_k!} p_1^{x_1} p_2^{x_2} \ldots p_k^{x_k}
$$

## 期待値

$$
E[X_i] = np_i
$$

## 分散

$$
Var[X_i] = np_i(1-p_i)
$$

## 共分散

$$
Cov[X_i, X_j] = -np_ip_j \quad (i \neq j)
$$

# ベイズ推定

- [乗法定理](https://bellcurve.jp/statistics/course/6442.html){:target="_blank"}
- [ベイズの定理](https://bellcurve.jp/statistics/course/6444.html){:target="_blank"}
