---
layout: post
title:  "正則化"
date:   2023-07-05 23:19:05 +0900
categories: regularization
---

機械学習における正則化とは、過学習を防ぐためにパラメータの値を制限する手法のこと。  

# ノルム

ノルムは、ベクトルの大きさを表す計算方法。  
L1 ノルムと L2 ノルムが使用される。

## L1 ノルム

L1 ノルムは、各要素の絶対値の和を表します。  
例えば、ベクトル x = `[1, -2, 3]` の L1 ノルムは、  
$$
|1| + |-2| + |3| = 6
$$

L1 ノルムは、ラッソ回帰による正則化に使用される。  
ラッソ回帰では、コスト関数に L1 ノルムを加えることで、モデルのパラメータを 0 に近づけることができる。

## L2 ノルム

L2 ノルムは、各要素の二乗和の平方根を表す。  
例えば、ベクトル x = `[1, -2, 3]` の L2 ノルムは、  
$$
\sqrt{(1^2 + (-2)^2 + 3^2)}
$$

L2 ノルムは、リッジ回帰による正則化に使用される。  
リッジ回帰では、コスト関数に L2 ノルムを加えることで、モデルのパラメータを大きくなりすぎないように制限することができる。  
L2 ノルムを加えると、重み減衰と呼ばれる、 **パラメータが原点から離れる** ことに対してペナルティを課すことができる。

# 正則化手法

リッジ回帰、ラッソ回帰、エラスティックネットは、過学習を防ぐために重要な役割を果たす。  
正則化の対象とするパラメータにはバイアスを含まないことが一般的。  
重みと異なり、バイアスが大きくなっても過剰適合に繋がることが少ないため。

## リッジ回帰(L2 正則化)

リッジ回帰は、コスト関数に L2 ノルムを加えることで実現される。  
この手法により、パラメータの値が大きくなりすぎないように制限される。

```python
def ridge_regression(X, y, alpha):
    n, d = X.shape
    I = np.eye(d) # 単位行列
    C = X.T.dot(X) + alpha * I
    return np.linalg.inv(C).dot(X.T.dot(y))
```
sk-learn を使用することでもっと簡潔に書ける。

## ラッソ回帰(L1 正則化)

ラッソ回帰は、コスト関数に L1 ノルムを加えることで実現される。  
この手法により、一部のパラメータが 0 になることがある。

```python
def lasso_regression(X, y, alpha, num_iter, lr):
    n, d = X.shape
    w = np.zeros(d)
    for _ in range(num_iter):
        grad = -2 * X.T.dot(y - X.dot(w)) + alpha * np.sign(w)
        w -= lr * grad
    return w
```
sk-learn を使用することでもっと簡潔に書ける。

## エラスティックネット

エラスティックネットは、リッジ回帰とラッソ回帰を組み合わせた手法。  
コスト関数に L1 ノルムと L2 ノルムの和を加えることで実現される。  
この手法により、リッジ回帰とラッソ回帰の両方の効果を得ることができる。

# 参考

- [ラッソ回帰とリッジ回帰の理論 - Qiita](https://qiita.com/oki_kosuke/items/fb8bb418167f2ab1744e)
- [リッジ回帰(L2正則化)を理解して実装する - Qiita](https://qiita.com/g-k/items/d3124eb00cb166f5b575)