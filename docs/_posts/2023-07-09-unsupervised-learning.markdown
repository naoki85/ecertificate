---
layout: post
title:  "教師なし学習"
date:   2023-07-08 21:14:00 +0900
categories: unsupervised-learning
---

# k-means 法

k-means 法は、データを k 個のクラスタに分ける手法。  
ちなみに、 k 近傍法はある点から K 個のポイントを集め、その K 個の中から最も多い性質を持つものに合わせる教師あり学習。  
[【Python】KNN(k近傍法)とk-means(k平均法)の違いと区別](https://dse-souken.com/2021/04/02/ai-16/)

## k-means++

k-means の初期値の設定方法を変更したアルゴリズム。  
他の代表ベクトルとの距離の 2 乗に比例する確率 $$p(x_i)$$ を使って確率的に代表ベクトルを選択する。  
このような選択方法をルーレット選択という。
