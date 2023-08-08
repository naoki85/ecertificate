---
layout: post
title:  "学習の最適化手法"
date:   2023-07-06 10:13:00 +0900
categories: optimizer
---

# Optimizer

## 確率的勾配降下法(SGD)

最も基本的な手法。勾配に学習率をかけた値でパラメータを更新する。

```python
class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]
```

## モーメンタム法

ボールが急斜面は加速して転がり、上り坂になると徐々に減速する、という発想を取り入れた手法。  
v が物体の速度にあたり、前回の値から `momentum` という値をかけ、今回の勾配を引くことで実現させている。

```python
class Momentum:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)

        for key in params.keys():
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            params[key] += self.v[key]
```

## AdaGrad

各パラメータに対して、学習率を個別に適応させる手法。  
更新が頻繁に行われるパラメータは学習率が低下し、更新が少ないパラメータは学習率が上がる。  

$$
G = G + ∇J(θ)^2
$$

$$
θ = θ - η* \frac{1}{\sqrt{(G + ε)}}*∇J(θ)
$$

```python
class AdaGrad:
    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)

        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-07)
```

## RMSProp

AdaGradが勾配の二乗和を無限に蓄積していくと、学習率が0に近づくという問題を解決する手法。  
このために、過去の勾配の情報は指数的に忘れられ、最近の情報がより重視される。

AdaGrad の時と学習率の更新度合いを決める式が変わる。

$$
G = γ*G + (1-γ)*∇J(θ)^2
$$

```python
class RMSprop:
    def __init__(self, lr=0.01, decay_rate=0.99):
        self.lr = lr
        self.decay_rate = decay_rate
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)

        for key in params.key():
            self.h[key] *= self.decay_rate
            self.h[key] += (1 - self.decay_rate) * grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)
```

## Adam

MomentumとRMSPropのアイデアを組み合わせた手法で、一般的には最も良好なパフォーマンスを発揮する。

$$
m_t = \beta_1m_{t-1} + (1-\beta_1)g_t
$$

$$
v_t = \beta_2v_{t-1} + (1-\beta_2)g_t^2
$$

$$
\hat{m_t} = \frac{m_t}{1-\beta_1^t}
$$

$$
\hat{v_t} = \frac{v_t}{1-\beta_2^t}
$$

$$
\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{\hat{v_t}}+\epsilon}\hat{m_t}
$$

- $$m_t$$ は1次モーメント
- $$v_t$$ は2次モーメント
- $$\beta_1$$ は1次モーメントの指数減衰率
- $$\beta_2$$ は2次モーメントの指数減衰率
- $$\hat{m_t}$$ は1次モーメントのバイアス補正
- $$\hat{v_t}$$ は2次モーメントのバイアス補正
- $$\theta_t$$ はt時点のパラメータ
- $$\alpha$$ は学習率
- $$\epsilon$$ は数値安定性のための定数

```python
class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None

    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)

        self.iter += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta2 ** self.iter) / (1.0 - self.beta1 ** self.iter)

        for key in params.keys():
            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
            self.v[key] += (1 - self.beta2) * (grads[key] ** 2 - self.v[key])
            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)
```

- [【決定版】スーパーわかりやすい最適化アルゴリズム -損失関数からAdamとニュートン法- - Qiita](https://qiita.com/omiita/items/1735c1d048fe5f611f80#7-adam)

# 初期値

## Xavierの初期値

Xavierの初期値は、ニューラルネットワークの重みを適切に初期化する手法のひとつです。
Xavierの初期値を用いることで、勾配消失や爆発を防ぎ、学習を安定化させることができます。
Xavierの初期値は、前の層のノード数をn、次の層のノード数をmとすると、以下のように計算されます。

$$
W \sim N(0, \frac{1}{n})
$$

ここで、 $$N(0, σ^2)$$ は平均0、分散 $$σ^2$$ の正規分布を表します。
前後のノードがある場合は、それぞれのノード数を $$n_1$$ 、 $$n_2$$ とすると、

$$
W \sim N(0, \frac{2}{n_1 + n_2})
$$

## Heの初期値

Heの初期値は、Xavierの初期値の改良版で、ReLUなどの活性化関数を用いる場合に適した初期値です。
Heの初期値を用いることで、勾配消失や爆発を防ぎ、より高速な学習が可能になります。
Heの初期値は、前の層のノード数をnとすると、以下のように計算されます。

$$
W \sim N(0, \frac{2}{n})
$$

Xavierの初期値とは異なり、Heの初期値では分散が2/nとなっています。

## バッチ正規化

バッチ正規化は、ニューラルネットワークの中間層の出力を正規化することで、学習を高速化し、精度を向上させる手法です。  
バッチサイズごとに平均と分散を計算し、それを用いて入力を正規化することで、学習の収束を早めることができます。  
[バッチ正規化](https://deepage.net/deep_learning/2016/10/26/batch_normalization.html)

$$
\mu = \frac{1}{m}\sum_{i=1}^{m} x_i
$$

$$
\sigma^2 = \frac{1}{m}\sum_{i=1}^{m} (x_i - \mu)^2
$$

$$
\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}
$$

$$
y_i = \gamma \hat{x}_i + \beta
$$

```py
def batch_normalization(data, epsilon=1e-6):
    # 特徴量ごとの平均を計算
    batch_means = data.mean(axis=0)
    # 特徴量ごとの分散を計算
    batch_vars = data.var(axis=0)

    data_hat = (data - batch_means)/np.sqrt(batch_vars+epsilon)
    data_output = gamma * data_hat + beta
    return data_output
```

## レイヤー正規化

レイヤー正規化は、バッチ正規化のように **中間層の出力** を正規化する手法ですが、バッチ正規化と異なり、バッチ内ではなく層内で正規化を行います。  
そのため、バッチサイズに依存しないモデルの学習が可能になります。  
[レイヤー正規化](https://cvml-expertguide.net/terms/dl/layers/batch-normalization-layer/layer-normalization/#:~:text=%E3%83%AC%E3%82%A4%E3%83%A4%E3%83%BC%E6%AD%A3%E8%A6%8F%E5%8C%96%20(Layer%20Normalization)%E3%81%A8%E3%81%AF%EF%BC%8C%E5%8F%AF%E5%A4%89%E9%95%B7,%E3%82%A2%E3%83%AC%E3%83%B3%E3%82%B8%E3%81%97%E3%81%9F%E3%82%82%E3%81%AE%E3%81%A7%E3%81%82%E3%82%8B%EF%BC%8E)

$$
\mu = \frac{1}{N}\sum_{i=1}^{N} x_i
$$

$$
\sigma^2 = \frac{1}{N}\sum_{j=1}^{N} (x_{ij} - \mu)^2
$$

$$
\hat{x}_{ij} = \frac{x_{ij} - \mu}{\sqrt{\sigma^2 + \epsilon}}
$$

```py
def layer_normalization(x):
    input_mean = np.mean(x, axis = 1)
    input_var = np.var(x, axis = 1)
    normed_input = (x - input_mean[...,None])/np.sqrt(input_var[...,None]+1e-8)
    return normed_input
```

## インスタンス正規化

インスタンス正規化は、バッチ正規化やレイヤー正規化のように、中間層の出力を正規化する手法ですが、 **バッチサイズや層内のグループごとではなく、各特徴マップごと** に正規化を行います。
そのため、畳み込み層の特徴マップに対しても適用可能であり、画像の局所的な特徴を捉えることができます。
[インスタンス正規化](https://qiita.com/sho12333/items/8a23f0fcdc03b91cbc04)

$$
\mu = \frac{1}{L}\sum_{h=1}^{L} x_h
$$

$$
\sigma^2 = \frac{1}{L}\sum_{h=1}^{L} (x_h - \mu)^2
$$

$$
\hat{x}_h = \frac{x_h - \mu}{\sqrt{\sigma^2 + \epsilon}}
$$

$$
y_h = \gamma \hat{x}_h + \beta
$$

```py
def instance_normalization(x, epsilon=1e-8):
    ''' x.shape = (batch_size, channel, data) '''
    mean = np.mean(x, axis=(2))
    var = np.var(x, axis=(2))
    x_normed = (x-mean[...,None])/np.sqrt(var[...,None] + epsilon)
    return x_normed
```

インスタンス正規化は画像データに対して使うケースが多いので、データを画像の縦・横で表した4次元（4重配列）データxを引数とする関数実装は以下のようになる。  

```py
def instance_normalization(x, epsilon=1e-8):
    ''' x.shape = (batch_size, channel, width, height) '''
    mean = np.mean(x, axis=(2,3))
    var = np.var(x, axis=(2,3))
    x_normed = (x-mean[...,None,None])/np.sqrt(var[...,None,None] + epsilon)
    return x_normed
```

## グループ正規化

グループ正規化は、 **レイヤー正規化の一種であり、層内の特徴マップをグループ単位で正規化する** 手法です。
グループ正規化を用いることで、層内での正規化がうまくいかない場合でも、グループ単位で正規化することで学習を安定化させることができます。  
バッチ次元に対して独立しているので、バッチサイズが変化してもサイズが変わらないという特徴がある。  
[グループ正規化](https://data-analytics.fun/2022/09/23/group-normalization/#toc7)

1. 特徴マップを G 個のグループに分割します。それぞれのグループは C/G 個のチャンネルを含みます（ここで C は特徴マップの全チャンネル数です）。
2. 各グループ内での平均 $$\mu_g$$ と分散 $$\sigma_g^2$$ を計算します。具体的には、グループ g の平均と分散は以下のように計算されます：

$$
\mu_g = \frac{1}{m} \sum_{i=1}^{m} x_{gi}
$$

$$
\sigma_g^2 = \frac{1}{m} \sum_{i=1}^{m} (x_{gi} - \mu_g)^2
$$

ここで、 $$x_{gi}$$ はグループ g 内のチャンネルの値を表し、m はグループ内のチャンネル数です。

3. 次に、各チャンネルを正規化します：

$$
\hat{x}_{gi} = \frac{x_{gi} - \mu_g}{\sqrt{\sigma_g^2 + \epsilon}}
$$

ここで、 $$\epsilon$$ は数値的安定性を保つための小さな値（通常 $$10^{-5}$$ 程度）です。

4. 最後に、スケーリングとシフトを行います：

$$
y_{gi} = \gamma_g \hat{x}_{gi} + \beta_g
$$

ここで、 $$\gamma_g$$ と $$\beta_g$$ は学習可能なパラメータで、それぞれスケールとシフトを制御します。これらはモデルの訓練プロセスの一部として学習されます。

# ハイパーパラメータのチューニング

主に、グリッドサーチ、ランダムサーチ、ベイズ最適化がある。

## ベイズ最適化

ベイズ最適化は、確率モデル（主に **ガウス過程** ）を使用してハイパーパラメータ探索を行う手法。  
これにより、過去の試行結果から得られた情報を活用して、次に試すべきハイパーパラメータの組み合わせを選択する。  
ベイズ最適化は、ランダムサーチやグリッドサーチよりも効率的に探索空間を探索し、少ない試行回数で最適解に収束できる場合がある。