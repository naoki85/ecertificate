---
layout: post
title:  "学習の最適化手法"
date:   2023-07-02 10:13:00 +0900
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

バッチ正規化は、ニューラルネットワークの中間層の出力を正規化することで、学習を高速化し、精度を向上させる手法です。バッチサイズごとに平均と分散を計算し、それを用いて入力を正規化することで、学習の収束を早めることができます。
バッチ正規化の詳細については、[こちらの記事](https://deepage.net/deep_learning/2016/10/26/batch_normalization.html)をご覧ください。

## レイヤー正規化

レイヤー正規化は、バッチ正規化のように **中間層の出力** を正規化する手法ですが、バッチ正規化と異なり、バッチ内ではなく層内で正規化を行います。そのため、バッチサイズに依存しないモデルの学習が可能になります。
レイヤー正規化の詳細については、[こちらの論文](https://arxiv.org/abs/1607.06450)をご覧ください。

## グループ正規化

グループ正規化は、 **レイヤー正規化の一種であり、層内の特徴マップをグループ単位で正規化する** 手法です。
グループ正規化を用いることで、層内での正規化がうまくいかない場合でも、グループ単位で正規化することで学習を安定化させることができます。
グループ正規化の詳細については、[こちらの論文](https://arxiv.org/abs/1803.08494)をご覧ください。

## インスタンス正規化

インスタンス正規化は、バッチ正規化やレイヤー正規化のように、中間層の出力を正規化する手法ですが、 **バッチサイズや層内のグループごとではなく、各特徴マップごと** に正規化を行います。
そのため、畳み込み層の特徴マップに対しても適用可能であり、画像の局所的な特徴を捉えることができます。
インスタンス正規化の詳細については、[こちらの論文](https://arxiv.org/abs/1607.08022)をご覧ください。