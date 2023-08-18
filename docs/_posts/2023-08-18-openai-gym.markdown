---
layout: post
title:  "ゼロから作るディープラーニング入門 4 をやっていたら、 OpenAI Gym が Gynmasium に移管していた"
date:   2023-08-18 23:12:00 +0900
categories: reinforcement-learning
---

ゼロから作るディープラーニングでは、 8 章の環境構築で、 OpenAI Gym を利用している。  
[https://github.com/openai/gym](https://github.com/openai/gym)  
  
ただ、ドキュメントを見ると、 [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) というリポジトリの方がメンテナンス対象になっているようだ。  
（OpenAI Gym の方も 2023 年 7 月時点では使用できている）  
  
とはいえ、基本的には移管後のライブラリをインストールして、 `gym` として利用すればいいだけのよう。

```shell
$ pip install gymnasium
```

```python
import gymnasium as gym
```
テキストの方では最低限のライブラリの解説だったため、下記のドキュメントも参考にした方が良さそう。  
https://gymnasium.farama.org/environments/classic_control/cart_pole/

## サンプルコードからの変更点

ステップの返り値は 5 要素のようである。  
https://gymnasium.farama.org/content/basic_usage/

```python
next_state, reward, done, truncated, info = env.step(action)
```
terminated は終了、 truncated は中断のようだ。  
  
CartPole を利用して render を利用するには、 `gymnasium[classic-control]` が必要のようだ。

```shell
$ pip install gymnasium[classic-control]
```

`env.reset()` で取得できる値は、 `[state, info]` の 2 要素らしい。  
https://gymnasium.farama.org/api/env/#gymnasium.Env.reset

```python
state, _ = env.reset()
```