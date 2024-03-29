---
layout: post
title:  "機械学習の基礎"
date:   2023-07-04 21:48:00 +0900
categories: machine-learning
---

[機械学習](https://ja.wikipedia.org/wiki/機械学習){:target="blank"}  
Mitchell の提唱した定義が一般的だが、 Googfellow の定義も問題に出たりする。

# 尤度

想定するパラメーターがある値をとる場合に観測している事柄や事象が起こりうる確率のこと。  
尤度はパラメーターの関数として表すことができるので尤度関数とも言う。  
  
例えば、「2枚のコインを投げて2枚とも表が出た」という観測結果が得られた場合、この結果が観測される確率はコインが表になる確率 p をパラメーターとする関数 $$L(p) = p^2$$ で表すことができる。  
このとき、p = 0.2 であれば、尤度は0.04である。  
  
- [【統計学】尤度って何？をグラフィカルに説明してみる。 - Qiita](https://qiita.com/kenmatsu4/items/b28d1b3b3d291d0cc698)
- [尤度関数](https://ja.wikipedia.org/wiki/尤度関数)

尤度関数は、確率関数にデータの各値を代入したもので書ける。

$$
L_D(p) = \prod_{i=1}^n f(x_i;p)
$$

また、負の対数尤度は、

$$
-log L_D(p) = -log \prod_{i=1}^n f(x_i;p)
$$

$$
            = -\sum_{i=1}^n log f(x_i;p)
$$

最尤推定値は、負の対数尤度を微分して求める。  
代表的な関数として交差エントロピーがあるが、この最尤推定量は、 $$\frac{1}{n} \displaystyle \sum_{i=1}^n x_i$$ になる。

# バイアス、バリアンス、ノイズ

[バイアス・バリアンスとは?図解で分かりやすく数式まで徹底解説!! 機械学習ナビ](https://nisshingeppo.com/ai/whats-bias-variance/){:target="blank"}  

- バイアス: モデルの表現力が不足していることによって生じる誤差。予測値の平均が理想値からどれくらい離れているか
- バリアンス: 訓練データの選び方によって生じる誤差。特定のデータが予測値全体からどれくらい離れているか
- ノイズ: データの測定誤差などによって生じる誤差

一般的に、ニューラルネットワークはノイズに対して脆弱である。  
入力にノイズを加えて学習することは、理論的には「パラメータのノルムペナルティを課すこと」と等価である。

# サンプリング方法

ブートストラップ法は、元のデータセットからランダムに再サンプリングを行う統計的手法。  
この方法は、データの性質を理解するための代替的な手段を提供する。  
 具体的には、推定値の不確実性（標準誤差や信頼区間）を推定したり、モデルの精度を検証したりする。
例えば、ブートストラップサンプルを用いて複数のモデルを訓練し、それらのモデルの平均的なパフォーマンスを評価することで、モデルの精度を検証する。  
これはブートストラップ・アグリゲーティング（バギング）と呼ばれ、ランダムフォレストのようなアンサンブル学習手法の基礎となっています。

# 転移学習、ファインチューニング

既存の学習済みモデルを新しいタスクに適用するための手法。

## 転移学習

- 一つのタスク（ソースタスク）で学習した知識を別のタスク（ターゲットタスク）に適用する手法。
- 深層学習においては、一般的に大規模なデータセット（例：ImageNet）で事前に訓練されたモデル（例：VGG16、ResNet、BERT等）を使用する。
- この学習済みモデルの重みは、新しいタスクでのモデル訓練の初期値として使用される。
- このことにより、ターゲットタスクで必要となる訓練データの量を減らすとともに、モデルの汎化能力を向上させることが期待される。

例: 犬の画像判定ができるモデルを利用し、猫の画像が判定できるモデルに利用する。  
転移学習の多くでは、新しい層を追加した場合にはその層のみ再学習を行う。（ファインチューニング）

## ファインチューニング

**転移学習の一部** としてよく用いられる手法で、転移学習によって初期化されたモデルの重みをターゲットタスクに特化させるために微調整する。

- 一部または全ての層の重みをターゲットタスクのデータで再訓練し、モデルの性能を向上させる。
- ここで注意すべき点は、ファインチューニングにはターゲットタスクのデータが一定量必要であり、データが少なすぎると過学習（overfitting）のリスクがある。

要するに、転移学習は学習済みモデルを新しいタスクに適用する広範な概念であり、ファインチューニングはその中の一手法で、新しいタスクによりよく適合するようモデルを微調整するという点で両者は異なる。

# 性能評価

## ホールドアウト法

データを訓練データ、（検証データ、）テストデータに分割して学習、評価を行う手法。

## k-分割交差検証

データをk個の重複しない集合に分割し、そのうちの1つをテストデータ、残りを訓練データとして訓練、精度計算をおこない、これをk回繰り返して平均を取ることで精度評価を行う。特にデータセットの数が少ない場合に用いられる。

# その他モデルとつく単語

## 隠れマルコフモデル (Hidden Markov Model, HMM)

隠れマルコフモデルは、系列データや時系列データのモデリングに広く使用される確率的なモデルの一つ。  
以前は音声認識や自然言語処理で利用されていたが、現在は LSTM に代わられている。

1. **状態と観測**: HMMは二つの主要なシーケンスを持つ。一つは隠れた状態のシーケンス、もう一つはそれに対応する観測（または出力）のシーケンス。
2. **遷移確率**: ある状態から次の状態への遷移の確率。これは一つの行列で示され、遷移確率行列と呼ばれる。
3. **出力確率**: ある状態で特定の観測が得られる確率。出力確率行列（または放出確率）で示される。

[時系列データ：隠れマルコフモデルの基礎と、リカレントネットの台等](https://www.hellocybernetics.tech/entry/2017/01/14/235811)

### ノンパラメトリックモデル

ノンパラメトリックモデルは、データの量が増えるとモデルのパラメータも増加するようなモデルを指す。

1. **柔軟性**: これらのモデルは、データの構造に自動的に適応する。そのため、特定の形や関数に制約されず、データの構造を捉えるのに非常に有効。
2. **カーネル法**: ノンパラメトリックの方法の一つ。例として、カーネル密度推定やカーネル回帰などがある。
3. **k-近傍法**: これもノンパラメトリックな方法の一つで、クラス分類や回帰の問題に使用される。
4. **利点**: 大きなデータセットに対して非常に柔軟で、予測の精度が高い場合がある。
5. **欠点**: 計算コストが高く、メモリ効率が低い場合がある。また、次元の呪いに影響されやすい。

# モデルの蒸留

学習済みのモデルから、小さなモデルに転移させる方法。  
学習済みの大きなモデルを先生、小さなモデルを生徒とする。  
先生は正解ラベルを用いて学習を行い、その後生徒の学習を行う。  
  
生徒の学習時、ソフトターゲット損失（ $$L_{soft}$$ ）とハードターゲット損失（ $$L_{hard}$$ ）を考える。  
ソフトターゲット損失は先生の予測、ハードターゲットは正解ラベルを目標分布にする。  
そしてソフトターゲットとハードターゲットの加重平均を最小化の対象にする。  
  
蒸留では、通常のソフトマックス関数の代わりに温度つきソフトマックス関数が用いられる。  
T を温度として、

$$
softmax(z)_i = \frac{exp(z_i/T)}{\sum_j exp(z_j/T)}
$$

で与えられる。  
温度 T を変えることで出力分布の滑らかさを調整できるが、ハードターゲットの場合は T = 1 とする。  
  
正解ラベルに対して、不正解ラベルの確率は本来とても小さいが、温度 T を高くすることで、不正解ラベルの確率を大きくすることができる。  
  
ソフトターゲットとハードターゲットの加重平均は、

$$
L = \frac{\lambda_1 T^2 L_{soft} + \lambda_2 L_{hard}}{\lambda_1 + \lambda_2}
$$
