---
layout: post
title:  "データの評価指標"
date:   2023-07-07 01:28:05 +0900
categories: metrics
---

# 混同行列 (Confusion matrix)

予測値と実績値の組み合わせは以下の 4 通りがある。
この 4 通りについて名前をつけた表のことを混合行列と言う。

|  | 正例（予測） | 負例（予測） |
| --- | --- | --- |
| 正例（実績） | 真陽性（True positive: TP） | 偽陰性（False negative: FN） |
| 負例（実績） | 偽陽性（False positive: FP） | 真陰性（True negative: TN） |

## 正解率 (accuracy)

全データ中、どれだけ予測が当たっていたかの割合。

$$
accuracy = \frac{TP + TN}{TP + FN + FP + TN}
$$

## 適合率 (precision)

予測が正の中で、実際に正であったものの割合。

$$
precision = \frac{TP}{TP + FP}
$$

## 再現率 (recall)

実際に正であるものの中で、正だと予測できた割合。

$$
recall = \frac{TP}{TP + FN}
$$

## F値 (F measure)

適合率と再現率の調和平均。適合率のみあるいは再現率のみで判断すると、予測が偏っているときも値が高くなってしまう。

$$
f = \frac{2 * precision * recall}{precision + recall}
$$

適合率と再現率はトレードオフの関係であるが、それらが一致する点がある（[PR 図](https://www.codexa.net/ml-evaluation-cls/)）
その場所をブレークイーブンポイントというが、それが F 値のようなもの。  
ブレークイーブンポイントの位置が **右上** に遷移するほど、適合率/精度(precision)と再現率(recall)が同時に高くなるので、良いモデルができたと結論づけることができる。

# scikit-learn で試す

scikit-learn に [confusion matrix](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html) を算出するメソッドがあるので試せる。
モデルには[サポートベクターマシン](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html?highlight=svc#sklearn.svm.SVC)を使用する。

```python
from sklearn.svm import SVC
 from sklearn.metrics import confusion_matrix
 
 cancer = load_breast_cancer()
 X_train, X_test, y_train, y_test = train_test_split(cancer.data,
                                                     cancer.target,
                                                     stratify=cancer.target,
                                                     random_state=66)
 model = SVC(gamma=0.001,C=1)
 model.fit(X_train,y_train)
 
 # For check score
 # print('{} train score: {:.3f}'.format(model.__class__.__name__, model.score(X_train,y_train)))
 # print('{} test score: {:.3f}'.format(model.__class__.__name__ , model.score(X_test,y_test)))
 
 y_pred = model.predict(X_test)
 confusion_matrix(y_test, y_pred)
```

|  | 予測 | 予測 |
| --- | --- | --- |
| 観測 | 48 | 5 |
| 観測 | 8 | 82 |

行列で取得できるので、 `confusion_matrix[0, 0]` みたいな感じで要素が取得できるので、正解率であれば、

```python
accuracy = (m[0, 0] + m[1, 1]) / m.sum()
```

ただ、だいたい scikit-learn 便利メソッドが用意されている。

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
print('正解率:{:.3f}'.format(accuracy_score(y_test, y_pred)))
print('適合率:{:.3f}'.format(precision_score(y_test, y_pred)))
print('再現率:{:.3f}'.format(recall_score(y_test, y_pred)))
print('F1値:{:.3f}'.format(f1_score(y_test, y_pred)))
```