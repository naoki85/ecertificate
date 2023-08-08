import numpy as np
from sklearn.datasets import load_wine


def pca(X, n_components):
    X = X - X.mean(axis=0)
    # 共分散行列の作成
    cov = np.cov(X, rowvar=False)
    # 固有値と固有ベクトルを計算
    l, v = np.linalg.eig(cov)

    # 固有値の大きい順に固有ベクトルを並べ替え
    l_index = np.argsort(l)[::-1]
    v_ = v[:, l_index]

    # n_components 個の固有ベクトルを取得
    components = v_[:, :n_components]

    # データを低次元空間へ射影
    T = np.dot(X, components)
    return T


if __name__ == '__main__':
    # scikit-learn内にあるワインの品質判定用データセットをwineという変数に代入。
    wine = load_wine()
    # Xにワインの特徴量を代入。特徴量は、ここではアルコール濃度や色などwineの属性データのこと。
    X = wine.data
    print(X.shape)
    X_ = pca(X, 4)
    print(X_.shape)
