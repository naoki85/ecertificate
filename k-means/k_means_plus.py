import numpy as np
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


def init_centroid(X, k, n_data):
    # 1 つ目のセントロイドをランダムに選択
    idx = np.random.choice(n_data, 1)
    centroids = X[idx]
    for i in range(k - 1):
        # 各データ点とセントロイドとの距離を計算
        distances = compute_distances(X, len(centroids), n_data, centroids)
        # 各データ点と最も近いセントロイドとの距離の二乗を計算
        closest_dist_sq = np.min(distances ** 2, axis=1)
        # 距離の二乗の和を計算
        weights = closest_dist_sq.sum()
        # [0, 1) の乱数と距離の二乗和を掛ける。random_sample と rand の違いは引数の指定方法の違いだけ
        rand_vals = np.random.random_sample() * weights
        # 距離の二乗の累積和を計算し、 rand_val と最も値が近いデータ点の index を取得
        # searchsorted はバイナリサーチ。rand_vals の値にちかい値のインデックスを返してくれる
        # cumsum は累積和
        candidate_ids = np.searchsorted(np.cumsum(closest_dist_sq), rand_vals)
        # 選ばれた点を新たなセントロイドとして追加
        centroids = np.vstack([centroids, X[candidate_ids]])
    return centroids


def compute_distances(X, k, n_data, centroids):
    distances = np.zeros((n_data, k))
    for idx_centroids in range(k):
        dist = np.sqrt(np.sum((X - centroids[idx_centroids]) ** 2, axis=1))
        distances[:, idx_centroids] = dist
    return distances


def k_means_plus(X, k, max_iter=300):
    n_data, n_features = X.shape
    # セントロイドの初期値
    centroids = init_centroid(X, k, n_data)
    # 新しいクラスタを格納するための配列
    new_cluster = np.zeros(n_data)
    # 各データの所属クラスタを保存する配列
    cluster = np.zeros(n_data)

    for epoch in range(max_iter):
        # 各データ点とセントロイドとの距離を計算
        distances = compute_distances(X, k, n_data, centroids)
        # 新たな所属クラスタを計算
        # データごとに最も近いクラスタの番号の配列ができる（この例だと 1 次元配列）
        new_cluster = np.argmin(distances, axis=1)

        # 全てのクラスタに対してセントロイドを再計算
        for idx_centroids in range(k):
            # new_cluster にはデータごとのクラスタ番号が入っているので、対象のクラスタのデータだけ取得した上で平均をとっている
            # データは各要素ごとの平均を取りたいので、データ軸で平均をとる
            centroids[idx_centroids] = X[new_cluster == idx_centroids].mean(axis=0)

        # クラスタによるグループ分けに変化がなかったら終了
        if (new_cluster == cluster).all():
            break
        cluster = new_cluster

    return cluster


if __name__ == '__main__':
    # scikit-learn内にあるワインの品質判定用データセットをwineという変数に代入。
    wine = load_wine()
    # Xにワインの特徴量を代入。特徴量は、ここではアルコール濃度や色などwineの属性データのこと。
    X = wine.data
    # yにワインの目的変数を代入。目的変数は、ここでは専門家によるワインの品質評価結果のこと。
    y = wine.target
    scaler = StandardScaler()
    scaler.fit(X)
    kmeans_c = k_means_plus(scaler.transform(X), 3)
    # 一致率（正解率）を算出
    print(accuracy_score(y, kmeans_c))
