import numpy as np


def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """
    :param input_data: (データ数, チャンネル, 高さ, 幅)の4次元配列からなる入力データ
    :param filter_h: フィルターの高さ
    :param filter_w: フィルターの幅
    :param stride: ストライド
    :param pad: パディング
    :return col: 2次元配列
    """
    N, C, H, W = input_data.shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1
    print('=============================')
    print('params:')
    print('input_data.shape: ', input_data.shape)
    print('filter_h: ', filter_h)
    print('filter_w: ', filter_w)
    print('stride: ', stride)
    print('pad: ', pad)
    print('out_h: ', out_h)
    print('out_w: ', out_w)
    print('=============================')

    # pad は行列にパディングをするメソッド
    # 第二引数にて、どの次元にどれだけパディングを詰めるか指定する
    # CNN では H と W 、3 次元目と 4 次元目に対してパディングをするのでこのようになっている。
    img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))
    print('img.shape: ', img.shape)
    print('col.shape: ', col.shape)
    print('=============================')

    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            # ここの代入ではブロードキャストが行われる
            # 例: x = 0, y = 0, stride = 1, out = 3 * 3 とすると
            # img[:, :, 0:3:1, 0:3:1] は (N, C, 3, 3) になる
            # 一方、 col は 3 次元（フィルターの高さ）、 4 次元（フィルターの幅）は 1 となる。
            # ここでブロードキャストが行われて、(N, C, 3, 3) が (N, C, 1, 1, 3, 3) として代入される
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    # col を並び替える
    # out のサイズを先に持ってきて、高さが N * out_h * out_w 、幅が C * filter_h * filter_w
    print('col.transpose: ', col.transpose(0, 4, 5, 1, 2, 3).shape)
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    print('output col: ', col.shape)
    return col


if __name__ == '__main__':
    x1 = np.random.rand(1, 3, 7, 7)
    col1 = im2col(x1, 5, 5, stride=1, pad=0)

    x2 = np.random.rand(10, 3, 7, 7)
    col2 = im2col(x2, 5, 5, stride=1, pad=0)