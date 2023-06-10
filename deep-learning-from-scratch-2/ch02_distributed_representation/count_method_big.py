import sys

import matplotlib.pyplot as plt
import numpy as np
sys.path.append('..')
from dataset import ptb
from common.util import preprocess, create_co_matrix, ppmi, most_similar

window_size = 2
wordvec_size = 100

corpus, word_to_id, id_to_word = ptb.load_data('train')
vocab_size = len(word_to_id)
print('counting co-occurrence ...')
C = create_co_matrix(corpus, vocab_size, window_size)
print('calculating PPMI ...')
W = ppmi(C, verbose=True)

print('calculating SVD ...')
try:
    # sklearn の SVD ライブラリの方が高速(Truncated SVD)
    # 特異値の大きいものだけに限定して計算する
    # wordvec_size=100 に圧縮するイメージ
    from sklearn.utils.extmatch import randomized_svd
    U, S, V = randomized_svd(W, n_components=wordvec_size, n_iter=5, random_state=None)
except ImportError:
    U, S, V = np.linalg.svd(W)

word_vecs = U[:, :wordvec_size]

queries = ['you', 'year', 'car', 'toyota']
for query in queries:
    most_similar(query, word_to_id, id_to_word, word_vecs, top=5)
