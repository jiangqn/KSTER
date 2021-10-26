import sys
sys.path.append("../..")

from joeynmt.knn import KNNElasticSearch, KNNFaissSearch
import time
import numpy as np

embeddings_path = "embeddings_4.npy"
embeddings = np.load(embeddings_path)
m_embeddings = np.load(embeddings_path, mmap_mode="r")

batch_size = 32
d = 512
n_run = 100

# es_knn = KNNElasticSearch(index="wmt14_en_de_base_v2",
#                             psm="byte.es.mt_corpus.service.hl")
# query = np.random.random((batch_size, d)).astype(np.float32)

# start = time.time()

# for i in range(n_run):
#         query = embeddings[i * batch_size: (i + 1) * batch_size]
#         res = es_knn.search(query)
# print("es_search delay (batch = %d): %.4f ms" % (batch_size, (time.time() - start) / n_run * 1000))

faiss_knn = KNNFaissSearch(index_path='trained.index',
                               token_path='token_map')
start = time.time()
for i in range(n_run):
        query = embeddings[i * batch_size: (i + 1) * batch_size]
        res = faiss_knn.search(query)
        indices = np.random.choice(embeddings.shape[0], batch_size, replace=True)
        s = m_embeddings[indices]
print("faiss_search delay (batch = %d): %.4f ms" % (batch_size, (time.time() - start) / n_run * 1000))