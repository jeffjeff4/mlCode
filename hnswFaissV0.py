#https://github.com/pinecone-io/examples/blob/master/learn/search/faiss-ebook/hnsw-faiss/hnsw_faiss.ipynb

import faiss
import numpy as np

folder = "/Users/shizhefu0/Desktop/ml/data/sift/"
# now define a function to read the fvecs file format of Sift1M dataset
#def read_fvecs(fp):
#    a = np.fromfile(fp, dtype='int32')
#    d = a[0]
#    return a.reshape(-1, d + 1)[:, 1:].copy().view('float32')

def read_fvecs(filename):
    with open(filename, 'rb') as f:
        vecs = []
        while True:
            dim_bin = f.read(4)
            if not dim_bin:
                break
            dim = np.frombuffer(dim_bin, dtype='int32')[0]
            vec = np.frombuffer(f.read(4*dim), dtype='float32')
            vecs.append(vec)
    return np.vstack(vecs)

def read_fvecs_chunked(filename, chunk_size=100000):
    dim = np.fromfile(filename, dtype='int32', count=1)[0]
    dtype = np.dtype([('dim', 'i4'), ('vec', 'f4', dim)])
    return np.fromfile(filename, dtype=dtype)['vec']

# 1M samples
#xb = read_fvecs('./sift/sift_base.fvecs')
#xb = faiss.read_fvec_format('./sift/sift_base.fvecs')
xb = read_fvecs_chunked(folder + "/sift_base.fvecs")
BASE_NUM_RECORDS = 1000
xb = xb[:BASE_NUM_RECORDS, ]


# queries
xq = read_fvecs(folder + "sift_query.fvecs")[0].reshape(1, -1)
xq_full = read_fvecs(folder + "sift_query.fvecs")
QUERY_NUM_RECORDS = 10
xq_full = xq_full[:QUERY_NUM_RECORDS, ]

# setup our HNSW parameters
d = 128  # vector size
M = 32
efSearch = 32  # number of entry points (neighbors) we use on each layer
efConstruction = 32  # number of entry points used on each layer
                     # during construction

index = faiss.IndexHNSWFlat(d, M)
print(index.hnsw)

# the HNSW index starts with no levels
index.hnsw.max_level

# and levels (or layers) are empty too
levels = faiss.vector_to_array(index.hnsw.levels)
np.bincount(levels)

index.hnsw.efConstruction = efConstruction
index.hnsw.efSearch = efSearch


index.add(xb)

# after adding our data we will find that the level
# has been set automatically
index.hnsw.max_level

# and levels (or layers) are now populated
levels = faiss.vector_to_array(index.hnsw.levels)
np.bincount(levels)

index.hnsw.entry_point

def set_default_probas(M: int, m_L: float):
    nn = 0  # set nearest neighbors count = 0
    cum_nneighbor_per_level = []
    level = 0  # we start at level 0
    assign_probas = []
    while True:
        # calculate probability for current level
        proba = np.exp(-level / m_L) * (1 - np.exp(-1 / m_L))
        # once we reach low prob threshold, we've created enough levels
        if proba < 1e-9: break
        assign_probas.append(proba)
        # neighbors is == M on every level except level 0 where == M*2
        nn += M*2 if level == 0 else M
        cum_nneighbor_per_level.append(nn)
        level += 1
    return assign_probas, cum_nneighbor_per_level

assign_probas, cum_nneighbor_per_level = set_default_probas(
    32, 1/np.log(32)
)
assign_probas, cum_nneighbor_per_level

# this is copy of HNSW::random_level function
def random_level(assign_probas: list, rng):
    # get random float from 'r'andom 'n'umber 'g'enerator
    f = rng.uniform()
    for level in range(len(assign_probas)):
        # if the random float is less than level probability...
        if f < assign_probas[level]:
            # ... we assert at this level
            return level
        # otherwise subtract level probability and try again
        f -= assign_probas[level]
    # below happens with very low probability
    return len(assign_probas) - 1

chosen_levels = []
rng = np.random.default_rng(12345)
for _ in range(1_000_000):
    chosen_levels.append(random_level(assign_probas, rng))
np.bincount(chosen_levels)

1/np.log(32)  # the previous value we used for m_L

set_default_probas(32, 0.09)

levels = faiss.vector_to_array(index.hnsw.levels)
np.bincount(levels)

del index
index = faiss.IndexHNSWFlat(d, 32)
index.hnsw.set_default_probas(32, 0.09)  # HNSW::set_default_probas(int M, float levelMult)
index.hnsw.efConstruction = efConstruction
index.add(xb)

levels = faiss.vector_to_array(index.hnsw.levels)
np.bincount(levels)

assign_probas, cum_nneighbor_per_level = set_default_probas(32, 0.0000001)
assign_probas, cum_nneighbor_per_level

chosen_levels = []
rng = np.random.default_rng(12345)
for _ in range(1_000_000):
    chosen_levels.append(random_level(assign_probas, rng))
np.bincount(chosen_levels)


del index
index = faiss.IndexHNSWFlat(d, 32)
index.hnsw.efConstruction = efConstruction
index.add(xb[:1_000])


levels = faiss.vector_to_array(index.hnsw.levels)
np.bincount(levels)

#----------------------------------------------
#Testing Faiss Parameters
#----------------------------------------------

recall_idx = []

index = faiss.IndexFlatL2(d)
index.add(xb)
D, recall_idx = index.search(xq_full[:1000], k=1)
import os

def get_memory(index):
    faiss.write_index(index, './temp.index')
    file_size = os.path.getsize('./temp.index')
    os.remove('./temp.index')
    return file_size
import pandas as pd
from tqdm.auto import trange
from datetime import datetime

results = pd.DataFrame({
    'M': [],
    'efConstruction': [],
    'efSearch': [],
    'recall@1': [],
    'build_time': [],
    'search_time': [],
    'memory_usage': []
})

for epoch in range(3):
    for M_bit in range(1, 10):
        M = 2 ** M_bit
        print(M)
        for ef_bit in trange(1, 6):
            efConstruction = 2 ** ef_bit
            index = faiss.IndexHNSWFlat(d, M)
            index.efConstruction = efConstruction
            start = datetime.now()
            index.add(xb)
            build_time = (datetime.now() - start).microseconds
            memory_usage = get_memory(index)
            for efSearch in [2, 4, 8, 16, 32]:
                index.efSearch = efSearch
                start = datetime.now()
                D, I = index.search(xq_full[:1000], k=1)
                search_time = (datetime.now() - start).microseconds
                recall = sum(I == recall_idx)[0]
                ##results = results.append({
                ##    'M': M,
                ##    'efConstruction': efConstruction,
                ##    'efSearch': efSearch,
                ##    'recall@1': recall,
                ##    'build_time': build_time,
                ##    'search_time': search_time,
                ##    'memory_usage': memory_usage
                ##}, ignore_index=True)
                results.loc[len(results)] = {
                    'M': M,
                    'efConstruction': efConstruction,
                    'efSearch': efSearch,
                    'recall@1': recall,
                    'build_time': build_time,
                    'search_time': search_time,
                    'memory_usage': memory_usage
                }

            del index

results.to_csv('./results.csv', sep='|', index=False)
import pandas as pd

results = pd.read_csv('./results.csv', sep='|')
results.head()

import matplotlib.pyplot as plt
import seaborn as sns
for efConstruction in [2, 16, 64]:
    sns.lineplot(data=results[results['efConstruction'] == efConstruction], x='efSearch', y='recall@1', hue='M')
    plt.show()

for efConstruction in [2, 16, 64]:
    sns.lineplot(data=results[results['efConstruction'] == efConstruction], x='efSearch', y='search_time', hue='M')
    plt.yscale('log')
    plt.ylim(0, 500_000)
    plt.show()

sns.lineplot(data=results, x='M', y='memory_usage')

