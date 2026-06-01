a = "flying fish flew by the space station"
b = "he will not allow you to bring your sticks of dynamite and pet armadillo along"
c = "he figured a few sticks of dynamite were easier than a fishing pole to catch an armadillo"

k = 2
for i in range(len(a) - k+1):
    print(a[i: i+k], end='|')

def shingle(text: str, k: int):
    shingle_set = []
    for i in range(len(text) - k+1):
        shingle_set.append(text[i:i+k])
    return set(shingle_set)

a = shingle(a, k)
b = shingle(b, k)
c = shingle(c, k)
print()
print("a after shigle:")
print(a)

vocab = list(a.union(b).union(c))
print()
print("vocab:")
print(vocab)

a_1hot = [1 if x in a else 0 for x in vocab]
b_1hot = [1 if x in b else 0 for x in vocab]
c_1hot = [1 if x in c else 0 for x in vocab]
print()
print("a_1hot:")
print(a_1hot)

hash_ex = list(range(1, len(vocab)+1))
print()
print("before shuffle, hash_ex:")
print(hash_ex)  # we haven't shuffled yet

from random import shuffle

shuffle(hash_ex)
print()
print("after shuffle, hash_ex:")
print(hash_ex)

print()
print("samples of hash_ex:")
for i in range(1, 5):
    print(f"{i} -> {hash_ex.index(i)}")

for i in range(1, len(vocab)+1):
    idx = hash_ex.index(i)
    signature_val = a_1hot[idx]
    print(f"{i} -> {idx} -> {signature_val}")
    if signature_val == 1:
        print('match!')
        break

hash_funcs = []

for _ in range(20):
    hash_ex = list(range(1, len(vocab)+1))
    shuffle(hash_ex)
    hash_funcs.append(hash_ex)

print()
for i in range(3):
    print(f"hash function {i+1}:")
    print(hash_funcs[i])

signature = []

for func in hash_funcs:
    for i in range(1, len(vocab)+1):
        idx = func.index(i)
        signature_val = a_1hot[idx]
        if signature_val == 1:
            signature.append(idx)
            break

print()
print("signature:")
print(signature)

def create_hash_func(size: int):
    # function for creating the hash vector/function
    hash_ex = list(range(1, size+1))
    shuffle(hash_ex)
    return hash_ex

def build_minhash_func(vocab_size: int, nbits: int):
    # function for building multiple minhash vectors
    hashes = []
    for _ in range(nbits):
        hashes.append(create_hash_func(vocab_size))
    return hashes

# we create 20 minhash vectors
minhash_func = build_minhash_func(len(vocab), 20)

def create_hash(vector: list):
    # use this function for creating our signatures (eg the matching)
    signature = []
    for func in minhash_func:
        for i in range(1, len(vocab)+1):
            idx = func.index(i)
            signature_val = vector[idx]
            if signature_val == 1:
                signature.append(idx)
                break
    return signature

# now create signatures
a_sig = create_hash(a_1hot)
b_sig = create_hash(b_1hot)
c_sig = create_hash(c_1hot)

print()
print("a_sig:")
print(a_sig)
print("b_sig:")
print(b_sig)

def jaccard(a: set, b: set):
    return len(a.intersection(b)) / len(a.union(b))

jaccard(a, b), jaccard(set(a_sig), set(b_sig))

jaccard(a, c), jaccard(set(a_sig), set(c_sig))

jaccard(b, c), jaccard(set(b_sig), set(c_sig))

def split_vector(signature, b):
    assert len(signature) % b == 0
    r = int(len(signature) / b)
    # code splitting signature in b parts
    subvecs = []
    for i in range(0, len(signature), r):
        subvecs.append(signature[i : i+r])
    return subvecs

band_b = split_vector(b_sig, 10)
print()
print("band_b:")
print(band_b)


band_c = split_vector(c_sig, 10)
print("band_c:")
print(band_c)

for b_rows, c_rows in zip(band_b, band_c):
    if b_rows == c_rows:
        print(f"Candidate pair: {b_rows} == {c_rows}")
        # we only need one band to match
        break

band_a = split_vector(a_sig, 10)

for a_rows, b_rows in zip(band_a, band_b):
    if a_rows == b_rows:
        print(f"Candidate pair: {a_rows} == {b_rows}")
        # we only need one band to match
        break

for a_rows, c_rows in zip(band_a, band_c):
    if a_rows == c_rows:
        print(f"Candidate pair: {b_rows} == {c_rows}")
        # we only need one band to match
        break

def probability(s, r, b):
    # s: similarity
    # r: rows (per band)
    # b: number of bands
    return 1 - (1 - s**r)**b
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
results = pd.DataFrame({
    's': [],
    'P': [],
    'r,b': []
})

list0 = []
for s in np.arange(0.01, 1, 0.01):
    total = 100
    for b in [100, 50, 25, 20, 10, 5, 4, 2, 1]:
        r = int(total/b)
        P = probability(s, r, b)
        #results = results.append({
        #    's': s,
        #    'P': P,
        #    'r,b': f"{r},{b}"
        #}, ignore_index=True)

        list0.append({
            's': s,
            'P': P,
            'r,b': f"{r},{b}"
        })

results = pd.DataFrame(list0)

sns.lineplot(data=results, x='s', y='P', hue='r,b')

plt.show()