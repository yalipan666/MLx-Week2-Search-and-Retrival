#
#
#
import datasets
import hashlib
import pickle
import random


#
#
#
ds = datasets.load_dataset('microsoft/ms_marco', 'v1.1')


#
#
#
for split in ds.keys():
    docs = {}
    qrys = {}
    for e in ds[split]:
        qrys[e['query_id']] = { 'text': e['query'], 'docs': [] }
        for p in e['passages']['passage_text']:
            hsh = hashlib.sha256(p.encode()).hexdigest()[:16]
            if hsh not in docs: docs[hsh] = p
            qrys[e['query_id']]['docs'].append(hsh)
    with open(f'./corpus/docs_{split}.pkl', 'wb') as f: pickle.dump(docs, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f'./corpus/qrys_{split}.pkl', 'wb') as f: pickle.dump(qrys, f, protocol=pickle.HIGHEST_PROTOCOL)


# #A hash is a fixed-length string (or number) generated from data (like text) using 
# a mathematical function (a hash function).
# Properties:
# Same input always gives the same output.
# Even a tiny change in input gives a very different output.
# Used for quick comparison, deduplication, and lookup.
# #
# #The full SHA-256 hash is 64 hex characters (256 bits). Taking the first 16 characters gives you 64 bits.
# Why is this usually OK?
# 64 bits is still a huge number of possible values (2^64 ≈ 18 quintillion).
# The chance of two different passages having the same first 16 characters (a "collision") 
# is extremely low for most practical dataset sizes.
# print("len(qrys)", len(qrys))
# print("len(docs)", len(docs))
# print(list(docs.keys())[:10])
# print(list(qrys.keys())[:10])


#
#
#
print(random.choice(list(docs.values())))
print(qrys[9655])