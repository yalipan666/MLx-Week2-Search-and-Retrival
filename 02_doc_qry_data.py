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
docs = {}
qrys = {}


#
#
#
for s in ds.keys():
  for e in ds[s]:
    qrys[e['query_id']] = { 'text': e['query'], 'docs': [] } # create an empty entry for each query
    for p in e['passages']['passage_text']:
      hsh = hashlib.sha256(p.encode()).hexdigest()[:16]
      # hash is quicker than text for lookup, and make sure no duplication for docs
      # p.encode() turn text into bytes, which is required by hash
      if hsh not in docs: docs[hsh] = p
      # Instead of storing the full text as a key (which is slow and memory-intensive), 
      # you use a short, fixed-length hash as the key.
      qrys[e['query_id']]['docs'].append(hsh)


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
print(docs['fdb37125d43984c2'])
print(random.choice(list(docs.values())))
print(qrys[9655])


#
#
#
with open('./corpus/docs.pkl', 'wb') as f: pickle.dump(docs, f, protocol=pickle.HIGHEST_PROTOCOL)
with open('./corpus/qrys.pkl', 'wb') as f: pickle.dump(qrys, f, protocol=pickle.HIGHEST_PROTOCOL)