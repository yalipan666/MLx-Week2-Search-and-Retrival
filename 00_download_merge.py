#
#
#
#
import datasets
import requests


#
#
#
#
ds = datasets.load_dataset('microsoft/ms_marco', 'v1.1')
docs = [passage for s in ds.keys() for e in ds[s] for passage in e['passages']['passage_text']]
# dict is non-order object, so the order of the keys are not
#  all the same everytime you run it
qrys = [e['query'] for s in ds.keys() for e in ds[s]]
with open('./corpus/msmarco.txt', 'w', encoding='utf-8') as f: f.write('\n'.join(set(docs + qrys)))
# set() to have a set of "unique" elements
# set(docs + qrys), first all docs then all queries

#
#
#
#
r = requests.get('https://huggingface.co/datasets/ardMLX/text8/resolve/main/text8')
with open('./corpus/text8.txt', 'wb') as f: f.write(r.content)
# Why use binary mode ('wb') instead of text mode ('w')?
# When downloading files from the internet, you often don't 
# know the encoding or file type.
# Writing in binary mode ensures you save the file exactly as it 
# was received, without any encoding/decoding errors or changes.
