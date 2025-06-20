import torch
import pickle
import random

class Triplets(torch.utils.data.Dataset):
    def __init__(self, embs, tkns, split='train'):
        self.embs = embs
        self.tkns = tkns
        self.qrys = pickle.load(open(f'./corpus/qrys_{split}.pkl', 'rb'))
        self.docs = pickle.load(open(f'./corpus/docs_{split}.pkl', 'rb'))
        self.q_keys = list(self.qrys.keys())
        self.d_keys = list(self.docs.keys())
        with open('./corpus/tokeniser.pkl', 'rb') as f: tkns = pickle.load(f)
        self.words_to_ids = tkns['words_to_ids']

    def __len__(self):
        return len(self.qrys)

    def __getitem__(self, idx):
        qry = self.qrys[self.q_keys[idx]]
        pos = self.docs[qry['docs'][0]]
        neg = self.docs[random.choice(self.d_keys)]
        qry = self.to_emb(qry['text'])
        pos = self.to_emb(pos)
        neg = self.to_emb(neg)
        return qry, pos, neg

    def to_emb(self, text):
        text = self.preprocess(text)
        tkns = [self.tkns[t] for t in text if t in self.tkns]
        if len(tkns) == 0: return
        device = self.embs.weight.device
        tkns = torch.tensor(tkns).to(device)
        embs = self.embs(tkns)
        return embs  # return the full sequence, not mean

    def preprocess(self, text):
        text = text.lower()
        text = text.replace('.',  ' <PERIOD> ')
        text = text.replace(',',  ' <COMMA> ')
        text = text.replace('"',  ' <QUOTATION_MARK> ')
        text = text.replace('"',  ' <QUOTATION_MARK> ')
        text = text.replace('"',  ' <QUOTATION_MARK> ')
        text = text.replace(';',  ' <SEMICOLON> ')
        text = text.replace('!',  ' <EXCLAMATION_MARK> ')
        text = text.replace('?',  ' <QUESTION_MARK> ')
        text = text.replace('(',  ' <LEFT_PAREN> ')
        text = text.replace(')',  ' <RIGHT_PAREN> ')
        text = text.replace('--', ' <HYPHENS> ')
        text = text.replace('?',  ' <QUESTION_MARK> ')
        text = text.replace(':',  ' <COLON> ')
        text = text.replace("'",  ' <APOSTROPHE> ')
        text = text.replace("'",  ' <APOSTROPHE> ')
        return text.split() 