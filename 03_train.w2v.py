#
#
#
import torch
import models
import dataset
import datetime
import pickle
# import wandb
import tqdm


#
#
#
torch.manual_seed(42)
ts = datetime.datetime.now().strftime('%Y_%m_%d__%H_%M_%S')


#
#
#
with open('./corpus/tokeniser.pkl', 'rb') as f: tkns = pickle.load(f)
words_to_ids, ids_to_words = tkns['words_to_ids'], tkns['ids_to_words']
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#
#
#Model: Creates a SkipGram (Word2Vec) model with vocabulary size and 128-dimensional embeddings.
w2v = models.SkipGram(voc=len(words_to_ids), emb=128).to(dev)
torch.save(w2v.state_dict(), f'./checkpoints/{ts}.0.w2v.pth')
print('w2v:', sum(p.numel() for p in w2v.parameters())) # 35,998,976
opt = torch.optim.Adam(w2v.parameters(), lr=0.003)
# wandb.init(project='mlx6-week-02-mrc')


#
#
# prepare dataset and dataloader
ds = dataset.Window('./corpus/tokens.txt')
dl = torch.utils.data.DataLoader(ds, batch_size=1024, shuffle=True)


#
#
#
for epoch in range(5):
  prgs = tqdm.tqdm(dl, desc=f"Epoch {epoch + 1}", leave=False)
  for idx, (inpt, trgs) in enumerate(prgs):
    inpt, trgs = inpt.to(dev), trgs.to(dev)
    rand = torch.randint(0, len(words_to_ids), (inpt.size(0), 2)).to(dev)
    # negative sampling, get the negative examples by randomly select 2 words
    opt.zero_grad()
    # this is not for initialization, it rests(zeros out) the gradients of all model parameters
    # In PyTorch, gradients accumulate by default (they are added up each time you call .backward()).
    # If you don’t zero them out at the start of each batch, you’ll be mixing gradients from multiple 
    # batches, which will break your training.

    loss = w2v(inpt, trgs, rand)
    loss.backward()
    opt.step()
    # wandb.log({'loss': loss.item()})
    if idx % 10_000 == 0: torch.save(w2v.state_dict(), f'./checkpoints/{ts}.{epoch}.{idx}.w2v.pth')


    # when generating negative examples using the line ot rand, we generate 2 negative examples,
    # which is difference/dependent of the window size, 
    # How many random words per (center, context) pair, it's a hyperparameter usually between 2-20  
    
    # Suppose your window size is 5 (context = 2 words before and after):
    # For the center word "cat" in "the quick brown cat jumps over", context words are ["quick", "brown", "jumps", "over"].
    # For each (center, context) pair, you generate (say) 2 negative samples, e.g., ["apple", "car"].
    # So, for "cat" and "quick" (positive pair), you might have:
    # Positive: (cat, quick)
    # Negative: (cat, apple), (cat, car)      


#
#
#
# wandb.finish()
