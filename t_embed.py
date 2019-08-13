f'''
这里用来训练词向量
'''
import util
from torch import optim
import torch
import numpy as np
from m_elmo import ELMoChar
from tqdm import tqdm

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
"""这里做预训练 """
X,word_to_ix = util.load_corp('datagrad')
print('complete loading')
num_epoch = 5
num_layer = 3
embed_dim = 64
hid_dim = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = ELMoChar(len(word_to_ix),num_layer,embed_dim,hid_dim)
model.train().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-6)

for epoch in range(num_epoch):    
    avg_loss = 0
    for i in tqdm(np.random.permutation(list(range(len(X))))):
        x_ts = torch.tensor(X[i], dtype=torch.long, device=device)
        x_batch = torch.unsqueeze(x_ts,dim=1)
        loss = model.loss(x_batch)
        loss.backward()
        optimizer.step()
        avg_loss += loss.item() / len(X)
    print('Epoch {}/{} \t avg_loss {:.4f}'.format(
            epoch + 1, num_epoch, avg_loss))
    util.savemodel(model, 'elmo_{}.pytorch'.format(epoch+1))
