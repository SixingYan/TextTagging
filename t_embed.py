f'''
这里用来训练词向量
'''
import util


#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
"""这里做预训练 """
X = util.load_corp()
trn_X, trn_y, tst, word_to_ix = util.load()
num_epoch = 5
num_layer = 4
embed_dim = 64
hid_dim = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for epoch in range(num_epoch):
    emlo = ELMoChar(len(word_to_ix),num_layer,embed_dim,hid_dim)
    emlo.train().to(device)
    optimizer = optim.Adam(emlo.parameters(), lr=0.1, weight_decay=1e-6)
    for i in :
        x_ts = torch.tensor(X[i], dtype=torch.long, device=device)
        x_batch = torch.unsqueeze(x_ts,dim=1)
        loss = emlo.loss(x_batch)
        loss.backward()
        optimizer.step()

















#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
"""这里做联合训练 """
