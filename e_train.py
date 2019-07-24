trn_X, trn_y, tst, word_to_ix = load()
max_length = max(len(x) for x in trn_X)

def train(trn_X, trn_ixs, model, optimizer):
    avg_loss = 0
    tsize = len(trn_ixs)
    for i in np.random.permutation(trn_ixs): #tqdm(np.random.permutation(trn_ixs)):#
        x = torch.tensor(trn_X[i], dtype=torch.long, device=device)
        y = torch.tensor(trn_y[i], dtype=torch.long, device=device)
        optimizer.zero_grad()
        loss = model.loss(x, y)
        avg_loss += loss.item() / tsize
        loss.backward()
        optimizer.step()
    return model, avg_loss



def evaluate(trn_X, vld_ixs, model):
    vld_loss = 0
    vsize = len(vld_ixs)
    y_preds, y_trues = [], []
    for i in np.random.permutation(vld_ixs):#tqdm(np.random.permutation(vld_ixs)):#
        x = torch.tensor(trn_X[i], dtype=torch.long, device=device)
        y = torch.tensor(trn_y[i], dtype=torch.long, device=device)
        y_pred = model(x)
        loss = model.loss(x, y)
        vld_loss += loss.item() / vsize
        y_preds.append(y_pred[:])
        y_trues.append(trn_y[i])
    return vld_loss, y_preds, y_trues


def main():

    pre_word_embeds = load_pretrained('cbow', word_to_ix)
    
    for sid in range(split_num):
        trn_ixs, vld_ixs = train_test_split(
            list(range(len(trn_X))), test_size=1 / split_num,
            shuffle=True, random_state=sid)
        print('SPLIT {} --------------------'.format(sid + 1))
    model = BiLSTM(len(word_to_ix), embed_dim=128, hid_dim=128,pre_word_embeds=None, use_subembed=False).to(device)
    optimizer = optim.Adam(
        model.parameters(), lr=0.01, weight_decay=1e-6)
    for epoch in range(epoch_num):
        model.train()
        model, avg_loss = train(trn_ixs, model, optimizer)
        model.eval()
        vld_loss, y_preds, y_trues = evaluate(vld_ixs, model)
        print('Epoch {}/{} \t avg_loss {:.4f} \t vld_loss {:.4f} \t'.format(
            epoch + 1, epoch_num, avg_loss, vld_loss))
        evalinfo(y_preds, y_trues)
    savemodel(model, 'bilstm_{}_notinitpresub.pytorch'.format(sid))








