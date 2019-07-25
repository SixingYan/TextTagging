
def test():
    trn_X, trn_y, tst, word_to_ix = util.load()
    # device =
    model = BiLSTM(len(word_to_ix), embed_dim=64, hid_dim=64).to(device)

    model.load_state_dict(torch.load(
        os.path.join(const.MODELPATH, 'bilstm_4_3.pytorch'), map_location='cpu'))
    tsttag = []
    model.eval()
    for i in tqdm(range(len(tst))):
        x = torch.tensor(tst[i], dtype=torch.long)
        y_pred = model(x)
        tsttag.append([ix_to_tag[ix] for ix in y_pred])
    util.output(tsttag, 'bilstm43')

