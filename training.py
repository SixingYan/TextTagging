

def train(train_loader, valid_loader, model, lossfunc, optimizer, epoch_num=30):

    for epoch in range(epoch_num):  # again, normally you would NOT do 300 epochs, it is toy data
        model.train()
        for x_batch, y_batch in tqdm(train_loader, disable=True):
            model.zero_grad()
            optimizer.zero_grad()

            tag_scores = model(sentence_in)

            loss = lossfunc(tag_scores, targets)
            loss.backward()
            optimizer.step()

        model.eval()
        for i, (x_batch, y_batch) in enumerate(valid_loader):
            pass

    return model
