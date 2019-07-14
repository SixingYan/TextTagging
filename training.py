from tqdm import tqdm
tqdm.pandas(desc='Progress')
from const import ix_to_tag


def train(train_loader,
          model, lossfunc, optimizer, epoch_num=1):

    for epoch in range(epoch_num):
        model.train()
        for x_batch, y_batch in tqdm(train_loader):  # , disable=True):
            model.zero_grad()
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = lossfunc(logits, y_batch)
            loss.backward()
            optimizer.step()
        '''
        model.eval()
        for i, (x_batch, y_batch) in enumerate(valid_loader):
            pass
        '''
    return model


def valid():
    """
    交叉训练时，用于验证当前模型的效果
    """
    pass


def test(tst_tensor, model):
    """
    使用测试集给出预测，
    返回预测结果
    """
    pred_tags = []
    model.eval()
    for x_tensor in tqdm(tst_tensor, disable=True):

        logits = model(x_tensor)
        tagix = logits.argmax(dim=1).numpy()
        tag = [ix_to_tag[ix] for ix in tagix]
        pred_tags.append(tag)
    return pred_tags
