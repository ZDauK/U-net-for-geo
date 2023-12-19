import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR
from torch import nn, optim


def loss_fn(y, y_hat):
    MSE_loss = nn.MSELoss(reduction='mean')
    mse_loss = MSE_loss(y, y_hat)
    loss = mse_loss
    return loss


def R2(x, y):
    x1 = x.reshape(x.shape[0], -1)
    y1 = y.reshape(y.shape[0], -1)
    x1 = torch.tensor(x1, requires_grad=False).to('cpu')
    x1 = x1.numpy()
    y1 = torch.tensor(y1, requires_grad=False).to('cpu')
    y1 = y1.numpy()
    x_mean = np.mean(x1, axis=0).reshape(-1, 1)
    over_num = np.sum(np.power(x1 - y1, 2))
    over_num = np.sum(over_num)
    under_sum = np.sum(np.power(x1 - x_mean, 2))
    R2 = over_num/under_sum
    R2_mean = np.mean(R2)
    return R2


def train(net, train_dataloader, valid, device, epochs,
          lr, optim='adam', init=True, scheduler_type='Step'):
    def init_xavier(m):
        if type(m) == nn.Linear:
            nn.init.xavier_normal_(m.weight)

    if init:
        net.apply(init_xavier)

    if optim == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=0)
    else:
        raise TypeError

    if scheduler_type == 'Step':
        scheduler = StepLR(optimizer, step_size=100, gamma=0.97)

    train_losses = []
    train_acc = []
    eval_acc = []

    best_acc = 0.0

    for epoch in range(epochs):
        print("------This is {} epoch------".format(epoch+1))
        train_loss = 0.0
        net.train()
        for x, y in train_dataloader:
            x = x.to(device)
            y = y.to(device)
            output = net(x)
            Loss = loss_fn(y, output)
            train_loss += Loss
            Loss.backward()
            optimizer.step()

        train_loss = torch.tensor(train_loss, requires_grad=False).to('cpu')
        train_loss = train_loss.numpy()
        train_losses.append(train_loss)
        print('The loss on valid set is {}'.format(train_losses[epoch]))
        scheduler.step()
        # net.eval()
        # eval_loss = []
        # eval_R2 = []
        # with torch.no_grad():
        #     valid_x = valid
        #     output, mu, log_var = net(valid_x)
        #     eval_loss.append(loss_fn(valid_x, output, mu, log_var))
        #     R2 = R2(valid_x, output)
        #     eval_R2.append(R2)
        #
        # print('The R2 on valid set is {}'.format(R2))
        # print('The loss on valid set is {}'.format(eval_loss[epoch]))
    return train_losses





