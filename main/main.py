import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import time
import argparse
import h5py
from U_net import U_net
from train import train
import matplotlib.pyplot as plt

t1 = time.time()


def dataset(path):
    f = h5py.File(path)
    n = 2000
    k = 1000
    x = f['x'][:, :n]
    x = x.T
    x = x.reshape(-1, 60, 60)
    # x = x.reshape(-1, x.shape[2], x.shape[3])
    x = np.expand_dims(x, 1)
    y = f['x'][:, n:n+k]
    x_sample = y.copy()
    y = y.T
    n_y = y.shape[1]
    dim_x = y.shape[-2]
    dim_y = y.shape[-1]

    x = x.reshape(n, -1)
    x_min = np.min(x, axis=0)
    x_max = np.max(x, axis=0)
    x_max = x_max + 1e-7
    # x = (x - np.tile(x_min, (x.shape[0], 1))) / np.tile(x_max - x_min, (x.shape[0], 1))
    # x = x.reshape(n, dim_x, dim_y)
    # x = np.expand_dims(x, 1)

    y = (y - np.tile(x_min, (y.shape[0], 1))) / np.tile(x_max - x_min, (y.shape[0], 1))
    y = y.reshape(k, 60, 60)
    y = np.expand_dims(y, 1)
    print(y.shape)
    return y


def initial_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=3000, help='Iteration number')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--input_channels', type=int, default=1, help='number of original channels')
    parser.add_argument('--hidden_channels', type=list, default=[32, 64, 256, 256], help='numbers of latent channels')
    parser.add_argument('--input_kernel_size', type=list, default=[4, 4, 4, 3], help='kernel size of PhyConvLstm')
    parser.add_argument('--input_stride', type=list, default=[2, 2, 2, 1], help='stride size of PhyConvLstm')
    parser.add_argument('--input_padding', type=list, default=[1, 1, 1, 1], help='padding size of PhyConvLstm')
    parser.add_argument('--num_layers', type=list, default=[3, 2, 1], help='number of layer of PhyConvLstm')
    parser.add_argument('--in_channels2', type=list, default=[10, 20, 30], help='number of in channels of D')
    parser.add_argument('--hidden_channels2', type=list, default=[20, 30, 1], help='number of hidden channels of D')
    parser.add_argument('--kernel_size2', type=list, default=[(2, 9, 9), (2, 9, 9), (1, 9, 9)], help='kernel size of D')
    parser.add_argument('--stride2', type=list, default=[2, 2, 2], help='stride of D')
    parser.add_argument('--padding2', type=list, default=[0, 0, 0], help='padding of D')
    parser.add_argument('--num_layers2', type=int, default=3, help='num layers of D')
    parser.add_argument('--step', type=int, default=1, help='number of simulation steps')
    parser.add_argument('--effective_step', type=list, default=[i for i in range(50)],
                        help='number of effective default steps')
    parser.add_argument('--batch_first', type=bool, default=True, help='Batch size first option')
    parser.add_argument('--bias', type=bool, default=True, help='bias option')
    parser.add_argument('--return_all_layers', type=bool, default=True, help='output type option')
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate of generator optimizer')
    parser.add_argument('--device', type=str, default='cuda', help='The run device')
    parser.add_argument('--alpha', type=float, default=1, help='lambda of central  loss')
    args = parser.parse_args()
    return args


class MyData(Dataset):
    def __init__(self, path):
        f = h5py.File(path)
        n = 2000
        x = f['x'][:, :n]
        x = x.T
        x = x.reshape(-1, 60, 60)
        # x = x.reshape(-1, x.shape[2], x.shape[3])
        x = np.expand_dims(x, 1)
        y = x.copy()
        n_y = y.shape[1]
        dim_x = y.shape[-2]
        dim_y = y.shape[-1]

        x = x.reshape(n, -1)
        x_min = np.min(x, axis=0)
        x_max = np.max(x, axis=0)
        x_max = x_max + 1e-7
        x = (x - np.tile(x_min, (x.shape[0], 1))) / np.tile(x_max - x_min, (x.shape[0], 1))
        x = x.reshape(n, dim_x, dim_y)
        x = np.expand_dims(x, 1)

        y = y.reshape(n, n_y, -1)
        y_min = np.min(y, axis=0)
        y_max = np.max(y, axis=0)
        y_max = y_max + 1e-7
        y = (y - np.tile(y_min, (y.shape[0], 1, 1))) / np.tile(y_max - y_min, (y.shape[0], 1, 1))
        y = y.reshape(n, n_y, dim_x, dim_y)

        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()

        self.x = x
        self.y = y

    def __getitem__(self, item):
        return self.x[item], self.y[item]

    def __len__(self):
        return self.x.size(0)


if __name__ == '__main__':
    path = 'C:/Users/86137/Desktop/test/proxy_model/ViT_FC1/data/data5000.h5'
    arg = initial_arg()
    datasets = MyData(path)
    dataloader = DataLoader(datasets, batch_size=arg.batch_size, shuffle=True)

    net = U_net(
        input_channels=arg.input_channels,
        hidden_channels=arg.hidden_channels,
        input_kernel_size=arg.input_kernel_size,
        input_stride=arg.input_stride,
        input_padding=arg.input_padding,
        num_layers=arg.num_layers,
        step=arg.step,
        effective_step=arg.effective_step
    ).to('cuda')
    valid = 0
    train_loss = train(net, dataloader, valid, arg.device, arg.epoch, arg.lr)
    torch.save(net.state_dict(), 'model_200.pkl')
    # net.load_state_dict(state_dict=torch.load('model_200.pkl'))

    # plot a sample
    x = dataset(path)
    x = x[200, 0, :, :]
    y = x.copy()
    x = np.expand_dims(x, 0)
    x = np.expand_dims(x, 0)
    x = torch.from_numpy(x).float()
    x = x.to('cuda')
    with torch.no_grad():
        y_hat = net(x)
    y_hat = torch.tensor(y_hat, requires_grad=False).to('cpu')
    y_hat = y_hat.numpy()
    y_hat = y_hat.reshape(60, 60)

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(y)
    ax[1].imshow(y_hat)
    plt.show()
