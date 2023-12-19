import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from torch.nn.utils import weight_norm


class ResidualBlocks(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlocks, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                               kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                               kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu = nn.ReLU()

    def forward(self, x):
        conv1 = self.conv1(x)
        relu1 = self.relu(conv1)
        conv2 = self.conv2(relu1)
        relu2 = self.relu(x + conv2)
        return relu2


class encoder_block(nn.Module):
    def __init__(self, input_channels, hidden_channels, input_kernel_size,
                 input_stride, input_padding):
        super(encoder_block, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.input_kernel_size = input_kernel_size
        self.input_stride = input_stride
        self.input_padding = input_padding

        self.conv = weight_norm(nn.Conv2d(self.input_channels,
                                          self.hidden_channels, self.input_kernel_size, self.input_stride,
                                          self.input_padding, bias=True, padding_mode='circular'))
        self.norm = nn.BatchNorm2d(self.hidden_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.norm(self.conv(x)))


class decoder_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride, output_padding=0):
        super(decoder_block, self).__init__()
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                               padding=padding, stride=stride, output_padding=1),
            nn.BatchNorm2d(out_channels),
            nn.Sigmoid()
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                               padding=padding, stride=stride, output_padding=0),
            nn.BatchNorm2d(out_channels),
            nn.Sigmoid()
        )
        self.down = nn.Sequential(
            nn.Conv2d(in_channels=in_channels * 2, out_channels=in_channels, kernel_size=(3, 3), padding=(1, 1),
                      stride=(1, 1)),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid()
        )

    def forward(self, x, out):
        x = torch.cat((x, out), dim=1)
        x = self.down(x)
        if x.shape[-1] % 5 != 0:
            x = self.up1(x)
        else:
            x = self.up2(x)
        return x


class U_net(nn.Module):
    def __init__(self, input_channels, hidden_channels, input_kernel_size,
                 input_stride, input_padding, num_layers, step=1, effective_step=None):
        super(U_net, self).__init__()
        if effective_step is None:
            self.effective_step = [1]
            self.input_channels = [input_channels + 1] + hidden_channels
            self.hidden_channels = hidden_channels
            self.decoder_input_channels = self.input_channels[::-1]
            self.decoder_input_channels = self.decoder_input_channels[:-1]
            self.decoder_hidden_channels = self.input_channels[::-1]
            self.input_kernel_size = input_kernel_size
            self.input_stride = input_stride
            self.input_padding = input_padding
            self.num_layers = num_layers
            self.step = step
        else:
            self.effective_step = effective_step
            self.input_channels = [input_channels + 1] + hidden_channels
            self.hidden_channels = hidden_channels
            self.decoder_hidden_channels = self.input_channels[::-1]
            self.decoder_hidden_channels = self.decoder_hidden_channels[2:]
            self.decoder_hidden_channels[-1] = self.decoder_hidden_channels[-1]-1
            self.decoder_input_channels = self.input_channels[::-1]
            self.decoder_input_channels = self.decoder_input_channels[1:-1]
            self.input_kernel_size = input_kernel_size
            self.input_stride = input_stride
            self.input_padding = input_padding
            self.num_layers = num_layers
            self.step = step

        self._all_layers = []
        self.num_encoder_layers = self.num_decoder_layers = num_layers[0]
        self.num_residual_layers = self.num_layers[1]
        self.num_ConvLstm_layers = num_layers[2]

        for i in range(self.num_encoder_layers):
            name = 'encoder{}'.format(i)
            cell = encoder_block(
                input_channels=self.input_channels[i],
                hidden_channels=self.hidden_channels[i],
                input_kernel_size=self.input_kernel_size[i],
                input_stride=self.input_stride[i],
                input_padding=self.input_padding[i]
            )
            setattr(self, name, cell)
            self._all_layers.append(cell)

        for i in range(self.num_encoder_layers, self.num_encoder_layers + self.num_residual_layers):
            name = 'residual{}'.format(i)
            residual_cell = ResidualBlocks(in_channels=self.hidden_channels[-1])
            setattr(self, name, residual_cell)
            self._all_layers.append(residual_cell)

        for i in range(self.num_decoder_layers):
            name = 'decoder{}'.format(i)
            decoder_cell = decoder_block(
                in_channels=self.decoder_input_channels[i],
                out_channels=self.decoder_hidden_channels[i],
                kernel_size=self.input_kernel_size[i],
                stride=self.input_stride[i],
                padding=self.input_padding[i]
            )
            setattr(self, name, decoder_cell)
            self._all_layers.append(decoder_cell)

    def forward(self, x):
        k = x.clone()
        encoder_out = []
        outputs = []
        for step in range(self.step):
            x = torch.cat((x, k), dim=1)

            for i in range(self.num_encoder_layers):
                name = 'encoder{}'.format(i)
                x = getattr(self, name)(x)
                encoder_out.append(x)
            encoder_out = encoder_out[::-1]

            for i in range(self.num_encoder_layers, self.num_encoder_layers + self.num_residual_layers):
                name = 'residual{}'.format(i)
                x = getattr(self, name)(x)

            for i in range(self.num_decoder_layers):
                name = 'decoder{}'.format(i)
                x = getattr(self, name)(encoder_out[i], x)

            outputs.append(x)
        output = torch.cat(outputs, dim=1)
        return output


if __name__ == '__main__':
    x = torch.randn(16, 1, 60, 60).to('cuda')
    steps = 50
    effective_step = [i for i in range(50)]
    model = U_net(
        input_channels=1,
        hidden_channels=[8, 32, 128, 128],
        input_kernel_size=[4, 4, 4, 3],
        input_stride=[2, 2, 2, 1],
        input_padding=[1, 1, 1, 1],
        num_layers=[3, 2, 1],
        step=steps,
        effective_step=effective_step).cuda()
    y = model(x)
    print(y.shape)
