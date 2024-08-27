import torch
import torch.nn as nn
from model.MyNet.block import block
from model.MyNet.pixelshuffle import pixelshuffle_block
import torch.nn.functional as F


class MyNet(nn.Module):
    def __init__(self, args, num_in_ch=3, num_out_ch=3):
        super().__init__()

        self.bias = args.bias
        self.up_scale = args.up_scale
        self.num_feat = args.num_feat
        self.block_num = args.block_num
        self.window_size = args.window_size

        residual_layer = []

        for _ in range(self.block_num):
            temp_res = block()
            residual_layer.append(temp_res)
        self.residual_layer = nn.Sequential(*residual_layer)

        self.input = nn.Conv2d(in_channels=num_in_ch, out_channels=self.num_feat, kernel_size=3, padding=1, bias=self.bias)
        self.up = pixelshuffle_block(self.num_feat, num_out_ch, self.up_scale, bias=self.bias)


    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'constant', 0)
        return x

    def forward(self, x):
        H, W = x.shape[2:]
        x = self.check_image_size(x)

        residual = self.input(x)
        out = self.residual_layer(residual)
        out = torch.add(out, residual)
        out = self.up(out)

        out = out[:, :, :H * self.up_scale, :W * self.up_scale]
        return out