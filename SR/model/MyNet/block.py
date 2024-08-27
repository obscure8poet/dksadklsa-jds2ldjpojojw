# sys.path.append('../')
from model.MyNet.esa import ESA
# from block.module import main
import torch
from torch import nn, einsum
from einops import rearrange
import torch.nn.functional as F
from model.MyNet.layernorm import LayerNorm2d


class block(nn.Module):
    def __init__(self, channel_num=64):
        super().__init__()

        self.residual_layer = main()
        esa_channel = max(channel_num // 4, 16)
        self.esa = ESA(channel_num, esa_channel)

    def forward(self, x):
        out = self.residual_layer(x)
        out = out + x
        return self.esa(out)


class Conv_PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = LayerNorm2d(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x)) + x


class Gated_Conv_FeedForward(nn.Module):
    def __init__(self, dim, mult=1, bias=False, dropout=0.):
        super().__init__()

        hidden_features = int(dim * mult)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, padding=1, groups=hidden_features * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


def MBConv(
        dim_in,
        dim_out,
        *,
        downsample,
        expansion_rate=4,
        dropout=0.
):
    hidden_dim = int(expansion_rate * dim_out)
    stride = 2 if downsample else 1

    net = nn.Sequential(
        nn.Conv2d(dim_in, hidden_dim, 1),
        nn.GELU(),
        nn.PixelShuffle(2),
        nn.Conv2d(hidden_dim // 4, hidden_dim // 4, 3, stride=2, padding=1, groups=hidden_dim // 4),
        nn.GELU(),
        nn.Conv2d(hidden_dim // 4, dim_out, 1),
    )

    return net


class Block_Attention(nn.Module):
    def __init__(
            self,
            dim,
            heads,
            bias=False,
            dropout=0.,
            window_size=7
    ):
        super(Block_Attention, self).__init__()
        self.heads = heads

        self.ps = window_size

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        tempw = w
        temph = h

        qkv = self.qkv_dwconv(self.qkv(x))

        while tempw % (self.ps * self.ps) != 0:
            tempw += 1
        while temph % (self.ps * self.ps) != 0:
            temph += 1

        qkv = F.interpolate(qkv, size=(temph, tempw), mode='bilinear', align_corners=True)

        qkv = qkv.chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, 'b (head d) (x h ph) (y w pw) -> (b x y) head d (w ph) (pw h)', w=self.ps, h=self.ps,
                                ph=self.ps, pw=self.ps, head=self.heads, x=temph // self.ps // self.ps,
                                y=tempw // self.ps // self.ps), qkv)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, '(b x y) head d (w ph) (pw h) -> b (head d) (x h ph) (y w pw)', w=self.ps, h=self.ps,
                        ph=self.ps, pw=self.ps, head=self.heads, x=temph // self.ps // self.ps,
                        y=tempw // self.ps // self.ps)

        out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=False)
        out = self.project_out(out)
        return out


class Channel_Attention(nn.Module):
    def __init__(
            self,
            dim,
            heads,
            bias=False,
            dropout=0.,
            window_size=7
    ):
        super(Channel_Attention, self).__init__()
        self.heads = heads
        self.ps = window_size

        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        tempw = w
        temph = h

        qkv = self.qkv_dwconv(self.qkv(x))

        while tempw % (self.ps * self.ps) != 0:
            tempw += 1
        while temph % (self.ps * self.ps) != 0:
            temph += 1

        qkv = F.interpolate(qkv, size=(temph, tempw), mode='bilinear', align_corners=True)

        qkv = qkv.chunk(3, dim=1)

        q, k, v = map(
            lambda t: rearrange(t, 'b (head d) (x h ph) (y w pw) -> (b x y) head d (w ph pw h)', w=self.ps, h=self.ps,
                                ph=self.ps, pw=self.ps, head=self.heads, x=temph // self.ps // self.ps,
                                y=tempw // self.ps // self.ps), qkv)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)

        out = rearrange(out, '(b x y) head d (w ph pw h) -> b (head d) (x h ph) (y w pw)', w=self.ps, h=self.ps,
                        ph=self.ps, pw=self.ps, head=self.heads, x=temph // self.ps // self.ps,
                        y=tempw // self.ps // self.ps)

        out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=False)

        out = self.project_out(out)

        return out

class bigKernelConv(nn.Module):
    def __init__(self, dim=64, dim_feat=16):
        super().__init__()
        self.covn1 = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=13, padding=6, groups=dim),
            nn.BatchNorm2d(dim)
        )

        self.covn2 = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1, groups=dim),
            nn.BatchNorm2d(dim)
        )
        self.fuseConv = nn.Conv2d(in_channels=dim * 2, out_channels=dim * 2, kernel_size=1)
        self.conv_x1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1)
        self.conv_x2 = nn.Sequential(
            nn.Conv2d(dim, dim_feat, kernel_size=1),
            nn.Conv2d(dim_feat, dim_feat, kernel_size=5, padding=2, stride=2),
            nn.MaxPool2d(kernel_size=7, stride=3),
            nn.Conv2d(dim_feat, dim, kernel_size=1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.covn1(x)
        x2 = self.covn2(x)
        x1, x2 = self.fuseConv(torch.cat((x1, x2), dim=1)).chunk(2, dim=1)
        x1 = self.conv_x1(x1)
        x_ = F.interpolate(self.conv_x2(x2), (x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        out = x1 * self.sigmoid(torch.add(x2, x_))
        return out
class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x)) + x

class main(nn.Module):
    def __init__(self, channel_num=64, window_size=8, dropout=0.0, with_pe=True):
        super().__init__()

        self.layer = nn.Sequential(
            MBConv(channel_num, channel_num, downsample=False, expansion_rate=1),
            Conv_PreNormResidual(channel_num, Gated_Conv_FeedForward(dim=channel_num, dropout=dropout)),
        )
        # 8 * 8
        self.block1 = nn.Sequential(
            Conv_PreNormResidual(channel_num, Block_Attention(dim=channel_num, heads=4, dropout=dropout, window_size=window_size)),
        )
        # lr * lr
        self.block2 = nn.Sequential(
            Conv_PreNormResidual(channel_num, Block_Attention(dim=channel_num, heads=4, dropout=dropout, window_size=window_size // 2)),
        )

        self.bigKernelConv = Conv_PreNormResidual(channel_num, bigKernelConv(dim=channel_num))
        self.GateConv = Conv_PreNormResidual(channel_num, Gated_Conv_FeedForward(dim=channel_num, dropout=dropout))

        # 通道注意力
        self.channels_Attention = nn.Sequential(
            # channel-like attention
            Conv_PreNormResidual(channel_num,  Channel_Attention(dim=channel_num, heads=4, dropout=dropout, window_size=window_size)),
            Conv_PreNormResidual(channel_num, Gated_Conv_FeedForward(dim=channel_num, dropout=dropout)),
        )

    def forward(self, x):
        x = self.layer(x)
        x1 = self.block1(x)
        x2 = self.block2(x)

        y1 = self.GateConv(torch.add(x1, x2))

        # 加入门控卷积
        y2 = self.channels_Attention(y1)

        out = self.bigKernelConv(y2)

        return out
