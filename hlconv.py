
import torch
import torch.nn as nn
def normlayer(x):
    return nn.GroupNorm(2,x)
def conv_bn(inp, oup, k=3, s=1, BatchNorm2d=normlayer):
    return nn.Sequential(
        nn.Conv2d(inp, oup, k, s, padding=k//2, bias=True),
        BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )

def dep_sep_conv_bn(inp, oup, k=3, s=1, BatchNorm2d=normlayer):
    return nn.Sequential(
        nn.Conv2d(inp, inp, k, s, padding=k//2, groups=inp, bias=True),
        BatchNorm2d(inp),
        nn.ReLU(inplace=True),
        nn.Conv2d(inp, oup, 1, 1, padding=0, bias=True),
        BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )

hlconv = {
    'std_conv': conv_bn,
    'dep_sep_conv': dep_sep_conv_bn
}