import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from torch_deform_conv.layers import ConvOffset2D


def fixed_padding(inputs, kernel_size, dilation):
    kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    padded_inputs = F.pad(inputs, (pad_beg, pad_end, pad_beg, pad_end))
    return padded_inputs


class SeparableConv2d(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, bias=False, BatchNorm=None):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size, stride, 0, dilation, groups=inplanes, bias=bias)
        self.bn = BatchNorm(inplanes)
        self.pointwise = nn.Conv2d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = fixed_padding(x, self.conv1.kernel_size[0], dilation=self.conv1.dilation[0])
        x = self.conv1(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x


# relu first
class SeparableConv2d2(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, bias=False, BatchNorm=None):
        super(SeparableConv2d2, self).__init__()

        self.relu = nn.ReLU(inplace=False)
        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size, stride, 0, dilation, groups=inplanes, bias=bias)
        self.bn = BatchNorm(inplanes)
        self.pointwise = nn.Conv2d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = fixed_padding(x, self.conv1.kernel_size[0], dilation=self.conv1.dilation[0])
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pointwise(x)
        return x


# deformable
class SeparableConv2d_deform(nn.Module):
    def __init__(self, inplanes, planes, bias=False, BatchNorm=None):
        super(SeparableConv2d_deform, self).__init__()

        self.relu = nn.ReLU(inplace=False)
        self.conv1 = ConvOffset2D(inplanes)
        self.bn = BatchNorm(inplanes)
        self.pointwise = nn.Conv2d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = fixed_padding(x, self.conv1.kernel_size[0], dilation=self.conv1.dilation[0])
        x = self.conv1(x)
        x = x[:, :, 1:-1, 1:-1]
        print("change", x.shape)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self, inplanes, planes, dilation=1, BatchNorm=None):
        super(Block, self).__init__()

        self.relu = nn.ReLU(inplace=False)

        self.SeparableConv2d1 = SeparableConv2d(inplanes, planes, 3, 1, dilation, BatchNorm=BatchNorm)
        self.bn1 = BatchNorm(planes)

        self.SeparableConv2d2 = SeparableConv2d(planes, planes, 3, 1, dilation, BatchNorm=BatchNorm)
        self.bn2 = BatchNorm(planes)

        self.SeparableConv2d3 = SeparableConv2d(planes, planes, 3, 1, dilation, BatchNorm=BatchNorm)
        self.bn3 = BatchNorm(planes)

    def forward(self, inp):
        x = self.relu(inp)
        x = self.SeparableConv2d1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.SeparableConv2d2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.SeparableConv2d3(x)
        x = self.bn3(x)
        x = x + inp

        return x


class Block_deform(nn.Module):
    def __init__(self, inplanes, planes, dilation=1, BatchNorm=None):
        super(Block_deform, self).__init__()

        self.relu = nn.ReLU(inplace=False)

        self.SeparableConv2d1 = SeparableConv2d_deform(inplanes, planes, BatchNorm=BatchNorm)
        self.bn1 = BatchNorm(planes)

        self.SeparableConv2d2 = SeparableConv2d_deform(planes, planes, BatchNorm=BatchNorm)
        self.bn2 = BatchNorm(planes)

        self.SeparableConv2d3 = SeparableConv2d_deform(planes, planes, BatchNorm=BatchNorm)
        self.bn3 = BatchNorm(planes)

    def forward(self, inp):
        print('inp', inp.shape)
        x = self.relu(inp)
        x = self.SeparableConv2d1(x)
        print('1', x.shape)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.SeparableConv2d2(x)
        print('2', x.shape)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.SeparableConv2d3(x)
        print('3', x.shape)
        x = self.bn3(x)
        x = x + inp

        return x


# Entry flow 1
class Block1(nn.Module):
    def __init__(self, inplanes, planes, dilation=1, BatchNorm=None):
        super(Block1, self).__init__()

        self.relu = nn.ReLU(inplace=False)

        self.SeparableConv2d1 = SeparableConv2d(inplanes, planes, 3, 1, dilation, BatchNorm=BatchNorm)
        self.bn1 = BatchNorm(planes)

        self.SeparableConv2d2 = SeparableConv2d(planes, planes, 3, 1, dilation, BatchNorm=BatchNorm)
        self.bn2 = BatchNorm(planes)

        self.SeparableConv2d3 = SeparableConv2d(planes, planes, 3, 2, dilation, BatchNorm=BatchNorm)
        self.bn3 = BatchNorm(planes)

        self.skip = nn.Conv2d(inplanes, planes, 1, stride=2, bias=False)
        self.skipbn = BatchNorm(planes)

    def forward(self, inp):

        x = self.relu(inp)
        x = self.SeparableConv2d1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.SeparableConv2d2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.SeparableConv2d3(x)
        x = self.bn3(x)

        skip = self.skip(inp)
        skip = self.skipbn(skip)

        x = x + skip

        return x


# Entry flow 2
class Block2(nn.Module):
    def __init__(self, inplanes, planes, dilation=1, BatchNorm=None):
        super(Block2, self).__init__()

        self.relu = nn.ReLU(inplace=False)

        self.SeparableConv2d1 = SeparableConv2d(inplanes, planes, 3, 1, dilation, BatchNorm=BatchNorm)
        self.bn1 = BatchNorm(planes)

        self.SeparableConv2d2 = SeparableConv2d(planes, planes, 3, 1, dilation, BatchNorm=BatchNorm)
        self.bn2 = BatchNorm(planes)

        self.SeparableConv2d3 = SeparableConv2d(planes, planes, 3, 2, dilation, BatchNorm=BatchNorm)
        self.bn3 = BatchNorm(planes)

        self.skip = nn.Conv2d(inplanes, planes, 1, stride=2, bias=False)
        self.skipbn = BatchNorm(planes)

    def forward(self, inp):

        x = self.relu(inp)
        x = self.SeparableConv2d1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.SeparableConv2d2(x)
        x = self.bn2(x)
        low_level_feat = x
        x = self.relu(x)
        x = self.SeparableConv2d3(x)
        x = self.bn3(x)

        skip = self.skip(inp)
        skip = self.skipbn(skip)

        x = x + skip

        return x, low_level_feat


# Entry flow 3
class Block3(nn.Module):
    def __init__(self, inplanes, planes, stride, dilation=1, BatchNorm=None):
        super(Block3, self).__init__()

        self.relu = nn.ReLU(inplace=False)

        self.SeparableConv2d1 = SeparableConv2d(inplanes, planes, 3, 1, dilation, BatchNorm=BatchNorm)
        self.bn1 = BatchNorm(planes)

        self.SeparableConv2d2 = SeparableConv2d(planes, planes, 3, 1, dilation, BatchNorm=BatchNorm)
        self.bn2 = BatchNorm(planes)

        self.SeparableConv2d3 = SeparableConv2d(planes, planes, 3, stride, dilation, BatchNorm=BatchNorm)
        self.bn3 = BatchNorm(planes)

        self.skip = nn.Conv2d(inplanes, planes, 1, stride, bias=False)
        self.skipbn = BatchNorm(planes)

    def forward(self, inp):

        x = self.relu(inp)
        x = self.SeparableConv2d1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.SeparableConv2d2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.SeparableConv2d3(x)
        x = self.bn3(x)

        skip = self.skip(inp)
        skip = self.skipbn(skip)

        x = x + skip

        return x


# Exit flow 4
class Block4(nn.Module):
    def __init__(self, inplanes, planes, dilation=1, BatchNorm=None):
        super(Block4, self).__init__()

        self.relu = nn.ReLU(inplace=False)

        self.SeparableConv2d1 = SeparableConv2d(inplanes, inplanes, 3, 1, dilation, BatchNorm=BatchNorm)
        self.bn1 = BatchNorm(inplanes)

        self.SeparableConv2d2 = SeparableConv2d(inplanes, planes, 3, 1, dilation, BatchNorm=BatchNorm)
        self.bn2 = BatchNorm(planes)

        self.SeparableConv2d3 = SeparableConv2d(planes, planes, 3, 1, dilation, BatchNorm=BatchNorm)
        self.bn3 = BatchNorm(planes)

        self.skip = nn.Conv2d(inplanes, planes, 1, stride=1, bias=False)
        self.skipbn = BatchNorm(planes)

    def forward(self, inp):

        x = self.relu(inp)
        x = self.SeparableConv2d1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.SeparableConv2d2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.SeparableConv2d3(x)
        x = self.bn3(x)

        skip = self.skip(inp)
        skip = self.skipbn(skip)

        x = x + skip

        return x


class AlignedXception(nn.Module):
    """
    Modified Alighed Xception
    """
    def __init__(self, output_stride, BatchNorm, pretrained=False):
        super(AlignedXception, self).__init__()

        if output_stride == 16:
            entry_block3_stride = 2
            middle_block_dilation = 1
            exit_block_dilations = (1, 2)
        elif output_stride == 8:
            entry_block3_stride = 1
            middle_block_dilation = 2
            exit_block_dilations = (2, 4)
        else:
            raise NotImplementedError

        # Entry flow
        self.conv1 = nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False)
        self.bn1 = BatchNorm(32)
        self.relu = nn.ReLU(inplace=False)

        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=False)
        self.bn2 = BatchNorm(64)

        self.block1 = Block1(64, 128, BatchNorm=BatchNorm)
        self.block2 = Block2(128, 256, BatchNorm=BatchNorm)
        self.block3 = Block3(256, 728, stride=entry_block3_stride, BatchNorm=BatchNorm)

        # Middle flow
        self.block4  = Block(728, 728, dilation=middle_block_dilation, BatchNorm=BatchNorm)
        self.block5  = Block(728, 728, dilation=middle_block_dilation, BatchNorm=BatchNorm)
        self.block6  = Block(728, 728, dilation=middle_block_dilation, BatchNorm=BatchNorm)
        self.block7  = Block(728, 728, dilation=middle_block_dilation, BatchNorm=BatchNorm)
        self.block8  = Block(728, 728, dilation=middle_block_dilation, BatchNorm=BatchNorm)
        self.block9  = Block(728, 728, dilation=middle_block_dilation, BatchNorm=BatchNorm)
        self.block10 = Block(728, 728, dilation=middle_block_dilation, BatchNorm=BatchNorm)
        self.block11 = Block(728, 728, dilation=middle_block_dilation, BatchNorm=BatchNorm)
        self.block12 = Block(728, 728, dilation=middle_block_dilation, BatchNorm=BatchNorm)
        self.block13 = Block(728, 728, dilation=middle_block_dilation, BatchNorm=BatchNorm)
        self.block14 = Block(728, 728, dilation=middle_block_dilation, BatchNorm=BatchNorm)
        self.block15 = Block(728, 728, dilation=middle_block_dilation, BatchNorm=BatchNorm)
        self.block16 = Block(728, 728, dilation=middle_block_dilation, BatchNorm=BatchNorm)
        self.block17 = Block(728, 728, dilation=middle_block_dilation, BatchNorm=BatchNorm)
        self.block18 = Block(728, 728, dilation=middle_block_dilation, BatchNorm=BatchNorm)
        self.block19 = Block_deform(728, 728, dilation=middle_block_dilation, BatchNorm=BatchNorm)

        # Exit flow
        self.block20 = Block4(728, 1024, dilation=exit_block_dilations[0], BatchNorm=BatchNorm)

        self.conv3 = SeparableConv2d2(1024, 1536, 3, stride=1, dilation=exit_block_dilations[1], BatchNorm=BatchNorm)
        self.bn3 = BatchNorm(1536)

        self.conv4 = SeparableConv2d2(1536, 1536, 3, stride=1, dilation=exit_block_dilations[1], BatchNorm=BatchNorm)
        self.bn4 = BatchNorm(1536)

        self.conv5 = SeparableConv2d2(1536, 2048, 3, stride=1, dilation=exit_block_dilations[1], BatchNorm=BatchNorm)
        self.bn5 = BatchNorm(2048)

        # Init weights
        self._init_weight()

        # Load pretrained model
        if pretrained:
            self._load_pretrained_model()

    def forward(self, x):

        # Entry flow
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.block1(x)
        x, low_level_feat = self.block2(x)
        x = self.block3(x)

        # Middle flow
        x = self.block4(x)
        x4 = x
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = self.block13(x)
        x = self.block14(x)
        x = self.block15(x)
        x = self.block16(x)
        x = self.block17(x)
        x = self.block18(x)
        x = self.block19(x)
        x = x + x4

        # Exit flow
        x = self.block20(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)

        return x, low_level_feat

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _load_pretrained_model(self):
        pretrain_dict = model_zoo.load_url('http://data.lip6.fr/cadene/pretrainedmodels/xception-b5690688.pth')
        model_dict = {}
        state_dict = self.state_dict()

        for k, v in pretrain_dict.items():
            if k in model_dict:
                if 'pointwise' in k:
                    v = v.unsqueeze(-1).unsqueeze(-1)
                if k.startswith('block11'):
                    model_dict[k] = v
                    model_dict[k.replace('block11', 'block12')] = v
                    model_dict[k.replace('block11', 'block13')] = v
                    model_dict[k.replace('block11', 'block14')] = v
                    model_dict[k.replace('block11', 'block15')] = v
                    model_dict[k.replace('block11', 'block16')] = v
                    model_dict[k.replace('block11', 'block17')] = v
                    model_dict[k.replace('block11', 'block18')] = v
                    model_dict[k.replace('block11', 'block19')] = v
                elif k.startswith('block12'):
                    model_dict[k.replace('block12', 'block20')] = v
                elif k.startswith('bn3'):
                    model_dict[k] = v
                    model_dict[k.replace('bn3', 'bn4')] = v
                elif k.startswith('conv4'):
                    model_dict[k.replace('conv4', 'conv5')] = v
                elif k.startswith('bn4'):
                    model_dict[k.replace('bn4', 'bn5')] = v
                else:
                    model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)


if __name__ == "__main__":
    import torch
    model = AlignedXception(BatchNorm=nn.BatchNorm2d, pretrained=True, output_stride=8)
    input = torch.rand(1, 3, 512, 512)
    output, low_level_feat = model(input)
    print(output.size())
    print(low_level_feat.size())