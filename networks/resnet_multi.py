"""ResNet in PyTorch.
ImageNet-Style ResNet
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
Adapted from: https://github.com/bearpaw/pytorch-classification
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
#from torchvision.models import resnet18, resnet34, resnet50, resnet101             #


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(BasicBlock, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(Bottleneck, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out


class ResNet_inter(nn.Module):
    def __init__(self, block, num_blocks, in_channel=3, zero_init_residual=False):
        super(ResNet_inter, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves
        # like an identity. This improves the model by 0.2~0.3% according to:
        # https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i in range(num_blocks):
            stride = strides[i]
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def adaptive_pool(self, features, attn_from=None):
        # features and attn_from are paired feature maps, of same size
        if attn_from is None:
            attn_from = features
        assert features.size() == attn_from.size()
        N, C, H, W = features.size()
        assert (attn_from >= 0).float().sum() == N*C*H*W
        attention = torch.einsum('nchw,nc->nhw', [attn_from, nn.functional.adaptive_avg_pool2d(attn_from, (1, 1)).view(N, C)])
        attention = attention / attention.view(N, -1).sum(1).view(N, 1, 1).repeat(1, H, W)
        attention = attention.view(N, 1, H, W)
        # output size: N, C
        return (features * attention).view(N, C, -1).sum(2)

    def forward(self, x, layer=100):
        out0 = F.relu(self.bn1(self.conv1(x)))
        out1 = self.layer1(out0)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out = self.adaptive_pool(out4) #self.avgpool(out4)
        out = torch.flatten(out, 1)
        return out1, out2, out3, out4, out


def resnet18_inter(**kwargs):
    return ResNet_inter(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet34_inter(**kwargs):
    return ResNet_inter(BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet50_inter(**kwargs):
    return ResNet_inter(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet101_inter(**kwargs):
    return ResNet_inter(Bottleneck, [3, 4, 23, 3], **kwargs)



model_dict_inter = {
    'resnet18': [resnet18_inter, 512],
    'resnet34': [resnet34_inter, 512],
    'resnet50': [resnet50_inter, 2048],
    'resnet101': [resnet101_inter, 2048],
}

class remove_fc(nn.Module):
    def __init__(self, model):
        super(remove_fc, self).__init__()
        self.model = model

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        return x


class LinearBatchNorm(nn.Module):
    """Implements BatchNorm1d by BatchNorm2d, for SyncBN purpose"""
    def __init__(self, dim, affine=True):
        super(LinearBatchNorm, self).__init__()
        self.dim = dim
        self.bn = nn.BatchNorm2d(dim, affine=affine)

    def forward(self, x):
        x = x.view(-1, self.dim, 1, 1)
        x = self.bn(x)
        x = x.view(-1, self.dim)
        return x


class SupConResNet_inter(nn.Module):
    """backbone + projection head"""
    def __init__(self, name='resnet18', head='mlp', feat_dim=128):
        super(SupConResNet_inter, self).__init__()
        pretrained = "./pretrained/resnet18-f37072fd.pth"
        model_fun, dim_in = model_dict_inter[name]
        self.encoder = model_fun()
        #self.encoder.load_state_dict(torch.load(pretrained))                 #
        #self.encoder = nn.Sequential(*list(self.encoder.children())[:-1])    #
        if head == 'linear':
            self.head = nn.Linear(dim_in, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

        self.out_conv1, self.out_conv2, self.out_fc = self.out_block(256, 128, 32, 8)
            
            
    def out_block(self, in_planes, out_planes1, out_planes2, size):
        
        out_conv1 = nn.Conv2d(in_planes, out_planes1, kernel_size=1)
        out_conv2 = nn.Conv2d(out_planes1, out_planes2, kernel_size=1)
        out_fc = nn.Linear(in_features=out_planes2*size*size, out_features=512)

        return out_conv1, out_conv2, out_fc


    def forward(self, x):
        feat = self.encoder(x)
        feat1, feat2, feat3, feat4, feat = feat
        feat = F.normalize(self.head(feat), dim=1)
        feat3 = self.out_conv2(self.out_conv1(feat3))
        feat3 = torch.flatten(feat3, 1)
        feat3 = self.out_fc(feat3)
        feat3 = F.normalize(self.head(feat3), dim=1)
        
        #feat = torch.cat([feat3, feat], dim=1)     # TODO cat mode
        
        return feat3, feat                            # in the order of from early to late

################################################################################################################


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, in_channels=3, zero_init_residual=False, k=1):
        super(ResNet, self).__init__()
        self.in_planes = 64*k

        self.conv1 = nn.Conv2d(in_channels, int(64*k), kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64*k)
        self.layer1 = self._make_layer(block, int(64*k), num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, int(128*k), num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, int(256*k), num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, int(512*k), num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves
        # like an identity. This improves the model by 0.2~0.3% according to:
        # https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i in range(num_blocks):
            stride = strides[i]
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def adaptive_pool(self, features, attn_from=None):
        if attn_from is None:
            attn_from = features
        assert features.size() == attn_from.size()
        N, C, H, W = features.size()
        assert (attn_from >= 0).float().sum() == N*C*H*W
        attention = torch.einsum('nchw,nc->nhw', [attn_from, nn.functional.adaptive_avg_pool2d(attn_from, (1, 1)).view(N, C)])
        attention = attention / attention.view(N, -1).sum(1).view(N, 1, 1).repeat(1, H, W)
        attention = attention.view(N, 1, H, W)
        return (features * attention).view(N, C, -1).sum(2)

    def forward(self, x, layer=100):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return out


def resnet18_end(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet34_end(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet50_end(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet101_end(**kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)




model_dict_end = {
    'resnet18': [resnet18_end, 512],
    'resnet34': [resnet34_end, 512],
    'resnet50': [resnet50_end, 2048],
    'resnet101': [resnet101_end, 2048],
}


class SupConResNet_end(nn.Module):
    """backbone + projection head"""
    def __init__(self, name='resnet18', head='mlp', feat_dim=128, in_channels=3, wide_k=1):
        super(SupConResNet_end, self).__init__()
        model_fun, dim_in = model_dict_end[name]
        dim_in = dim_in * wide_k
        self.encoder = model_fun(in_channels=in_channels, k=wide_k)
        if head == 'linear':
            self.head1 = nn.Linear(dim_in, feat_dim)
            self.head2 = nn.Linear(dim_in, feat_dim)
            self.head3 = nn.Linear(dim_in, feat_dim)
        elif head == 'mlp':
            self.head1 = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim))
            self.head2 = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim))
            self.head3 = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim))
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

    def forward(self, x):
        feat = self.encoder(x)
        feat1 = F.normalize(self.head1(feat), dim=1)
        feat2 = F.normalize(self.head2(feat), dim=1)
        feat3 = F.normalize(self.head3(feat), dim=1)
        return feat1, feat2, feat3
