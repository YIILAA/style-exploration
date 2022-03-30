# 网络结构的定义
# mostly borrowed from https://github.com/pytorch/examples/blob/master/fast_neural_style/neural_style/vgg.py
# add a class for Conditional Instance Normalization

import torch

class TransformerNet(torch.nn.Module):
    def __init__(self, style_num):
        super(TransformerNet, self).__init__()
        # convolution layers
        self.conv1 = ConvLayer(3, 32, kernel_size = 9, stride = 1) 
        self.in1 = ConditionalInstanceNorm2d(32, style_num)

        self.conv2 = ConvLayer(32, 64, kernel_size = 3, stride = 2)
        self.in2 = ConditionalInstanceNorm2d(64, style_num)

        self.conv3 = ConvLayer(64, 128, kernel_size = 3, stride = 2)
        self.in3 = ConditionalInstanceNorm2d(128, style_num)

        # Residual layers
        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.res3 = ResidualBlock(128)
        self.res4 = ResidualBlock(128)
        self.res5 = ResidualBlock(128)
        # Upsampling Layers
        self.deconv1 = UpsampleConvLayer(128, 64, kernel_size = 3, stride = 1, upsample = 2)
        self.in4 = ConditionalInstanceNorm2d(64, style_num)
        self.deconv2 = UpsampleConvLayer(64, 32, kernel_size = 3, stride = 1, upsample = 2)
        self.in5 = ConditionalInstanceNorm2d(32, style_num)
        self.deconv3 = ConvLayer(32, 3, kernel_size = 9, stride = 1)
        # Non-linearities
        self.relu = torch.nn.ReLU()

    def forward(self, X, style_id, blend=False, style_blend_weights=None):
        # 当blend=False时，一个forward处理一个batch_size的图片，style_id列表为batch中每个图片对应的风格
        # 当blend=True时，一个forward处理第一维=1的一张图片，style_id列表为需要融合的风格列表
        y = self.relu(self.in1(self.conv1(X), style_id, blend, style_blend_weights))
        y = self.relu(self.in2(self.conv2(y), style_id, blend, style_blend_weights))
        y = self.relu(self.in3(self.conv3(y), style_id, blend, style_blend_weights))
        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        y = self.res4(y)
        y = self.res5(y)
        y = self.relu(self.in4(self.deconv1(y), style_id, blend, style_blend_weights))
        y = self.relu(self.in5(self.deconv2(y), style_id, blend, style_blend_weights))
        y = self.deconv3(y)         
        return y


class ConditionalInstanceNorm2d(torch.nn.Module):
    # torch.nn.InstanceNorm2d不含affine参数
    # torch.nn.InstanceNorm2d之后加上torch.nn.Embedding层，每种风格对应一个嵌入向量，每一组嵌入向量shape为(1, channel*2)

    # 风格融合时，索引到对应的嵌入向量，做凸组合
    def __init__(self, num_features, num_classes):
        # num_features输入通道，num_classes风格数量
        super(ConditionalInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.inst_norm = torch.nn.InstanceNorm2d(num_features, affine=False) # 注意affine=False 这里IN层本身不带参数
        self.embed = torch.nn.Embedding(num_classes, num_features * 2) # 将原本IN层的affine参数取出，独立作为参数。每种风格 channel个缩放&偏移参数
        ## torch.nn.Embedding(num_embeddings, embedding_dim)
        ## 输入为一个编号列表，输出为对应的符号嵌入向量列表
        self.embed.weight.data[:, :num_features].normal_(1, 0.02)  # 缩放参数 Initialise scale at N(1, 0.02)
        self.embed.weight.data[:, num_features:].zero_()  # 偏移参数 Initialise bias at 0

    def forward(self, x, style_index, blend=False, style_blend_weights=None):
        # 先经过一层普通的IN层
        # 加上一个embed层，参数为(num_classes, c * 2), 每一行匹配一种风格，前后分别表示系数和偏移
        out = self.inst_norm(x)

        # new
        if blend: # blend=True, 风格融合，需对参数进行凸组合
            blend_style_embedding = self.embed(style_index) * style_blend_weights.unsqueeze(dim=1) # embed行 * 每个weight
            gamma, beta = blend_style_embedding.sum(dim=0, keepdim=True).chunk(2, 1)
        else: # 处理一个batch的图片
            gamma, beta = self.embed(style_index).chunk(2, 1) # chunks=2, dim=1
            # gamma, beta为处理一个batch的列表，元素的shape为(batch, c)
            # x.shape: (batch, c, h, w), gamma/beta reshape为(batch, c, 1, 1)

        out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(-1, self.num_features, 1, 1)
        return out


class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2 # same dimension after padding
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride) # remember this dimension

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class ResidualBlock(torch.nn.Module):
    """ResidualBlock
    introduced in: https://arxiv.org/abs/1512.03385
    recommended architecture: http://torch.ch/blog/2016/02/04/resnets.html
    """

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in2 = torch.nn.InstanceNorm2d(channels, affine=True)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out = out + residual # need relu right after
        return out


class UpsampleConvLayer(torch.nn.Module):
    """UpsampleConvLayer
    Upsamples the input and then does a convolution. This method gives better results
    compared to ConvTranspose2d.
    ref: http://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        if upsample:
            self.upsample_layer = torch.nn.Upsample(mode='nearest', scale_factor=upsample)
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = self.upsample_layer(x_in)
        out = self.reflection_pad(x_in)
        out = self.conv2d(out)
        return out
