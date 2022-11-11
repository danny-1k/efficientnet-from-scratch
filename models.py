import os
import torch
import torch.nn as nn
from math import ceil
from utils import calculate_factors, get_image_size, get_dropout

# base model

#Stage | Operator              | Resolution (output) | Channels | Layers
#------+-----------------------+---------------------+----------+-------
#1     | Conv 3x3              | 224 x 224           | 32       | 1
#------+-----------------------+---------------------+----------+-------
#2     | MBConv1 3x3           | 112 x 112           | 16       | 1
#------+-----------------------+---------------------+----------+-------
#3     |MBConv6 3x3            | 112 x 112           | 24       | 2
#------+-----------------------+---------------------+----------+-------
#4     | MBConv6 5x5           | 56 x 56             | 40       | 2
#------+-----------------------+---------------------+----------+-------
#5     | MBConv6 3x3           | 28 x 28             | 80       | 3
#------+-----------------------+---------------------+----------+-------
#6     |MBConv6 5x5            | 14 x 14             | 112      | 3
#------+-----------------------+---------------------+----------+-------
#7     | MBConv6 5x5           | 14 x 14             | 192      | 4
#------+-----------------------+---------------------+----------+-------
#8     | MBConv6 3x3           | 7 x 7               | 320      | 1
#------+-----------------------+---------------------+----------+-------
#9     | Conv 1x1 Pooling & FC | 7 x 7               | 1280     | 1
#------+-----------------------+---------------------+----------+-------

# expand ratio, kernel_size, stride, channels, repeat

base_model_architecture = [
    [1, 3, 1, 16, 1],
    [6, 3, 2, 24, 2],
    [6, 5, 2, 40, 2],
    [6, 3, 2, 80, 3],
    [6, 5, 1, 112, 3],
    [6, 5, 2, 192, 4],
    [6, 3, 1, 320, 1],
]


def Swish():
    # Swish was initially called SiLU
    return nn.SiLU()


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups):
        super().__init__()


        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False
        )

        self.swish = Swish()
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.swish(x)
        
        return x


class SqueezeExcitationLayer(nn.Module):
    def __init__(self, in_channels, squeeze):
        super().__init__()

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=in_channels, out_channels=squeeze, kernel_size=1),
            Swish(),
            nn.Conv2d(in_channels=squeeze, out_channels=in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x * self.se(x) # channel-wise weights
        
        return x


class DropConnect(nn.Module):
    def __init__(self, drop):
        super().__init__()

        self.survival_probability = 1-drop

    def forward(self, x):

        
        if self.training:
            drop_mask = torch.rand(x.shape[0], 1, 1, 1, requires_grad=False, device=x.device) < self.survival_probability
            # rescale

            x = x/self.survival_probability

            x = x * drop_mask

        return x


class InvertedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio, drop_connect, squeeze=.25):
        super().__init__()

        self.drop_connect = DropConnect(drop_connect)
        self.residual = (in_channels == out_channels) and stride == 1
        self.expand = in_channels != (in_channels*expand_ratio)

        if self.expand: # expansion
            self.expansion_layer = ConvBlock(
                in_channels=in_channels,
                out_channels=in_channels*expand_ratio,
                kernel_size=3, 
                stride=1, padding=1, 
                groups=1
            )


        self.depth_conv_layer = nn.Sequential(
            ConvBlock(
                in_channels=in_channels*expand_ratio,
                out_channels=in_channels*expand_ratio,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size//2,
                groups=in_channels*expand_ratio
            ),

            SqueezeExcitationLayer(in_channels=in_channels*expand_ratio,
                squeeze=int(in_channels*squeeze)
            )
        )


        self.projection_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels*expand_ratio,
                out_channels=out_channels,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
        )


    def forward(self, x):
        if self.expand:
            x = self.expansion_layer(x)

        out = self.depth_conv_layer(x)
        out = self.projection_layer(out)

        if self.residual:
            out = out + x

        return out


class RepeatedInvertedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio, drop_connect=.2, repeat=2):
        super().__init__()

        self.layers = nn.Sequential(

            *[InvertedResidualBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                expand_ratio=expand_ratio,
                drop_connect=drop_connect,
            )] + \

            [
                InvertedResidualBlock(
                    in_channels=out_channels, 
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    expand_ratio=1,
                    drop_connect=drop_connect
                )

                for i in range(repeat-1)

            ]
        )
        

    def forward(self, x):
        x = self.layers(x)

        return x

    def __len__(self):
        return len(self.layers)

    def __getitem__(self, idx):
        return self.layers[idx]


class EfficientNet(nn.Module):
    def __init__(self, v='b0', num_classes=1000):
        super().__init__()

        self.num_classes = num_classes

        self.image_size = get_image_size(v)

        self.layers, channels, output_channels = self.construct_layers(v)

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(channels)
        )

        self.features = []

        for layer in self.layers:

            self.features.append(
                RepeatedInvertedResidualBlock(
                    in_channels=layer[0],
                    out_channels=layer[1],
                    kernel_size=layer[2],
                    stride=layer[3],
                    expand_ratio=layer[4],
                    repeat=layer[5]
                )
            )

        self.features.append(
            ConvBlock(in_channels=self.layers[-1][1],
                    out_channels = output_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    groups=1
            ),
        )

        self.features.append(
            nn.BatchNorm2d(output_channels)
        )


        self.features = nn.Sequential(*self.features)

        self.pool = nn.AdaptiveAvgPool2d(1)


        self.classifier = nn.Sequential(
            nn.Dropout(p=get_dropout(v)),
            nn.Linear(output_channels, num_classes)
        )


    def forward(self, x):
        assert x.shape[-1] == self.image_size and x.shape[-2] == self.image_size, f'Expected Input size of ({self.image_size} x {self.image_size}).'

        x = self.stem(x)

        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)

        return x


    @classmethod
    def from_pretrained(cls, v='b0'):

        model = EfficientNet(v)
        model.load_pretrained_weights(v)

        return model


    def construct_layers(self, v='b0'):
        depth_factor, width_factor = calculate_factors(v)

        layers = []


        first_channel = 4 * ceil((width_factor*32)/4)
        output_channel = 4 * ceil((width_factor*1280)/4)

        in_channel = first_channel

        for layer in base_model_architecture:
            
            new_layer = []

            expand_ratio, kernel_size, stride, out_channel, repeat = layer 
            
            out_channel = 4 * ceil((width_factor*out_channel)/4) # make sure divisible by 4
            repeat = ceil(depth_factor*repeat)

            new_layer.append(in_channel)
            new_layer.append(out_channel)
            new_layer.append(kernel_size)
            new_layer.append(stride)
            new_layer.append(expand_ratio)
            new_layer.append(repeat)

            in_channel = out_channel

            layers.append(new_layer)

        return layers, first_channel, output_channel


    def load_pretrained_weights(self, v='b0'):
        try:
            state_dict = torch.load(f'pretrained/{v}.pth')
            self.load_state_dict(state_dict)

        except FileNotFoundError:

            state_dict = torch.load(f'pretrained/{v}.pth')
            self.load_state_dict(state_dict)