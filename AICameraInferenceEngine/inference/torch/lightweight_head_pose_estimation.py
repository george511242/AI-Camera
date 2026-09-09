from ..model import Result
from ..utils import img_utils
from .model import TorchModel
import torch.nn as nn
import torch
import math
import cv2


def conv_bn(inp, oup, stride, conv_layer=nn.Conv2d, norm_layer=nn.BatchNorm2d, nlin_layer=nn.ReLU):
    return nn.Sequential(
        conv_layer(inp, oup, 3, stride, 1, bias=False),
        norm_layer(oup),
        nlin_layer(inplace=True)
    )


def adjust_feature_order(a, groups=2):
    idx = []
    for i in range(groups):
        for k in range(a):
            if k % groups == i:
                idx.append(k)
    return idx


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride, exp_rate=2, activation=nn.ReLU, residual=False, groups=1, sort=False):
        super(Bottleneck, self).__init__()
        assert stride in [1, 2]
        assert kernel in [3, 5]
        exp_channels = int(in_channels * exp_rate)
        self.residual = residual
        self.sort = sort
        self.new_order = adjust_feature_order(exp_channels)
        padding = (kernel - 1) // 2
        conv_layer = nn.Conv2d
        norm_layer = nn.BatchNorm2d
        nlin_layer = activation
        # expand-linear
        self.exp_conv = nn.Sequential(conv_layer(in_channels, exp_channels, 1, 1, 0, bias=False, groups=groups), norm_layer(exp_channels), nlin_layer(inplace=True))
        # dw-linear
        self.dw_conv = nn.Sequential(conv_layer(exp_channels, exp_channels, kernel, stride, padding, groups=exp_channels, bias=False), norm_layer(exp_channels), nlin_layer(inplace=True))
        # pw-linear
        self.pw_conv = nn.Sequential(conv_layer(exp_channels, out_channels, 1, 1, 0, bias=False, groups=1), norm_layer(out_channels), nlin_layer(inplace=True))

    def forward(self, x):
        y = self.exp_conv(x)
        y = self.dw_conv(y)
        if self.sort:
            y = y[:, self.new_order]
        y = self.pw_conv(y)
        if self.residual:
            y = x + y
        return y


class LightweightHeadPoseEstimationModule(nn.Module):
    def __init__(self, num_bins=66, M=99, bin_train=False, base=16, width_mult=1):
        super(LightweightHeadPoseEstimationModule, self).__init__()
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.bin_train = bin_train
        self.M = M
        self.num_bins = num_bins
        input_shape = (1, 3, 224, 224)
        multiplier = [1, 2, 4, 6, 6, 8, 8, 8]
        kernel = [3, 3, 3, 5, 5, 5, 3, 0]
        stride = [2, 2, 1, 2, 1, 2, 1, 0]

        bandwidth = [base * m for m in multiplier]
        for i in range(3, len(bandwidth)):
            bandwidth[i] = int(bandwidth[i] * width_mult)

        self.features = []
        self.features.append(conv_bn(3, bandwidth[0], 2))
        for i in range(len(bandwidth)-1):
            groups = 4
            sort = False
            activation = nn.ReLU if i < 3 else h_swish
            self.features.append(Bottleneck(bandwidth[i], bandwidth[i+1], kernel=kernel[i], exp_rate=2, stride=stride[i], groups=groups, sort=sort, activation=activation))
        self.features.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.features.append(nn.ReLU())
        self.features = nn.Sequential(*self.features)

        self.feature_size = self.forward_feature(torch.zeros(*input_shape)).view(-1).size(0)
        self.fc_yaw = nn.Linear(self.feature_size, num_bins)
        self.fc_pitch = nn.Linear(self.feature_size, num_bins)
        self.fc_roll = nn.Linear(self.feature_size, num_bins)
        self.softmax = nn.Softmax(dim=1)

        self.idx_tensor = torch.FloatTensor([idx for idx in range(num_bins)]).to(device=device)
        self.bins = self.gen_bins()

        self._initialize_weights()

    def gen_bins(self):
        return self.idx_tensor * 2 * self.M / self.num_bins + self.M / self.num_bins - self.M

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward_feature(self, x):
        x = self.features(x)
        return x.view(x.size()[0], -1)

    def forward(self, x):
        x = self.forward_feature(x)
        pre_roll = self.fc_roll(x)
        pre_yaw = self.fc_yaw(x)
        pre_pitch = self.fc_pitch(x)
        if self.bin_train:
            return pre_roll, pre_yaw, pre_pitch
        else:
            roll = torch.sum(self.softmax(pre_roll) * self.bins, 1)
            yaw = torch.sum(self.softmax(pre_yaw) * self.bins, 1)
            pitch = torch.sum(self.softmax(pre_pitch) * self.bins, 1)
            return roll, yaw, pitch


class LightweightHeadPoseEstimation(TorchModel):
    def __init__(self, weight_path, margin=.6) -> None:
        super().__init__(model=LightweightHeadPoseEstimationModule(bin_train=False), weight_path=weight_path)
        self.margin = margin
        self.input_size = (224, 224)

    def predict_single(self, img):
        face = img_utils.pad_zero(img, 1)
        face = img_utils.resize(face, self.input_size)
        face = img_utils.z_norm(face, [0., 0., 0.], [255., 255., 255.])
        face = img_utils.z_norm(face, self.imagenet_mean, self.imagenet_std)
        face = img_utils.channel_first(face)
        faces = img_utils.to_batch([face])

        predict = self.model(torch.from_numpy(faces).to(device=self.device))
        roll, yaw, pitch = predict
        roll, yaw, pitch = roll.cpu().detach().numpy()[0], yaw.cpu().detach().numpy()[0], pitch.cpu().detach().numpy()[0]
        return roll, yaw, pitch

    def __call__(self, img: cv2.typing.MatLike, results: 'list[Result]') -> 'list[Result]':
        if not len(results):
            return results

        # preprocess
        faces = list()
        for result in results:
            if result.crop_info is not None:
                face = img_utils.crop(img, result.crop_info, self.margin)
            else:
                face = img_utils.crop(img, result.bbox, self.margin)
            face = img_utils.pad_zero(face, 1)
            face = img_utils.resize(face, self.input_size)
            face = img_utils.z_norm(face, [0., 0., 0.], [255., 255., 255.])
            face = img_utils.z_norm(face, self.imagenet_mean, self.imagenet_std)
            face = img_utils.channel_first(face)
            faces.append(face)
        faces = img_utils.to_batch(faces)

        # inference
        predict = self.model(torch.from_numpy(faces).to(device=self.device))

        # postprocess
        roll, yaw, pitch = predict
        roll, yaw, pitch = roll.cpu().detach().numpy(), yaw.cpu().detach().numpy(), pitch.cpu().detach().numpy()

        for result, r, y, p in zip(results, roll, yaw, pitch):
            result.set(
                yaw=y,
                roll=r,
                pitch=p,
            )
        return results
