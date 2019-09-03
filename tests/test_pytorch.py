import os

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F


base_dir = os.path.join('out', 'torch')


def _export(model, inputs, name, input_names, output_names, **kwargs):
    out_dir = os.path.join(base_dir, name)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_name = os.path.join(out_dir, 'model.onnx')
    torch.onnx.export(
        model, inputs, out_name, verbose=True,
        input_names=input_names, output_names=output_names, **kwargs)


def test_multiple_inputs():
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.prelu = nn.PReLU()

        def forward(self, x, y, z):
            return F.relu(x) + self.prelu(y) * z

    inputs = [torch.randn(1, 5), torch.randn(1, 5), torch.randn(1, 5)]
    input_names = ['x', 'y', 'z']
    _export(Net(), inputs, 'multiple_inputs', input_names, [])


def test_implicit_input():
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.frac = torch.tensor([2], dtype=torch.float)

        def forward(self, x):
            return x / self.frac

    inputs = torch.tensor([1], dtype=torch.float)
    _export(Net(), inputs, 'implicit_input', [], [])


def test_temporary_input():
    class Net(nn.Module):
        def forward(self, x):
            return x + torch.tensor([3], dtype=torch.float)

    inputs = torch.tensor([5], dtype=torch.float)
    _export(Net(), inputs, 'temporary_input', [], [])


def test_mnist():
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 20, 5, 1)
            self.conv2 = nn.Conv2d(20, 50, 5, 1)
            self.fc1 = nn.Linear(4*4*50, 500)
            self.fc2 = nn.Linear(500, 10)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.max_pool2d(x, 2, 2)
            x = F.relu(self.conv2(x))
            x = F.max_pool2d(x, 2, 2)
            x = x.view(-1, 4*4*50)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return F.log_softmax(x, dim=1)

    dummy_input = torch.randn(1, 1, 28, 28, device='cpu')
    model = Net()
    input_names = ['actual_input_1']
    output_names = ['output1']

    _export(model, dummy_input, 'mnsit_conv', input_names, output_names)


def test_resnet():
    dummy_input = torch.randn(10, 3, 224, 224, device='cpu')
    model = torchvision.models.resnet50(pretrained=True)
    input_names = ['actual_input_1']
    output_names = ['output1']

    _export(model, dummy_input, 'resnet', input_names, output_names)


def test_vgg():
    dummy_input = torch.randn(10, 3, 224, 224, device='cpu')
    model = torchvision.models.vgg16(pretrained=True)
    input_names = ['actual_input_1']
    output_names = ['output1']

    _export(model, dummy_input, 'vgg', input_names, output_names)


def test_alexnet():
    dummy_input = torch.randn(10, 3, 224, 224, device='cpu')
    model = torchvision.models.alexnet(pretrained=True)
    input_names = ['actual_input_1'] + ['learned_%d' % i for i in range(16)]
    output_names = ['output1']

    _export(model, dummy_input, 'alexnet', input_names, output_names)


def test_fasterrcnn():
    from torchvision.models.detection import FasterRCNN
    from torchvision.models.detection.rpn import AnchorGenerator
    backbone = torchvision.models.mobilenet_v2(pretrained=True).features
    backbone.out_channels = 1280
    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),))
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=[0], output_size=7, sampling_ratio=2)
    model = FasterRCNN(
        backbone, num_classes=2, rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler)
    model.eval()
    x = [torch.rand(3, 300, 400, device='cpu'), torch.rand(3, 500, 400, device='cpu')]

    _export(model, x, 'fasterrcnn', ['input_0'], ['output_0'])