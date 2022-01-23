from torchvision.models import resnet50, vgg16_bn, mobilenetv3
import torch.nn as nn


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def create_resnet_model(num_classes):
    model = resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model


def create_mobilenet_large_model(num_classes):
    model = mobilenetv3.mobilenet_v3_large(pretrained=True)
    model.classifier = nn.Sequential(
        nn.Linear(960, 1280),
        nn.Hardswish(inplace=True),
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(1280, num_classes),
    )
    return model


def create_mobilenet_small_model(num_classes):
    model = mobilenetv3.mobilenet_v3_small(pretrained=True)
    model.classifier = nn.Sequential(
        nn.Linear(576, 1024),
        nn.Hardswish(inplace=True),
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(1024, num_classes),
    )
    return model


def create_vgg16_model(num_classes):
    model = vgg16_bn(pretrained=True)
    model.classifier = nn.Sequential(
        nn.Linear(512 * 7 * 7, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, num_classes),
    )
    return model


def create_model(model_arch, num_classes):
    if model_arch == 'mobilenet_large':
        return create_mobilenet_large_model(num_classes)
    elif model_arch == 'mobilenet_small':
        return create_mobilenet_small_model(num_classes)
    elif model_arch == 'vgg16':
        return create_vgg16_model(num_classes)
    elif model_arch == 'resnet':
        return create_resnet_model(num_classes)
    else:
        raise "%s architecture not defined" % model_arch
