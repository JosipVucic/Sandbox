from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

# default genotype, 13 tuples, one tuple defines one block of the network
# (conv_output_channels, pool_layer_present, conv_kernel_size)
DEFAULT_GENOTYPE = [
    (57, 0, 3),
    (96, 0, 1),
    (36, 0, 1),
    (104, 0, 1),
    (97, 0, 3),
    (18, 0, 1),
    (60, 0, 3),
    (66, 1, 3),
    (82, 0, 3),
    (94, 0, 3),
    (210, 1, 3),
    (17, 1, 1),
    (100, 0, 3)]


class GACNN(nn.Module):
    """
    Model architecture consisting of 13 blocks, defined by genotype found using genetic algorithm evolution.
    """
    def __init__(
            self,
            genotype=None,
            num_classes: int = 10,
            in_chans: int = 1,

    ):
        """Instantiates a GACNN model comprised of 13 blocks, each block is defined by number of convolutional output
        channels, presence of a pooling layer, and kernel size for the convolution

        Args:
            genotype (list, optional): list of 13 3tuples defining the model architecture
            num_classes (int, optional): number of classes. Defaults to 10.
            in_chans (int, optional): number of input channels. Defaults to 1.
            """
        super(GACNN, self).__init__()

        if genotype is None:
            genotype = DEFAULT_GENOTYPE

        self.num_classes = num_classes
        self.in_chans = in_chans

        self.features = self._make_layers(genotype)
        self.classifier = nn.Linear(round(genotype[-1][0]), num_classes)

    def forward(self, x):
        """
        The input is processed by the model body, then a global pooling operation processes the feature maps changing
        their size to 1x1, they get concatenated into a feature vector which enters the linear classifier.

        :param x: model input
        :return: model output
        """
        out = self.features(x)
        out = F.max_pool2d(out, kernel_size=out.size()[2:])
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, genotype):
        """
        Makes the model according to the provided genotype.

        :param genotype: model genotype - list containing 13 3tuples of parameters
        :return: The model.
        """

        layers: List[nn.Module] = []
        input_channel = self.in_chans
        for idx, (layer, pool, kernel_size) in enumerate(
                genotype
        ):
            if pool == 1:
                layers += [
                    nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
                ]
            else:
                filters = round(layer)

                layers += [
                    nn.Conv2d(input_channel, filters, kernel_size=kernel_size, stride=1, padding=1),
                    nn.BatchNorm2d(filters, eps=1e-05, momentum=0.05, affine=True),
                    nn.ReLU(inplace=True),
                ]

                input_channel = filters

        model = nn.Sequential(*layers)
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain("relu"))
        return model

    def classify_input(self, img):
        """Returns digit classification for the provided input.
        :param img: Tensor containing the preprocessed image object.
        :return: integer, one of the digits 0 to 9"""
        with torch.no_grad():
            output = self(img)

            _, predicted = torch.max(output.data, 1)

        return predicted.item()
    @classmethod
    def get_trained(cls):
        """Retrieves a pretrained model from ./digits/neural/trained_models/default.pth
        :return: The loaded GACNN model."""
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model = GACNN()
        model.load_state_dict(torch.load("digits/neural/trained_models/default.pth", map_location=device))
        model.eval()

        return model
