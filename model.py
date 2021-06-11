import torch
import torch.nn as nn

class ConvModel(torch.nn.Module):
    """4-layer Convolutional Neural Network architecture from [1].

    Parameters
    ----------
    in_channels : int
        Number of channels for the input images.

    out_features : int
        Number of classes (output of the model).

    hidden_size : int (default: 64)
        Number of channels in the intermediate representations.

    feature_size : int (default: 64)
        Number of features returned by the convolutional head.

    References
    ----------
    .. [1] Finn C., Abbeel P., and Levine, S. (2017). Model-Agnostic Meta-Learning
           for Fast Adaptation of Deep Networks. International Conference on
           Machine Learning (ICML) (https://arxiv.org/abs/1703.03400)
    """
    def __init__(self, in_channels, out_features, hidden_size=64):
        super(ConvModel, self).__init__()
        self.in_channels = in_channels
        self.out_features = out_features
        self.hidden_size = hidden_size

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, hidden_size, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(),
            nn.MaxPool2d(2),            
            nn.Conv2d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(),
            nn.MaxPool2d(2),  
            nn.Conv2d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(),
            nn.MaxPool2d(2)                        
        )

        self.classifier = nn.Linear(5*5*hidden_size, out_features, bias=True)

    def forward(self, inputs):
        features = self.features(inputs)
        features = features.view((features.size(0), -1))
        logits = self.classifier(features)
        return logits

if __name__ == '__main__':
    model = ConvModel(in_channels=3, out_features=5)