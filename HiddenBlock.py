import torch
import torch.nn as nn


class HiddenBlock(nn.Module):
    def __init__(self, ni, no, dropout=True, p=0.5, bias=True, relu=True, activation=True):
        super().__init__()
        lyr = nn.Linear(ni, no, bias)
        torch.nn.init.kaiming_normal_(lyr.weight)
        
        if bias: lyr.bias.data.zero_().add_(0.1)
        layers = [lyr]
        
        if activation:
            if relu: 
                layers += [nn.ReLU()]
            else:
                layers += [nn.Sigmoid()]

        if dropout: layers += [nn.Dropout(p)]

        self.seq = nn.Sequential(*layers)


    def forward(self, x):
        return self.seq(x)