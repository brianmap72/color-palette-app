import torch
import torch.nn as nn


class PaletteModel(nn.Module):
    def __init__(self, ni=15, no=15, hidden_seq=[64, 64, 32], lr=0.0001):
        super().__init__()
        self.ni = ni
        self.no = no
        self.hidden_seq = hidden_seq
        self.lr = lr
        self.hidden = self._get_hidden()
        self.out = HiddenBlock(self.hidden_seq[-1], self.no, activation=True, relu=False, dropout=False)
        self.loss = nn.L1Loss()
        #     self.loss = nn.MSELoss()
        self.optim = torch.optim.Adam(self.parameters(), lr=self.lr)

    def _get_hidden(self):
        hidden = []
        ni = self.ni
        for i in range(len(self.hidden_seq)):
            hidden.append(HiddenBlock(ni, self.hidden_seq[i], p=0.5))
            ni = self.hidden_seq[i]
    
        return nn.Sequential(*hidden)
  
    def forward(self, x):
        x = self.out(self.hidden(x))
        return x

    def reset_lr(self, lr):
        for p in self.optim.param_groups:
            p['lr'] = lr

    def fit(self, x, y_true):
        self.optim.zero_grad()
        y_pred = self.forward(x)
        loss = self.loss(y_pred, y_true)
        loss.backward()
        self.optim.step()
        return y_pred, loss