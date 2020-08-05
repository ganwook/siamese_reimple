import torch
import torch.nn as nn
import torch.nn.functional as F

class SiameseNN(nn.Module):

    def __init__(self):
        super(SiameseNN, self).__init__()
        self.encoder = nn.Sequential(
                            nn.Conv2d(1, 64, 10),
                            nn.ReLU(),
                            nn.MaxPool2d(2, 2),
                            nn.Conv2d(64, 128, 7),
                            nn.ReLU(),
                            nn.MaxPool2d(2, 2),
                            nn.Conv2d(128, 128, 4),
                            nn.ReLU(),
                            nn.MaxPool2d(2, 2),
                            nn.Conv2d(128, 256, 4),
                            nn.ReLU()
                            )
        self.linear = nn.Sequential(nn.Linear(256 * 6 * 6, 4096), nn.Sigmoid())
        self.out    = nn.Linear(4096, 1)

    def forward_each(self, x):
        x = self.encoder(x)
        x = x.view(x.size()[0], -1)
        x = self.linear(x)
        
        return x
    
    def forward(self, img1, img2):
        hidden1 = self.forward_each(img1)
        hidden2 = self.forward_each(img2)
        dist    = torch.abs(hidden1 - hidden2)
        out     = self.out(dist)

        return out

if __name__ == '__main__':
    model = SiameseNN()
    print(model)
    print(model.state_dict().keys())