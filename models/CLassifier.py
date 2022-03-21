from torch import nn

class Classifier(nn.Module):
    def __init__(self, config) -> None:
        super(Classifier, self).__init__()
        self.fc_net = nn.Sequential(nn.Dropout(config['dropout']), \
                            nn.Linear(in_features=config['linear_in'], out_features=config['n_classes']))

    def forward(self, x):
        x = x.view(x.shape[0],-1)
        x = self.fc_net(x)
        return x
