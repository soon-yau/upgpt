from torch import nn

class LinearProject(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.model = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.model(x)
