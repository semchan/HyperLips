import torch.nn as nn

class HyperNetwork(nn.Module):
  """Hypernetwork architecture."""
  def __init__(self, in_dim=1, h_dim=32):
    """
    Args:
      in_dim : Input dimension
      h_dim : Hidden dimension
    """
    super(HyperNetwork, self).__init__()
    
    # Network layers
    self.lin1 = nn.Linear(in_dim, h_dim)
    self.lin2 = nn.Linear(h_dim, h_dim)
    # self.lin3 = nn.Linear(h_dim, h_dim)
    # self.lin4 = nn.Linear(h_dim, h_dim)

    # Activations
    self.relu = nn.LeakyReLU(inplace=True)

  def forward(self, x):
    """
    Args:
      x : Hyperparameter values (batch_size, num_hyperparams)
    """
    x = self.relu(self.lin1(x))
    x = self.relu(self.lin2(x))
    # x = self.relu(self.lin3(x))
    # x = self.relu(self.lin4(x))
    return x