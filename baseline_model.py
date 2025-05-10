import torch
import torch.nn as nn
import torch.nn.functional as F # allows us to use activation functions like ReLU in line.

class BaselineModel(nn.Module):
    """
    A simple baseline model for semantic segmentation.
    This model consists of three fully connected layers.
    The first two layers are followed by ReLU activation functions.
    The last layer outputs the logits for each class."""
    def __init__(self, input_dim: int =3, hidden_dim: int =128, num_classes: int =3):
        """
        A simple baseline model for semantic segmentation.
        Args:
            input_dim (int): Number of input channels (default: 3 for RGB).
            hidden_dim (int): Number of hidden units in the fully connected layers, hyperparameter.
            num_classes (int): Number of output classes.
        """
        super(BaselineModel, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W) where B is batch size, C is number of channels,
                              H is height, and W is width.
        Returns:
            torch.Tensor: Output tensor of shape (B, num_classes, H, W).
                              """
        B, C, H, W = x.shape
        # reshape the input tensor
        x = x.permute(0, 2, 3, 1) # reorder the dimensions (B, H, W, C)
        x= x.reshape(-1, C) # flatten spatial dimensions, shape (B*H*W, C)

        # apply the fully connected layers
        x = F.relu(self.layer1(x)) # apply relu to make negative values zero, introduces non-linearity
        x = F.relu(self.layer2(x))
        logits = self.layer3(x)  # (B*H*W, 19) # no RelU here, for cross entropy loss

        # reshape back to original dimensions
        logits = logits.view(B, H, W, -1) 
        logits = logits.permute(0, 3, 1, 2)
        return logits
    


model = BaselineModel()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss = nn.CrossEntropyLoss()

for images, masks in data: # replace data with our data thingy
    # images: (B, 3, H, W), masks: (B, H, W)
    logits = model(images)  # (B, 19, H, W)
    loss = loss(logits, masks)

    optimizer.zero_grad() # clear previous gradients
    loss.backward() # compute gradients
    optimizer.step() # update weights
