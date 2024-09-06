import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import torch.nn.functional as F



class AOD_CNN(nn.Module):
    """
    A simple CNN model for image-based tasks, featuring convolutional layers with max-pooling, 
    followed by global average pooling and fully connected layers.

    Architecture:
    - Three convolutional layers (with ReLU activations and max-pooling).
    - Global average pooling to reduce spatial dimensions.
    - Three fully connected layers for output.
    """
    
    def __init__(self):
        """
        Initializes the AOD_CNN model with convolutional, pooling, and fully connected layers.
        """
        super(AOD_CNN, self).__init__()
        
        # First convolutional layer: input channels = 3, output channels = 32
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        
        # Second convolutional layer: input channels = 32, output channels = 64
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        
        # Third convolutional layer: input channels = 64, output channels = 128
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)

        # Max pooling layer: reduces the spatial dimensions by half
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Global average pooling: reduces the output to (1, 1) spatial dimensions
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layers
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        """
        Defines the forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, height, width)
        
        Returns:
            torch.Tensor: Output tensor after passing through the network.
        """
        # Apply conv1 -> ReLU -> MaxPool
        x = self.pool(F.relu(self.conv1(x)))  # Output: 32x64x64
        
        # Apply conv2 -> ReLU -> MaxPool
        x = self.pool(F.relu(self.conv2(x)))  # Output: 64x32x32
        
        # Apply conv3 -> ReLU -> MaxPool
        x = self.pool(F.relu(self.conv3(x)))  # Output: 128x16x16
        
        # Apply global average pooling to reduce to (1x1)
        x = self.global_avg_pool(x)  # Output: 128x1x1
        
        # Flatten the tensor
        x = x.view(-1, 128)  # Output: (batch_size, 128)
        
        # Fully connected layers with ReLU activation
        x = F.relu(self.fc1(x))  # Output: (batch_size, 64)
        x = F.relu(self.fc2(x))  # Output: (batch_size, 32)
        x = F.relu(self.fc3(x))  # Output: (batch_size, 1)
        
        return x
    
    
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
 
        # First convolutional layer: 1 input channel, 16 output channels, 3x3 kernel
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3)
 
        # Second convolutional layer: 16 input channels, 32 output channels, 3x3 kernel
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
 
        # Fully connected layer: assuming input images of size 28x28, after two rounds of conv + max pool,
        # the feature map size will be reduced to 5x5 (with stride 2 in pooling).
        self.fc1 = nn.Linear(in_features=28800, out_features=1)
 
    def forward(self, x):
        # First convolutional layer with ReLU and 2x2 max pooling
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
 
        # Second convolutional layer with ReLU and 2x2 max pooling
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
 
 
        # Flatten the output from the convolutional layers
        x = x.view(x.size(0), -1)
        # print(x.shape)
        # Fully connected layer followed by ReLU
        x = F.relu(self.fc1(x))
 
        return x
