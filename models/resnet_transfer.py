import torch
import torch.nn as nn
import torchvision.models as models

class TransferResNet18(nn.Module):
    """
    ResNet-18 NN model adapted for Transfer Learning and Knowledge Distillation on the CIFAR-10 dataset.
    
    This class provides three modes of operation:
    1. 'resize_freeze': Freezes early layers and only trains a new fully connected (FC) layer.
    2. 'modify_finetune': Adapts the first convolutional layer for 32x32 images and fine-tunes the entire network.
    3. 'scratch': Initializes a fresh ResNet-18 (adapted for 32x32 images) with no pre-trained weights.
    """
    
    def __init__(self, num_classes: int = 10, transfer_mode: str = "modify_finetune") -> None:
        """
        Initializes the TransferResNet18 model based on the selected transfer mode.

        Args:
            num_classes (int): The number of output classes for the dataset (default: 10 for CIFAR-10).
            transfer_mode (str): The strategy to use. Must be one of 'resize_freeze', 'modify_finetune', or 'scratch'.
        """
        super(TransferResNet18, self).__init__()
        self.transfer_mode = transfer_mode
        
        # Load pre-trained weights on ImageNet
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        if transfer_mode == "resize_freeze":
            # METHOD 1: Freeze the weights of early layers
            for param in self.model.parameters():
                param.requires_grad = False
                
            # Modify only the final Fully Connected (FC) layer for CIFAR-10 (10 classes)
            # Newly added layers have requires_grad=True by default.
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, num_classes)
            
        elif transfer_mode == "modify_finetune":
            # METHOD 2: Modify the early convolutional layer to suit CIFAR-10's 32x32 size
            # Original ResNet conv1: Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
            # For CIFAR-10, we change it to 3x3 kernel and stride=1 to avoid immediate resolution drop.
            self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            
            # Disable the initial MaxPool layer since CIFAR-10 images are already very small
            self.model.maxpool = nn.Identity()
            
            # Modify the final FC layer for 10 classes
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, num_classes)
            
        elif transfer_mode == "scratch":
            # For Knowledge Distillation (Part B), in case we want to train the NN from scratch
            self.model = models.resnet18(weights=None)
            self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.model.maxpool = nn.Identity()
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the network.

        Args:
            x (torch.Tensor): A batch of input images.

        Returns:
            torch.Tensor: The output logits of the network.
        """
        return self.model(x)