import copy
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Dict, Any, Tuple

def get_loaders(params: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    """
    Creates and returns training and validation dataloaders for the CIFAR-10 dataset.
    Applies specific transforms (Resize to 224x224 vs keeping 32x32) based on the transfer_mode.
    Integrates AugMix data augmentation if flagged in parameters.

    Args:
        params (Dict[str, Any]): A dictionary containing hyperparameters and configurations.

    Returns:
        Tuple[DataLoader, DataLoader]: A tuple containing the training and validation dataloaders.
    """
    mean, std = params["mean"], params["std"]
    normalize = transforms.Normalize(mean, std)
    use_augmix = params.get("use_augmix", False)

    # Base transformations depending on transfer mode
    if params["transfer_mode"] == "resize_freeze":
        # ImageNet standard: resize images to 224x224
        train_tf_list = [
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
        ]
        val_tf = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        # modify_finetune or scratch: keep 32x32 but add basic augmentations
        train_tf_list = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ]
        val_tf = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

    # HW3: Inject AugMix augmentation if requested
    if use_augmix:
        train_tf_list.append(transforms.AugMix())

    # Finalize training transforms
    train_tf_list.extend([
        transforms.ToTensor(),
        normalize,
    ])
    train_tf = transforms.Compose(train_tf_list)

    # Load CIFAR-10 dataset
    train_ds = datasets.CIFAR10(params["data_dir"], train=True, download=True, transform=train_tf)
    val_ds   = datasets.CIFAR10(params["data_dir"], train=False, download=True, transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=params["batch_size"],
                              shuffle=True, num_workers=params["num_workers"])
    val_loader   = DataLoader(val_ds, batch_size=params["batch_size"],
                              shuffle=False, num_workers=params["num_workers"])
    
    return train_loader, val_loader


def train_one_epoch(
    model: nn.Module, 
    loader: DataLoader, 
    optimizer: torch.optim.Optimizer, 
    criterion: nn.Module, 
    device: torch.device, 
    params: Dict[str, Any]
) -> Tuple[float, float]:
    """
    Trains the neural network model for a single epoch.

    Args:
        model (nn.Module): The PyTorch model to be trained.
        loader (DataLoader): The dataloader providing the training data batches.
        optimizer (torch.optim.Optimizer): The optimizer algorithm (e.g., Adam).
        criterion (nn.Module): The loss function.
        device (torch.device): The device (CPU or CUDA) to perform computations on.
        params (Dict[str, Any]): Hyperparameters including logging intervals.

    Returns:
        Tuple[float, float]: The average training loss and accuracy for the epoch.
    """
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    log_interval = params.get("log_interval", 100)
    
    for batch_idx, (imgs, labels) in enumerate(loader):
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, labels)
        
        loss.backward()
        optimizer.step()

        total_loss += loss.detach().item() * imgs.size(0)
        correct    += out.argmax(1).eq(labels).sum().item()
        n          += imgs.size(0)

        if (batch_idx + 1) % log_interval == 0:
            print(f"  [{batch_idx+1}/{len(loader)}] "
                  f"loss: {total_loss/n:.4f}  acc: {correct/n:.4f}")

    return total_loss / n, correct / n


def validate(
    model: nn.Module, 
    loader: DataLoader, 
    criterion: nn.Module, 
    device: torch.device
) -> Tuple[float, float]:
    """
    Evaluates the neural network model on the validation dataset.

    Args:
        model (nn.Module): The PyTorch model to be evaluated.
        loader (DataLoader): The dataloader providing the validation data batches.
        criterion (nn.Module): The loss function.
        device (torch.device): The device (CPU or CUDA) to perform computations on.

    Returns:
        Tuple[float, float]: The average validation loss and accuracy.
    """
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out  = model(imgs)
            loss = criterion(out, labels)
            
            total_loss += loss.detach().item() * imgs.size(0)
            correct    += out.argmax(1).eq(labels).sum().item()
            n          += imgs.size(0)
            
    return total_loss / n, correct / n


def kd_loss_fn(
    student_logits: torch.Tensor, 
    teacher_logits: torch.Tensor, 
    labels: torch.Tensor, 
    T: float = 2.0, 
    alpha: float = 0.5
) -> torch.Tensor:
    """
    Computes the standard Knowledge Distillation (KD) Loss.

    Args:
        student_logits (torch.Tensor): The raw output predictions from the student model.
        teacher_logits (torch.Tensor): The raw output predictions from the teacher model.
        labels (torch.Tensor): The ground truth labels.
        T (float): Temperature parameter to soften the probabilities (default: 2.0).
        alpha (float): Weight balancing the soft loss and hard loss (default: 0.5).

    Returns:
        torch.Tensor: The computed scalar loss.
    """
    soft_loss = F.kl_div(F.log_softmax(student_logits / T, dim=1),
                         F.softmax(teacher_logits / T, dim=1),
                         reduction='batchmean') * (T * T)
    hard_loss = F.cross_entropy(student_logits, labels)
    return alpha * soft_loss + (1. - alpha) * hard_loss


def custom_mobilenet_kd_loss(
    student_logits: torch.Tensor, 
    teacher_logits: torch.Tensor, 
    labels: torch.Tensor, 
    T: float = 2.0, 
    alpha: float = 0.5
) -> torch.Tensor:
    """
    Custom Knowledge Distillation Loss. The teacher assigns its exact probability 
    to the true class, and distributes the remaining probability equally among other classes.

    Args:
        student_logits (torch.Tensor): The raw output predictions from the student model.
        teacher_logits (torch.Tensor): The raw output predictions from the teacher model.
        labels (torch.Tensor): The ground truth labels.
        T (float): Temperature parameter (default: 2.0).
        alpha (float): Weight balancing the soft loss and hard loss (default: 0.5).

    Returns:
        torch.Tensor: The computed scalar custom KD loss.
    """
    teacher_probs = F.softmax(teacher_logits / T, dim=1)
    batch_size, num_classes = teacher_probs.shape
    
    # Create the custom target distribution
    target_probs = torch.zeros_like(teacher_probs)
    for i in range(batch_size):
        true_class = labels[i]
        true_prob = teacher_probs[i, true_class].item()
        
        # Distribute remaining prob equally
        remaining_prob = 1.0 - true_prob
        target_probs[i, :] = remaining_prob / (num_classes - 1)
        target_probs[i, true_class] = true_prob # Set true class prob
        
    soft_loss = F.kl_div(F.log_softmax(student_logits / T, dim=1),
                         target_probs,
                         reduction='batchmean') * (T * T)
    hard_loss = F.cross_entropy(student_logits, labels)
    return alpha * soft_loss + (1. - alpha) * hard_loss


def run_training(model: nn.Module, params: Dict[str, Any], device: torch.device) -> None:
    """
    Executes the main training loop including Early Stopping, Learning Rate Scheduling, 
    and plotting the loss curve.

    Args:
        model (nn.Module): The PyTorch model to train.
        params (Dict[str, Any]): The configuration dictionary with hyperparameters.
        device (torch.device): The target computation device (CPU or CUDA).
    """
    train_loader, val_loader = get_loaders(params)
    criterion = nn.CrossEntropyLoss()
    
    # We only pass trainable parameters to the optimizer (crucial for "resize_freeze")
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    optimizer = torch.optim.Adam(trainable_params,
                                 lr=params["learning_rate"],
                                 weight_decay=params["weight_decay"])
                                 
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_loss = float('inf') 
    best_acc  = 0.0
    best_weights = None
    epochs_no_improve = 0    
    patience = params.get("patience", 5)

    history_train_loss = []
    history_val_loss = []
    
    use_augmix = params.get("use_augmix", False)
    aug_status = "(with AugMix)" if use_augmix else ""

    print(f"Starting training with mode: {params['transfer_mode']} {aug_status}...")

    for epoch in range(1, params["epochs"] + 1):
        print(f"\nEpoch {epoch}/{params['epochs']}")
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, params)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        history_train_loss.append(tr_loss)
        history_val_loss.append(val_loss)

        print(f"  Train loss: {tr_loss:.4f}  acc: {tr_acc:.4f}")
        print(f"  Val   loss: {val_loss:.4f}  acc: {val_acc:.4f}")

        if val_loss < best_loss:
            best_loss    = val_loss
            best_acc     = val_acc
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(best_weights, params["save_path"])
            epochs_no_improve = 0 
            print(f"  Saved best model (val_loss={best_loss:.4f}, val_acc={best_acc:.4f})")
        else:
            epochs_no_improve += 1
            print(f"  No improvement in validation loss for {epochs_no_improve} epoch(s).")
            
            if epochs_no_improve >= patience:
                print(f"\nEarly stopping triggered! Training stopped at epoch {epoch}.")
                break

    # Load best weights before returning
    if best_weights is not None:
        model.load_state_dict(best_weights)
    print(f"\nTraining done. Best val loss: {best_loss:.4f}, Best val accuracy: {best_acc:.4f}")

    # Plot and save the loss curve dynamically based on AugMix
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(history_train_loss) + 1), history_train_loss, label='Train Loss', marker='o')
    plt.plot(range(1, len(history_val_loss) + 1), history_val_loss, label='Validation Loss', marker='s')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    
    title_suffix = " + AugMix" if use_augmix else ""
    plt.title(f'Loss Curve - ResNet18 ({params["transfer_mode"]}{title_suffix})')
    plt.legend()
    plt.grid(True)
    
    file_suffix = "_augmix" if use_augmix else ""
    plot_filename = f'loss_curve_{params["transfer_mode"]}{file_suffix}.png'
    
    plt.savefig(plot_filename)
    plt.close()
    print(f"Graph saved as {plot_filename}.")