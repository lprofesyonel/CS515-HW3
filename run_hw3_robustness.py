import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
from typing import Dict, Any, Tuple
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# HW3 New Attack and Visualization Libraries
import torchattacks
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# Project imports
from train import get_loaders
from models.resnet_transfer import TransferResNet18
from models.CNN import SimpleCNN

# --- 1. CIFAR-10-C LOADER ---
def get_cifar_c_loader(cifar_c_dir: str, batch_size: int = 128) -> DataLoader:
    """
    Loads the corrupted CIFAR-10-C dataset. 
    Assumes numpy files (.npy) for images and labels are extracted in the given directory.
    """
    print(f"Loading CIFAR-10-C from: {cifar_c_dir}...")
    
    images_path = os.path.join(cifar_c_dir, 'gaussian_noise.npy')
    labels_path = os.path.join(cifar_c_dir, 'labels.npy')
    
    if not os.path.exists(images_path) or not os.path.exists(labels_path):
        raise FileNotFoundError(
            f"\n[CRITICAL ERROR] CIFAR-10-C data NOT FOUND at {cifar_c_dir}!\n"
            f"Please download the dataset and extract it to this folder, or "
            f"comment out the CIFAR-C evaluation lines if you want to skip it."
        )
        
    images = np.load(images_path) # Shape: (50000, 32, 32, 3)
    labels = np.load(labels_path)
    
    # Convert to PyTorch tensors and permute to (N, C, H, W)
    images = torch.tensor(images, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    labels = torch.tensor(labels, dtype=torch.long)
    
    # Standard CIFAR-10 normalization
    normalize = transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
    images = normalize(images)
    
    dataset = TensorDataset(images, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# --- 2. EVALUATION & ADVERSARIAL ATTACKS ---
def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device, attack: Any = None) -> float:
    """
    Evaluates the model on a given dataloader. If an attack object is provided, 
    evaluates adversarial robustness.
    """
    model.eval()
    correct = 0
    total = 0
    
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        
        if attack is not None:
            imgs = attack(imgs, labels) # Generate adversarial samples
            
        with torch.no_grad():
            outputs = model(imgs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    return 100.0 * correct / total

# --- 3. VISUALIZATION: GRAD-CAM & TSNE ---
def plot_grad_cam(model: nn.Module, imgs: torch.Tensor, labels: torch.Tensor, attack: Any, device: torch.device):
    """
    Generates and saves Grad-CAM heatmaps comparing clean and adversarial samples.
    """
    print("\n--- Generating Grad-CAM Heatmaps ---")
    model.eval()
    
    # Target the last convolutional layer of ResNet18
    target_layers = [model.model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
    
    # Get a single batch
    imgs, labels = imgs.to(device), labels.to(device)
    adv_imgs = attack(imgs, labels)
    
    # Forward pass to find a misclassified adversarial example
    with torch.no_grad():
        clean_preds = model(imgs).argmax(dim=1)
        adv_preds = model(adv_imgs).argmax(dim=1)
    
    # Find an index where the model was correct on clean, but fooled by adversarial
    fooled_indices = (clean_preds == labels) & (adv_preds != labels)
    if not fooled_indices.any():
        print("No samples were fooled in this batch for Grad-CAM. Try another batch.")
        return
        
    idx = fooled_indices.nonzero(as_tuple=True)[0][0]
    
    clean_img = imgs[idx:idx+1]
    adv_img = adv_imgs[idx:idx+1]
    
    # Generate CAM masks
    clean_mask = cam(input_tensor=clean_img, targets=None)[0]
    adv_mask = cam(input_tensor=adv_img, targets=None)[0]
    
    # Prepare images for plotting (Denormalize)
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1).to(device)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1).to(device)
    
    def denorm_and_rgb(tensor_img):
        img = tensor_img[0] * std + mean
        img = img.permute(1, 2, 0).cpu().numpy()
        return np.clip(img, 0, 1)

    clean_rgb = denorm_and_rgb(clean_img)
    adv_rgb = denorm_and_rgb(adv_img)
    
    clean_cam_img = show_cam_on_image(clean_rgb, clean_mask, use_rgb=True)
    adv_cam_img = show_cam_on_image(adv_rgb, adv_mask, use_rgb=True)
    
    # Plotting
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(clean_cam_img)
    axs[0].set_title(f"Clean (Pred: {clean_preds[idx].item()})")
    axs[0].axis('off')
    
    axs[1].imshow(adv_cam_img)
    axs[1].set_title(f"Adversarial (Pred: {adv_preds[idx].item()})")
    axs[1].axis('off')
    
    plt.tight_layout()
    plt.savefig('grad_cam_comparison.png', dpi=300)
    print("Grad-CAM saved to 'grad_cam_comparison.png'")

def plot_tsne_adversarial(model: nn.Module, loader: DataLoader, attack: Any, device: torch.device, num_samples: int = 1000):
    """
    Visualizes the feature space of clean vs adversarial samples using T-SNE.
    """
    print(f"\n--- Generating T-SNE for Clean vs Adversarial ({num_samples} samples) ---")
    model.eval()
    
    features, labels_list, types = [], [], []
    collected = 0
    
    for imgs, labels in loader:
        if collected >= num_samples: break
        imgs, labels = imgs.to(device), labels.to(device)
        adv_imgs = attack(imgs, labels)
        
        with torch.no_grad():
            clean_feats = model(imgs).cpu().numpy()
            adv_feats = model(adv_imgs).cpu().numpy()
            
        features.append(clean_feats)
        labels_list.append(labels.cpu().numpy())
        types.extend(['Clean'] * imgs.size(0))
        
        features.append(adv_feats)
        labels_list.append(labels.cpu().numpy())
        types.extend(['Adversarial'] * imgs.size(0))
        
        collected += imgs.size(0)
        
    features = np.concatenate(features)
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features)
    
    plt.figure(figsize=(10, 8))
    clean_idx = [i for i, t in enumerate(types) if t == 'Clean']
    adv_idx = [i for i, t in enumerate(types) if t == 'Adversarial']
    
    plt.scatter(features_2d[clean_idx, 0], features_2d[clean_idx, 1], c='blue', label='Clean', alpha=0.5, s=10)
    plt.scatter(features_2d[adv_idx, 0], features_2d[adv_idx, 1], c='red', label='Adversarial', alpha=0.5, s=10, marker='x')
    
    plt.title("T-SNE: Clean vs Adversarial Samples")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig('tsne_adversarial.png', dpi=300)
    print("T-SNE saved to 'tsne_adversarial.png'")

# --- 4. MAIN ORCHESTRATOR FOR HW3 ---
def run_robustness_evaluations(model: nn.Module, params: Dict[str, Any], device: torch.device):
    """
    Executes all requirements for HW3: Robustness, Attacks, KD, and Transferability.
    This function is called by main.py when mode is 'test' or 'attack'.
    """
    print("\n" + "="*50)
    print(" HW3: ROBUSTNESS & ADVERSARIAL EVALUATIONS")
    print("="*50)
    
    _, clean_loader = get_loaders(params)
    cifarc_loader = get_cifar_c_loader(params["cifar_c_dir"], params["batch_size"])
    
    # 1. Define Attacks (PGD-20 L_inf and L_2)
    pgd_linf = torchattacks.PGD(model, eps=params["eps_linf"], alpha=params["alpha"], steps=params["pgd_steps"])
    pgd_l2 = torchattacks.PGDL2(model, eps=params["eps_l2"], alpha=params["alpha"], steps=params["pgd_steps"])
    
    # 2. Evaluate Standard Model
    print("\n--- Evaluating Standard Fine-tuned Model ---")
    acc_clean = evaluate_model(model, clean_loader, device)
    print(f"Accuracy on Clean Test Set: {acc_clean:.2f}%")
    
    if cifarc_loader:
        acc_corrupt = evaluate_model(model, cifarc_loader, device)
        print(f"Accuracy on CIFAR-10-C (Corrupted): {acc_corrupt:.2f}%")
        
    acc_linf = evaluate_model(model, clean_loader, device, attack=pgd_linf)
    print(f"Accuracy under PGD L_inf Attack: {acc_linf:.2f}%")
    
    acc_l2 = evaluate_model(model, clean_loader, device, attack=pgd_l2)
    print(f"Accuracy under PGD L_2 Attack: {acc_l2:.2f}%")
    
    # 3. Visualizations (Using the first batch of the clean loader)
    imgs, labels = next(iter(clean_loader))
    plot_grad_cam(model, imgs, labels, pgd_linf, device)
    plot_tsne_adversarial(model, clean_loader, pgd_linf, device, num_samples=500)
    
    # 4. Transferability and Knowledge Distillation Logic
    # In a full run, you would load the AugMix teacher weights here, 
    # distill to a student, and test transferability.
    print("\n--- Evaluating Transferability ---")
    student = SimpleCNN(num_classes=10).to(device)
    # Note: Normally student would be trained here using the standard_kd_loss.
    # For demonstration of transferability, we attack the teacher and evaluate the student.
    
    student.eval()
    transfer_correct, total = 0, 0
    for imgs, labels in clean_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        adv_imgs = pgd_linf(imgs, labels) # Generated by TEACHER
        
        with torch.no_grad():
            preds = student(adv_imgs).argmax(dim=1)
            total += labels.size(0)
            transfer_correct += preds.eq(labels).sum().item()
            
    transfer_acc = 100.0 * transfer_correct / total
    print(f"Student Accuracy on Teacher's Adversarial Samples (Transferability): {transfer_acc:.2f}%")
    print("\nRobustness evaluation complete!")