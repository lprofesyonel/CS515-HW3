import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any

# Projemizin güncel veri yükleyicisi
from train import get_loaders

@torch.no_grad()
def run_test(model: nn.Module, params: Dict[str, Any], device: torch.device) -> None:
    """
    Evaluates the trained model on the clean test dataset and calculates 
    overall and class-wise accuracy.

    Args:
        model (nn.Module): The PyTorch neural network model.
        params (Dict[str, Any]): The configuration dictionary.
        device (torch.device): The computation device (CPU or CUDA).
    """
    print("\n" + "="*40)
    print(" STANDARD TEST EVALUATION")
    print("="*40)
    
    _, test_loader = get_loaders(params)

    try:
        model.load_state_dict(torch.load(params["save_path"], map_location=device))
        print(f"Successfully loaded best weights from: {params['save_path']}")
    except FileNotFoundError:
        print(f"[Warning] Weights not found at '{params['save_path']}'. Testing with current memory weights.")

    model.eval()

    correct, n = 0, 0
    class_correct = [0] * params["num_classes"]
    class_total   = [0] * params["num_classes"]

    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        preds = model(imgs).argmax(1)
        
        correct += preds.eq(labels).sum().item()
        n       += imgs.size(0)
        
        for p, t in zip(preds, labels):
            class_correct[t] += (p == t).item()
            class_total[t]   += 1

    print(f"\n=== Test Results ===")
    print(f"Overall accuracy: {correct/n:.4f}  ({correct}/{n})\n")
    
    for i in range(params["num_classes"]):
        if class_total[i] > 0:
            acc = class_correct[i] / class_total[i]
            print(f"  Class {i}: {acc:.4f}  ({class_correct[i]}/{class_total[i]})")