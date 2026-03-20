import os
import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
import csv

# Dataclass
from dataclass import ArrowSegmentationDataset

# Models
from Models.deeplabv3plus import DeepLabV3Plus
from Models.Resnet18_FPN import FPN_Segmentation
from Models.SegFormer_B0 import SegFormerB0

# --------------------------- X --------------------------- # 
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def compute_metrics(pred, target, threshold=0.5):
    pred_bin = (pred > threshold).float()
    target = target.float()

    TP = (pred_bin * target).sum().item()
    TN = ((1 - pred_bin) * (1 - target)).sum().item()
    FP = (pred_bin * (1 - target)).sum().item()
    FN = ((1 - pred_bin) * target).sum().item()

    eps = 1e-6
    dice = (2 * TP) / (2 * TP + FP + FN + eps)
    iou = TP / (TP + FP + FN + eps)
    accuracy = (TP + TN) / (TP + TN + FP + FN + eps)
    sensitivity = TP / (TP + FN + eps)
    specificity = TN / (TN + FP + eps)

    return {
        'dice': dice,
        'iou': iou,
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity
    }

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (
            inputs.sum() + targets.sum() + self.smooth
        )
        return 1 - dice

class BCEDiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.bce = nn.BCELoss()
        self.dice = DiceLoss(smooth)

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        dice_loss = self.dice(inputs, targets)
        return bce_loss + dice_loss

# --------------------------- ######## --------------------------- #
# --------------------------- Training --------------------------- #
# --------------------------- ######## --------------------------- #    

def train(seed):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using Device : ", device)

    results_base_root= r"D:\FFR\Skin\Train_result"
    run_name = f"Experiment_name_{seed}"
    results_base = os.path.join(results_base_root, run_name)
    os.makedirs(results_base)

    # CSV Setup  
    metrics_csv_path = os.path.join(results_base, "metrics.csv")
    csv_file = open(metrics_csv_path, mode='w', newline='')
    csv_writer = None

    # ISIC Data directories
    TRimg = r"D:\FFR\Skin\_SK_ALL_Data_Resized\ISIC2018\resized\tr"
    TRlbl = r"D:\FFR\Skin\_SK_ALL_Data_Resized\ISIC2018\resized\tr.lbl"
    Vimg = r"D:\FFR\Skin\_SK_ALL_Data_Resized\ISIC2018\resized\v"
    Vlbl = r"D:\FFR\Skin\_SK_ALL_Data_Resized\ISIC2018\resized\v.lbl"

    # (_SK_ALL_Data_Resized : original datasets images resized to HxW = 192x256)
    #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #

    # Datasets and loaders   
    train_dataset = ArrowSegmentationDataset(TRimg, TRlbl, is_train=True) 
    val_dataset = ArrowSegmentationDataset(Vimg, Vlbl, is_train=False)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    # Model, optimizer, loss   
    model = SegFormerB0().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    criterion = BCEDiceLoss()
    #
    total_nbr_epochs = 100
    early_stop_patience = 10
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(total_nbr_epochs):
        
        model.train()
        total_loss = 0
        metric_totals_train = {k: 0 for k in ['dice', 'iou', 'accuracy', 'sensitivity', 'specificity']}

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_nbr_epochs}")
        for images, masks, id in loop:
            images, masks = images.to(device), masks.to(device)
            output_final = model(images)
            output_final = torch.sigmoid(output_final)
            loss = criterion(output_final, masks) 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            #
            metrics = compute_metrics(output_final, masks)
            for k in metric_totals_train:
                metric_totals_train[k] += metrics[k]
            loop.set_postfix(train_loss=loss.item())
        #
        avg_train_loss = total_loss / len(train_loader)
        for k in metric_totals_train:
            metric_totals_train[k] /= len(train_loader)
        #
        print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}")
        print(f"→ Train Metrics - Dice: {metric_totals_train['dice']:.4f} | IoU: {metric_totals_train['iou']:.4f} | "
              f"Acc: {metric_totals_train['accuracy']:.4f} | Sn: {metric_totals_train['sensitivity']:.4f} | Sp: {metric_totals_train['specificity']:.4f}")

        # --------------------------- Validation --------------------------- # 
        model.eval()
        val_loss = 0
        total_val_images = 0
        metric_totals = {k: 0 for k in ['dice', 'iou', 'accuracy', 'sensitivity', 'specificity']}

        with torch.no_grad():
            for images, masks, id in val_loader:
                images, masks = images.to(device), masks.to(device)
                preds = model(images)
                preds = torch.sigmoid(preds)
                loss = criterion(preds, masks)

                batch_size_now = images.size(0)
                val_loss += loss.item() * batch_size_now
                total_val_images += batch_size_now

                metrics = compute_metrics(preds, masks)
                for k in metric_totals:
                    metric_totals[k] += metrics[k] * batch_size_now

        avg_val_loss = val_loss / total_val_images
        for k in metric_totals:
            metric_totals[k] /= total_val_images

        print(f"Epoch {epoch+1} - Validation Loss: {avg_val_loss:.4f}")
        print(f"→ Dice: {metric_totals['dice']:.4f} | IoU: {metric_totals['iou']:.4f} | "
              f"Acc: {metric_totals['accuracy']:.4f} | Sn: {metric_totals['sensitivity']:.4f} | Sp: {metric_totals['specificity']:.4f}")

        # Scheduler and Early Stopping 
        scheduler.step(avg_val_loss)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(results_base, "model.pth"))
            print("New best model (saved).")
        else:
            patience_counter += 1
            print(f"Validation loss did not improve. [Patience: {patience_counter}/{early_stop_patience}]")
            if patience_counter >= early_stop_patience:
                print("Early stopping triggered.")
                break

        # Save metrics to CSV
        row = {
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'train_dice': metric_totals_train['dice'],
            'val_dice': metric_totals['dice'],
            'train_iou': metric_totals_train['iou'],
            'val_iou': metric_totals['iou'],
            'train_acc': metric_totals_train['accuracy'],
            'val_acc': metric_totals['accuracy'],
            'train_sensitivity': metric_totals_train['sensitivity'],
            'val_sensitivity': metric_totals['sensitivity'],
            'train_specificity': metric_totals_train['specificity'],
            'val_specificity': metric_totals['specificity'],
        }
        if csv_writer is None:
            csv_writer = csv.DictWriter(csv_file, fieldnames=row.keys())
            csv_writer.writeheader()
        csv_writer.writerow(row)
        csv_file.flush()

    csv_file.close()
    print(f"\n Saved metrics to: {metrics_csv_path}")
    print(f"\n Training completed for seed {seed}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True, help="Random seed for reproducibility")
    args = parser.parse_args()
    train(args.seed)
