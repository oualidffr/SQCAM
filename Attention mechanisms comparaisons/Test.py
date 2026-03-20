import os
import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
from PIL import Image
import csv

# Dataclass
from dataclass import ArrowSegmentationDataset

# Models used in attention mechanisms comparaison
from Models.UNet import SmallUNet # Baseline
from Models.UNet_SQCAM import SmallUNet_AllCA
from Models.UNet_ECA import SmallUNet_AllCA_vECA
from Models.UNet_SE import SmallUNet_AllCA_vSE
from Models.UNet_CBAM import SmallUNet_AllCA_vCBAM

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

#---------------------------------------- ############# ---------------------------------------------# 
#---------------------------------------- Test function ---------------------------------------------# 
#---------------------------------------- ############# ---------------------------------------------# 
def test(seed):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using Device : ", device)

    train_results_base_root = r"D:\FFR\Skin\Train_result"
    test_results_base_root = r"D:\FFR\Skin\Test_result"

    #ISIC
    test_img = r"D:\FFR\Skin\_SK_ALL_Data_Resized\ISIC2018\resized\t"   
    test_lbl_resized = r"D:\FFR\Skin\_SK_ALL_Data_Resized\ISIC2018\resized\t.lbl" 
    test_lbl_original = r"D:\FFR\Skin\_SK_ALL_Data_\ISIC2018\t.lbl" 

    #PH2
    #test_img = r"D:\FFR\Skin\_SK_ALL_Data_Resized\PH2\resized\all"         
    #test_lbl_resized = r"D:\FFR\Skin\_SK_ALL_Data_Resized\PH2\resized\all.lbl" 
    #test_lbl_original = r"D:\FFR\Skin\_SK_ALL_Data_\PH2\all.lbl"
    
    # (_SK_ALL_Data_ : original datasets) (_SK_ALL_Data_Resized : original datasets images resized to HxW = 192x256)
    #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   

    run_name = f"Experiment_name_{seed}"
    run_model_root = os.path.join(train_results_base_root, run_name)
    test_csv_root = os.path.join(test_results_base_root, run_name)
    os.makedirs(test_csv_root)

    # Load trained model      
    model = SmallUNet().to(device) # Baseline exemple.
    model_path = os.path.join(run_model_root, "unet_model.pth")
    assert os.path.exists(model_path), f"Model not found at {model_path}"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Loaded best model from: {model_path}")

    # Dataset & loader  
    test_dataset = ArrowSegmentationDataset(test_img, test_lbl_resized, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    # 
    criterion = BCEDiceLoss()
    #
    metric_totals_small = {k: 0 for k in ['dice', 'iou', 'accuracy', 'sensitivity', 'specificity']}
    metric_totals_large = {k: 0 for k in ['dice', 'iou', 'accuracy', 'sensitivity', 'specificity']}
    total_test_images = 0
    total_test_loss = 0.0

    with torch.no_grad():
        for images, masks_resized, img_ids in tqdm(test_loader, desc="Testing"):
            images, masks_resized = images.to(device), masks_resized.to(device)
            preds = model(images)
            loss = criterion(preds, masks_resized)
            total_test_loss += loss.item() * images.size(0)
            preds_cpu = preds.cpu()
            masks_resized_cpu = masks_resized.cpu()

            # Process each sample in the batch
            for b in range(images.size(0)):
                pred_b = preds_cpu[b:b+1] 
                gt_resized = masks_resized_cpu[b:b+1]

                # ---- Small output metrics (not reported in paper) ----
                metrics_small = compute_metrics(pred_b, gt_resized)
                for k in metric_totals_small:
                    metric_totals_small[k] += metrics_small[k]

                # ---- Large output metrics (original GT) ----
                orig_mask_path = os.path.join(test_lbl_original, img_ids[b] + "_segmentation.png") # ISIC
                #orig_mask_path = os.path.join(test_lbl_original, img_ids[b] + ".bmp")             # PH2
                
                orig_mask = Image.open(orig_mask_path).convert("L")
                orig_size = orig_mask.size[::-1]

                gt = torch.from_numpy(np.array(orig_mask)).float().unsqueeze(0).unsqueeze(0) / 255.
                gt_bin = (gt > 0.5).float()

                resized_prob = torch.nn.functional.interpolate(
                    pred_b, size=orig_size, mode="bilinear", align_corners=False
                )

                metrics_large = compute_metrics(resized_prob, gt_bin)
                for k in metric_totals_large:
                    metric_totals_large[k] += metrics_large[k]

                total_test_images += 1

    # Average loss & metrics
    avg_test_loss = total_test_loss / total_test_images
    for k in metric_totals_small:
        metric_totals_small[k] /= total_test_images
        metric_totals_large[k] /= total_test_images

    # Print results
    print("\nTest Results:")
    print(f"Loss: {avg_test_loss:.4f}")
    print("\n--- Small Output (Resized) ---")
    for k, v in metric_totals_small.items():
        print(f"{k.capitalize()}: {v:.4f}")
    print("\n--- Large Output (Original Size) ---")
    for k, v in metric_totals_large.items():
        print(f"{k.capitalize()}: {v:.4f}")

    # Save to CSV 
    test_csv_path = os.path.join(test_csv_root, "test_metrics_original.csv")
    with open(test_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["loss"] +
                                [f"small_{k}" for k in metric_totals_small.keys()] +
                                [f"large_{k}" for k in metric_totals_large.keys()])
        writer.writeheader()
        writer.writerow({
            "loss": avg_test_loss,
            **{f"small_{k}": v for k, v in metric_totals_small.items()},
            **{f"large_{k}": v for k, v in metric_totals_large.items()},
        })

    print(f"\nSaved test metrics to: {test_csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True, help="Seed")
    args = parser.parse_args()
    test(args.seed)
