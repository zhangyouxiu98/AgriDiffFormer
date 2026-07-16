"""
Run inference on Test folder with trained checkpoint and save visualized results.
"""
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from skimage import io
import matplotlib.pyplot as plt
from datasets import RS_ST as RS
from models.DiffFormer_SCD import BTSCD as Net
from utils.SCD_misc import ConfuseMatrixMeter

# ---------- config ----------
datapath = r"D:\Desktop\SCD_for_lodging_bdd\Test"
ckpt_path = r"D:\Desktop\SCD_for_lodging_bdd\checkpoints\E64_iou75.00_Sek50.74.pth"
out_dir = r"D:\Desktop\SCD_for_lodging_bdd\Test\results"
num_classes = 5

# class names and colors (Landsat)
CLASSES = ['unchanged', 'farmland', 'desert', 'building', 'water']
COLORS = np.array([[255,255,255], [0,155,0], [255,165,0], [230,30,100], [0,170,240]], dtype=np.uint8)
CD_COLORS = np.array([[255,255,255], [255,0,0]], dtype=np.uint8)  # unchanged=white, changed=red

os.makedirs(out_dir, exist_ok=True)

# ---------- load model ----------
print("Loading model...")
net = Net(3, num_classes=num_classes)
state = torch.load(ckpt_path, map_location='cpu', weights_only=True)
if "diff_former.pos_embed" in state:
    del state["diff_former.pos_embed"]
net.load_state_dict(state, strict=False)
net.eval()
print("Model loaded.")

# ---------- data ----------
test_set = RS.Data(datapath, 'test')
test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

tool = ConfuseMatrixMeter(n_class=num_classes)

# ---------- inference ----------
print(f"Running inference on {len(test_set)} images...")
with torch.no_grad():
    for imgs_A, imgs_B, labels_A, labels_B, _, _, name in tqdm(test_loader):
        imgs_A = imgs_A.float()
        imgs_B = imgs_B.float()
        labels_A = labels_A.long()
        labels_B = labels_B.long()

        out_change, outputs_A, outputs_B = net(imgs_A, imgs_B)

        change_mask = torch.argmax(out_change, dim=1)
        preds_A = torch.argmax(outputs_A, dim=1) * change_mask.long()
        preds_B = torch.argmax(outputs_B, dim=1) * change_mask.long()

        # metrics
        pred_all = torch.cat([preds_A, preds_B], dim=0)
        label_all = torch.cat([labels_A, labels_B], dim=0)
        tool.update_cm(pr=pred_all.cpu().numpy(), gt=label_all.cpu().numpy())

        # to numpy
        img_a = (imgs_A.squeeze(0).permute(1,2,0).cpu().numpy() * RS.STD_A + RS.MEAN_A).clip(0,255).astype(np.uint8)
        img_b = (imgs_B.squeeze(0).permute(1,2,0).cpu().numpy() * RS.STD_B + RS.MEAN_B).clip(0,255).astype(np.uint8)
        gt_a = labels_A.squeeze(0).cpu().numpy().astype(np.uint8)
        gt_b = labels_B.squeeze(0).cpu().numpy().astype(np.uint8)
        pr_a = preds_A.squeeze(0).cpu().numpy().astype(np.uint8)
        pr_b = preds_B.squeeze(0).cpu().numpy().astype(np.uint8)
        ch_gt = ((gt_a != 0) | (gt_b != 0)).astype(np.uint8)
        ch_pr = change_mask.squeeze(0).cpu().numpy().astype(np.uint8)

        # colorize
        gt_a_rgb = COLORS[gt_a]
        gt_b_rgb = COLORS[gt_b]
        pr_a_rgb = COLORS[pr_a]
        pr_b_rgb = COLORS[pr_b]
        ch_gt_rgb = CD_COLORS[ch_gt]
        ch_pr_rgb = CD_COLORS[ch_pr]

        sample_name = name[0]

        # ---------- save individual maps ----------
        sub = os.path.join(out_dir, sample_name)
        os.makedirs(sub, exist_ok=True)
        io.imsave(os.path.join(sub, "img_A.png"), img_a)
        io.imsave(os.path.join(sub, "img_B.png"), img_b)
        io.imsave(os.path.join(sub, "label_A_gt.png"), gt_a_rgb)
        io.imsave(os.path.join(sub, "label_B_gt.png"), gt_b_rgb)
        io.imsave(os.path.join(sub, "label_A_pred.png"), pr_a_rgb)
        io.imsave(os.path.join(sub, "label_B_pred.png"), pr_b_rgb)
        io.imsave(os.path.join(sub, "change_gt.png"), ch_gt_rgb)
        io.imsave(os.path.join(sub, "change_pred.png"), ch_pr_rgb)

        # ---------- composite figure ----------
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes[0,0].imshow(img_a);          axes[0,0].set_title('Image A', fontsize=10)
        axes[0,1].imshow(img_b);          axes[0,1].set_title('Image B', fontsize=10)
        axes[0,2].imshow(ch_gt_rgb);      axes[0,2].set_title('Change GT', fontsize=10)
        axes[0,3].imshow(ch_pr_rgb);      axes[0,3].set_title('Change Pred', fontsize=10)
        axes[1,0].imshow(gt_a_rgb);       axes[1,0].set_title('Label A GT', fontsize=10)
        axes[1,1].imshow(gt_b_rgb);       axes[1,1].set_title('Label B GT', fontsize=10)
        axes[1,2].imshow(pr_a_rgb);       axes[1,2].set_title('Label A Pred', fontsize=10)
        axes[1,3].imshow(pr_b_rgb);       axes[1,3].set_title('Label B Pred', fontsize=10)
        for ax in axes.flat:
            ax.axis('off')
        plt.tight_layout(pad=0.5)
        plt.savefig(os.path.join(sub, "overview.png"), dpi=150, bbox_inches='tight')
        plt.close()

        # ---------- legend ----------
        fig2, ax2 = plt.subplots(figsize=(6, 2))
        ax2.axis('off')
        patches = [plt.Rectangle((i*0.12, 0), 0.1, 0.1, facecolor=c/255, edgecolor='gray')
                   for i, c in enumerate(COLORS)]
        ax2.legend(patches, CLASSES, ncol=5, loc='center', fontsize=9,
                   title='Semantic Classes (white=unchanged)')
        fig2.savefig(os.path.join(sub, "legend.png"), dpi=100, bbox_inches='tight')
        plt.close()

# ---------- metrics ----------
print()
score = tool.get_scores()
print(f"OA={score['acc']:.4f}  mIoU={score['mIoU']:.4f}  Sek={score['Sek']:.4f}  Fscd={score['Fscd']:.4f}  Pre={score['Pre']:.4f}  Rec={score['Rec']:.4f}")
print(f"\nResults saved to: {out_dir}")
