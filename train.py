import os
os.environ["TORCH_ONNX_DISABLE_DIAGNOSTICS"] = "1"

import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys

from utils.SCD_misc import ConfuseMatrixMeter, AverageMeter
from datasets import RS_ST as RS
from models.DiffFormer_SCD import BTSCD as Net


# =====================================================
# ================= Safe BCE (重要) ===================
# =====================================================
def safe_bce(pred, target, eps=1e-6):
    pred = torch.clamp(pred, eps, 1.0 - eps)
    target = torch.clamp(target, eps, 1.0 - eps)
    loss = F.binary_cross_entropy(pred, target)
    return torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)


# =====================================================
# ============== Distillation helpers =================
# =====================================================
def semantic_boundary_map(P_sem):
    """
    P_sem: [B, C, H, W] soft probabilities
    return: [B, 1, H, W]
    """
    m1 = F.max_pool2d(P_sem, 3, 1, 1)
    m2 = -F.max_pool2d(-P_sem, 3, 1, 1)
    bmap = torch.max(m1 - m2, dim=1, keepdim=True)[0]
    bmap = torch.clamp(bmap, 0.0, 1.0)
    return torch.nan_to_num(bmap, nan=0.0, posinf=1.0, neginf=0.0)


def loss_sem_to_cd(P_sem1, P_sem2, P_cd, alpha=10.0, tau=0.05):
    """
    Semantic -> Change distillation
    """
    D_sem = torch.mean(torch.abs(P_sem1 - P_sem2), dim=1, keepdim=True)
    target = torch.sigmoid(alpha * (D_sem - tau))
    target = torch.nan_to_num(target, nan=0.0, posinf=1.0, neginf=0.0)

    pred_change = P_cd[:, 1:2, :, :]
    return safe_bce(pred_change, target)


def loss_cd_to_sem(
    P_sem1, P_sem2, P_cd,
    lambda_cons=1.0,
    lambda_bdy=0.2,
    detach_cd=True
):
    """
    Change -> Semantic distillation
    """
    eps = 1e-6

    # ===== consistency (unchanged regions) =====
    log_p1 = torch.log(torch.clamp(P_sem1, eps, 1.0))
    kl = F.kl_div(log_p1, P_sem2, reduction='none').sum(dim=1, keepdim=True)
    M_unc = 1.0 - P_cd[:, 1:2, :, :]
    consistency = (M_unc * kl).mean()

    # ===== boundary alignment =====
    b_sem1 = semantic_boundary_map(P_sem1)
    b_sem2 = semantic_boundary_map(P_sem2)
    b_sem = torch.clamp(b_sem1 + b_sem2, 0.0, 1.0)
    b_sem = torch.nan_to_num(b_sem, nan=0.0, posinf=1.0, neginf=0.0)

    b_target = P_cd[:, 1:2, :, :]
    if detach_cd:
        b_target = b_target.detach()

    boundary_loss = safe_bce(b_sem, b_target)

    return lambda_cons * consistency + lambda_bdy * boundary_loss


# =====================================================
# ================= LR scheduler ======================
# =====================================================
def adjust_lr(optimizer, curr_iter, all_iter, init_lr, lr_decay_power):
    scale = (1. - float(curr_iter) / all_iter) ** lr_decay_power
    for param_group in optimizer.param_groups:
        param_group['lr'] = init_lr * scale


# =====================================================
# ======================= Main ========================
# =====================================================
def main(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    writer = SummaryWriter(args.chkpt_dir)

    net = Net(3, num_classes=args.num_classes).cuda()
    print(f"Model params: {sum(p.numel() for p in net.parameters())}")

    train_set = RS.Data(args.datapath, 'train', augmentation=True)
    val_set = RS.Data(args.datapath, 'val')

    train_loader = DataLoader(train_set, batch_size=args.train_batchsize, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.val_batchsize, shuffle=False, num_workers=0)

    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, net.parameters()),
        lr=args.lr, momentum=0.9, weight_decay=5e-4, nesterov=True
    )

    train(train_loader, val_loader, net, optimizer, writer, args)
    writer.close()
    print("Training finished.")


def train(train_loader, val_loader, net, optimizer, writer, args):
    CE = nn.CrossEntropyLoss(ignore_index=0)
    CE_change = nn.CrossEntropyLoss()

    tool4metric = ConfuseMatrixMeter(n_class=args.num_classes)
    bestscore = 0.0

    T = args.distill_T

    all_iters = float(len(train_loader) * args.epoch)

    for epc in range(args.epoch):
        # ==================== TRAIN ====================
        net.train()
        torch.cuda.empty_cache()

        meter = AverageMeter()
        loop = tqdm(train_loader, file=sys.stdout)

        for i, (imgs_A, imgs_B, labels_A, labels_B, _, _, _) in enumerate(loop):
            curr_iter = epc * len(train_loader) + i + 1
            adjust_lr(optimizer, curr_iter, all_iters, args.lr, args.lr_decay_power)

            imgs_A = imgs_A.cuda().float()
            imgs_B = imgs_B.cuda().float()
            labels_A = labels_A.cuda().long()
            labels_B = labels_B.cuda().long()
            labels_bn = (labels_A > 0).long()

            optimizer.zero_grad()

            out_change, outputs_A, outputs_B = net(imgs_A, imgs_B)

            loss_seg = 0.5 * (CE(outputs_A, labels_A) + CE(outputs_B, labels_B))
            loss_cd = CE_change(out_change, labels_bn)

            P_sem1 = F.softmax(outputs_A / T, dim=1)
            P_sem2 = F.softmax(outputs_B / T, dim=1)
            P_cd = F.softmax(out_change, dim=1)

            L_s2c = loss_sem_to_cd(P_sem1, P_sem2, P_cd,
                                   args.distill_alpha, args.distill_tau)

            # ===== CD→Sem warm-up =====
            if epc < 5:
                L_c2s = torch.zeros(1, device=imgs_A.device)
            else:
                L_c2s = loss_cd_to_sem(
                    P_sem1, P_sem2, P_cd,
                    args.lambda_c2s_cons, args.lambda_c2s_bdy, detach_cd=True
                )

            loss = (
                loss_seg +
                loss_cd +
                args.lambda_s2c * L_s2c +
                args.lambda_c2s * L_c2s
            )

            loss.backward()
            optimizer.step()

            meter.update(loss.item())
            loop.set_postfix(loss=meter.val)

        writer.add_scalar('train/loss', meter.val, epc)

        # ==================== VAL ====================
        net.eval()
        tool4metric.clear()

        with torch.no_grad():
            for imgs_A, imgs_B, labels_A, labels_B, _, _, _ in val_loader:
                imgs_A = imgs_A.cuda().float()
                imgs_B = imgs_B.cuda().float()
                labels_A = labels_A.cuda().long()
                labels_B = labels_B.cuda().long()

                out_change, outputs_A, outputs_B = net(imgs_A, imgs_B)

                change_mask = torch.argmax(out_change, dim=1)
                preds_A = torch.argmax(outputs_A, dim=1) * change_mask
                preds_B = torch.argmax(outputs_B, dim=1) * change_mask

                tool4metric.update_cm(
                    pr=torch.cat([preds_A, preds_B]).cpu().numpy(),
                    gt=torch.cat([labels_A, labels_B]).cpu().numpy()
                )

        score = tool4metric.get_scores()
        print(f"[Epoch {epc}] mIoU={score['mIoU']:.4f}  Sek={score['Sek']:.4f}")

        writer.add_scalar('val/mIoU', score['mIoU'], epc)
        writer.add_scalar('val/Sek', score['Sek'], epc)

        if score['Sek'] > bestscore:
            bestscore = score['Sek']
            torch.save(
                net.state_dict(),
                os.path.join(args.chkpt_dir,
                             f"E{epc}_mIoU{score['mIoU']*100:.1f}_Sek{score['Sek']*100:.1f}.pth")
            )


# =====================================================
# ===================== Entry =========================
# =====================================================
if __name__ == '__main__':
    working_path = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser("DiffFormer + Bidirectional Distillation")
    parser.add_argument("--dataname", default="Landsat")
    parser.add_argument("--modelname", default="DiffFormer_ABD")
    parser.add_argument("--datapath", default=r"E:\zjl\SCD\dataset\Landsat_BT")
    parser.add_argument("--num_classes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--lr_decay_power", type=float, default=1.5)
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--train_batchsize", type=int, default=4)
    parser.add_argument("--val_batchsize", type=int, default=4)

    parser.add_argument("--distill_T", type=float, default=2.0)
    parser.add_argument("--distill_alpha", type=float, default=10.0)
    parser.add_argument("--distill_tau", type=float, default=0.05)
    parser.add_argument("--lambda_s2c", type=float, default=0.2)
    parser.add_argument("--lambda_c2s", type=float, default=0.2)
    parser.add_argument("--lambda_c2s_cons", type=float, default=1.0)
    parser.add_argument("--lambda_c2s_bdy", type=float, default=0.2)

    args = parser.parse_args()

    chkpt_dir = os.path.join(working_path, "checkpoints", args.dataname, args.modelname)
    os.makedirs(chkpt_dir, exist_ok=True)

    run_id = len([d for d in os.listdir(chkpt_dir) if d.startswith("run_")])
    args.chkpt_dir = os.path.join(chkpt_dir, f"run_{run_id:04d}")
    os.makedirs(args.chkpt_dir, exist_ok=True)

    main(args)
