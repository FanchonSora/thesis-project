import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────────────────────
class ResidualSEBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.1):
        super().__init__()

        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm3d(out_channels)

        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm3d(out_channels)

        self.act = nn.LeakyReLU(0.01, inplace=True)
        self.dropout = nn.Dropout3d(dropout_rate)

        self.residual = (
            nn.Conv3d(in_channels, out_channels, 1, bias=False)
            if in_channels != out_channels else nn.Identity()
        )

        # SE nhẹ hơn (ratio = 32)
        se_ch = max(out_channels // 32, 1)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(out_channels, se_ch, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(se_ch, out_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        identity = self.residual(x)

        out = self.act(self.norm1(self.conv1(x)))
        out = self.dropout(out)
        out = self.norm2(self.conv2(out))

        out = out * self.se(out)
        return self.act(out + identity)

class ImprovedAttentionGate3D(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()

        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, 1, bias=False),
            nn.InstanceNorm3d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, 1, bias=False),
            nn.InstanceNorm3d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, 1),
            nn.Sigmoid()
        )

        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(F_l, max(F_l // 16, 1), 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(max(F_l // 16, 1), F_l, 1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='trilinear', align_corners=False)

        psi = self.psi(self.relu(g1 + x1))
        return x * psi * self.channel_gate(x)

class DeepSupervisionHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // 2, 3, padding=1, bias=False),
            nn.InstanceNorm3d(in_channels // 2),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(in_channels // 2, num_classes, 1)
        )

    def forward(self, x, target_size=None):
        x = self.conv(x)
        if target_size and x.shape[2:] != target_size:
            x = F.interpolate(x, size=target_size, mode='trilinear', align_corners=False)
        return x

class ConvBlock3D(nn.Module):
    """Double convolution block for UNet"""
    def __init__(self, in_channels, out_channels, dropout_rate=0.1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.InstanceNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.InstanceNorm3d(out_channels)
        self.dropout = nn.Dropout3d(dropout_rate)
        self.act = nn.LeakyReLU(0.01, inplace=True)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.dropout(x)
        
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act(x)
        return x

class DownBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.1):
        super().__init__()
        self.pool = nn.MaxPool3d(2)
        self.conv = ResidualSEBlock3D(in_channels, out_channels, dropout_rate)

    def forward(self, x):
        return self.conv(self.pool(x))

class UpBlock3D(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, dropout_rate=0.1, use_attention=True):
        super().__init__()

        self.use_attention = use_attention
        self.upsample = nn.ConvTranspose3d(in_channels, out_channels, 2, stride=2)

        if use_attention:
            self.attention = ImprovedAttentionGate3D(
                out_channels, skip_channels, out_channels // 2
            )

        self.conv = ResidualSEBlock3D(out_channels + skip_channels, out_channels, dropout_rate)

    def forward(self, x, skip):
        x = self.upsample(x)

        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=False)

        if self.use_attention:
            skip = self.attention(x, skip)

        return self.conv(torch.cat([x, skip], dim=1))

# ─────────────────────────────────────────────────────────────────────────────
# UNet3D
# ─────────────────────────────────────────────────────────────────────────────

class UNet3D(nn.Module):
    def __init__(
        self,
        input_shape=(4, 64, 64, 64),
        num_classes=4,
        base_filters=32,
        depth=4,
        dropout_rate=0.2,
        use_deep_supervision=True
    ):
        super().__init__()

        self.use_deep_supervision = use_deep_supervision
        in_channels = input_shape[0]

        filters = [base_filters * (2 ** i) for i in range(depth + 1)]  # lên 512

        self.init_conv = ResidualSEBlock3D(in_channels, filters[0], dropout_rate)

        self.down_blocks = nn.ModuleList([
            DownBlock3D(filters[i], filters[i + 1], dropout_rate)
            for i in range(depth)
        ])

        # Bottleneck gọn hơn
        self.bottleneck = ResidualSEBlock3D(filters[depth], filters[depth], dropout_rate)

        self.up_blocks = nn.ModuleList([
            UpBlock3D(
                filters[depth - i],
                filters[depth - i - 1],
                filters[depth - i - 1],
                dropout_rate,
                use_attention=(i < 2)  # 👈 attention chỉ ở deep levels
            )
            for i in range(depth)
        ])

        if use_deep_supervision:
            self.ds_heads = nn.ModuleList([
                DeepSupervisionHead(filters[depth - i - 1], num_classes)
                for i in range(depth - 1)
            ])

        self.output_conv = nn.Conv3d(filters[0], num_classes, 1)

    def forward(self, x, return_deep_supervision=None):
        if return_deep_supervision is None:
            return_deep_supervision = self.training and self.use_deep_supervision

        skips = []
        x = self.init_conv(x)
        skips.append(x)

        for down in self.down_blocks:
            x = down(x)
            skips.append(x)

        x = self.bottleneck(x)
        skips = skips[:-1]

        ds_outputs = []
        target_size = skips[0].shape[2:]

        for i, up in enumerate(self.up_blocks):
            x = up(x, skips[-(i + 1)])
            if return_deep_supervision and i < len(self.ds_heads):
                ds_outputs.append(self.ds_heads[i](x, target_size))

        main_out = self.output_conv(x)
        return (main_out, ds_outputs) if return_deep_supervision else main_out
# ─────────────────────────────────────────────────────────────────────────────
# Loss functions & metrics (unchanged from training)
# ─────────────────────────────────────────────────────────────────────────────

EPSILON = 1e-6

def main_loss(y_true, y_pred, lambda_weight=15.0, class_weights=None,
              label_smoothing=0.1):
    if class_weights is None:
        class_weights = [1.0, 1.5, 1.2, 2.0]
    cw = torch.tensor(class_weights, device=y_pred.device, dtype=torch.float32)

    target = y_true.argmax(dim=1)
    if label_smoothing > 0:
        n_c = y_pred.shape[1]
        y_smooth = y_true * (1 - label_smoothing) + label_smoothing / n_c
        ce_loss  = -(y_smooth * F.log_softmax(y_pred, dim=1)).sum(dim=1).mean()
    else:
        ce_loss = F.cross_entropy(y_pred, target, weight=cw, reduction='mean')

    y_soft     = torch.softmax(y_pred, dim=1)
    dice_terms = []
    for c in range(y_pred.shape[1]):
        ytc, ypc = y_true[:, c], y_soft[:, c]
        present  = ytc.sum(dim=(1, 2, 3)) > 0
        if present.any():
            inter  = (ytc * ypc).sum(dim=(1, 2, 3))
            union  = ytc.sum(dim=(1, 2, 3)) + ypc.sum(dim=(1, 2, 3))
            dice   = (2 * inter + EPSILON) / (union + EPSILON)
            dice_c = (1 - dice[present].mean()) * cw[c]
        else:
            dice_c = ypc.mean() * cw[c] * 0.5
        dice_terms.append(dice_c)

    return ce_loss + lambda_weight * torch.stack(dice_terms).mean()

def dice_coef(y_true, y_pred, smooth=EPSILON, class_weights=None):
    if class_weights is None:
        class_weights = torch.tensor([1.0, 1.5, 1.2, 2.0], device=y_pred.device)
    probs = torch.softmax(y_pred, dim=1) if y_pred.dim() == 5 else y_pred
    inter = torch.sum(y_true * probs, dim=(2, 3, 4))
    union = torch.sum(y_true, dim=(2, 3, 4)) + torch.sum(probs, dim=(2, 3, 4))
    dice  = (2 * inter + smooth) / (union + smooth)
    w     = class_weights[1:]
    return (dice[:, 1:].mean(dim=0) * w).sum() / w.sum()

def _dice_class(y_true, y_pred, c, smooth=EPSILON):
    probs = torch.softmax(y_pred, dim=1) if y_pred.dim() == 5 else y_pred
    ytc, ypc = y_true[:, c], probs[:, c]
    present  = ytc.sum(dim=(1, 2, 3)) > 0
    if not present.any():
        ok = (ypc.reshape(ypc.shape[0], -1).max(dim=1)[0] < 0.5).all()
        return torch.tensor(1.0 if ok else 0.0, device=y_pred.device)
    inter = (ytc * ypc).sum(dim=(1, 2, 3))
    union = ytc.sum(dim=(1, 2, 3)) + ypc.sum(dim=(1, 2, 3))
    return ((2 * inter + smooth) / (union + smooth))[present].mean()

def dice_coef_necrotic(y_true, y_pred, smooth=EPSILON):
    return _dice_class(y_true, y_pred, 1, smooth)

def dice_coef_edema(y_true, y_pred, smooth=EPSILON):
    return _dice_class(y_true, y_pred, 2, smooth)

def dice_coef_enhancing(y_true, y_pred, smooth=EPSILON):
    return _dice_class(y_true, y_pred, 3, smooth)

def precision(y_true, y_pred):
    pl = y_pred.argmax(dim=1) > 0
    tl = y_true.argmax(dim=1) > 0
    if tl.sum() == 0:
        return torch.tensor(1.0 if pl.sum() == 0 else 0.0, device=y_pred.device)
    tp = (pl & tl).sum()
    fp = (pl & ~tl).sum()
    return tp.float() / (tp + fp + EPSILON)

def sensitivity(y_true, y_pred):
    pl = y_pred.argmax(dim=1) > 0
    tl = y_true.argmax(dim=1) > 0
    if tl.sum() == 0:
        return torch.tensor(1.0, device=y_pred.device)
    tp = (pl & tl).sum()
    fn = (~pl & tl).sum()
    return tp.float() / (tp + fn + EPSILON)

def specificity(y_true, y_pred):
    pl = y_pred.argmax(dim=1) > 0
    tl = y_true.argmax(dim=1) > 0
    bg = ~tl
    tn = (~pl & bg).sum()
    fp = (pl & bg).sum()
    return tn.float() / (tn + fp + EPSILON)

# ─────────────────────────────────────────────────────────────────────────────
# UNET_Curriculum wrapper
# ─────────────────────────────────────────────────────────────────────────────
EPSILON = 1e-6

class UNET_Curriculum(nn.Module):
    def __init__(self, base_model, class_weights=None):
        super().__init__()
        self.base_model = base_model

        if class_weights is None:
            self.class_weights = [1.0, 1.5, 1.2, 2.0]
        else:
            self.class_weights = class_weights

        self.use_t1mri_balancing = False
        self.t1mri_balancer = nn.Identity()
    # Forward
    def forward(self, inputs):
        if self.use_t1mri_balancing:
            inputs = self.t1mri_balancer(inputs)
        return self.base_model(inputs)
    # Utils
    def _extract_logits(self, outputs):
        if isinstance(outputs, (tuple, list)):
            return outputs[0]
        return outputs

    def _ensure_mask_and_pred_dims(self, masks, logits):
        num_classes = logits.shape[1]
        spatial_size = logits.shape[2:]

        if masks.dim() == 3:
            masks = masks.unsqueeze(0)

        if masks.shape[1:] != spatial_size:
            masks = F.interpolate(
                masks.unsqueeze(1).float(),
                size=spatial_size,
                mode="nearest"
            ).squeeze(1).long()

        masks_onehot = F.one_hot(
            masks,
            num_classes=num_classes
        ).permute(0, 4, 1, 2, 3).float()

        return masks_onehot, logits

    # Metrics
    def _compute_metrics(self, masks_onehot, logits, loss):
        dice_score = dice_coef(
            masks_onehot,
            logits,
            class_weights=torch.tensor(
                self.class_weights,
                device=logits.device
            )
        )

        dice_nec = dice_coef_necrotic(masks_onehot, logits)
        dice_ede = dice_coef_edema(masks_onehot, logits)
        dice_enh = dice_coef_enhancing(masks_onehot, logits)

        prec = precision(masks_onehot, logits)
        sens = sensitivity(masks_onehot, logits)
        spec = specificity(masks_onehot, logits)

        def safe_scalar(x):
            if isinstance(x, (int, float)):
                return float(x)
            return float(x.detach().mean().cpu().item())

        return {
            "loss": safe_scalar(loss),
            "dice_coefficient": safe_scalar(dice_score),
            "precision": safe_scalar(prec),
            "sensitivity": safe_scalar(sens),
            "specificity": safe_scalar(spec),
            "dice_coef_necrotic": safe_scalar(dice_nec),
            "dice_coef_edema": safe_scalar(dice_ede),
            "dice_coef_enhancing": safe_scalar(dice_enh),
        }
    # Training step
    def train_step(self, batch, optimizer, criterion, scaler=None):
        images, masks = batch
        device = next(self.parameters()).device
        images = images.float().to(device)        # (B,4,D,H,W)
        masks  = masks.long().to(device)          # (B,D,H,W)

        self.train()
        optimizer.zero_grad(set_to_none=True)
        # Forward + Loss
        with torch.amp.autocast(
            device_type='cuda',
            dtype=torch.float16 if scaler is not None else torch.float32
        ):
            outputs = self.forward(images)
            loss = criterion(outputs, masks)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            optimizer.step()
        with torch.no_grad():
            # Extract main logits only
            if isinstance(outputs, (tuple, list)):
                logits = outputs[0]
            else:
                logits = outputs
            if masks.shape != logits.shape[2:]:
                masks_resized = F.interpolate(
                    masks.unsqueeze(1).float(),
                    size=logits.shape[2:],
                    mode="nearest"
                ).squeeze(1).long()
            else:
                masks_resized = masks

            masks_onehot = F.one_hot(
                masks_resized,
                num_classes=logits.shape[1]
            ).permute(0, 4, 1, 2, 3).float()

            metrics = self._compute_metrics(
                masks_onehot,
                logits,
                loss
            )
        # Cleanup
        del outputs, logits, masks_onehot
        torch.cuda.empty_cache()

        return metrics
# ─────────────────────────────────────────────────────────────────────────────
# CombinedLoss  
# ─────────────────────────────────────────────────────────────────────────────

class CombinedLoss(nn.Module):
    def __init__(self, num_classes=4, class_weights=None, ds_weights=None):
        super().__init__()
        self.num_classes = num_classes
        self.eps = 1e-6

        if class_weights is None:
            class_weights = [1.0, 1.0, 1.0, 1.0]

        self.register_buffer(
            "class_weights",
            torch.tensor(class_weights, dtype=torch.float32)
        )

        if ds_weights is None:
            ds_weights = [0.5, 0.3, 0.2]
        self.ds_weights = ds_weights
    # CORE LOSS
    def compute_loss(self, logits, target):
        # Cross Entropy 
        ce_loss = F.cross_entropy(
            logits,
            target,
            weight=self.class_weights,
            ignore_index=-1
        )
        # Dice (foreground only) 
        probs = torch.softmax(logits, dim=1)
        dice_loss = 0.0
        valid_classes = 0

        for c in range(1, self.num_classes): 
            gt = (target == c).float()

            if gt.sum() == 0:
                continue  
            pred = probs[:, c]

            intersection = (pred * gt).sum()
            union = pred.sum() + gt.sum()

            dice = (2.0 * intersection + self.eps) / (union + self.eps)
            dice_loss += (1.0 - dice)
            valid_classes += 1

        if valid_classes > 0:
            dice_loss /= valid_classes

        return ce_loss + dice_loss
    # FORWARD (deep supervision)
    def forward(self, outputs, target):
        if isinstance(outputs, tuple):
            main_out, ds_outs = outputs

            loss = self.compute_loss(main_out, target)

            for i, ds in enumerate(ds_outs):
                w = self.ds_weights[i] if i < len(self.ds_weights) else 0.1
                loss += w * self.compute_loss(ds, target)

            return loss
        else:
            return self.compute_loss(outputs, target)
        
def create_unet_curriculum(input_shape=(4, 64, 64, 64),num_classes=4,base_lr=2e-4,weight_decay=1e-4,class_weights=None,use_deep_supervision=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"CREATING UNET")
    print(f"Device: {device}")
    if class_weights is None:
        class_weights = [1.0, 1.5, 1.2, 2.0]
        print(f"Using RECOMMENDED class weights: {class_weights}")
    enhanced_config = {
        'base_filters': 32, 
        'depth': 4,
        'dropout_rate': 0.2,
        'use_deep_supervision': use_deep_supervision
    }
    # Create base model
    base_model = UNet3D(
        input_shape=input_shape,
        num_classes=num_classes,
        **enhanced_config
    )
    model = UNET_Curriculum(base_model=base_model)
    model = model.to(device)
    criterion = CombinedLoss(
        num_classes=num_classes,
        class_weights=class_weights,
    ).to(device)
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=base_lr,
        weight_decay=weight_decay
    )
    # Print summary
    params_count = sum(p.numel() for p in model.parameters())
    print(f"\n📊 Training Configuration:")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Optimizer: AdamW")
    print(f"   LR: {base_lr}")
    print(f"   Weight decay: {weight_decay}")
    print(f"   Deep supervision: {use_deep_supervision}")
    return model, criterion, optimizer, device