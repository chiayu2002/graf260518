import torch
# from torch.nn import functional as F
import torch.utils.data
import torch.utils.data.distributed
from torch import autograd
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np
import os
import torchvision.models as models


class MCE_Loss(nn.Module):
    def __init__(self):
        super(MCE_Loss, self).__init__()

    def __call__(self, n_each_task, output, target):
        loss = []
        for i, n in enumerate(n_each_task):
            if i == 0:
                loss.append(nn.CrossEntropyLoss()(output[:, :n], target[:, :n]))
            else:
                summation = sum(n_each_task[:i])
                loss.append(nn.CrossEntropyLoss()(output[:, summation:summation+n], target[:, summation:summation+n]))
        
        return sum(loss)

class CCSRLoss(nn.Module):
    def __init__(self, device='cuda:0'):
        super().__init__()
        self.mse = nn.MSELoss()
        # 加載預訓練 VGG 作為知覺損失提取器
        vgg = models.vgg19(pretrained=True).features[:16].to(device).eval()
        for p in vgg.parameters(): p.requires_grad = False
        self.vgg = vgg

    def forward(self, sr_patch, real_patch):
        """
        sr_patch: CCSR 輸出的高品質 Patch [B, 3, H, W]
        real_patch: 從資料集採樣的真實高品質 Patch [B, 3, H, W]
        """
        # 1. Pixel Loss (維持顏色正確)
        l1_loss = F.l1_loss(sr_patch, real_patch)

        # 將範圍從 [-1, 1] 映射到 [0, 1]
        sr_patch_norm = (sr_patch + 1) / 2.0
        real_patch_norm = (real_patch + 1) / 2.0
        
        # 2. Perceptual Loss (提升細節紋理)
        sr_feat = self.vgg(sr_patch_norm)
        real_feat = self.vgg(real_patch_norm)
        perceptual_loss = F.mse_loss(sr_feat, real_feat)
        
        # 3. [修正] VGG perceptual weight 從 0.5 降到 0.01
        #    VGG 在 64×64 小圖上 receptive field 過大，高權重會造成不穩定梯度
        return l1_loss + 0.01 * perceptual_loss


def compute_loss(d_outs, target_value):
    """
    優化後的損失計算函數
    支援標籤平滑 (Label Smoothing)：target_value 可以是 float 或 (min, max) tuple
    """
    d_outs = [d_outs] if not isinstance(d_outs, list) else d_outs
    loss = 0
    device = d_outs[0].device
    
    for d_out in d_outs:
        # 檢查 target_value 是否為範圍 (例如 (0.9, 1.0) 或 (0.0, 0.1))
        if isinstance(target_value, tuple):
            low, high = target_value
            # 產生與 d_out 形狀相同的隨機平滑標籤
            targets = torch.empty_like(d_out).uniform_(low, high)
        else:
            # 傳統硬標籤或固定平滑標籤
            targets = torch.full_like(d_out, target_value)
        
        # 使用 binary_cross_entropy_with_logits 更加簡潔，不需重複實例化 nn.Module
        loss += F.binary_cross_entropy_with_logits(d_out, targets)
        
    return loss / len(d_outs)


def compute_grad2(d_outs, x_in):
    d_outs = [d_outs] if not isinstance(d_outs, list) else d_outs
    reg = 0
    for d_out in d_outs:
        batch_size = x_in.size(0)
        grad_dout = autograd.grad(
            outputs=d_out.sum(), inputs=x_in,
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        grad_dout2 = grad_dout.pow(2)
        assert(grad_dout2.size() == x_in.size())
        reg += grad_dout2.view(batch_size, -1).sum(1)
    return reg / len(d_outs)

def wgan_gp_reg(discriminator, x_real, x_fake, y, center=1.):
        batch_size = y.size(0)
        device = torch.device("cuda:0")
        y = y.to(device)

        samples_per_batch = x_real.size(0) // batch_size
        x_real_batched = x_real.reshape(batch_size, samples_per_batch, 3)
        x_fake_batched = x_fake.reshape(batch_size, samples_per_batch, 3)

        eps = torch.rand(batch_size, device=y.device).view(batch_size, 1, 1)
        x_interp_batched = (1 - eps) * x_real_batched + eps * x_fake_batched
        x_interp = x_interp_batched.reshape(-1, 3)
        x_interp = x_interp.detach()
        x_interp.requires_grad_()
        d_out = discriminator(x_interp, y)

        reg = (compute_grad2(d_out, x_interp).sqrt() - center).pow(2).mean()

        return reg

def toggle_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)

def save_data(label, rays, iteration, save_dir='./saved_data'):
    """
    簡單的函數用於儲存標籤和光線
    """
    save_dir = os.path.join(save_dir, f'iter_{iteration}')
    os.makedirs(save_dir, exist_ok=True)
    
    # 儲存為 numpy 格式
    label_np = label.detach().cpu().numpy() if isinstance(label, torch.Tensor) else label
    rays_np = rays.detach().cpu().numpy() if isinstance(rays, torch.Tensor) else rays
    
    np.save(os.path.join(save_dir, 'labels.npy'), label_np)
    np.save(os.path.join(save_dir, 'rays.npy'), rays_np)

    with open(os.path.join(save_dir, 'rays_values.csv'), 'w') as f:
        f.write("batch,index,x,y,z\n")  # CSV 標頭
        for batch_idx in range(rays_np.shape[0]):
            for ray_idx in range(rays_np.shape[1]):
                x, y, z = rays_np[batch_idx, ray_idx]
                f.write(f"{batch_idx},{ray_idx},{x},{y},{z}\n")

    with open(os.path.join(save_dir, 'labels_full.txt'), 'w') as f:
        # 設置 numpy 顯示選項以顯示所有元素
        np.set_printoptions(threshold=np.inf, precision=8, suppress=True)
        f.write("Labels (Shape: {}):\n".format(label_np.shape))
        f.write(np.array2string(label_np))

    # 恢復 numpy 的默認顯示選項
    np.set_printoptions(threshold=1000)