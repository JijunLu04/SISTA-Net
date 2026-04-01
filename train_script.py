import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from PIL import Image
from pathlib import Path
from model.Res2MMNet import Res2MMNet
from model.ProximalNet import ProximalMamba_18

# ------------------- parameters -------------------
batch_size = 1
lr0 = 0.0001 
lrend = 0.000005
model_ctor = Res2MMNet
model_ctor2 = ProximalMamba_18
mask_dir = Path('./speckles_800')
Steps = 4000
num_patterns = 6000
print_interval = 5
save_interval_pre = 10
save_interval_pos = 100

lamda_fidelity = 10
lamda_proximal_KL = 1
lamda_sparsity_L1 = 0.1
save_path_name = 'save_name'
target_img_dir = Path('/targets')
results_root = Path(f'results_path/{save_path_name}')
device = torch.device('cuda:0')

def load_masks(mask_dir_path: Path, width: int, height: int, num_patterns: int = None) -> np.ndarray:
    exts = {'.bmp', '.png', '.jpg', '.jpeg'}
    files = [p for p in sorted(mask_dir_path.iterdir(), key=lambda x: x.name) if p.suffix.lower() in exts]
    if len(files) == 0:
        raise RuntimeError(f'Empty mask directory: {mask_dir_path}')

    # If num_patterns is set, only use the first num_patterns files
    if num_patterns is not None:
        if num_patterns > len(files):
            print(f'Warning: num_masks ({num_patterns}) > files ({len(files)}), using all files')
            files = files[:len(files)]
        else:
            files = files[:num_patterns]

    masks = []
    for p in files:
        img = Image.open(p).convert('L')
        img = img.resize((width, height), Image.NEAREST)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        masks.append(arr)
    masks = np.stack(masks, axis=-1)  # [H, W, M]
    return masks


def save_image(array_2d: np.ndarray, save_path: Path) -> None:
    arr = np.clip(array_2d * 255.0, 0, 255).astype('uint8')
    save_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).convert('L').save(save_path)


def load_gray_image(image_path: Path, width: int, height: int) -> np.ndarray:
    img = Image.open(image_path).convert('L')
    if img.size != (width, height):
        img = img.resize((width, height), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0  # [H, W]
    return arr


def process_single_image(img_H, img_W, target_img_path, masks_hw_m, actual_num_patterns, device):
    img_name = target_img_path.stem
    result_save_path = results_root / img_name
    result_save_path.mkdir(parents=True, exist_ok=True)

    target_hw = load_gray_image(target_img_path, img_W, img_H)

    y_vec_full = (target_hw[..., None] * masks_hw_m).sum(axis=(0, 1))
    y_vec_full = (y_vec_full - y_vec_full.mean()) / (y_vec_full.std() + 1e-8)
    y_real_full = torch.tensor(y_vec_full, dtype=torch.float32, device=device).view(
        batch_size, 1, 1, actual_num_patterns
    )

    A_flat = torch.tensor(masks_hw_m, dtype=torch.float32, device=device)  # [H, W, M]
    A_flat = A_flat.permute(2, 0, 1).reshape(actual_num_patterns, img_H * img_W).unsqueeze(0)  # [B, M, H*W]

    inpt = torch.rand((batch_size, 1, img_H, img_W), dtype=torch.float32, device=device)

    model_fidelity = model_ctor(img_W=img_W, img_H=img_H).to(device)
    model_proximal = model_ctor2().to(device)

    optimizer = optim.Adam(
        list(model_fidelity.parameters()) + list(model_proximal.parameters()),
        lr=lr0, betas=(0.9, 0.999), eps=1e-8
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(Steps * 0.01), eta_min=lrend
    )

    print(f'Starting reconstruction for {img_name}...')
    for step in range(Steps):
        model_fidelity.train()
        model_proximal.train()
        optimizer.zero_grad()

        x_pred_fidelity_stage1 = model_fidelity(inpt)  # [B,1,H,W]
        x_flat_fidelity_stage1 = x_pred_fidelity_stage1.view(batch_size, 1, img_H * img_W)
        out_y_fidelity = torch.bmm(x_flat_fidelity_stage1, A_flat.transpose(1, 2))
        out_y_fidelity = out_y_fidelity.view(batch_size, 1, 1, actual_num_patterns)
        out_y_fidelity = (out_y_fidelity - out_y_fidelity.mean()) / (out_y_fidelity.std() + 1e-8)

        input_for_proximal = x_pred_fidelity_stage1.to(device)
        x_pred_proximal, sparse_feat = model_proximal(input_for_proximal)

        x_flat_proximal = x_pred_proximal.view(batch_size, 1, img_H * img_W).to(device)
        out_y_proximal = torch.bmm(x_flat_proximal, A_flat.transpose(1, 2))
        out_y_proximal = out_y_proximal.view(batch_size, 1, 1, actual_num_patterns)
        out_y_proximal = (out_y_proximal - out_y_proximal.mean()) / (out_y_proximal.std() + 1e-8)

        loss_Fidelity_MSE = F.mse_loss(out_y_fidelity, y_real_full)

        pred_logits = out_y_proximal.view(batch_size, -1)
        real_logits = y_real_full.view(batch_size, -1)
        pred_log_probs = F.log_softmax(pred_logits, dim=-1)
        real_probs = F.softmax(real_logits, dim=-1)
        loss_proximal_KL = F.kl_div(pred_log_probs, real_probs, reduction='batchmean')

        loss_sparsity_L1 = sparse_feat.abs().mean()
        loss_all = lamda_fidelity*loss_Fidelity_MSE + lamda_proximal_KL*loss_proximal_KL + lamda_sparsity_L1*loss_sparsity_L1

        loss_all.backward()
        optimizer.step()

        if step % print_interval == 0:
            print(f"step:{step} total:{loss_all:.10f} fidelity_mse:{loss_Fidelity_MSE.item():.10f} proximal_KL:{loss_proximal_KL.item():.10f} sparsity_L1:{0.1*loss_sparsity_L1.item():.10f} "
                  f"lr:{optimizer.param_groups[0]['lr']:.8f}")
        if step < 200:
            save_interval = save_interval_pre
        else:
            save_interval = save_interval_pos

        if step % save_interval == 0 or step == Steps-1:
            with torch.no_grad():
                x_out_proximal = x_pred_proximal.detach().cpu().numpy()[0, 0, :, :]
                x_stage1_fidelity = x_pred_fidelity_stage1.detach().cpu().numpy()[0, 0, :, :]
                save_image(x_out_proximal, result_save_path / f'MP_recon_step_{step}.bmp')
                save_image(x_stage1_fidelity, result_save_path / f'MF_recon_stage1_step_{step}.bmp')

        if step % 100 == 0 and step > 0 and step <= 5000:
            scheduler.step()

        if step == Steps - 1:
            with torch.no_grad():
                x_out_proximal = x_pred_proximal.detach().cpu().numpy()[0, 0, :, :]
                x_stage1_fidelity = x_pred_fidelity_stage1.detach().cpu().numpy()[0, 0, :, :]
                save_image(x_out_proximal, result_save_path / 'MP_recon_final.bmp')
                save_image(x_stage1_fidelity, result_save_path / 'MF_recon_final.bmp')

    print(f'{img_name} completed!')
    del model_fidelity, model_proximal, optimizer, scheduler
    if device.type == 'cuda':
        torch.cuda.empty_cache()

def main():
    results_root.mkdir(parents=True, exist_ok=True)
    exts = {'.bmp', '.png', '.jpg', '.jpeg'}
    paths = sorted(
        [p for p in target_img_dir.iterdir() if p.is_file() and p.suffix.lower() in exts],
        key=lambda x: x.name
    )
    if not paths:
        print(f'Warning: No image files found in directory {target_img_dir}')
        return

    with Image.open(paths[0]) as img:
        w, h = img.size
    masks_hw_m = load_masks(mask_dir, w, h, num_patterns)
    m = masks_hw_m.shape[-1]

    print(f'Found {len(paths)} image(s), masks={m}')
    for p in paths:
        process_single_image(w, h, p, masks_hw_m, m, device)

if __name__ == "__main__":
    main()
