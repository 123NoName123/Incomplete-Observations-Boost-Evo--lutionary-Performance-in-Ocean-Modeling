import torch
import numpy as np
from matplotlib import gridspec
import torch.nn as nn
import matplotlib.pyplot as plt
import os
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
up_architecture = [
    (256, 2, 2),
    (256, 2, 2),
    (256, 4, 4),
    (256, 4, 4),  
    (256, 8, 8),
    (128, 8, 8), 
    (128, 16, 16),
    (64, 16, 16),
    (64, 32, 32),
    (32, 32, 32),
    (32, 64, 64),
    (32, 64, 64),
]

DEVICE                  = "cuda" if torch.cuda.is_available() else "cpu"

def save_sst_visualizations(
    groundtruth_s0, groundtruth_s1, generated_s0, generated_s1,
    batch_idx, iteration,
    save_dir="results1/paper",
    cmap='coolwarm',
    vmin=15,
    vmax=35,
    figsize=(3, 3),
    dpi=30,
    error_cmap='bwr',        # 新增：误差色图
    error_max=2.0,
    obs_error_scale=0.1,   # ✅ 新增：观测区域误差缩放系数（0~1）# 新增：误差最大范围 ±2°C
):
    """
    保存 SST 图像：
    - Observed: 按 Dataset 裁剪规则模拟观测（裁剪部分保留，其余置 NaN）
    - Generated: GAN 生成的完整 SST
    - GroundTruth: 完整 SST
    - ErrorMap: |Generated - GroundTruth|
    """
    import os
    import numpy as np
    import torch
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    def denormalize(x):
        return (x + 1) / 2 * (vmax - vmin) + vmin

    def to_numpy(x):
        if isinstance(x, list):
            x = x[0]
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        if x.ndim == 4:  # [1,1,H,W]
            x = x[0,0]
        elif x.ndim == 3:  # [1,H,W]
            x = x[0]
        return denormalize(np.nan_to_num(x, nan=vmin, posinf=vmax, neginf=vmin))

    def crop_by_rule_tensor(frame, rule_idx):
        H, W = frame.shape[-2], frame.shape[-1]
        c = 21
        rule = rule_idx % 6
        if rule == 0:                
            return frame[..., :c]
        elif rule == 1:              
            return frame[..., -c:]
        elif rule == 2:             
            return frame[..., -c:, :]
        elif rule == 3:             
            start = (W - c) // 2
            return frame[..., start:start + c]
        elif rule == 4:              
            return frame[..., :c, :]
        elif rule == 5:             
            start = (H - c) // 2
            return frame[..., start:start + c, :]
        else:
            raise ValueError("Invalid rule index")

    def expand_crop_to_full(crop, rule_idx, full_shape=(64,64), fill_value=1.0):
        if isinstance(crop, torch.Tensor):
            crop = crop.detach().cpu().numpy()
        H_full, W_full = full_shape
        H_crop, W_crop = crop.shape[-2], crop.shape[-1]
        canvas = np.full(full_shape, np.nan, dtype=crop.dtype)
        rule = rule_idx % 6
    
        if rule == 0:
            canvas[:, :W_crop] = crop
        elif rule == 1:
            canvas[:, -W_crop:] = crop
        elif rule == 2:
            canvas[-H_crop:, :] = crop
        elif rule == 3:
            start = (W_full - W_crop)//2
            canvas[:, start:start+W_crop] = crop
        elif rule == 4:
            canvas[:H_crop, :] = crop
        elif rule == 5:
            start = (H_full - H_crop)//2
            canvas[start:start+H_crop, :] = crop
    
        return canvas

    # 创建文件夹
    base_dir = os.path.join(save_dir, f"iter{iteration:03d}_batch{batch_idx:03d}")
    sub_dirs = {
        "Observed": os.path.join(base_dir, "Observed"),
        "Generated": os.path.join(base_dir, "Generated"),
        "GroundTruth": os.path.join(base_dir, "GroundTruth"),
        "ErrorMap": os.path.join(base_dir, "ErrorMap"),  # 新增
    }
    for d in sub_dirs.values():
        os.makedirs(d, exist_ok=True)

    B = len(groundtruth_s0) if isinstance(groundtruth_s0, list) else groundtruth_s0.shape[0]
    for b in range(B):
        gt0_np = to_numpy(groundtruth_s0[b:b+1])
        gt1_np = to_numpy(groundtruth_s1[b:b+1])
        gen0_np = to_numpy(generated_s0[b:b+1])
        gen1_np = to_numpy(generated_s1[b:b+1])

        # Observed 图像
        obs0_crop = crop_by_rule_tensor(gt0_np, b + 0)
        obs1_crop = crop_by_rule_tensor(gt1_np,  b + 1)
        obs0_np = expand_crop_to_full(obs0_crop, b + 0)
        obs1_np = expand_crop_to_full(obs1_crop, b + 1)

        # ✅ 生成观测掩码
        obs_mask0 = ~np.isnan(obs0_np)
        obs_mask1 = ~np.isnan(obs1_np)

        def save_image(img, subdir, tname, use_error=False):
            masked_img = np.ma.masked_invalid(img)
            fig, ax = plt.subplots(figsize=figsize)
            cmap_used = error_cmap if use_error else cmap
            if use_error:
                im = ax.imshow(img, cmap=cmap_used, origin='lower', vmin=-error_max, vmax=error_max)
            else:
                cmap_custom = cm.get_cmap(cmap_used).copy()
                cmap_custom.set_bad(color='white')
                im = ax.imshow(img, cmap=cmap_custom, origin='lower',
                               vmin=vmin, vmax=vmax)
            ax.axis('off')
            path = os.path.join(sub_dirs[subdir], f"sample{b:02d}_{tname}.png")
            plt.savefig(path, dpi=dpi, bbox_inches='tight', pad_inches=0.00)
            plt.close(fig)
            print(f"✅ 保存 {subdir}/{tname}: {path}")

        # 保存
        save_image(obs0_np, "Observed", "t0")
        save_image(obs1_np, "Observed", "t1")
        save_image(gen0_np, "Generated", "t0")
        save_image(gen1_np, "Generated", "t1")
        save_image(gt0_np, "GroundTruth", "t0")
        save_image(gt1_np, "GroundTruth", "t1")

        # ✅ 保存误差图
        err0_np = gen0_np - gt0_np
        err1_np = gen1_np - gt1_np

        save_image(err0_np, "ErrorMap", "t0", use_error=True)
        save_image(err1_np, "ErrorMap", "t1", use_error=True)

def normalize_sst(x, min_val=15.0, max_val=35.0):
    return 2 * (x - min_val) / (max_val - min_val) - 1

def denormalize_sst(normalized_tensor):
    # 确保输入是 torch.Tensor 类型
    if isinstance(normalized_tensor, torch.Tensor):
        # ----------------------------------------------------
        #  ↓↓↓  在这里添加 .cpu()，将数据移至 CPU  ↓↓↓
        # ----------------------------------------------------
        normalized_tensor = normalized_tensor.cpu().detach().numpy()
        # ----------------------------------------------------
        
    return (normalized_tensor * 10.0) + 25.0 # [-1, 1] -> [15, 35]

# @torch.no_grad()
# def random_sampling(generator, predictor, data_loader, num_batches=1):
#     generator.eval()
#     predictor.eval()


#     for batch_idx, batch in enumerate(data_loader):
#         if batch_idx >= num_batches:
#             break
        
#         B = len(batch['t0'])
#         c0 = torch.zeros(B, 0, device=DEVICE)  # 没标签

#         # 🔹随机初始化潜在变量和噪声
#         z0 = torch.rand(B, 512, device=DEVICE) * 2 - 1
#         z1 = torch.rand(B, 64 * 64, device=DEVICE) * 2 - 1
#         z2 = torch.rand(B, 64 * 64, device=DEVICE) * 2 - 1


#         # 🔹前向生成
#         s0 = generator(z0, c0)
#         s1 = predictor(s0, z1)
#         s2 = predictor(s1, z2)


#         save_sst_visualizations_jet_discrete(s0, s1, s2, batch_idx, folder_path="saved_examples/examples")
#         print(f"[Batch {batch_idx+1}] Random samples generated.")


def crop_by_rule_tensor(frame, rule_idx):
    """
    根据 rule_idx 对 frame 执行与 Dataset 一致的裁剪规则。
    支持 6 种轮换方式：
      0 → 左 2/3（列方向）
      1 → 右 2/3（列方向）
      2 → 下 2/3（行方向）
      3 → 中间 2/3（列方向，水平居中裁剪）
      4 → 上 2/3（行方向）
      5 → 中间 2/3（行方向，垂直居中裁剪）
    """

    H, W = frame.shape[-2], frame.shape[-1]
    c = 21          # 注意变量名拼写

    rule = rule_idx % 6          # 共有 6 种规则

    if rule == 0:                # 左 2/3
        return frame[..., :c]
    elif rule == 1:              # 右 2/3
        return frame[..., -c:]
    elif rule == 2:              # 下 2/3
        return frame[..., -c:, :]
    elif rule == 3:              # 中间 2/3（宽度方向）
        start = (W - c) // 2
        return frame[..., start:start + c]
    elif rule == 4:              # 上 2/3
        return frame[..., :c, :]
    elif rule == 5:              # 中间 2/3（高度方向）
        start = (H - c) // 2
        return frame[..., start:start + c, :]
    else:
        raise ValueError("Invalid rule index")

def adaptive_grad_clip_per_tensor(
    noise_list, 
    base_norm=5.0, 
    min_clip_norm=1e-9,
    strength=1.0,
):
    """
    对噪声列表中的【每个张量】独立地进行自适应梯度裁剪。

    每个张量的梯度裁剪强度由其自身的范数决定。

    Args:
        noise_list (list of torch.Tensor): 需要进行梯度裁剪的张量列表 (例如 n0)。
        base_norm (float): 单个张量的基准范数。
        min_clip_norm (float): 允许的最小梯度范数。
        strength (float): 裁剪强度因子。
        verbose (bool): 是否打印每个张量的裁剪详情。

    Returns:
        None: 这个函数是 in-place 操作，直接修改张量的梯度，不返回任何值。
    """
    if not noise_list:
        return

    # 遍历列表中的每一个噪声张量
    for i, noise_tensor in enumerate(noise_list):
        if noise_tensor.grad is None:
            continue

        # 1. 计算当前【单个张量】的范数
        with torch.no_grad():
            current_norm = torch.linalg.norm(noise_tensor.detach()).item()

        # 2. 动态计算【该张量】的 max_norm
        max_norm_to_use = (base_norm ** strength) / (current_norm ** strength + 1e-9)
        max_norm_to_use = max(min_clip_norm, max_norm_to_use)
        
        # 3. 对【该张量】的梯度应用裁剪
        torch.nn.utils.clip_grad_norm_(noise_tensor, max_norm=max_norm_to_use)

def langevin_sampling(generator, predictor, data_loader, iteration, steps = 2500, sigma = 0.5, lr = 0.5):
    generator.eval()
    predictor.eval()

    all_t0, all_t1, all_t2 = [], [], []
    

    for batch_idx, batch in enumerate(data_loader):
        
        best_loss = float('inf')
        best_state = {}
        
        loss_records = {
            "loss0": [],
            "loss1": [],
            "loss2": [],
            "loss_total": []
        }
        
        B = len(batch['t0'])

        for k in batch:
            batch[k] = [x.to(DEVICE) for x in batch[k]]

        c0 = torch.zeros(B, 0, device=DEVICE)  # 没标签
        z0 = (torch.randn(B, 512, device=DEVICE) * 5).requires_grad_()
        z1 = (torch.randn(B, 64 * 64, device=DEVICE) * 5).requires_grad_()


        n0 = get_noises_list(B,device=DEVICE, generator = generator)
        n1 = get_noises(B, predictor)

            
        # 重新组织参数分组
        optimizer = torch.optim.Adam([
            {"params": n0 + [z0], "lr": 0.5},   # 第一组: n0, z0
            {"params": n1 + [z1], "lr": 5.0},   # 第二组: n1, z1
        ])

        batch_start_idx = batch_idx * data_loader.batch_size

        for i in range(steps):
            
            optimizer.zero_grad()

            s0 = generator(z0, c0, noises = n0)
            s1 = predictor(s0, z1, noises = n1)

            batch_loss0 = 0
            batch_loss1 = 0

            loss_total = 0
            for b in range(B):
                
                idx_base = batch_start_idx + b
                obs_t0 = batch['t0'][b]
                obs_t1 = batch['t1'][b]

                s0_crop = crop_by_rule_tensor(s0[b:b+1], idx_base + 0).expand_as(obs_t0)
                s1_crop = crop_by_rule_tensor(s1[b:b+1], idx_base + 1).expand_as(obs_t1)

                loss0 = ((s0_crop - obs_t0) ** 2).view(obs_t0.shape[0], -1).sum(dim=1).mean()
                loss1 = ((s1_crop - obs_t1) ** 2).view(obs_t1.shape[0], -1).sum(dim=1).mean()

                batch_loss0 += loss0.item()
                batch_loss1 += loss1.item()
                
                loss_total += loss0 + loss1

            # --- G.S. - Start: 添加高斯先验 ---
            # 计算所有噪声/潜在变量的L2范数平方的均值，作为高斯先验的惩罚项。
            prior_loss_z0 = z0.pow(2).mean() * 0.000
            prior_loss_z1 = z1.pow(2).mean() * 0.02
            prior_loss_n0 = sum(n.pow(2).mean() for n in n0) * 0.000
            prior_loss_n1 = sum(n.pow(2).mean() for n in n1) * 0.003

            total_prior_loss = prior_loss_z0 + prior_loss_z1 + prior_loss_n0 + prior_loss_n1 
            
            # 将带权重的先验损失加入总损失。乘以 B 来匹配 loss_total 的尺度。
            loss_total += 0.1 * total_prior_loss * B
            
            loss = loss_total / B / (2 * sigma ** 2)

            loss.backward()

            clip_params = [z0]  + [z1] #+ list(n1) 
            for param in clip_params:
                if param.grad is not None:
                    param.grad = torch.nan_to_num(param.grad, nan=0.0, posinf=1, neginf=-1)
                    
            torch.nn.utils.clip_grad_norm_(clip_params, max_norm=0.2)       
            # torch.nn.utils.clip_grad_norm_(n0, max_norm=1) # <--- 在这里设置一个专门给 n0 的阈值
            # 使用“精准打击”的裁剪方式
            adaptive_grad_clip_per_tensor(
                noise_list=n1,
                base_norm=15.0, # 单个张量的范数软目标
                strength=1.5
            )
            adaptive_grad_clip_per_tensor(
                noise_list=n1,
                base_norm=15.0, # 单个张量的范数软目标
                strength=1.5
            )
            optimizer.step()




            if loss.item() <= 1:
                # print("切换学习率：loss <= 0.1")
                optimizer.param_groups[0]['lr'] = 0.1  # n0, z0
                optimizer.param_groups[1]['lr'] = 1 # n1, z1
            elif 1 < loss.item() < 2:
                # print("切换学习率：0.1 < loss < 0.5")
                optimizer.param_groups[0]['lr'] = 0.1  # n0, z0
                optimizer.param_groups[1]['lr'] = 1  # n1, z1
            else: 
                # print("使用初始学习率：loss >= 0.5")
                optimizer.param_groups[0]['lr'] = 0.5  # n0, z0
                optimizer.param_groups[1]['lr'] = 1  # n1, z1

            # 保存当前最优
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_state = {
                    "z0": z0.detach().clone(),
                    "z1": z1.detach().clone(),
                    "n0": [n.detach().clone() for n in n0],
                    "n1": [n.detach().clone() for n in n1],
                }
            

            if i % 50 == 0:
                loss_records["loss0"].append(batch_loss0 / B)
                loss_records["loss1"].append(batch_loss1 / B)
                loss_records["loss_total"].append(loss.item())
   
            # G.S. Mod: 更新打印语句以包含所有损失项
            if i % 500 == 0 or i == steps - 1:
                # 将 Tensor 转换为 Python 数字以便打印
                prior_z0_p = prior_loss_z0.item()
                prior_z1_p = prior_loss_z1.item()
                prior_n0_p = prior_loss_n0.item()
                prior_n1_p = prior_loss_n1.item()
                
                print(
                    f"[Batch {batch_idx+1}/{len(data_loader)}] Langevin step {i}, "
                    f"Total Loss: {loss.item():.4f}"
                )
                print(
                    f"    ├─ Data Loss: "
                    f"L0: {batch_loss0:.4f}, L1: {batch_loss1:.4f}"
                )
                # 打印加权前的单个先验损失
                print(
                    f"    ├─ Prior Terms (unweighted): "
                    f"P(z0): {prior_z0_p:.4f}, P(z1): {prior_z1_p:.4f},"
                    f"P(n0): {prior_n0_p:.4f}, P(n1): {prior_n1_p:.4f}"
                )
            
                # G.S. Mod: 扩展范围打印，并找出 n0 中包含最大值的张量
                with torch.no_grad():
                    # --- z0 和 z1 的 min/max (保持不变) ---
                    z0_min, z0_max = z0.min().item(), z0.max().item()
                    z1_min, z1_max = z1.min().item(), z1.max().item()
                
                    # --- 找出 n0 中包含全局最大值的张量 ---
                    global_max_n0 = -float('inf')  # 初始化全局最大值为负无穷
                    max_val_tensor_shape = None    # 记录包含最大值张量的形状
                    max_val_tensor_range = (0.0, 0.0) # 记录该张量的(min, max)范围
                    global_min_n0 = float('inf')   # 同时我们也可以找到全局最小值
                
                    # 遍历 n0 列表中的每一个噪声张量
                    for noise_tensor in n0:
                        current_min = noise_tensor.min().item()
                        current_max = noise_tensor.max().item()
                        
                        # 更新全局最小值
                        if current_min < global_min_n0:
                            global_min_n0 = current_min
                            
                        # 如果当前张量的最大值比我们记录的全局最大值还要大
                        if current_max > global_max_n0:
                            # 更新记录
                            global_max_n0 = current_max
                            max_val_tensor_shape = noise_tensor.shape
                            max_val_tensor_range = (current_min, current_max)
                    
                    # --- 计算 n1 的全局 min/max (保持原样) ---
                    all_n1_tensors = torch.cat([n.detach().flatten() for n in n1])
                    n1_global_min, n1_global_max = all_n1_tensors.min().item(), all_n1_tensors.max().item()

                
                    # --- 新的、信息更丰富的打印语句 ---
                    print(
                        f"        └─> Ranges: "
                        f"z0:[{z0_min:.2f},{z0_max:.2f}], "
                        f"z1:[{z1_min:.2f},{z1_max:.2f}], "
                        f"n1:[{n1_global_min:.2f},{n1_global_max:.2f}],"
                    )
                    
                    # 打印 n0 的整体范围
                    print(
                        f"        └─> n0 Global Range: [{global_min_n0:.2f},{global_max_n0:.2f}]"
                    )
                    
                    # 单独一行打印关于 n0 中最大值所在张量的详细信息
                    if max_val_tensor_shape:
                        print(
                            f"        └─> n0 Tensor w/ Max Val: "
                            f"Shape={str(list(max_val_tensor_shape)):<18} | " # <18 用于对齐
                            f"Its Range=[{max_val_tensor_range[0]:.2f},{max_val_tensor_range[1]:.2f}]"
                        )

        with torch.no_grad():
            final_s0 = generator(best_state["z0"], c0, noises=best_state["n0"])
            final_s1 = predictor(final_s0, best_state["z1"], noises=best_state["n1"])
            save_sst_visualizations(
                groundtruth_s0=batch['t3'],
                groundtruth_s1=batch['t4'],
                generated_s0=final_s0,
                generated_s1=final_s1,
                batch_idx=batch_idx,
                iteration=iteration
            )


        print(f"[Batch {batch_idx+1}/{len(data_loader)}] Best loss: {best_loss:.6f}")

        
        # # ========================================================================
        # # |                  开始评估 (Evaluation Module Starts Here)                  |
        # # ========================================================================
        # # --- 1. 准备工作：获取真实数据并初始化评估工具 ---
        # # 确保评估在无梯度的环境下进行，以节省内存和计算资源
        # with torch.no_grad():
        #  # --- 1. 准备 GT 数据，无论输入是列表还是张量，都强制转换为正确的形状 ---
            
        #     # 从 batch 中获取 gt_t0 的数据源
        #     gt_t0_source = batch['t2']
        #     gt_t1_source = batch['t3']

        #     # 判断源是列表还是张量，并统一处理
        #     if isinstance(gt_t0_source, list):
        #         # 如果是列表，先 stack 成一个批次张量
        #         gt_t0_tensor = torch.stack(gt_t0_source, dim=0)
        #     else:
        #         # 如果已经是张量，直接使用
        #         gt_t0_tensor = gt_t0_source
            
        #     # 对 gt_t1 进行同样的处理
        #     if isinstance(gt_t1_source, list):
        #         gt_t1_tensor = torch.stack(gt_t1_source, dim=0)
        #     else:
        #         gt_t1_tensor = gt_t1_source

        #     # --- 2. 强制形状统一为 [B, C, H, W] ---

        #     # 我们已知 final_s0/s1 的形状是 [B, 1, H, W]
        #     # 我们需要将 gt_t0/t1 也变成这个形状
            
        #     # 使用 .view() 来进行最稳健的形状重塑
        #     # 它会处理输入是 [B, H, W] 或 [B, 1, H, W] 或 [B, 1, 1, H, W] 等多种情况
        #     B = final_s0.shape[0]
        #     C, H, W = 1, final_s0.shape[2], final_s0.shape[3]

        #     gt_t0_4d = gt_t0_tensor.view(B, C, H, W)
        #     gt_t1_4d = gt_t1_tensor.view(B, C, H, W)

        #     # --- 3. (可选但强烈推荐) 添加断言，确保形状绝对一致 ---
        #     assert final_s0.shape == gt_t0_4d.shape, \
        #         f"Shape mismatch after final fix! Preds: {final_s0.shape}, Target: {gt_t0_4d.shape}"
        #     assert final_s1.shape == gt_t1_4d.shape, \
        #         f"Shape mismatch after final fix! Preds: {final_s1.shape}, Target: {gt_t1_4d.shape}"

        #     def convert_to_physical_range(tensor, target_min=15.0, target_max=35.0):
        #         """将 [-1, 1] 范围的张量线性映射到 [target_min, target_max] 范围。"""
        #         tensor_0_1 = (tensor.clamp(-1, 1) + 1) / 2.0
        #         return tensor_0_1 * (target_max - target_min) + target_min

        #     final_s0 = convert_to_physical_range(final_s0)
        #     gt_t0_4d = convert_to_physical_range(gt_t0_4d)
            
        #     final_s1 = convert_to_physical_range(final_s1)
        #     gt_t1_4d = convert_to_physical_range(gt_t1_4d)
        #     # --- 4. 初始化和评估 (这部分现在应该是完全安全的) ---
        #     psnr_metric = PeakSignalNoiseRatio(data_range=20.0).to(DEVICE)
        #     ssim_metric = StructuralSimilarityIndexMeasure(data_range=20.0).to(DEVICE)

        #     # 进行定量评估
        #     rmse_t0 = torch.sqrt(torch.mean((final_s0 - gt_t0_4d) ** 2)).item()
        #     mae_t0 = torch.mean(torch.abs(final_s0 - gt_t0_4d)).item()
        #     psnr_t0 = psnr_metric(final_s0, gt_t0_4d).item()
        #     ssim_t0 = ssim_metric(final_s0, gt_t0_4d).item()

        #     rmse_t1 = torch.sqrt(torch.mean((final_s1 - gt_t1_4d) ** 2)).item()
        #     mae_t1 = torch.mean(torch.abs(final_s1 - gt_t1_4d)).item()
        #     psnr_t1 = psnr_metric(final_s1, gt_t1_4d).item()
        #     ssim_t1 = ssim_metric(final_s1, gt_t1_4d).item()
        
        #     # --- 4. 打印格式化、清晰的评估结果 (保持不变) ---
        #     print("\n" + "="*60)
        #     print(f"| Evaluation for Batch {batch_idx+1}/{len(data_loader)}")
        #     print("="*60)
        #     print(f"| State t0:")
        #     print(f"|   ├─ RMSE: {rmse_t0:.4f}")
        #     print(f"|   ├─ MAE:  {mae_t0:.4f}")
        #     print(f"|   ├─ SSIM: {ssim_t0:.4f}")
        #     print(f"|   └─ PSNR: {psnr_t0:.2f} dB")
        #     print("├" + "-"*59)
        #     print(f"| State t1:")
        #     print(f"|   ├─ RMSE: {rmse_t1:.4f}")
        #     print(f"|   ├─ MAE:  {mae_t1:.4f}")
        #     print(f"|   ├─ SSIM: {ssim_t1:.4f}")
        #     print(f"|   └─ PSNR: {psnr_t1:.2f} dB")
        #     print("="*60 + "\n")
        
        all_t0.append(final_s0)
        all_t1.append(final_s1)



    return {
        "t0": torch.cat(all_t0, dim=0),
        "t1": torch.cat(all_t1, dim=0),
    }



def get_noises(batch_size, unet_generator, device='cuda'):
    """
    生成 n1 噪声列表。
    
    该函数会首先从预训练的 unet_generator 中提取出【逐通道】的噪声权重，
    然后用这些权重来对新生成的标准高斯噪声进行缩放，使其分布与
    模型训练时一致。

    Args:
        batch_size (int): 批次大小。
        unet_generator (torch.nn.Module): 预训练的 UnetGenerator 模型。
        device: 目标设备。

    Returns:
        list of torch.Tensor: 根据预训练的逐通道权重初始化好的 n1 噪声张量列表。
                              列表中的每个张量都设置了 requires_grad=True。
    """


    # --- 2. 提取逐通道权重 (直接使用我们之前写的函数逻辑) ---
    print("[*] 正在从 UNet 中提取逐通道噪声权重...")
    channelwise_weights = []
    up_blocks = [
        unet_generator.up1, unet_generator.up2, unet_generator.up3,
        unet_generator.up4, unet_generator.up5, unet_generator.up6
    ]
    for block in up_blocks:
        # detach().clone() 确保我们得到的是干净的权重副本
        channelwise_weights.append(block.conv.noise1.weight.detach().clone())
        channelwise_weights.append(block.conv.noise2.weight.detach().clone())
    
    if len(channelwise_weights) != len(up_architecture):
        raise ValueError("提取的权重数量与架构层数不匹配!")
    print(f"✅ 成功提取了 {len(channelwise_weights)} 个权重张量。")

    # --- 3. 生成噪声并应用权重 ---
    print("[*] 正在使用提取的权重初始化 n1 噪声...")
    noises = []
    hardcoded_std = 5.0
    
    # 使用 zip 将形状、权重配对
    for (channels, height, width), weight_tensor in zip(up_architecture, channelwise_weights):
        shape = (batch_size, channels, height, width)
        
        # a. 生成标准高斯噪声 (裸噪声)
        raw_noise = torch.randn(shape, device=device)
        
        # b. 计算最终的逐通道标准差
        # weight_tensor 的形状是 [1, C, 1, 1], 可以直接和 raw_noise [B, C, H, W] 进行广播
        effective_std_tensor = hardcoded_std * weight_tensor
        
        # c. 将裸噪声与计算出的标准差相乘
        noise = (raw_noise * effective_std_tensor).requires_grad_()
        
        noises.append(noise)
        
    print(f"✅ n1 噪声列表创建完成。")
    return noises

import matplotlib.pyplot as plt
import os

import os
import matplotlib.pyplot as plt



    
def extract_noise_strengths_as_tensor(generator, device):
    """从预训练的生成器中提取所有 noise_strength 参数，并按顺序返回一个张量。"""
    strengths = []
    synthesis_net = generator.synthesis
    
    for res in synthesis_net.block_resolutions:
        block = getattr(synthesis_net, f'b{res}')
        if hasattr(block, 'conv0') and block.conv0.use_noise:
            strengths.append(block.conv0.noise_strength.item())
        if hasattr(block, 'conv1') and block.conv1.use_noise:
            strengths.append(block.conv1.noise_strength.item())
            
    if not strengths:
        raise ValueError("未能从生成器中提取到任何 'noise_strength' 参数！")
        
    strengths_tensor = torch.tensor(strengths, device=device)
    print(f"✅ 成功提取了 {len(strengths)} 个 noise_strength 值。")
    return strengths_tensor


def get_noises_list(batch_size, device, generator):
    """
    使用从预训练模型中提取的权重来生成噪声列表。
    
    Args:
        batch_size (int): 批次大小。
        device: 目标设备。
        learned_strengths (torch.Tensor): 包含每层 noise_strength 的一维张量。
    
    Returns:
        list of torch.Tensor: 根据预训练权重初始化好的噪声张量列表。
    """
    # 这个 shapes 列表必须与 learned_strengths 的长度严格对应
    # 您需要根据您的模型结构精确定义它
    shapes = [
        (batch_size, 256, 4, 4),
        (batch_size, 256, 8, 8),
        (batch_size, 256, 8, 8),
        (batch_size, 256, 16, 16),
        (batch_size, 256, 16, 16),
        (batch_size, 256, 32, 32),
        (batch_size, 256, 32, 32),
        (batch_size, 256, 64, 64),
        (batch_size, 256, 64, 64),
    ]
    # 示例，您需要根据 SynthesisNetwork 的 channels_dict 来确定每个分辨率的通道数
    learned_strengths = extract_noise_strengths_as_tensor(generator, device)
    # 健壮性检查
    if len(learned_strengths) != len(shapes):
        raise ValueError(
            f"提取的权重数量 ({len(learned_strengths)}) 与"
            f"期望的噪声层数 ({len(shapes)}) 不匹配!"
        )

    print(f"[*] 正在使用 {len(learned_strengths)} 个预训练权重初始化噪声...")

    noises = []
    for shape, strength in zip(shapes, learned_strengths):
        # 记住：在您的模型中，实际的标准差是 5 * noise_strength
        # strength 是从模型中提取的 self.noise_strength.item()
        # print(f"Value of 'self.noise_strength' (.item()):\t {strength.item():.6f}")
        actual_std = 5.0 * strength.item()
        
        # 生成噪声并应用计算出的标准差
        noise = (torch.randn(shape, device=device) * actual_std).requires_grad_()
        noises.append(noise)
        
    print(f"    - 完成。噪声初始标准差范围: [{5*learned_strengths.min():.4f}, {5*learned_strengths.max():.4f}]")
    return noises



def save_sst_visualizations_jet_discrete(
    s0, s1, batch_idx,
    folder_path="saved_examples/lang_sampling_gao_jet",
    cmap='jet'
):
    import os
    os.makedirs(folder_path, exist_ok=True)
    batch_folder = os.path.join(folder_path, f"batch_{batch_idx:02d}")
    os.makedirs(batch_folder, exist_ok=True)

    B = s0.size(0)

    for i in range(B):
        s0_img = denormalize_sst(s0[i, 0].cpu().detach().numpy())
        s1_img = denormalize_sst(s1[i, 0].cpu().detach().numpy())

        fig = plt.figure(figsize=(13.5, 4))
        gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.05)

        axes = [plt.subplot(gs[i]) for i in range(3)]
        cax = plt.subplot(gs[3])  # 单独的 colorbar 区域

        images = [s0_img, s1_img]
        labels = ['t0', 't1']

        for ax, img, label in zip(axes, images, labels):
            im = ax.imshow(img, cmap=cmap, origin='lower', vmin=15, vmax=35)
            ax.set_title(f"{label} index {i}", fontsize=12)
            ax.axis('off')

        # colorbar 独立画在 cax 上
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label("SST (°C)")

        plt.savefig(os.path.join(batch_folder, f"sst_{i:03d}.png"), bbox_inches='tight', pad_inches=0.1)
        plt.close()



def visualize_sst_samples(s0, s1, s2, index=None, cmap='jet'):
    """
    可视化 SST 三个时间步（反归一化），样式参考样例：每张子图单独展示，统一 colormap，无 colorbar。
    """
    if index is None:
        index = random.randint(0, s0.size(0) - 1)

    # 提取第 index 个样本，并反归一化和 squeeze
    s0_img = denormalize_sst(s0[index, 0].detach().cpu()).squeeze().numpy()
    s1_img = denormalize_sst(s1[index, 0].detach().cpu()).squeeze().numpy()
    s2_img = denormalize_sst(s2[index, 0].detach().cpu()).squeeze().numpy()
    
    # 绘图
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    titles = [r"$t_0$", r"$t_1$", r"$t_2$"]
    for i, (img, title) in enumerate(zip([s0_img, s1_img, s2_img], titles)):
        axes[i].imshow(img, cmap=cmap, vmin=15, vmax=35)
        axes[i].set_title(title, fontsize=10)
        axes[i].axis('off')

    plt.suptitle("SST Prediction Example", fontsize=14)
    plt.tight_layout()
    plt.show()

def plot_loss_curves(loss_records, save_path_prefix):
    os.makedirs(os.path.dirname(save_path_prefix), exist_ok=True)

    # 绘制 loss0, loss1, loss2
    plt.figure(figsize=(10, 6))
    plt.plot(loss_records['loss0'], label='loss0')
    plt.plot(loss_records['loss1'], label='loss1')
    plt.plot(loss_records['loss2'], label='loss2')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Loss Curve: loss0, loss1, loss2')
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 1)   # ✅ 限制 y 轴范围
    plt.tight_layout()
    plt.savefig(f"{save_path_prefix}_loss012.png")
    plt.close()

    # 绘制 loss_total
    plt.figure(figsize=(10, 6))
    plt.plot(loss_records['loss_total'], label='loss_total', linestyle='--', linewidth=2, color='red')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Loss Curve: total')
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 1)   # ✅ 限制 y 轴范围
    plt.tight_layout()
    plt.savefig(f"{save_path_prefix}_loss_total.png")
    plt.close()

def save_sst_visualizations_grayscale(
    s0, s1, batch_idx,
    folder_path="saved_examples/lang_sampling_vis_gray",
    # 默认 cmap 已改为 'gray'
    cmap='gray' 
):
    """
    将 t0 和 t1 两个时间步的 SST 图像并排以灰度图形式可视化并保存。
    """
    os.makedirs(folder_path, exist_ok=True)
    batch_folder = os.path.join(folder_path, f"batch_{batch_idx:02d}")
    os.makedirs(batch_folder, exist_ok=True)

    B = s0.size(0)

    for i in range(B):
        s0_img = denormalize_sst(s0[i, 0])
        s1_img = denormalize_sst(s1[i, 0])

        # --- 布局修改 ---
        # 我们需要 2 个子图用于图像，1 个用于 colorbar
        fig = plt.figure(figsize=(9.5, 4))
        gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.1)

        # 创建子图区域
        ax0 = plt.subplot(gs[0])
        ax1 = plt.subplot(gs[1])
        cax = plt.subplot(gs[2])

        axes = [ax0, ax1]
        images = [s0_img, s1_img]
        labels = ['t0 (Generated)', 't1 (Predicted)']
        im = None
        
        for ax, img, label in zip(axes, images, labels):
            # --- 核心修改点：使用传入的 cmap (默认为 'gray') ---
            im = ax.imshow(img, cmap=cmap, origin='lower', vmin=15, vmax=35)
            ax.set_title(f"{label} (Sample {i})", fontsize=10)
            ax.axis('off')

        if im:
            cbar = fig.colorbar(im, cax=cax)
            cbar.set_label("SST (°C)")

        save_file = os.path.join(batch_folder, f"sst_gray_{i:03d}.png")
        plt.savefig(save_file, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)