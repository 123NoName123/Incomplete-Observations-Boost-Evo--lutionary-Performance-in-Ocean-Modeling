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

def get_mask_by_rule_tensor(frame, rule_idx):
    """
    根据 rule_idx 对 frame 执行裁剪规则：
    裁剪区域保持原值，其他区域置为 0（黑色）。
    
    支持 6 种裁剪方式：
      0 → 左 2/3（列方向）
      1 → 右 2/3（列方向）
      2 → 下 2/3（行方向）
      3 → 中间 2/3（列方向，水平居中裁剪）
      4 → 上 2/3（行方向）
      5 → 中间 2/3（行方向，垂直居中裁剪）
    """
    masked = torch.zeros_like(frame)
    H, W = frame.shape[-2], frame.shape[-1]
    c = 21  # 裁剪大小

    rule = rule_idx % 6

    if rule == 0:                # 左 2/3
        masked[..., :c] = frame[..., :c]
    elif rule == 1:              # 右 2/3
        masked[..., -c:] = frame[..., -c:]
    elif rule == 2:              # 下 2/3
        masked[..., -c:, :] = frame[..., -c:, :]
    elif rule == 3:              # 中间 2/3（宽度方向）
        start = (W - c) // 2
        masked[..., start:start + c] = frame[..., start:start + c]
    elif rule == 4:              # 上 2/3
        masked[..., :c, :] = frame[..., :c, :]
    elif rule == 5:              # 中间 2/3（高度方向）
        start = (H - c) // 2
        masked[..., start:start + c, :] = frame[..., start:start + c, :]
    else:
        raise ValueError("Invalid rule index")

    return masked

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

def langevin_sampling(generator, predictor, data_loader, iteration, steps = 6000, sigma = 0.5, lr = 0.5):
    generator.eval()
    predictor.eval()

    all_t0, all_t1, all_t2 = [], [], []
    
    print("Data loader length:", len(data_loader))

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
        # z0 = (torch.rand(B, 512, device=DEVICE) * 2 - 1).requires_grad_()
        # z1 = (torch.rand(B, 64 * 64, device=DEVICE) * 2 - 1).requires_grad_()
        z0 = (torch.randn(B, 512, device=DEVICE) * 5).requires_grad_()
        z1 = (torch.randn(B, 64 * 64, device=DEVICE) * 5).requires_grad_()



        # n0 = get_noises_list(B,device=DEVICE)
        # n1 = get_noises(B)
        n0 = get_noises_list(B,device=DEVICE, generator = generator)
        n1 = get_noises(B, predictor)

            
        # 重新组织参数分组
        optimizer = torch.optim.Adam([
            {"params": n0 + [z0], "lr": 0.1},   # 第一组: n0, z0
            {"params": n1 + [z1], "lr": 0.1},   # 第二组: n1, z1
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
            prior_loss_z0 = z0.pow(2).mean() * 0.0001
            prior_loss_z1 = z1.pow(2).mean() * 0.02
            prior_loss_n0 = sum(n.pow(2).mean() for n in n0) * 0.005
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




            # if loss.item() <= 1:
            #     # print("切换学习率：loss <= 0.1")
            #     optimizer.param_groups[0]['lr'] = 1  # n0, z0
            #     optimizer.param_groups[1]['lr'] = 1 # n1, z1
            # elif 1 < loss.item() < 2:
            #     # print("切换学习率：0.1 < loss < 0.5")
            #     optimizer.param_groups[0]['lr'] = 1  # n0, z0
            #     optimizer.param_groups[1]['lr'] = 1  # n1, z1
            # else: 
            #     # print("使用初始学习率：loss >= 0.5")
            #     optimizer.param_groups[0]['lr'] = 1  # n0, z0
            #     optimizer.param_groups[1]['lr'] = 1  # n1, z1

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

                print(
                    f"[Batch {batch_idx+1}/{len(data_loader)}] Langevin step {i}, "
                    f"Total Loss: {loss.item():.4f}"
                )
                print(
                    f"    ├─ Data Loss: "
                    f"L0: {batch_loss0:.4f}, L1: {batch_loss1:.4f}"
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
            
            save_sst_visualizations_jet_discrete(final_s0, final_s1, batch_idx)
            #     # 为每个 batch 创建单独文件夹
            # batch_folder = f"saved_examples/lang_sampling_observation/batch_{batch_idx:04d}"
            # os.makedirs(batch_folder, exist_ok=True)
            
            # # 复制 colormap 并设置 masked 区域为黑色
            # cmap = plt.cm.get_cmap('jet').copy()
            # cmap.set_bad(color='black')

            # # 保存参数
            # figsize = (8, 8)  # 英寸大小，可调
            # dpi = 30         # DPI，可调
            
            # for b in range(B):
            #     idx_base = batch_start_idx + b
            
            #     # === Step 1: 取出单张样本 ===
            #     s0 = final_s0[b:b+1]  # [1, 1, H, W]
            #     s1 = final_s1[b:b+1]
            
            #     # === Step 2: 获取观测掩码 (1=观测, 0=未观测) ===
            #     mask0 = get_mask_by_rule_tensor(s0, idx_base + 0)  # 返回 0/1 张量
            #     mask1 = get_mask_by_rule_tensor(s1, idx_base + 1)
            
            #     # === Step 3: 反归一化 ===
            #     s0_denorm = denormalize_sst(s0[0, 0].cpu().numpy())
            #     s1_denorm = denormalize_sst(s1[0, 0].cpu().numpy())
            
            #     # === Step 4: 创建 MaskedArray，未观测区域显示黑色 ===
            #     s0_masked = np.ma.masked_where(mask0[0, 0].cpu().numpy() == 0, s0_denorm)
            #     s1_masked = np.ma.masked_where(mask1[0, 0].cpu().numpy() == 0, s1_denorm)
            
            #     # === Step 5: 保存观测图像 ===
            #     for img, t in zip([s0_masked, s1_masked], ['t0', 't1']):
            #         fig, ax = plt.subplots(figsize=figsize)
            #         ax.imshow(img, cmap=cmap, origin='lower', vmin=15, vmax=35)
            #         ax.axis('off')
            #         plt.tight_layout(pad=0)
            #         save_path = f"{batch_folder}/s{t}_obs_{idx_base:03d}.png"
            #         plt.savefig(save_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
            #         plt.close(fig)

        print(f"[Batch {batch_idx+1}/{len(data_loader)}] Best loss: {best_loss:.6f}")

        

        all_t0.append(final_s0)
        all_t1.append(final_s1)


    return {
        "t0": torch.cat(all_t0, dim=0),
        "t1": torch.cat(all_t1, dim=0),
    }



# def get_noises(batch_size, device='cuda'):
#     """
#     生成 n1 噪声列表（均匀噪声 [-1,1]）。
    
#     Args:
#         batch_size (int): 批次大小。
#         up_architecture (list of tuple): 每层噪声的形状信息 [(C,H,W), ...]。
#         device: 目标设备。

#     Returns:
#         list of torch.Tensor: 均匀噪声列表，每个张量设置 requires_grad=True。
#     """
#     print("[*] 正在生成均匀 [-1,1] 噪声...")
#     noises = []
    
#     for (channels, height, width) in up_architecture:
#         shape = (batch_size, channels, height, width)
        
#         # 生成均匀分布 [-1,1] 噪声
#         noise = torch.empty(shape, device=device).uniform_(-1, 1).requires_grad_()
#         noises.append(noise)
    
#     print(f"✅ 均匀噪声列表创建完成，共 {len(noises)} 个张量。")
#     return noises

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


# def get_noises_list(batch_size, device):
#     """
#     生成均匀 [-1,1] 噪声列表，不依赖预训练权重。
    
#     Args:
#         batch_size (int): 批次大小。
#         device: 目标设备。
    
#     Returns:
#         list of torch.Tensor: 均匀噪声列表，每个张量 requires_grad=True。
#     """
#     # 定义每层噪声形状（必须与模型架构对应）
#     shapes = [
#         (batch_size, 256, 4, 4),
#         (batch_size, 256, 8, 8),
#         (batch_size, 256, 8, 8),
#         (batch_size, 256, 16, 16),
#         (batch_size, 256, 16, 16),
#         (batch_size, 256, 32, 32),
#         (batch_size, 256, 32, 32),
#         (batch_size, 256, 64, 64),
#         (batch_size, 256, 64, 64),
#     ]

#     noises = []
#     for shape in shapes:
#         noise = torch.empty(shape, device=device).uniform_(-1, 1).requires_grad_()
#         noises.append(noise)
    
#     print(f"✅ 均匀噪声列表生成完成，共 {len(noises)} 个张量。")
#     return noises

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



# def save_sst_visualizations_jet_discrete(
#     s0, s1, batch_idx,
#     folder_path="saved_examples/lang_sampling_vis_jet",
#     cmap='jet'
# ):
#     import os
#     os.makedirs(folder_path, exist_ok=True)
#     batch_folder = os.path.join(folder_path, f"batch_{batch_idx:02d}")
#     os.makedirs(batch_folder, exist_ok=True)

#     B = s0.size(0)

#     for i in range(B):
#         s0_img = denormalize_sst(s0[i, 0].cpu().detach().numpy())
#         s1_img = denormalize_sst(s1[i, 0].cpu().detach().numpy())

#         fig = plt.figure(figsize=(13.5, 4))
#         gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.05)

#         axes = [plt.subplot(gs[i]) for i in range(3)]
#         cax = plt.subplot(gs[3])  # 单独的 colorbar 区域

#         images = [s0_img, s1_img]
#         labels = ['t0', 't1']

#         for ax, img, label in zip(axes, images, labels):
#             im = ax.imshow(img, cmap=cmap, origin='lower', vmin=15, vmax=35)
#             ax.set_title(f"{label} index {i}", fontsize=12)
#             ax.axis('off')

#         # colorbar 独立画在 cax 上
#         cbar = fig.colorbar(im, cax=cax)
#         cbar.set_label("SST (°C)")

#         plt.savefig(os.path.join(batch_folder, f"sst_{i:03d}.png"), bbox_inches='tight', pad_inches=0.1)
#         plt.close()

def save_sst_visualizations_jet_discrete(
    s0, s1, batch_idx,
    base_folder="saved_examples/lang_sampling_gaosi",
    cmap='coolwarm',
    dpi=30,
    figsize=(6, 6)
):
    import os
    import matplotlib.pyplot as plt

    # === 每个 batch 独立文件夹 ===
    batch_folder = os.path.join(base_folder, f"batch_{batch_idx:04d}")
    os.makedirs(batch_folder, exist_ok=True)

    B = s0.size(0)

    for i in range(B):
        s0_img = denormalize_sst(s0[i, 0].cpu().detach().numpy())
        s1_img = denormalize_sst(s1[i, 0].cpu().detach().numpy())

        # ---- 保存 t0 ----
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(s0_img, cmap=cmap, origin='lower', vmin=15, vmax=35)
        ax.axis('off')
        plt.savefig(
            os.path.join(batch_folder, f"sst_{batch_idx:02d}_{i:03d}_t0.png"),
            bbox_inches='tight',
            pad_inches=0.01,
            dpi=dpi
        )
        plt.close(fig)

        # ---- 保存 t1 ----
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(s1_img, cmap=cmap, origin='lower', vmin=15, vmax=35)
        ax.axis('off')
        plt.savefig(
            os.path.join(batch_folder, f"sst_{batch_idx:02d}_{i:03d}_t1.png"),
            bbox_inches='tight',
            pad_inches=0.01,
            dpi=dpi
        )
        plt.close(fig)

    print(f"[Batch {batch_idx}] 已保存 {B * 2} 张图片到 {batch_folder}")







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