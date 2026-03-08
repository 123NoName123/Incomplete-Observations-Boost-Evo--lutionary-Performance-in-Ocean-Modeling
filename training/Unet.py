import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import os
from tqdm import tqdm
import logging
import math
from training.lang_dynamic import save_sst_visualizations_jet_discrete
from torch.utils.data import Dataset, DataLoader
##############################################################################
# Classes
##############################################################################

# 上采样+拼接
factors = [1, 1, 1, 1, 1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 32]
class InjectNoise(nn.Module):
    def __init__(self, channels, enabled=True):
        super().__init__()
        self.weight = nn.Parameter(torch.full((1, channels, 1, 1), 0.02))
        self.enabled = enabled

    def forward(self, x, noise=None):
        if not self.enabled:
            return x
        if noise is None:
            # noise = torch.rand_like(x) * 2 - 1
            noise = torch.randn_like(x) * 5
            # print(">>> noise shape:", noise.shape)   # 打印噪声的形状
        return x + self.weight * noise

        
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, inject_noise=True):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.relu1 = nn.Tanh()
        self.noise1 = InjectNoise(mid_channels, enabled=inject_noise)

        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.Tanh()
        self.noise2 = InjectNoise(out_channels, enabled=inject_noise)

    def forward(self, x, noises=None):
        n1 = None if noises is None else noises[0]
        n2 = None if noises is None else noises[1]

        x = self.conv1(x)
        x = self.noise1(x, n1)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.noise2(x, n2)
        x = self.bn2(x)
        x = self.relu2(x)

        return x



class Down(nn.Module):
    def __init__(self, in_channels, out_channels, inject_noise=False):
        super().__init__()
        self.downconv = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.Tanh()
        self.double_conv = DoubleConv(out_channels, out_channels, inject_noise=inject_noise)

    def forward(self, x):
        x = self.downconv(x)
        x = self.bn(x)
        x = self.relu(x)
        return self.double_conv(x)



class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True, inject_noise=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, mid_channels=in_channels // 2)


    def forward(self, x1, x2, noises = None):
        x1 = self.up(x1)
        x = torch.cat([x1, x2], dim=1)
        return self.conv(x, noises)


class UnetGenerator(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True, base_channel=32):
        super().__init__()
        self.in_channels = in_channels + 1
        self.out_channels = out_channels
        self.bilinear = bilinear

        factor = 2 if bilinear else 1

        max_channels = 512

        def c(ch):
            return min(ch, max_channels)

        self.in_conv = DoubleConv(self.in_channels, base_channel,inject_noise = False)
        self.down1 = Down(base_channel, c(base_channel * 2))           # 32 -> 64
        self.down2 = Down(c(base_channel * 2), c(base_channel * 4))    # 64 -> 128
        self.down3 = Down(c(base_channel * 4), c(base_channel * 8))    # 128 -> 256
        self.down4 = Down(c(base_channel * 8), c(base_channel * 8))    # 256 -> 256
        self.down5 = Down(c(base_channel * 8), c(base_channel * 8))    # 256 -> 256
        self.down6 = Down(c(base_channel * 8), c(base_channel * 8))    # 256 -> 256

        self.up1 = Up(c(base_channel * 8) * 2, c(base_channel * 8), bilinear)  
        self.up2 = Up(c(base_channel * 8) * 2, c(base_channel * 8), bilinear)
        self.up3 = Up(c(base_channel * 8) * 2, c(base_channel * 4), bilinear)
        self.up4 = Up(c(base_channel * 4) * 2, c(base_channel * 2), bilinear)
        self.up5 = Up(c(base_channel * 2) * 2, c(base_channel * 1), bilinear)
        self.up6 = Up(c(base_channel * 1) * 2, base_channel, bilinear)

        self.out_conv = nn.Conv2d(base_channel, out_channels, kernel_size=1)

    def forward(self, x, z, noises=None):
        B, _, H, W = x.shape
        z = z.view(B, 1, H, W)
        x = torch.cat([x, z], dim=1)

        # 下采样并保存特征图
        x1 = self.in_conv(x)    # [B, 64, 64, 64]
        x2 = self.down1(x1)     # [B, 128, 32, 32]
        x3 = self.down2(x2)     # [B, 256, 16, 16]
        x4 = self.down3(x3)     # [B, 512, 8, 8]
        x5 = self.down4(x4)     # [B, 512, 4, 4]
        x6 = self.down5(x5)     # [B, 512, 2, 2]
        x7 = self.down6(x6)     # [B, 512, 1, 1] 关键修改：使用所有下采样层

        # 上采样并拼接对应特征
        if noises is not None:
            x = self.up1(x7, x6, noises=[noises[0], noises[1]])  # [B, 512, 2, 2]
            x = self.up2(x, x5, noises=[noises[2], noises[3]])   # [B, 512, 4, 4]
            x = self.up3(x, x4, noises=[noises[4], noises[5]])   # [B, 512, 8, 8]
            x = self.up4(x, x3, noises=[noises[6], noises[7]])   # [B, 256, 16, 16]
            x = self.up5(x, x2, noises=[noises[8], noises[9]])   # [B, 128, 32, 32]
            x = self.up6(x, x1, noises=[noises[10], noises[11]]) # [B, 64, 64, 64]
        else:
            x = self.up1(x7, x6)  # [B, 512, 2, 2]
            x = self.up2(x, x5)   # [B, 512, 4, 4]
            x = self.up3(x, x4)   # [B, 512, 8, 8]
            x = self.up4(x, x3)   # [B, 256, 16, 16]
            x = self.up5(x, x2)   # [B, 128, 32, 32]
            x = self.up6(x, x1)   # [B, 64, 64, 64]


        return self.out_conv(x)  # [B, out_channels, 64, 64]

class WSConv2d(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size=3, stride=1, padding=1
    ):
        super(WSConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.scale = (2 / (in_channels * (kernel_size ** 2))) ** 0.5
        self.bias = self.conv.bias
        self.conv.bias = None

        # initialize conv layer
        nn.init.normal_(self.conv.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return self.conv(x * self.scale) + self.bias.view(1, self.bias.shape[0], 1, 1)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = WSConv2d(in_channels, out_channels)
        self.conv2 = WSConv2d(out_channels, out_channels)
        self.leaky = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.leaky(self.conv1(x))
        x = self.leaky(self.conv2(x))
        return x


class PredicateD(nn.Module):
    def __init__(self, in_channels, img_channels=1):
        super(PredicateD, self).__init__()
        self.prog_blocks, self.rgb_layers = nn.ModuleList([]), nn.ModuleList([])
        self.leaky = nn.LeakyReLU(0.2)

        # here we work back ways from factors because the discriminator
        # should be mirrored from the generator. So the first prog_block and
        # rgb layer we append will work for input size 1024x1024, then 512->256-> etc
        for i in range(len(factors) - 1, 0, -1):
            conv_in = int(in_channels * factors[i])
            conv_out = int(in_channels * factors[i - 1])
            self.prog_blocks.append(ConvBlock(conv_in, conv_out))
            self.rgb_layers.append(
                WSConv2d(img_channels, conv_in, kernel_size=1, stride=1, padding=0)
            )

        # perhaps confusing name "initial_rgb" this is just the RGB layer for 4x4 input size
        # did this to "mirror" the generator initial_rgb
        self.initial_rgb = WSConv2d(
            img_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.rgb_layers.append(self.initial_rgb)
        self.avg_pool = nn.AvgPool2d(
            kernel_size=2, stride=2
        )  # down sampling using avg pool

        # this is the block for 4x4 input size
        self.final_block = nn.Sequential(
            # +1 to in_channels because we concatenate from MiniBatch std
            WSConv2d(in_channels + 1, in_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            WSConv2d(in_channels, in_channels, kernel_size=4, padding=0, stride=1),
            nn.LeakyReLU(0.2),
            WSConv2d(
                in_channels, 1, kernel_size=1, padding=0, stride=1
            ),  # we use this instead of linear layer
        )

    def fade_in(self, alpha, downscaled, out):
        """Used to fade in downscaled using avg pooling and output from CNN"""
        # alpha should be scalar within [0, 1], and upscale.shape == generated.shape
        return alpha * out + (1 - alpha) * downscaled

    def minibatch_std(self, x):
        batch_statistics = (
            torch.std(x, dim=0).mean().repeat(x.shape[0], 1, x.shape[2], x.shape[3])
        )
        # we take the std for each example (across all channels, and pixels) then we repeat it
        # for a single channel and concatenate it with the image. In this way the discriminator
        # will get information about the variation in the batch/image
        return torch.cat([x, batch_statistics], dim=1)

    def forward(self, x, alpha, steps):
        # where we should start in the list of prog_blocks, maybe a bit confusing but
        # the last is for the 4x4. So example let's say steps=1, then we should start
        # at the second to last because input_size will be 8x8. If steps==0 we just
        # use the final block
        # x = torch.cat([x, c], dim=1)
        cur_step = len(self.prog_blocks) - steps

        # convert from rgb as initial step, this will depend on
        # the image size (each will have it's on rgb layer)
        out = self.leaky(self.rgb_layers[cur_step](x))

        if steps == 0:  # i.e, image is 4x4
            out = self.minibatch_std(out)
            return self.final_block(out).view(out.shape[0], -1)

        # because prog_blocks might change the channels, for down scale we use rgb_layer
        # from previous/smaller size which in our case correlates to +1 in the indexing
        downscaled = self.leaky(self.rgb_layers[cur_step + 1](self.avg_pool(x)))
        out = self.avg_pool(self.prog_blocks[cur_step](out))

        # the fade_in is done first between the downscaled and the input
        # this is opposite from the generator
        out = self.fade_in(alpha, downscaled, out)

        for step in range(cur_step + 1, len(self.prog_blocks)):
            out = self.prog_blocks[step](out)
            out = self.avg_pool(out)

        out = self.minibatch_std(out)
        return self.final_block(out).view(out.shape[0], -1)



# class NLayerDiscriminator(nn.Module):
#     def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
#         super().__init__()
#         use_bias = norm_layer == nn.InstanceNorm2d

#         self.model = nn.Sequential(
#             # Layer 1: RF = 1 (1x1)
#             nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
#             nn.LeakyReLU(0.2, True),

#             # Layer 2: RF = 2 (1 + (2-1)*1)
#             nn.Conv2d(ndf, ndf * 2, kernel_size=2, stride=1, padding=0, bias=use_bias),
#             norm_layer(ndf * 2),
#             nn.LeakyReLU(0.2, True),

#             # Layer 3: RF = 3
#             nn.Conv2d(ndf * 2, ndf * 4, kernel_size=2, stride=1, padding=0, bias=use_bias),
#             norm_layer(ndf * 4),
#             nn.LeakyReLU(0.2, True),

#             # Layer 4: RF = 4
#             nn.Conv2d(ndf * 4, 1, kernel_size=1, stride=1, padding=0),
#         )

#     def forward(self, x, c):
#         out = torch.cat([x, c], dim=1)  # 在通道维度拼接
#         return self.model(out)

class UNetPredictionDataset(Dataset):
    """
    一个为 U-Net 时间序列预测任务准备数据的 PyTorch Dataset 类。
    它加载一个包含时间序列图像的 .pt 文件，并返回 (t, t+1) 数据对。
    """
    def __init__(self, pt_path):
        super().__init__()
        
        print(f"[*] Initializing UNetPredictionDataset...")
        print(f"    - Loading data from: {os.path.basename(pt_path)}")
        
        # 1. 加载预处理好的 uint8 数据
        dataset_dict = torch.load(pt_path, weights_only=False)
        self.data = dataset_dict['data'] # 这是一个 [N, 1, 64, 64] 的 uint8 Tensor
        
        # 验证数据
        assert self.data.dtype == torch.uint8, "数据必须是 torch.uint8 类型"
        print(f"✅ Data loaded successfully. Found {len(self.data)} total frames.")

    def __len__(self):
        """
        返回可用的 (t, t+1) 数据对的数量。
        比总帧数少 1，因为最后一帧没有对应的 t+1。
        """
        return len(self.data) - 1

    def __getitem__(self, idx):
        """
        获取一个 (输入, 目标) 数据对，并进行格式转换。
        
        返回:
            (torch.Tensor, torch.Tensor): 两个都是 float32, [-1, 1] 范围的 Tensor。
        """
        # 获取输入图像 (t) 和目标图像 (t+1)
        input_img_uint8 = self.data[idx]
        target_img_uint8 = self.data[idx + 1]
        
        # --- 数据转换函数 ---
        def transform(image_uint8):
            # 1. 转换为 float32
            image_float = image_uint8.to(torch.float32)
            # 2. 归一化到 [-1, 1]
            return (image_float / 127.5) - 1.0

        # 应用转换
        input_img_normalized = transform(input_img_uint8)
        target_img_normalized = transform(target_img_uint8)
        
        return input_img_normalized, target_img_normalized

class UNET:
    def __init__(self, img_size, logger = None, opt = None, device='cuda'):
        self.G = UnetGenerator(in_channels=1, out_channels=1, bilinear=True, base_channel=32).to(device)
        self.D = PredicateD(256, img_channels = 1).to(device)
        # self.D = NLayerDiscriminator(input_nc=2, ndf=64, norm_layer=nn.BatchNorm2d).to(device)
        self.logger = logger
        self.opt = opt
        self.device = device
        self.img_size = img_size
        self.optimizer_G = torch.optim.Adam(self.G.parameters(), lr=opt['lr'], betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.Adam(self.D.parameters(), lr=opt['lr'] * 0.1, betas=(0.5, 0.999))

        self.criterion_GAN = nn.BCEWithLogitsLoss()

        # 你需要一个梯度惩罚的辅助函数
    def compute_gradient_penalty(self, D, real_samples, fake_samples, device):
        """Calculates the gradient penalty loss for WGAN GP.
           Warning: It's a conditional GAN, so we need the condition 'current_step' as well.
           Let's assume an unconditional GAN first to match your current code structure.
        """
        # Random weight term for interpolation between real and fake samples
        batch_size = real_samples.size(0)
        alpha_gp = torch.rand((batch_size, 1, 1, 1), device=device)
        
        # Get random interpolation between real and fake samples
        interpolates = (alpha_gp * real_samples + ((1 - alpha_gp) * fake_samples)).requires_grad_(True)
        
        d_interpolates = D(interpolates, alpha=1, steps=4)
        fake_grad_output = torch.ones_like(d_interpolates, device=device, requires_grad=False)
    
        # Get gradient w.r.t. interpolates
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake_grad_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty
        
    def sst_loss_fn(self, pred, target):
        # """
        # SST专用损失函数（无陆地，全温度场）
        # 1. 普通 MSE 损失
        # 2. 空间梯度保持损失（确保温度连续平滑）
        # """
        # 基础 MSE
        mse_loss = F.l1_loss(pred, target)
    
        # 空间梯度损失（保持温度场平滑性）
        pred_grad_x = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        target_grad_x = target[:, :, :, 1:] - target[:, :, :, :-1]
        grad_loss_x = F.mse_loss(pred_grad_x, target_grad_x)
    
        pred_grad_y = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        target_grad_y = target[:, :, 1:, :] - target[:, :, :-1, :]
        grad_loss_y = F.mse_loss(pred_grad_y, target_grad_y)
    
        # 总损失：MSE + 0.1 * (x方向梯度损失 + y方向梯度损失)
        return mse_loss + 0.1 * (grad_loss_x + grad_loss_y)

    def get_prediction_dataloader(self, pt_path, batch_size = 64, shuffle=False, num_workers=4):
        """
        一个便捷的函数，用于创建 U-Net 预测任务所需的数据加载器。
        """
        # 1. 创建数据集实例
        dataset = UNetPredictionDataset(pt_path=pt_path)
        
        # 2. 创建 DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,        # 训练时通常需要打乱
            num_workers=num_workers,# 使用多进程加速数据加载
            pin_memory=True         # 如果使用 GPU，可以加速内存拷贝
        )
        
        print(f"\n[INFO] DataLoader created.")
        print(f"       - Number of training pairs (t, t+1): {len(dataset)}")
        print(f"       - Batch size: {batch_size}")
        print(f"       - Batches per epoch: {len(dataloader)}")
        
        return dataloader

    def get_loader(self, image_size, results):
        
        batch_size = 64
        datasets = {}
        loaders = {}
    
        for t_key in ["t0", "t1"]:
            data = results[t_key]  # Tensor: [N, 1, H, W]
    
            if t_key == "t0":
                _, _, H, W = data.shape
                if H != image_size or W != image_size:
                    data = F.interpolate(data, size=(image_size, image_size), mode='bilinear', align_corners=False)
    
            datasets[t_key] = TensorDataset(data, torch.zeros(len(data)))
            loaders[t_key] = DataLoader(datasets[t_key], batch_size=batch_size, shuffle=False)
        return loaders, datasets
    
    def load_model(self, load_path):
        print(f"✅ UNET 模型恢复训练")
        checkpoint = torch.load(load_path, map_location=self.device)
        self.G.load_state_dict(checkpoint["G_state_dict"])
        self.D.load_state_dict(checkpoint["D_state_dict"])
        self.optimizer_G.load_state_dict(checkpoint["optimizer_G"])
        self.optimizer_D.load_state_dict(checkpoint["optimizer_D"])
        print(f"UNET 模型已从 {load_path} 加载完成")
    
    def train(self, iteration, results, epochs=1000):

        print(f"✅ UNET 预测器正在训练")

        # train_loader = self.get_prediction_dataloader(
        #     pt_path="datasets/test.pt", 
        #     batch_size=16,
        #     shuffle=False 
        # )
        # for epoch in range(epochs):
        #     # --- 2. 迭代训练数据 (已修改) ---
        #     # 直接遍历 train_loader，它会产生 (t0, t1) 数据对
        #     loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        #     for batch_idx, (t0, t1) in enumerate(loop):
        #         self.optimize_predicate(t0, t1, epoch, batch_idx)
        
        loaders, datasets = self.get_loader(self.img_size, results)  
        for epoch in range(epochs):
            logging.info(f"Epoch [{epoch}]")
            loop = tqdm(zip(loaders["t0"], loaders["t1"]), total=len(loaders["t0"]), disable=True)

            for batch_idx, ((t0, _), (t1, _)) in enumerate(loop):
                self.optimize_predicate(t0, t1, epoch, batch_idx)
        

        # ✅ 保存路径带 iteration
        save_path = f"results/output/predicate/iter_{iteration}/unet_predicate.pth"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({
            "G_state_dict": self.G.state_dict(),
            "D_state_dict": self.D.state_dict(),
            "optimizer_G": self.optimizer_G.state_dict(),
            "optimizer_D": self.optimizer_D.state_dict(),
        }, save_path)
        print(f"UNET 模型已保存到: {save_path}")

    def optimize_predicate(self, current_step, next_step, epoch, batch_idx):

        current_step = current_step.to(self.device)
        next_step = next_step.to(self.device)
     
        batch_size = current_step.size(0)
        latent_dim = self.opt['latent_dim']
        
        # ======== Train Discriminator (Pix2Pix Style) ========
        self.optimizer_D.zero_grad()
        
        # --- Real Image Loss ---
        # 判别器对真实的 next_step 进行判断
        real_output = self.D(next_step, alpha=1, steps=4)
        # 真实样本的目标标签是 1
        real_labels = torch.ones(real_output.size(), device=self.device)
        d_loss_real = self.criterion_GAN(real_output, real_labels)
        
        # --- Fake Image Loss ---
        # 生成器根据 current_step 生成假的 next_step
        with torch.no_grad():
            z = torch.randn(batch_size, latent_dim, device=self.device) * 5
            fake_imgs = self.G(current_step, z)
        
        # 判别器对生成的 fake_imgs 进行判断
        fake_output = self.D(fake_imgs.detach(), alpha=1, steps=4)
        # 伪造样本的目标标签是 0
        fake_labels = torch.zeros(fake_output.size(), device=self.device)
        d_loss_fake = self.criterion_GAN(fake_output, fake_labels)
        
        # --- Total Discriminator Loss ---
        # 综合真实和伪造样本的损失
        d_loss = (d_loss_real + d_loss_fake) * 0.5 # 乘以 0.5 是常见做法，用于平衡损失
        
        d_loss.backward()
        self.optimizer_D.step()
        
        # ======== Train Generator ========
        # 生成器的训练部分可以保持不变，因为它也适用于Pix2Pix的框架
        # （一个对抗性损失 + 一个重建损失）
        for _ in range(5): # 您这里的循环次数可以根据需要调整
            self.optimizer_G.zero_grad()
            z = torch.randn(batch_size, latent_dim, device=self.device) * 5
        
            gen_imgs = self.G(current_step, z)
            # 为了计算生成器的对抗损失，我们希望判别器将生成图像误判为“真实”
            g_adv_output = self.D(gen_imgs, alpha=1, steps=4)
            # 生成器的目标是让判别器输出接近1的概率
            g_adv_loss = self.criterion_GAN(g_adv_output, real_labels) # 复用上面创建的 real_labels
            
            # 重建损失（L1 Loss 在 Pix2Pix 中更常用，但您的 sst_loss_fn 也可以）
            g_recon_loss = self.sst_loss_fn(gen_imgs, next_step) * 100
        
            g_total_loss = g_adv_loss + g_recon_loss
            g_total_loss.backward()
            self.optimizer_G.step()
        
        # ======== Logging ========
        if batch_idx % self.opt['log_interval'] == 0:
            mse = F.mse_loss(gen_imgs, next_step).item()
            psnr = 10 * torch.log10(1 / torch.tensor(mse)) if mse > 0 else float('inf')
        
            self.logger.info(
                f"📊 Epoch {epoch} | Batch {batch_idx:04d}\n"
                f"  ├── D Loss: {d_loss.item():.4f}\n"
                f"  └── PSNR: {psnr:.2f} dB"
            )
        
        if epoch % self.opt["image_log_interval"] == 0 and batch_idx == 0:
            save_sst_visualizations_jet_discrete(
                s0=current_step,
                s1=gen_imgs,
                batch_idx=batch_idx,
                folder_path=os.path.join(self.opt['image_log_dir'], f"epoch_{epoch:03d}"),
                cmap='jet'
            )