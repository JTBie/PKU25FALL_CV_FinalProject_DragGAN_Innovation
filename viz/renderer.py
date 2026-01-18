# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from socket import has_dualstack_ipv6
import sys
import copy
import traceback
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.cm
import dnnlib
from torch_utils.ops import upfirdn2d
import legacy # pylint: disable=import-error
from raft import OpticalFlowCalculator #only for ONE pair of images

#----------------------------------------------------------------------------

class CapturedException(Exception):
    def __init__(self, msg=None):
        if msg is None:
            _type, value, _traceback = sys.exc_info()
            assert value is not None
            if isinstance(value, CapturedException):
                msg = str(value)
            else:
                msg = traceback.format_exc()
        assert isinstance(msg, str)
        super().__init__(msg)

#----------------------------------------------------------------------------

class CaptureSuccess(Exception):
    def __init__(self, out):
        super().__init__()
        self.out = out

#----------------------------------------------------------------------------

def add_watermark_np(input_image_array, watermark_text="AI Generated"):
    image = Image.fromarray(np.uint8(input_image_array)).convert("RGBA")

    # Initialize text image
    txt = Image.new('RGBA', image.size, (255, 255, 255, 0))
    font = ImageFont.truetype('arial.ttf', round(25/512*image.size[0]))
    d = ImageDraw.Draw(txt)

    text_width, text_height = font.getsize(watermark_text)
    text_position = (image.size[0] - text_width - 10, image.size[1] - text_height - 10)
    text_color = (255, 255, 255, 128)  # white color with the alpha channel set to semi-transparent

    # Draw the text onto the text canvas
    d.text(text_position, watermark_text, font=font, fill=text_color)

    # Combine the image with the watermark
    watermarked = Image.alpha_composite(image, txt)
    watermarked_array = np.array(watermarked)
    return watermarked_array

#----------------------------------------------------------------------------

class Renderer:
    def __init__(self, disable_timing=False):
        self._device        = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        self._dtype         = torch.float32 if self._device.type == 'mps' else torch.float64
        self._pkl_data      = dict()    # {pkl: dict | CapturedException, ...}
        self._networks      = dict()    # {cache_key: torch.nn.Module, ...}
        self._pinned_bufs   = dict()    # {(shape, dtype): torch.Tensor, ...}
        self._cmaps         = dict()    # {name: torch.Tensor, ...}
        self._is_timing     = False
        ## RAFT相关初始化
        self.raft_calculator = OpticalFlowCalculator(device=self._device)
        self.reference_frame = None
        self.reference_points = None
        self.frame_count = 0
        self.max_frames_before_reset = 999
        ##
        if not disable_timing:
            self._start_event   = torch.cuda.Event(enable_timing=True)
            self._end_event     = torch.cuda.Event(enable_timing=True)
        self._disable_timing = disable_timing
        self._net_layers    = dict()    # {cache_key: [dnnlib.EasyDict, ...], ...}

    def render(self, **args):
        if self._disable_timing:
            self._is_timing = False
        else:
            self._start_event.record(torch.cuda.current_stream(self._device))
            self._is_timing = True
        res = dnnlib.EasyDict()
        try:
            init_net = False
            if not hasattr(self, 'G'):
                init_net = True
            if hasattr(self, 'pkl'):
                if self.pkl != args['pkl']:
                    init_net = True
            if hasattr(self, 'w_load'):
                if self.w_load is not args['w_load']:
                    init_net = True
            if hasattr(self, 'w0_seed'):
                if self.w0_seed != args['w0_seed']:
                    init_net = True
            if hasattr(self, 'w_plus'):
                if self.w_plus != args['w_plus']:
                    init_net = True
            #
            if hasattr(self, 'is_raft'):
                if self.is_raft != args.get('is_raft', False):
                    init_net = True
            #
            if args['reset_w']:
                init_net = True
            res.init_net = init_net
            if init_net:
                self.init_network(res, **args)
            self._render_drag_impl(res, **args)
        except:
            res.error = CapturedException()
        if not self._disable_timing:
            self._end_event.record(torch.cuda.current_stream(self._device))
        if 'image' in res:
            res.image = self.to_cpu(res.image).detach().numpy()
            res.image = add_watermark_np(res.image, 'AI Generated')
        if 'stats' in res:
            res.stats = self.to_cpu(res.stats).detach().numpy()
        if 'error' in res:
            res.error = str(res.error)
        # if 'stop' in res and res.stop:

        if self._is_timing and not self._disable_timing:
            self._end_event.synchronize()
            res.render_time = self._start_event.elapsed_time(self._end_event) * 1e-3
            self._is_timing = False
        return res

    def get_network(self, pkl, key, **tweak_kwargs):
        data = self._pkl_data.get(pkl, None)
        if data is None:
            print(f'Loading "{pkl}"... ', end='', flush=True)
            try:
                with dnnlib.util.open_url(pkl, verbose=False) as f:
                    data = legacy.load_network_pkl(f)
                print('Done.')
            except:
                data = CapturedException()
                print('Failed!')
            self._pkl_data[pkl] = data
            self._ignore_timing()
        if isinstance(data, CapturedException):
            raise data

        orig_net = data[key]
        cache_key = (orig_net, self._device, tuple(sorted(tweak_kwargs.items())))
        net = self._networks.get(cache_key, None)
        if net is None:
            try:
                if 'stylegan2' in pkl:
                    from training.networks_stylegan2 import Generator
                elif 'stylegan3' in pkl:
                    from training.networks_stylegan3 import Generator
                elif 'stylegan_human' in pkl:
                    from stylegan_human.training_scripts.sg2.training.networks import Generator
                else:
                    raise NameError('Cannot infer model type from pkl name!')

                print(data[key].init_args)
                print(data[key].init_kwargs)
                if 'stylegan_human' in pkl:
                    net = Generator(*data[key].init_args, **data[key].init_kwargs, square=False, padding=True)
                else:
                    net = Generator(*data[key].init_args, **data[key].init_kwargs)
                net.load_state_dict(data[key].state_dict())
                net.to(self._device)
            except:
                net = CapturedException()
            self._networks[cache_key] = net
            self._ignore_timing()
        if isinstance(net, CapturedException):
            raise net
        return net

    def _get_pinned_buf(self, ref):
        key = (tuple(ref.shape), ref.dtype)
        buf = self._pinned_bufs.get(key, None)
        if buf is None:
            buf = torch.empty(ref.shape, dtype=ref.dtype).pin_memory()
            self._pinned_bufs[key] = buf
        return buf

    def to_device(self, buf):
        return self._get_pinned_buf(buf).copy_(buf).to(self._device)

    def to_cpu(self, buf):
        return self._get_pinned_buf(buf).copy_(buf).clone()

    def _ignore_timing(self):
        self._is_timing = False

    def _apply_cmap(self, x, name='viridis'):
        cmap = self._cmaps.get(name, None)
        if cmap is None:
            cmap = matplotlib.cm.get_cmap(name)
            cmap = cmap(np.linspace(0, 1, num=1024), bytes=True)[:, :3]
            cmap = self.to_device(torch.from_numpy(cmap))
            self._cmaps[name] = cmap
        hi = cmap.shape[0] - 1
        x = (x * hi + 0.5).clamp(0, hi).to(torch.int64)
        x = torch.nn.functional.embedding(x, cmap)
        return x

    def init_network(self, res,
        pkl             = None,
        w0_seed         = 0,
        w_load          = None,
        w_plus          = True,
        noise_mode      = 'const',
        trunc_psi       = 0.7,
        trunc_cutoff    = None,
        input_transform = None,
        lr              = 0.001,
        #
        is_raft         = False,
        #
        **kwargs
        ):
        # Dig up network details.
        self.pkl = pkl
        G = self.get_network(pkl, 'G_ema')
        self.G = G
        res.img_resolution = G.img_resolution
        res.num_ws = G.num_ws
        res.has_noise = any('noise_const' in name for name, _buf in G.synthesis.named_buffers())
        res.has_input_transform = (hasattr(G.synthesis, 'input') and hasattr(G.synthesis.input, 'transform'))

        # Set input transform.
        if res.has_input_transform:
            m = np.eye(3)
            try:
                if input_transform is not None:
                    m = np.linalg.inv(np.asarray(input_transform))
            except np.linalg.LinAlgError:
                res.error = CapturedException()
            G.synthesis.input.transform.copy_(torch.from_numpy(m))

        # Generate random latents.
        self.w0_seed = w0_seed
        self.w_load = w_load

        if self.w_load is None:
            # Generate random latents.
            z = torch.from_numpy(np.random.RandomState(w0_seed).randn(1, 512)).to(self._device, dtype=self._dtype)

            # Run mapping network.
            label = torch.zeros([1, G.c_dim], device=self._device)
            w = G.mapping(z, label, truncation_psi=trunc_psi, truncation_cutoff=trunc_cutoff)
        else:
            w = self.w_load.clone().to(self._device)

        self.w0 = w.detach().clone()
        self.w_plus = w_plus
        if w_plus:
            self.w = w.detach()
        else:
            self.w = w[:, 0, :].detach()
        self.w.requires_grad = True
        self.w_optim = torch.optim.Adam([self.w], lr=lr)
        #
        self.is_raft = is_raft
        #
        self.feat_refs = None
        self.points0_pt = None
        # 新增：用于硬约束背景保护的特征缓存
        self.feat_refs_dict = None  # {resolution: feature_tensor}，存储原始特征 F_original^(l)
        self.mask_blurred_full = None  # 平滑后的全分辨率mask
        # 新增：用于软约束背景保护的原始图像缓存
        self.img_original = None  # 原始图像，用于L1 loss计算

    def update_lr(self, lr):

        del self.w_optim
        self.w_optim = torch.optim.Adam([self.w], lr=lr)
        print(f'Rebuild optimizer with lr: {lr}')
        print('    Remain feat_refs and points0_pt')

    def _prepare_mask_with_blur(self, mask, img_resolution, blur_sigma=3.0):
        """
        对 mask 进行高斯模糊处理，使边缘平滑
        
        参数:
        - mask: [H, W] 或 [1, H, W]，1为保护区域（背景），0为可编辑区域（前景）
        - img_resolution: 图像分辨率
        - blur_sigma: 高斯模糊的标准差
        
        返回:
        - mask_blurred: [1, 1, H, W]，平滑后的mask，1为前景，0为背景
        """
        device = mask.device if isinstance(mask, torch.Tensor) else self._device
        
        # 转换为tensor
        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask, dtype=torch.float32, device=device)
        
        # 确保是 [1, 1, H, W] 格式
        if mask.dim() == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif mask.dim() == 3:
            mask = mask.unsqueeze(0)
        if mask.shape[0] != 1:
            mask = mask[0:1, :, :]
        if mask.shape[1] != 1:
            mask = mask[:, 0:1, :, :]
        
        # 调整到目标分辨率
        if mask.shape[2] != img_resolution or mask.shape[3] != img_resolution:
            mask = F.interpolate(mask, size=(img_resolution, img_resolution), 
                               mode='bilinear', align_corners=False)
        
        # 反转mask：原mask中1是保护（背景），0是可编辑（前景）
        # 新mask中1是前景（可编辑），0是背景（保护）
        mask_foreground = 1.0 - mask
        
        # 高斯模糊
        kernel_size = int(6 * blur_sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # 使用 separable 高斯模糊
        channels = mask_foreground.shape[1]
        kernel_1d = self._gaussian_kernel_1d(kernel_size, blur_sigma, device)
        kernel_1d = kernel_1d.view(1, 1, kernel_size, 1).repeat(channels, 1, 1, 1)
        
        # 先水平模糊
        mask_blurred = F.conv2d(mask_foreground, kernel_1d, padding=(0, kernel_size//2), groups=channels)
        # 再垂直模糊
        kernel_1d = kernel_1d.transpose(2, 3)
        mask_blurred = F.conv2d(mask_blurred, kernel_1d, padding=(kernel_size//2, 0), groups=channels)
        
        return mask_blurred
    
    def _gaussian_kernel_1d(self, size, sigma, device):
        """生成1D高斯核"""
        coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        return g
    
    def _cache_reference_features(self, G, ws, blend_resolutions=None):
        """
        缓存原始图像在关键层的特征图 F_original^(l)
        
        参数:
        - G: Generator
        - ws: 初始的 latent code w_orig
        - blend_resolutions: 需要缓存的分辨率列表，如 [64, 128, 256]
        
        返回:
        - feat_refs_dict: {resolution: feature_tensor}
        """
        if blend_resolutions is None:
            blend_resolutions = [64, 128, 256]  # 默认值
        
        label = torch.zeros([1, G.c_dim], device=self._device)
        
        # 运行生成器获取特征（使用原始w）
        with torch.no_grad():
            img, features = G(ws, label, truncation_psi=0.7, noise_mode='const', 
                            input_is_w=True, return_feature=True)
        
        # 构建分辨率到特征的映射
        feat_refs_dict = {}
        block_resolutions = G.synthesis.block_resolutions
        
        for i, res in enumerate(block_resolutions):
            if res in blend_resolutions:
                # 获取对应层的特征并持久化存储
                feat_refs_dict[res] = features[i].detach().clone()
        
        return feat_refs_dict

    def _render_drag_impl(self, res,
        points          = [],
        targets         = [],
        mask            = None,
        lambda_mask     = 10,
        reg             = 0,
        feature_idx     = 5,
        r1              = 3,
        r2              = 12,
        random_seed     = 0,
        noise_mode      = 'const',
        trunc_psi       = 0.7,
        force_fp32      = False,
        layer_name      = None,
        sel_channels    = 3,
        base_channel    = 0,
        img_scale_db    = 0,
        img_normalize   = False,
        untransform     = False,
        is_drag         = False,
        reset           = False,
        to_pil          = False,
        **kwargs
    ):
        G = self.G
        ws = self.w
        if ws.dim() == 2:
            ws = ws.unsqueeze(1).repeat(1,6,1)
        ws = torch.cat([ws[:,:6,:], self.w0[:,6:,:]], dim=1)
        if hasattr(self, 'points'):
            if len(points) != len(self.points):
                reset = True
        if reset:
            self.feat_refs = None
            self.points0_pt = None
            self.feat_refs_dict = None  # 重置特征缓存
            self.mask_blurred_full = None  # 重置mask
            self.img_original = None  # 重置原始图像缓存
            ## 重置RAFT相关变量
            self.reference_frame = None
            self.reference_points = None
            self.frame_count = 0
            ##
        self.points = points

        # ========== 预处理：特征缓存（在开始优化前） ==========
        # 如果使用硬约束且mask存在，在第一次迭代时缓存原始特征
        use_hard_constraint = kwargs.get('use_hard_constraint', False)
        blend_resolutions = kwargs.get('blend_resolutions', [64, 128, 256])
        mask_blur_sigma = kwargs.get('mask_blur_sigma', 3.0)
        
        if is_drag and use_hard_constraint and mask is not None:
            if self.feat_refs_dict is None or reset:
                # 使用原始w缓存参考特征
                print(f"Caching reference features at resolutions: {blend_resolutions}")
                self.feat_refs_dict = self._cache_reference_features(
                    G, self.w0, blend_resolutions=blend_resolutions
                )
                
                # 处理 mask：转换为前景mask并平滑
                mask_tensor = torch.tensor(mask, dtype=torch.float32, device=self._device)
                if mask_tensor.max() > 1.0:
                    mask_tensor = mask_tensor / 255.0
                
                # 平滑处理
                self.mask_blurred_full = self._prepare_mask_with_blur(
                    mask_tensor, 
                    G.img_resolution, 
                    blur_sigma=mask_blur_sigma
                )
                print(f"Mask blurred with sigma={mask_blur_sigma}")
        # ====================================================

        # ========== 预处理：缓存原始图像（用于软约束L1 loss） ==========
        # 如果不使用硬约束且mask存在，在第一次迭代时缓存原始图像
        if is_drag and not use_hard_constraint and mask is not None:
            if self.img_original is None or reset:
                # 使用原始w生成原始图像
                with torch.no_grad():
                    label_orig = torch.zeros([1, G.c_dim], device=self._device)
                    img_orig = G(self.w0, label_orig, truncation_psi=trunc_psi, 
                                noise_mode=noise_mode, input_is_w=True, return_feature=False)
                    # 转换为与当前图像相同的格式
                    img_orig = img_orig[0]
                    if img_normalize:
                        img_orig = img_orig / img_orig.norm(float('inf'), dim=[1,2], keepdim=True).clip(1e-8, 1e8)
                    img_orig = img_orig * (10 ** (img_scale_db / 20))
                    self.img_original = img_orig  # 保存为 [C, H, W] 格式
                    print("Caching original image for soft constraint L1 loss")
        # ====================================================

        # Run synthesis network.
        label = torch.zeros([1, G.c_dim], device=self._device)
        
        # 准备特征融合参数
        synthesis_kwargs = {}
        # 即使 is_drag=False（已停止），只要启用了硬约束且特征缓存存在，就继续应用硬约束
        # 这样可以确保最后一步渲染时背景仍然受到保护
        if use_hard_constraint and self.feat_refs_dict is not None:
            synthesis_kwargs['feat_refs_dict'] = self.feat_refs_dict
            synthesis_kwargs['mask_blurred_full'] = self.mask_blurred_full
            synthesis_kwargs['blend_resolutions'] = blend_resolutions
        
        img, feat = G(ws, label, truncation_psi=trunc_psi, noise_mode=noise_mode, 
                     input_is_w=True, return_feature=True, **synthesis_kwargs)

        h, w = G.img_resolution, G.img_resolution

        if is_drag:
            X = torch.linspace(0, h, h)
            Y = torch.linspace(0, w, w)
            xx, yy = torch.meshgrid(X, Y)
            feat_resize = F.interpolate(feat[feature_idx], [h, w], mode='bilinear')
            if self.feat_refs is None:
                self.feat0_resize = F.interpolate(feat[feature_idx].detach(), [h, w], mode='bilinear')
                self.feat_refs = []
                for point in points:
                    py, px = round(point[0]), round(point[1])
                    self.feat_refs.append(self.feat0_resize[:,:,py,px])
                self.points0_pt = torch.Tensor(points).unsqueeze(0).to(self._device) # 1, N, 2
            #
            if not self.is_raft:
                # Point tracking with feature matching
                with torch.no_grad():
                    for j, point in enumerate(points):
                        r = round(r2 / 512 * h)
                        up = max(point[0] - r, 0)
                        down = min(point[0] + r + 1, h)
                        left = max(point[1] - r, 0)
                        right = min(point[1] + r + 1, w)
                        feat_patch = feat_resize[:,:,up:down,left:right]
                        L2 = torch.linalg.norm(feat_patch - self.feat_refs[j].reshape(1,-1,1,1), dim=1)
                        _, idx = torch.min(L2.view(1,-1), -1)
                        width = right - left
                        point = [idx.item() // width + up, idx.item() % width + left]
                        points[j] = point

                res.points = [[point[0], point[1]] for point in points]

            else:
                # Point tracking with RAFT
                current_img = img[0].cpu().detach()
                current_img = (current_img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                current_img = current_img.permute(1, 2, 0).numpy()

                from PIL import Image
                current_img_pil = Image.fromarray(current_img)

                _reset = False

                if self.reference_frame is None: # initialization
                    self.reference_frame = current_img_pil.copy()
                    self.reference_points = self.points.copy()
                    tracked_points = self.points.copy()

                else:
                    flow_tensor = self.raft_calculator.calculate_optical_flow(
                        self.reference_frame,
                        current_img_pil
                    )
                    flow_np = flow_tensor[0].cpu().numpy()

                    tracked_points = []
                    for point_idx, ref_point in enumerate(self.reference_points):
                        y, x = ref_point[0], ref_point[1]
                        h_flow, w_flow = flow_np.shape[1], flow_np.shape[2]
                        y0, x0 = int(np.floor(y)), int(np.floor(x))
                        y1, x1 = min(y0 + 1, h_flow - 1), min(x0 + 1, w_flow - 1)
                        wy1, wx1 = y - y0, x - x0
                        wy0, wx0 = 1 - wy1, 1 - wx1
                        u = (
                                flow_np[0, y0, x0] * wy0 * wx0 +
                                flow_np[0, y0, x1] * wy0 * wx1 +
                                flow_np[0, y1, x0] * wy1 * wx0 +
                                flow_np[0, y1, x1] * wy1 * wx1
                        )
                        v = (
                                flow_np[1, y0, x0] * wy0 * wx0 +
                                flow_np[1, y0, x1] * wy0 * wx1 +
                                flow_np[1, y1, x0] * wy1 * wx0 +
                                flow_np[1, y1, x1] * wy1 * wx1
                        )
                        new_y = y + v
                        new_x = x + u

                        tracked_points.append([new_y, new_x])

                    # 奇妙的小小操作：
                    # 注意到有时相邻图片会追踪失败（噪音或是什么）
                    # 于是选择在每一次大幅跳动（局部最优估计）后重置参考点
                    # 注意到有时大幅跳动会是错误匹配，所以需要在大幅跳动后的第二次追踪点并未大跳
                    _reset = True
                    for _ in range(len(tracked_points)):
                        if not ((tracked_points[_][0] - self.points[_][0])**2 + (tracked_points[_][1] - self.points[_][1])**2 <= 2 * max(2 / 512 * h, 2) ** 2 and (self.reference_points[_][0] - self.points[_][0])**2 + (self.reference_points[_][1] - self.points[_][1])**2 >= 100 * max(2 / 512 * h, 2) ** 2):
                            _reset = False
                            break

                self.frame_count += 1

                points = tracked_points.copy()
                res.points = [[point[0], point[1]] for point in points]

                if self.frame_count > self.max_frames_before_reset or _reset:
                    self.reference_frame = current_img_pil.copy()
                    self.reference_points = points.copy()
                    self.frame_count = 0
            #

            # Motion supervision
            loss_motion = 0
            res.stop = True
            for j, point in enumerate(points):
                direction = torch.Tensor([targets[j][1] - point[1], targets[j][0] - point[0]])
                if torch.linalg.norm(direction) > 1 * max(2 / 512 * h, 2):
                    res.stop = False
                if torch.linalg.norm(direction) > 1:
                    distance = ((xx.to(self._device) - point[0])**2 + (yy.to(self._device) - point[1])**2)**0.5
                    relis, reljs = torch.where(distance < round(r1 / 512 * h))
                    direction = direction / (torch.linalg.norm(direction) + 1e-7)
                    gridh = (relis+direction[1]) / (h-1) * 2 - 1
                    gridw = (reljs+direction[0]) / (w-1) * 2 - 1
                    grid = torch.stack([gridw,gridh], dim=-1).unsqueeze(0).unsqueeze(0)
                    target = F.grid_sample(feat_resize.float(), grid, align_corners=True).squeeze(2)
                    loss_motion += F.l1_loss(feat_resize[:,:,relis,reljs].detach(), target)

            loss = loss_motion
            
            # ========== 背景保护的L1 Loss（软约束模式） ==========
            # 如果不使用硬约束，使用L1 Loss来约束背景区域
            if not use_hard_constraint and mask is not None and self.img_original is not None:
                # 准备mask：1为背景（保护），0为前景（可编辑）
                mask_tensor = torch.tensor(mask, dtype=torch.float32, device=self._device)
                if mask_tensor.max() > 1.0:
                    mask_tensor = mask_tensor / 255.0
                
                # 调整mask到图像分辨率
                if mask_tensor.dim() == 2:
                    mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
                elif mask_tensor.dim() == 3:
                    mask_tensor = mask_tensor.unsqueeze(0)  # [1, 1, H, W]
                
                if mask_tensor.shape[2] != h or mask_tensor.shape[3] != w:
                    mask_tensor = F.interpolate(mask_tensor, size=(h, w), 
                                              mode='bilinear', align_corners=False)
                
                # 当前图像（需要转换为与原始图像相同的格式）
                img_current = img[0]  # [C, H, W]
                if img_normalize:
                    img_current = img_current / img_current.norm(float('inf'), dim=[1,2], keepdim=True).clip(1e-8, 1e8)
                img_current = img_current * (10 ** (img_scale_db / 20))
                
                # 确保尺寸匹配
                if img_current.shape[1] != h or img_current.shape[2] != w:
                    img_current = F.interpolate(img_current.unsqueeze(0), size=(h, w), 
                                              mode='bilinear', align_corners=False)[0]
                if self.img_original.shape[1] != h or self.img_original.shape[2] != w:
                    img_orig_resized = F.interpolate(self.img_original.unsqueeze(0), size=(h, w), 
                                                   mode='bilinear', align_corners=False)[0]
                else:
                    img_orig_resized = self.img_original
                
                # 计算背景区域的L1 loss
                # mask: 1为背景，0为前景
                mask_bg = mask_tensor.squeeze(0).squeeze(0)  # [H, W]
                # 扩展到与图像相同的通道数 [C, H, W]
                if mask_bg.dim() == 2:
                    mask_bg = mask_bg.unsqueeze(0).repeat(img_current.shape[0], 1, 1)  # [C, H, W]
                
                # 只在背景区域计算L1 loss
                loss_mask = lambda_mask * F.l1_loss(
                    img_current * mask_bg, 
                    img_orig_resized * mask_bg
                )
                loss += loss_mask
            # ====================================================

            loss += reg * F.l1_loss(ws, self.w0)  # latent code regularization
            if not res.stop:
                self.w_optim.zero_grad()
                loss.backward()
                self.w_optim.step()

        # Scale and convert to uint8.
        img = img[0]
        if img_normalize:
            img = img / img.norm(float('inf'), dim=[1,2], keepdim=True).clip(1e-8, 1e8)
        img = img * (10 ** (img_scale_db / 20))
        img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8).permute(1, 2, 0)
        if to_pil:
            from PIL import Image
            img = img.cpu().numpy()
            img = Image.fromarray(img)
        res.image = img
        res.w = ws.detach().cpu().numpy()

#----------------------------------------------------------------------------
