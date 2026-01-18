"""
测试脚本：对比软约束和硬约束背景保护效果

使用方法：
1. 直接运行：python test_hard_constraint.py
2. 会生成对比图像和评估报告
"""

import torch
import numpy as np
from PIL import Image
import dnnlib
import os
import argparse
from viz.renderer import Renderer

def create_test_mask(img_resolution, mask_type='center'):
    """
    创建测试用的mask
    
    参数:
    - img_resolution: 图像分辨率
    - mask_type: 'center' (中心可编辑) 或 'edge' (边缘可编辑)
    
    返回:
    - mask: [H, W], 1为保护区域（背景），0为可编辑区域（前景）
    """
    mask = np.ones((img_resolution, img_resolution), dtype=np.uint8)
    
    if mask_type == 'center':
        # 中心区域设为可编辑（前景），边缘为保护（背景）
        center_x, center_y = img_resolution // 2, img_resolution // 2
        radius = img_resolution // 3
        y, x = np.ogrid[:img_resolution, :img_resolution]
        mask_circle = (x - center_x)**2 + (y - center_y)**2 <= radius**2
        mask[mask_circle] = 0  # 中心区域可编辑
    elif mask_type == 'edge':
        # 边缘区域设为可编辑（前景），中心为保护（背景）
        center_x, center_y = img_resolution // 2, img_resolution // 2
        radius = img_resolution // 3
        y, x = np.ogrid[:img_resolution, :img_resolution]
        mask_circle = (x - center_x)**2 + (y - center_y)**2 <= radius**2
        mask[mask_circle] = 1  # 中心区域保护
        # 边缘区域已经是0（可编辑）
    
    return mask

def evaluate_background_preservation(original_img, edited_img, mask):
    """
    评估背景保护效果
    
    参数:
    - original_img: 原始图像 [H, W, 3], numpy array, 0-255
    - edited_img: 编辑后图像 [H, W, 3], numpy array, 0-255
    - mask: 背景mask [H, W], 1为背景，0为前景
    
    返回:
    - metrics: 包含各种指标的字典
    """
    # 提取背景区域
    bg_mask = mask > 0.5
    original_bg = original_img[bg_mask]
    edited_bg = edited_img[bg_mask]
    
    # 提取前景区域
    fg_mask = mask < 0.5
    original_fg = original_img[fg_mask]
    edited_fg = edited_img[fg_mask]
    
    # 计算背景区域的L2距离
    bg_l2 = np.mean((original_bg.astype(np.float32) - edited_bg.astype(np.float32)) ** 2)
    
    # 计算前景区域的L2距离（用于对比）
    fg_l2 = np.mean((original_fg.astype(np.float32) - edited_fg.astype(np.float32)) ** 2)
    
    # 计算背景区域的L1距离
    bg_l1 = np.mean(np.abs(original_bg.astype(np.float32) - edited_bg.astype(np.float32)))
    
    # 计算前景区域的L1距离
    fg_l1 = np.mean(np.abs(original_fg.astype(np.float32) - edited_fg.astype(np.float32)))
    
    # 保护比率（越小越好，表示背景变化小）
    preservation_ratio = bg_l2 / (fg_l2 + 1e-8)
    
    return {
        'background_l2': bg_l2,
        'foreground_l2': fg_l2,
        'background_l1': bg_l1,
        'foreground_l1': fg_l1,
        'preservation_ratio': preservation_ratio,
    }

def test_hard_constraint(pkl_path, seed=42, output_dir='test_results'):
    """
    测试硬约束背景保护功能
    
    参数:
    - pkl_path: 模型路径
    - seed: 随机种子
    - output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("测试硬约束背景保护功能")
    print("=" * 60)
    
    # 初始化渲染器
    print("\n1. 初始化渲染器...")
    renderer = Renderer(disable_timing=True)
    
    # 加载模型
    print(f"2. 加载模型: {pkl_path}")
    res = dnnlib.EasyDict()
    renderer.init_network(
        res,
        pkl=pkl_path,
        w0_seed=seed,
        w_plus=True,
        lr=0.001,
    )
    
    img_resolution = renderer.G.img_resolution
    print(f"   图像分辨率: {img_resolution}x{img_resolution}")
    
    # 生成原始图像
    print("\n3. 生成原始图像...")
    renderer.w = renderer.w0.clone()
    label = torch.zeros([1, renderer.G.c_dim], device=renderer._device)
    img_orig, _ = renderer.G(renderer.w0, label, truncation_psi=0.7, 
                           noise_mode='const', input_is_w=True)
    img_orig = (img_orig[0] * 127.5 + 128).clamp(0, 255).to(torch.uint8).permute(1, 2, 0)
    img_orig_np = img_orig.cpu().numpy()
    img_orig_pil = Image.fromarray(img_orig_np)
    img_orig_pil.save(os.path.join(output_dir, 'original_image.png'))
    print(f"   原始图像已保存: {output_dir}/original_image.png")
    
    # 创建测试mask
    print("\n4. 创建测试mask...")
    mask = create_test_mask(img_resolution, mask_type='center')
    mask_vis = Image.fromarray((mask * 255).astype(np.uint8))
    mask_vis.save(os.path.join(output_dir, 'test_mask.png'))
    print(f"   测试mask已保存: {output_dir}/test_mask.png")
    
    # 定义测试点（在可编辑区域内）
    center = img_resolution // 2
    points = [[center - 30, center - 30]]  # 起点
    targets = [[center + 30, center + 30]]  # 目标点
    print(f"\n5. 测试点: {points[0]} -> {targets[0]}")
    
    # 测试软约束（原始方法）
    print("\n" + "=" * 60)
    print("测试软约束方法（Loss-based）")
    print("=" * 60)
    renderer.feat_refs = None
    renderer.points0_pt = None
    renderer.feat_refs_dict = None
    renderer.mask_blurred_full = None
    renderer.w = renderer.w0.clone()
    renderer.w.requires_grad = True
    renderer.w_optim = torch.optim.Adam([renderer.w], lr=0.001)
    
    res_soft = dnnlib.EasyDict()
    max_iterations = 50
    print(f"   运行 {max_iterations} 次迭代...")
    for i in range(max_iterations):
        renderer._render_drag_impl(
            res_soft,
            points=points,
            targets=targets,
            mask=mask,
            lambda_mask=20,  # 软约束的lambda
            is_drag=True,
            use_hard_constraint=False,  # 软约束
            to_pil=False,
        )
        if res_soft.stop:
            print(f"   在第 {i+1} 次迭代时停止（点已到达目标）")
            break
        if (i + 1) % 10 == 0:
            print(f"   迭代 {i+1}/{max_iterations}")
    
    img_soft = res_soft.image
    if isinstance(img_soft, Image.Image):
        img_soft_np = np.array(img_soft)
    else:
        img_soft_np = img_soft
    img_soft_pil = Image.fromarray(img_soft_np)
    img_soft_pil.save(os.path.join(output_dir, 'result_soft_constraint.png'))
    print(f"   软约束结果已保存: {output_dir}/result_soft_constraint.png")
    
    # 评估软约束
    metrics_soft = evaluate_background_preservation(img_orig_np, img_soft_np, mask)
    
    # 测试硬约束（新方法）
    print("\n" + "=" * 60)
    print("测试硬约束方法（Feature Blending）")
    print("=" * 60)
    renderer.feat_refs = None
    renderer.points0_pt = None
    renderer.feat_refs_dict = None
    renderer.mask_blurred_full = None
    renderer.w = renderer.w0.clone()
    renderer.w.requires_grad = True
    renderer.w_optim = torch.optim.Adam([renderer.w], lr=0.001)
    
    res_hard = dnnlib.EasyDict()
    print(f"   运行 {max_iterations} 次迭代...")
    for i in range(max_iterations):
        renderer._render_drag_impl(
            res_hard,
            points=points,
            targets=targets,
            mask=mask,
            lambda_mask=20,  # 硬约束模式下不使用，但保留参数
            is_drag=True,
            use_hard_constraint=True,  # 硬约束
            blend_resolutions=[64, 128, 256],
            mask_blur_sigma=3.0,
            to_pil=False,
        )
        if res_hard.stop:
            print(f"   在第 {i+1} 次迭代时停止（点已到达目标）")
            break
        if (i + 1) % 10 == 0:
            print(f"   迭代 {i+1}/{max_iterations}")
    
    img_hard = res_hard.image
    if isinstance(img_hard, Image.Image):
        img_hard_np = np.array(img_hard)
    else:
        img_hard_np = img_hard
    img_hard_pil = Image.fromarray(img_hard_np)
    img_hard_pil.save(os.path.join(output_dir, 'result_hard_constraint.png'))
    print(f"   硬约束结果已保存: {output_dir}/result_hard_constraint.png")
    
    # 评估硬约束
    metrics_hard = evaluate_background_preservation(img_orig_np, img_hard_np, mask)
    
    # 打印对比结果
    print("\n" + "=" * 60)
    print("评估结果对比")
    print("=" * 60)
    print("\n软约束方法（Loss-based）:")
    print(f"  背景L2距离: {metrics_soft['background_l2']:.2f}")
    print(f"  前景L2距离: {metrics_soft['foreground_l2']:.2f}")
    print(f"  背景L1距离: {metrics_soft['background_l1']:.2f}")
    print(f"  保护比率 (背景L2/前景L2): {metrics_soft['preservation_ratio']:.4f} (越小越好)")
    
    print("\n硬约束方法（Feature Blending）:")
    print(f"  背景L2距离: {metrics_hard['background_l2']:.2f}")
    print(f"  前景L2距离: {metrics_hard['foreground_l2']:.2f}")
    print(f"  背景L1距离: {metrics_hard['background_l1']:.2f}")
    print(f"  保护比率 (背景L2/前景L2): {metrics_hard['preservation_ratio']:.4f} (越小越好)")
    
    print("\n" + "=" * 60)
    print("改进情况:")
    print("=" * 60)
    bg_l2_improvement = (metrics_soft['background_l2'] - metrics_hard['background_l2']) / metrics_soft['background_l2'] * 100
    preservation_improvement = (metrics_soft['preservation_ratio'] - metrics_hard['preservation_ratio']) / metrics_soft['preservation_ratio'] * 100
    print(f"  背景L2距离改进: {bg_l2_improvement:+.2f}%")
    print(f"  保护比率改进: {preservation_improvement:+.2f}%")
    
    print(f"\n所有结果已保存到: {output_dir}/")
    print("=" * 60)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='测试硬约束背景保护功能')
    parser.add_argument('--pkl', type=str, 
                       default='checkpoints/stylegan2-ffhq-512x512.pkl',
                       help='模型路径')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--output', type=str, default='test_results',
                       help='输出目录')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.pkl):
        print(f"错误: 模型文件不存在: {args.pkl}")
        print("请先下载模型或指定正确的路径")
        exit(1)
    
    test_hard_constraint(args.pkl, args.seed, args.output)

