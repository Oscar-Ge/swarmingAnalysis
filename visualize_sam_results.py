import os
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import re

def get_original_image_path(mask_path):
    """
    从mask路径推断原始图片路径
    例如: ./2025-06-05/up/1/4.png -> images/2025-06-05/up/1.bmp
    """
    # 提取信息: date/direction/experiment_id
    pattern = r'(\d{4}-\d{2}-\d{2})[/\\](\w+)[/\\](\d+)[/\\]'
    match = re.search(pattern, mask_path)

    if match:
        date, direction, exp_id = match.groups()
        original_path = f'images/{date}/{direction}/{exp_id}.bmp'
        return original_path
    return None

def fix_mask_path(mask_path):
    """
    修正mask路径，将相对路径转换为实际路径
    ./2025-06-05/up/1/4.png -> images/analysis/2025-06-05/up/1/4.png
    """
    if mask_path.startswith('./'):
        return 'images/analysis/' + mask_path[2:]
    return mask_path

def create_visualization(original_img, mask_img, metadata, save_path=None):
    """
    创建包含原图、mask overlay和单独mask的可视化
    """
    # 创建图形
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 1. 原始图像
    axes[0].imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')

    # 2. Mask overlay on original image
    overlay = original_img.copy()
    # 创建彩色mask (红色半透明)
    colored_mask = np.zeros_like(original_img)
    colored_mask[mask_img > 0] = [0, 255, 255]  # 黄色 (BGR格式)

    # 混合
    overlay = cv2.addWeighted(original_img, 0.7, colored_mask, 0.3, 0)

    # 绘制边界框
    bbox_x, bbox_y, bbox_w, bbox_h = metadata['bbox_x0'], metadata['bbox_y0'], metadata['bbox_w'], metadata['bbox_h']
    cv2.rectangle(overlay, (int(bbox_x), int(bbox_y)),
                  (int(bbox_x + bbox_w), int(bbox_y + bbox_h)),
                  (0, 0, 255), 3)  # 红色边界框

    axes[1].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axes[1].set_title('Segmentation Overlay', fontsize=14, fontweight='bold')
    axes[1].axis('off')

    # 3. 单独的mask
    axes[2].imshow(mask_img, cmap='gray')
    axes[2].set_title('Segmentation Mask', fontsize=14, fontweight='bold')
    axes[2].axis('off')

    # 添加整体标题
    fig.suptitle('Segment Anything Model - Colony Detection Results',
                 fontsize=16, fontweight='bold', y=0.98)

    # 将元数据信息添加到右上角
    info_text = (f"Date: {metadata['date']}\n"
                 f"Direction: {metadata['direction']}\n"
                 f"Experiment: {metadata['experiment_id']}\n"
                 f"Area: {metadata['area']:.0f} px\n"
                 f"Ellipticity: {metadata.get('ellipticity', 'N/A'):.3f}\n"
                 f"IOU: {metadata['predicted_iou']:.3f}")

    fig.text(0.98, 0.95, info_text, ha='right', va='top', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5, pad=0.5))

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to: {save_path}")

    return fig

def select_and_visualize_sample(csv_path='images/analysis/colonies_data.csv',
                                 output_dir='visualization_output'):
    """
    从CSV中选择一个示例并创建可视化
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 读取CSV数据
    df = pd.read_csv(csv_path)

    print(f"Total colonies in CSV: {len(df)}")
    print("\nAvailable samples:")
    print(df[['date', 'direction', 'experiment_id', 'area', 'ellipticity']].head(10))

    # 选择一个质量较好的样本（面积适中、椭圆度接近1、IOU高）
    # 过滤条件：面积在10万到100万之间，椭圆度小于2（比较圆）
    filtered_df = df[
        (df['area'] > 100000) &
        (df['area'] < 1000000) &
        (df['ellipticity'] < 2.0) &
        (df['predicted_iou'] > 0.99)
    ].copy()

    if len(filtered_df) == 0:
        print("\nNo samples meet the criteria, using first sample from CSV...")
        sample = df.iloc[0]
    else:
        # 按IOU排序，选择最好的
        filtered_df = filtered_df.sort_values('predicted_iou', ascending=False)
        sample = filtered_df.iloc[0]

    print(f"\n{'='*60}")
    print(f"Selected sample:")
    print(f"  Date: {sample['date']}")
    print(f"  Direction: {sample['direction']}")
    print(f"  Experiment ID: {sample['experiment_id']}")
    print(f"  Mask path: {sample['mask_path']}")
    print(f"  Area: {sample['area']:.0f} pixels")
    print(f"  Ellipticity: {sample['ellipticity']:.3f}")
    print(f"  IOU Score: {sample['predicted_iou']:.3f}")
    print(f"{'='*60}\n")

    # 获取原始图片路径
    mask_path = sample['mask_path']
    mask_path = fix_mask_path(mask_path)  # 修正mask路径
    original_img_path = get_original_image_path(mask_path)

    if not original_img_path:
        print(f"Error: Could not parse mask path: {mask_path}")
        return None

    print(f"Original image path: {original_img_path}")
    print(f"Mask path: {mask_path}")

    # 读取图片
    if not os.path.exists(original_img_path):
        print(f"Error: Original image not found: {original_img_path}")
        return None

    if not os.path.exists(mask_path):
        print(f"Error: Mask image not found: {mask_path}")
        return None

    original_img = cv2.imread(original_img_path)
    mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if original_img is None:
        print(f"Error: Could not read original image: {original_img_path}")
        return None

    if mask_img is None:
        print(f"Error: Could not read mask image: {mask_path}")
        return None

    print(f"Original image shape: {original_img.shape}")
    print(f"Mask image shape: {mask_img.shape}")

    # 创建可视化
    metadata = sample.to_dict()
    output_filename = f"{output_dir}/sam_visualization_{sample['date']}_{sample['direction']}_{sample['experiment_id']}.png"

    fig = create_visualization(original_img, mask_img, metadata, output_filename)

    # 显示
    plt.show()

    return fig, sample

def batch_visualize_top_samples(csv_path='images/analysis/colonies_data.csv',
                                 output_dir='visualization_output',
                                 n_samples=5,
                                 direction=None):
    """
    批量生成多个高质量样本的可视化

    参数:
        csv_path: CSV数据文件路径
        output_dir: 输出目录
        n_samples: 生成样本数量
        direction: 指定方向 ('up', 'down', 'vertical', None表示所有方向)
    """
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(csv_path)

    # 选择多个高质量样本（放宽条件）
    filtered_df = df[
        (df['area'] > 100000) &
        (df['area'] < 1000000) &
        (df['ellipticity'] < 3.5) &  # 放宽椭圆度限制
        (df['predicted_iou'] > 0.95)  # 放宽IOU限制
    ].copy()

    # 如果指定了方向，只选择该方向的样本
    if direction:
        filtered_df = filtered_df[filtered_df['direction'] == direction].copy()
        print(f"Filtering for direction: {direction}")

    filtered_df = filtered_df.sort_values('predicted_iou', ascending=False).head(n_samples)

    print(f"Generating visualizations for {len(filtered_df)} samples...")

    for idx, (_, sample) in enumerate(filtered_df.iterrows()):
        print(f"\n[{idx+1}/{len(filtered_df)}] Processing: {sample['date']} - {sample['direction']} - Exp {sample['experiment_id']}")

        mask_path = sample['mask_path']
        mask_path = fix_mask_path(mask_path)  # 修正mask路径
        original_img_path = get_original_image_path(mask_path)

        if not original_img_path or not os.path.exists(original_img_path) or not os.path.exists(mask_path):
            print(f"  Original: {original_img_path} (exists: {os.path.exists(original_img_path) if original_img_path else False})")
            print(f"  Mask: {mask_path} (exists: {os.path.exists(mask_path)})")
            print(f"  Skipping due to missing files...")
            continue

        original_img = cv2.imread(original_img_path)
        mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if original_img is None or mask_img is None:
            print(f"  Skipping due to read error...")
            continue

        metadata = sample.to_dict()
        output_filename = f"{output_dir}/sam_viz_{idx+1}_{sample['date']}_{sample['direction']}_exp{sample['experiment_id']}.png"

        create_visualization(original_img, mask_img, metadata, output_filename)
        plt.close()

    print(f"\nAll visualizations saved to: {output_dir}/")

if __name__ == "__main__":
    import sys

    # 可以通过命令行参数选择模式
    if len(sys.argv) > 1 and sys.argv[1] == 'batch':
        # 批量模式
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        direction = sys.argv[3] if len(sys.argv) > 3 else None
        batch_visualize_top_samples(n_samples=n, direction=direction)
    else:
        # 单个样本模式（交互式）
        select_and_visualize_sample()
