import os
import pandas as pd
import cv2
import numpy as np
from datetime import datetime

# --- 配置 ---
IMAGE_ROOT = './image'  # 菌落图像根目录
OUTPUT_CSV = 'colonies_analysis_result.csv'  # 输出CSV文件名
DIRECTIONS = ['down', 'up', 'vertical']  # 需要分析的方向

def calculate_ellipticity(mask_path):
    """
    加载一个mask图片，找到最大的轮廓，拟合一个椭圆，并计算其椭圆度。
    椭圆度定义为：长轴 / 短轴。一个完美的圆椭圆度为1。

    参数:
        mask_path (str): mask图片的文件路径。

    返回:
        dict: 包含椭圆相关信息的字典，如果无法计算则返回None。
    """
    try:
        # 以灰度模式读取图像
        image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"    - Warning: Could not read image {mask_path}")
            return None

        # 二值化处理（确保图像是黑白的）
        _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

        # 查找轮廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            print(f"    - Warning: No contours found in {mask_path}")
            return None

        # 找到面积最大的轮廓
        main_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(main_contour)
        perimeter = cv2.arcLength(main_contour, True)

        # 拟合椭圆需要轮廓至少有5个点
        if len(main_contour) < 5:
            print(f"    - Warning: Not enough points in contour to fit ellipse for {mask_path}")
            return None

        # 拟合椭圆
        # 返回值: ((center_x, center_y), (minor_axis, major_axis), angle)
        (center, axes, angle) = cv2.fitEllipse(main_contour)

        major_axis = max(axes)
        minor_axis = min(axes)

        # 避免除以零错误
        if minor_axis == 0:
            ellipticity = np.inf
        else:
            ellipticity = major_axis / minor_axis

        return {
            'area': area,
            'perimeter': perimeter,
            'ellipticity': ellipticity,
            'major_axis': major_axis,
            'minor_axis': minor_axis,
            'center_x': center[0],
            'center_y': center[1],
            'ellipse_angle': angle
        }

    except Exception as e:
        print(f"    - Error calculating ellipticity for {mask_path}: {e}")
        return None


def analyze_colony_shape(mask_path):
    """
    加载菌落的mask图片，分析其轮廓形状，计算从重心到轮廓各点的距离统计信息。

    参数:
        mask_path (str): mask图片的文件路径。

    返回:
        dict: 包含形状分析指标的字典，如果无法分析则返回None。
    """
    try:
        # 以灰度模式读取图像
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"    - Warning: Could not read image {mask_path}")
            return None

        # 二值化处理
        _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        # 查找轮廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            print(f"    - Warning: No contours found in {mask_path}")
            return None

        # 找到面积最大的轮廓 (即菌落的主体)
        main_contour = max(contours, key=cv2.contourArea)

        # 对轮廓进行平滑处理，减少像素化带来的噪声
        epsilon = 0.001 * cv2.arcLength(main_contour, True)
        smoothed_contour = cv2.approxPolyDP(main_contour, epsilon, True)

        # 计算轮廓的矩
        M = cv2.moments(smoothed_contour)
        if M["m00"] == 0:
            print(f"    - Warning: Moment is zero, cannot calculate centroid for {mask_path}")
            return None

        # 计算重心 (cx, cy)
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        # 计算从重心到轮廓上每个点的距离
        distances = []
        for point in smoothed_contour:
            px, py = point[0]
            dist = np.sqrt((px - cx)**2 + (py - cy)**2)
            distances.append(dist)

        if not distances:
            return None

        distances = np.array(distances)

        # 计算统计指标
        max_radius = np.max(distances)
        min_radius = np.min(distances)
        mean_radius = np.mean(distances)
        std_radius = np.std(distances)

        # 找到最大和最小半径对应的点，并计算其角度
        max_idx = np.argmax(distances)
        min_idx = np.argmin(distances)

        max_point = smoothed_contour[max_idx][0]
        min_point = smoothed_contour[min_idx][0]

        # 使用 atan2 计算角度 (范围在 -pi 到 pi)，然后转换为度 (0 到 360)
        max_angle = np.degrees(np.arctan2(max_point[1] - cy, max_point[0] - cx))
        min_angle = np.degrees(np.arctan2(min_point[1] - cy, min_point[0] - cx))

        # 将角度调整到 0-360 度范围
        if max_angle < 0:
            max_angle += 360
        if min_angle < 0:
            min_angle += 360

        return {
            "centroid_x": cx,
            "centroid_y": cy,
            "max_radius": max_radius,
            "max_radius_angle": max_angle,
            "min_radius": min_radius,
            "min_radius_angle": min_angle,
            "mean_radius": mean_radius,
            "std_radius": std_radius,
            "radius_ratio": max_radius / min_radius if min_radius > 0 else np.inf
        }

    except Exception as e:
        print(f"    - Error analyzing shape for {mask_path}: {e}")
        return None


def analyze_single_colony(image_path, direction):
    """
    分析单个菌落图像，返回完整的分析结果。

    参数:
        image_path (str): 图像文件路径。
        direction (str): 方向标签 (down/up/vertical)。

    返回:
        dict: 包含所有分析指标的字典。
    """
    # 基础信息
    filename = os.path.basename(image_path)
    colony_id = os.path.splitext(filename)[0]

    result = {
        'colony_id': colony_id,
        'direction': direction,
        'image_path': image_path,
        'filename': filename
    }

    # 计算椭圆度及相关指标
    ellipse_metrics = calculate_ellipticity(image_path)
    if ellipse_metrics:
        result.update(ellipse_metrics)
    else:
        # 如果计算失败，填充空值
        result.update({
            'area': None,
            'perimeter': None,
            'ellipticity': None,
            'major_axis': None,
            'minor_axis': None,
            'center_x': None,
            'center_y': None,
            'ellipse_angle': None
        })

    # 计算形状指标
    shape_metrics = analyze_colony_shape(image_path)
    if shape_metrics:
        result.update(shape_metrics)
    else:
        # 如果计算失败，填充空值
        result.update({
            "centroid_x": None,
            "centroid_y": None,
            "max_radius": None,
            "max_radius_angle": None,
            "min_radius": None,
            "min_radius_angle": None,
            "mean_radius": None,
            "std_radius": None,
            "radius_ratio": None
        })

    return result


def process_all_colonies():
    """
    遍历所有方向的菌落图像，进行批量分析并保存结果。
    """
    print("=" * 60)
    print("菌落图像批量分析工具")
    print("=" * 60)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"图像根目录: {IMAGE_ROOT}")
    print(f"输出文件: {OUTPUT_CSV}")
    print()

    # 检查根目录是否存在
    if not os.path.isdir(IMAGE_ROOT):
        print(f"错误: 根目录 '{IMAGE_ROOT}' 不存在!")
        return

    all_results = []
    total_images = 0
    success_count = 0

    # 遍历每个方向
    for direction in DIRECTIONS:
        dir_path = os.path.join(IMAGE_ROOT, direction)

        if not os.path.isdir(dir_path):
            print(f"警告: 方向文件夹 '{dir_path}' 不存在，跳过。")
            continue

        # 获取所有BMP文件
        image_files = sorted([f for f in os.listdir(dir_path) if f.lower().endswith('.bmp')])

        if not image_files:
            print(f"警告: 在 '{direction}' 文件夹中未找到BMP图像。")
            continue

        print(f"\n处理方向: {direction} (共 {len(image_files)} 张图像)")
        print("-" * 60)

        # 处理每张图像
        for idx, filename in enumerate(image_files, 1):
            image_path = os.path.join(dir_path, filename)
            print(f"  [{idx}/{len(image_files)}] 处理: {filename} ...", end=' ')

            try:
                result = analyze_single_colony(image_path, direction)
                all_results.append(result)

                # 检查是否成功计算了主要指标
                if result['ellipticity'] is not None and result['max_radius'] is not None:
                    success_count += 1
                    print("[OK]")
                else:
                    print("[WARN]")

                total_images += 1

            except Exception as e:
                print(f"[ERROR]: {e}")
                total_images += 1

    # 保存结果到CSV
    print("\n" + "=" * 60)
    print("分析完成，正在保存结果...")

    if all_results:
        df = pd.DataFrame(all_results)

        # 重新排列列顺序，使其更易读
        column_order = [
            'colony_id', 'direction', 'filename', 'image_path',
            'area', 'perimeter', 'ellipticity',
            'major_axis', 'minor_axis', 'center_x', 'center_y', 'ellipse_angle',
            'centroid_x', 'centroid_y',
            'max_radius', 'max_radius_angle',
            'min_radius', 'min_radius_angle',
            'mean_radius', 'std_radius', 'radius_ratio'
        ]

        # 确保所有列都存在
        for col in column_order:
            if col not in df.columns:
                df[col] = None

        df = df[column_order]

        # 保存到CSV
        df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')

        print(f"\n结果已保存到: {OUTPUT_CSV}")
        print(f"总计处理: {total_images} 张图像")
        print(f"成功分析: {success_count} 张图像")
        print(f"失败/部分失败: {total_images - success_count} 张图像")

        # 显示统计信息
        print("\n" + "=" * 60)
        print("各方向统计:")
        direction_stats = df['direction'].value_counts()
        for direction, count in direction_stats.items():
            print(f"  {direction}: {count} 张")

        # 显示数据摘要
        print("\n" + "=" * 60)
        print("数据摘要 (前5行):")
        print(df[['colony_id', 'direction', 'area', 'ellipticity', 'radius_ratio']].head())

    else:
        print("错误: 没有成功分析任何图像。")

    print("\n" + "=" * 60)
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)


if __name__ == "__main__":
    process_all_colonies()
