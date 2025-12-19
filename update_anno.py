import pandas as pd
import cv2
import numpy as np
import os

def analyze_colony_shape(mask_path):
    """
    加载菌落的mask图片，分析其轮廓形状，计算从重心到轮廓各点的距离统计信息。

    参数:
        mask_path (str): mask图片的文件路径。

    返回:
        dict: 包含形状分析指标的字典，如果无法分析则返回None。
    """
    # 检查文件是否存在
    if not os.path.exists(mask_path):
        print(f"  - 警告: 文件未找到 {mask_path}")
        return None

    try:
        # 以灰度模式读取图像
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"  - 警告: 无法读取图片 {mask_path}")
            return None

        # 查找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            print(f"  - 警告: 在 {mask_path} 中未找到轮廓")
            return None

        # 找到面积最大的轮廓 (即菌落的主体)
        main_contour = max(contours, key=cv2.contourArea)
        
        # 对轮廓进行平滑处理，减少像素化带来的噪声
        # 使用cv2.arcLength计算轮廓的周长
        # 0.001 是一个经验值，用于控制平滑的程度，可以根据需要调整
        epsilon = 0.001 * cv2.arcLength(main_contour, True)
        smoothed_contour = cv2.approxPolyDP(main_contour, epsilon, True)


        # 计算轮廓的矩
        M = cv2.moments(smoothed_contour)
        if M["m00"] == 0:
            print(f"  - 警告: 轮廓矩为0，无法计算重心 {mask_path}")
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
            "max_radius": max_radius,
            "max_radius_angle": max_angle,
            "min_radius": min_radius,
            "min_radius_angle": min_angle,
            "mean_radius": mean_radius,
            "std_radius": std_radius,
            "radius_ratio": max_radius / min_radius if min_radius > 0 else np.inf
        }

    except Exception as e:
        print(f"  - 错误: 处理 {mask_path} 时发生异常: {e}")
        return None

def main():
    """
    主函数，用于读取CSV，处理每一行，并保存更新后的CSV。
    """
    input_csv = 'colonies_data_0713.csv'
    output_csv = 'colonies_data_0713_updated.csv'

    # 检查输入文件是否存在
    if not os.path.exists(input_csv):
        print(f"错误: 输入文件 '{input_csv}' 未找到。请先运行你原来的脚本生成该文件。")
        return

    # 加载CSV
    try:
        df = pd.read_csv(input_csv)
    except pd.errors.EmptyDataError:
        print(f"错误: 文件 '{input_csv}' 为空。")
        return
        
    print(f"成功加载 '{input_csv}', 共 {len(df)} 条记录。")

    # 用于存储新计算指标的列表
    new_metrics_list = []

    # 遍历DataFrame的每一行
    for index, row in df.iterrows():
        mask_path = row['mask_path']
        print(f"\n正在处理第 {index + 1} 条记录: {mask_path}")
        
        # 分析形状
        metrics = analyze_colony_shape(mask_path)
        
        if metrics:
            new_metrics_list.append(metrics)
            print("  - 分析完成。")
        else:
            # 如果分析失败，添加一个空字典以保持行数一致
            new_metrics_list.append({})
            print("  - 分析失败。")

    # 将新指标转换为DataFrame
    metrics_df = pd.DataFrame(new_metrics_list, index=df.index)

    # 将新旧DataFrame合并
    # 为了避免重复添加列，先删除可能已存在的旧列
    cols_to_drop = [
        'max_radius', 'max_radius_angle', 'min_radius', 
        'min_radius_angle', 'mean_radius', 'std_radius', 'radius_ratio'
    ]
    df_cleaned = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    
    # 合并
    updated_df = pd.concat([df_cleaned, metrics_df], axis=1)

    # 保存到新的CSV文件
    updated_df.to_csv(output_csv, index=False)
    print(f"\n处理完成！更新后的数据已保存到 '{output_csv}'。")


if __name__ == "__main__":
    main()