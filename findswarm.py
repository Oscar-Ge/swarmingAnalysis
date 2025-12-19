import cv2
import numpy as np
import math
import os
import glob

# --- 可调参数 (您可以根据需要调整这些值) ---

# 1. 面积筛选阈值 (用来定义 "不是太大也不是太小")
#    - 目标面积占总面积的最小比例，用于排除噪点或太小的物体 (如 5.png, 6.png)
MIN_AREA_RATIO = 0.005  # 0.5%
#    - 目标面积占总面积的最大比例，用于排除背景或太大的物体 (如 0.png, 1.png)
MAX_AREA_RATIO = 0.25   # 20%

# 2. 中心距离筛选阈值 (用来定义 "离中心点足够近")
#    - 目标中心点与图像中心点的最大允许距离，以图像短边长度的比例计算
MAX_DISTANCE_RATIO = 0.1 # 距离不超过短边的 25%

# --- 脚本主逻辑 ---

# 获取当前文件夹中所有png图像文件
# 注意：请确保此脚本与您的图片文件在同一个文件夹下
input_folder = './images/analysis/2025-06-05/down/5' 
image_files = glob.glob(os.path.join(input_folder, '*.png'))
print(f"找到 {len(image_files)} 张待处理图像: {image_files}")

# --- 修改点 1: 变量初始化 ---
# 用于存储最佳候选者信息
best_candidate_file = None
# 我们要找面积最大的，所以初始值设为0
max_found_area = 0 

# 遍历每一张图片
for image_path in image_files:
    print(f"\n--- 正在分析: {os.path.basename(image_path)} ---")
    
    # 读取图像为灰度图
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"错误：无法读取图像 {image_path}")
        continue
    
    # 获取图像尺寸和中心点
    height, width = image.shape
    total_area = height * width
    center_x, center_y = width // 2, height // 2

    # 寻找图像中的轮廓
    # 因为输入是黑白二值图，可以直接找轮廓
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 如果没有找到轮廓，则跳过
    if not contours:
        print("结果: 未找到任何轮廓。")
        continue

    # 通常我们假设每张图只有一个主要物体，选择其中面积最大的轮廓进行分析
    main_contour = max(contours, key=cv2.contourArea)
    
    # --- 1. 应用面积筛选 ---
    contour_area = cv2.contourArea(main_contour)
    area_ratio = contour_area / total_area
    print(f"信息: 物体面积 = {contour_area} 像素, 占总面积 {area_ratio:.2%}")

    # 判断面积是否在定义的“适中”范围内
    if not (MIN_AREA_RATIO < area_ratio < MAX_AREA_RATIO):
        print(f"结果: 筛选失败 (面积不符)。目标范围: {MIN_AREA_RATIO:.2%} ~ {MAX_AREA_RATIO:.2%}")
        continue
    
    print("状态: ✔ 面积符合要求。")

    # --- 2. 应用中心距离筛选 ---
    # 计算轮廓的几何中心（质心）
    M = cv2.moments(main_contour)
    if M["m00"] == 0:
        print("结果: 无法计算质心。")
        continue
    
    obj_center_x = int(M["m10"] / M["m00"])
    obj_center_y = int(M["m01"] / M["m00"])

    # 计算物体中心与图像中心的距离
    distance = math.sqrt((obj_center_x - center_x)**2 + (obj_center_y - center_y)**2)
    print(f"信息: 物体中心 ({obj_center_x}, {obj_center_y}), 离图像中心距离 {distance:.2f} 像素")
    
    # 判断距离是否“足够近”
    max_allowed_distance = min(height, width) * MAX_DISTANCE_RATIO
    if distance > max_allowed_distance:
        print(f"结果: 筛选失败 (距离太远)。最大允许距离: {max_allowed_distance:.2f} 像素")
        continue

    print("状态: ✔ 距离符合要求。")
    print("状态: ★★★ 这是一张合格的候选图像！ ★★★")

    # --- 修改点 2: 最终选择逻辑 ---
    # 在所有合格的候选者中，我们选择面积最大的那一个
    if contour_area > max_found_area:
        print(f"更新: 发现一个更好的候选者 (面积更大: {contour_area:.0f} > {max_found_area:.0f})。")
        max_found_area = contour_area
        best_candidate_file = image_path

# --- 输出最终结果 ---
print("\n======================================")
if best_candidate_file:
    print(f"✅ 分析完成！最终选择的菌群图像是: {os.path.basename(best_candidate_file)}")
else:
    print("❌ 分析完成，但没有找到任何一张同时满足所有条件的图像。")
    print("   您可以尝试放宽脚本开头的 MIN/MAX_AREA_RATIO 或 MAX_DISTANCE_RATIO 参数。")
print("======================================")