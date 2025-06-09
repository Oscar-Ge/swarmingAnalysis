import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- 方案一：优化后的霍夫圆变换 ---
# 核心思路：
# 1. 缩小图像：在处理前先将图像缩小，可以极大减少计算量。
# 2. 调整参数：使用更合理的半径范围和累加器阈值。
def create_petri_dish_mask_hough_optimized(image_path, resize_width=500):
    """
    使用优化后的霍夫圆变换来识别培养皿。
    返回: 
    - result_image: 带有绿色检测圆圈的原图副本。
    - mask: 黑白的蒙版图像。
    - masked_original: 应用了蒙版的原图。
    """
    # 1. 加载图像
    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"图片加载失败，请检查路径: {image_path}")
        return None, None, None
        
    # 2. 预处理：缩小图像尺寸以加快处理速度
    original_height, original_width = original_image.shape[:2]
    scale = resize_width / original_width
    resized_height = int(original_height * scale)
    resized_image = cv2.resize(original_image, (resize_width, resized_height))
    
    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    
    # 使用高斯模糊代替中值模糊，对于霍夫变换通常效果更好且稍快
    gray_blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    # 3. 执行霍夫圆检测 (参数已调整)
    circles = cv2.HoughCircles(
        gray_blurred, 
        cv2.HOUGH_GRADIENT, 
        dp=1.2, 
        minDist=int(resize_width / 2),
        param1=100, 
        param2=40,
        minRadius=int(resize_width * 0.3), 
        maxRadius=int(resize_width * 0.55)
    )

    mask = None
    masked_original = None
    result_image = original_image.copy() # 在原图上绘制结果
    
    if circles is not None:
        # 将检测到的圆的坐标和半径按比例还原到原始图像尺寸
        best_circle = circles[0, 0]
        center_resized = (int(best_circle[0]), int(best_circle[1]))
        radius_resized = int(best_circle[2])
        
        # 还原到原图坐标
        center_original = (int(center_resized[0] / scale), int(center_resized[1] / scale))
        radius_original = int(radius_resized / scale)
        
        # 4. 在原始尺寸的空mask上绘制白色实心圆
        mask = np.zeros(original_image.shape[:2], dtype=np.uint8)
        cv2.circle(mask, center_original, radius_original, 255, -1)
        
        # 在原图上画出检测到的圆以供可视化
        cv2.circle(result_image, center_original, radius_original, (0, 255, 0), 10)

        # 5. 【新增】使用蒙版提取原图中的培养皿区域
        masked_original = cv2.bitwise_and(original_image, original_image, mask=mask)

    return result_image, mask, masked_original

# --- 方案二：更快、更推荐的方法：轮廓检测 ---
# 核心思路：
# 1. 阈值分割：将图像变为黑白二值图，让培养皿变成一个白色区域。
# 2. 查找轮廓：找到所有独立的白色区域的边界。
# 3. 筛选轮廓：根据面积筛选出最大的那个轮廓，即培养皿。
# 4. 创建Mask：根据找到的轮廓创建一个填充的mask。
def create_petri_dish_mask_contour(image_path):
    """
    使用轮廓检测来识别培养皿，速度通常远快于霍夫变换。
    返回:
    - result_image: 带有绿色检测轮廓的原图副本。
    - mask: 黑白的蒙版图像。
    - masked_original: 应用了蒙版的原图。
    """
    # 1. 加载图像并转为灰度图
    image = cv2.imread(image_path)
    if image is None:
        print(f"图片加载失败，请检查路径: {image_path}")
        return None, None, None
        
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 2. 阈值处理
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 3. 查找轮廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    result_image = image.copy()
    mask = None
    masked_original = None
    
    if contours:
        # 4. 找到面积最大的轮廓
        main_contour = max(contours, key=cv2.contourArea)
        
        # 5. 创建一个与原图同样大小的黑色图像 (mask)
        mask = np.zeros_like(gray)
        
        # 在 mask 上将轮廓内部填充为白色
        cv2.drawContours(mask, [main_contour], -1, 255, thickness=cv2.FILLED)
        
        # 为了可视化，可以在原图上画出轮廓
        cv2.drawContours(result_image, [main_contour], -1, (0, 255, 0), 10)

        # 6. 【新增】使用蒙版提取原图中的培养皿区域
        masked_original = cv2.bitwise_and(image, image, mask=mask)

    return result_image, mask, masked_original


# --- 使用示例 ---
# 请确保你的图片路径是正确的
try:
    image_path = '/data/coding/images/2025-06-05/vertical/1.bmp' 
    original_for_display = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
except:
    print("无法加载示例图片，请修改为您的本地图片路径")
    original_for_display = np.zeros((400, 400, 3), dtype=np.uint8)
    image_path = None


if image_path:
    # --- 测试优化后的霍夫变换 ---
    result_image_hough, result_mask_hough, masked_image_hough = create_petri_dish_mask_hough_optimized(image_path)

    # --- 测试轮廓检测法 ---
    result_image_contour, result_mask_contour, masked_image_contour = create_petri_dish_mask_contour(image_path)


    # --- 使用 matplotlib 显示结果对比 ---
    plt.figure(figsize=(20, 10)) # 调整画布大小以容纳更多图像
    plt.suptitle("不同方法的性能对比", fontsize=16)

    # --- 霍夫变换行 ---
    plt.subplot(2, 4, 1)
    plt.imshow(original_for_display)
    plt.title('原始图像')
    plt.axis('off')

    if result_image_hough is not None:
        plt.subplot(2, 4, 2)
        plt.imshow(cv2.cvtColor(result_image_hough, cv2.COLOR_BGR2RGB))
        plt.title('优化霍夫变换')
        plt.axis('off')

    if result_mask_hough is not None:
        plt.subplot(2, 4, 3)
        plt.imshow(result_mask_hough, cmap='gray')
        plt.title('霍夫变换-Mask')
        plt.axis('off')

    if masked_image_hough is not None:
        plt.subplot(2, 4, 4)
        plt.imshow(cv2.cvtColor(masked_image_hough, cv2.COLOR_BGR2RGB))
        plt.title('霍夫变换-蒙版后图像')
        plt.axis('off')

    # --- 轮廓检测行 ---
    plt.subplot(2, 4, 5)
    plt.imshow(original_for_display) 
    plt.title('原始图像')
    plt.axis('off')

    if result_image_contour is not None:
        plt.subplot(2, 4, 6)
        plt.imshow(cv2.cvtColor(result_image_contour, cv2.COLOR_BGR2RGB))
        plt.title('轮廓检测 (推荐)')
        plt.axis('off')

    if result_mask_contour is not None:
        plt.subplot(2, 4, 7)
        plt.imshow(result_mask_contour, cmap='gray')
        plt.title('轮廓检测-Mask')
        plt.axis('off')
    
    if masked_image_contour is not None:
        plt.subplot(2, 4, 8)
        plt.imshow(cv2.cvtColor(masked_image_contour, cv2.COLOR_BGR2RGB))
        plt.title('轮廓检测-蒙版后图像')
        plt.axis('off')
        
        # 【新增】保存应用了蒙版的原图
        save_path = 'masked_petri_dish_contour.png'
        cv2.imwrite(save_path, masked_image_contour)
        print(f"成功保存蒙版后的图像到: {save_path}")
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    plt.savefig("comp.png")