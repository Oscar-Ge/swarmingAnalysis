import cv2
import numpy as np
import math # 需要 math 模块用于 pi

def analyze_red_area_detailed(image_path):
    """
    分析图片中的红色区域面积和形状，并计算圆形度、凸性系数和进行椭圆拟合。

    参数:
    image_path (str): 图片文件的路径。

    返回:
    None: 直接打印分析结果。
    """
    try:
        # 1. 加载图片
        img = cv2.imread(image_path)
        if img is None:
            print(f"错误：无法加载图片 {image_path}")
            return

        # 创建一个副本用于绘制，避免修改原图数据
        output_img = img.copy()

        # 2. 颜色空间转换 BGR -> HSV
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # 3. 定义红色的HSV阈值
        lower_red1 = np.array([0, 70, 70])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 70, 70])
        upper_red2 = np.array([180, 255, 255])

        # 4. 创建掩码
        mask1 = cv2.inRange(hsv_img, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv_img, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)

        kernel = np.ones((5,5),np.uint8)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

        # 5. 寻找轮廓
        contours, hierarchy = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            print("图中未检测到红色区域。")
            return

        print(f"检测到 {len(contours)} 个红色区域。\n")

        for i, contour in enumerate(contours):
            print(f"--- 红色区域 {i+1} ---")

            # 基本属性
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            print(f"面积: {area:.2f} 像素")
            print(f"周长: {perimeter:.2f} 像素")

            if area == 0 or perimeter == 0:
                print("面积或周长为0，无法计算形状描述符。")
                cv2.drawContours(output_img, [contour], -1, (0, 255, 0), 2)
                x, y, w, h = cv2.boundingRect(contour)
                cv2.putText(output_img, f"Region {i+1}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
                print("-" * 20)
                continue


            # 形状推断 (基于多边形逼近)
            epsilon = 0.02 * perimeter
            approx_poly = cv2.approxPolyDP(contour, epsilon, True)
            num_vertices = len(approx_poly)
            print(f"近似多边形的顶点数: {num_vertices}")

            # 1. 圆形度 (Circularity)
            # 圆形度 = (4 * pi * 面积) / (周长^2)
            # 值越接近1越接近圆形
            circularity = (4 * math.pi * area) / (perimeter**2) if perimeter > 0 else 0
            print(f"圆形度: {circularity:.4f} (越接近1越圆)")

            # 2. 凸性 (Convexity)
            # 凸性 = 面积 / 凸包面积
            # 值越接近1越凸
            convex_hull = cv2.convexHull(contour)
            convex_hull_area = cv2.contourArea(convex_hull)
            convexity = area / convex_hull_area if convex_hull_area > 0 else 0
            print(f"凸性系数: {convexity:.4f} (越接近1越凸)")

            # 3. 椭圆拟合 (Ellipse Fitting)
            # 轮廓点数必须大于等于5才能进行椭圆拟合
            if len(contour) >= 5:
                # ((center_x, center_y), (minor_axis, major_axis), angle)
                ellipse = cv2.fitEllipse(contour)
                (el_center_x, el_center_y), (el_minor_axis, el_major_axis), el_angle = ellipse

                print(f"椭圆拟合:")
                print(f"  中心点: ({el_center_x:.2f}, {el_center_y:.2f})")
                print(f"  轴长: 短轴={el_minor_axis:.2f}, 长轴={el_major_axis:.2f}")
                print(f"  旋转角度: {el_angle:.2f} 度")

                # 各向异性 (Anisotropy) - 可以定义为长轴与短轴的比率
                anisotropy = el_major_axis / el_minor_axis if el_minor_axis > 0 else float('inf')
                print(f"  各向异性 (长轴/短轴): {anisotropy:.4f}")

                # 在图像上绘制拟合的椭圆
                cv2.ellipse(output_img, ellipse, (255, 0, 0), 2) # 蓝色椭圆
            else:
                print("椭圆拟合: 轮廓点数不足 (<5)")


            # 在图像上绘制轮廓和基本信息
            x_rect, y_rect, w_rect, h_rect = cv2.boundingRect(contour) # 用于文本定位
            cv2.drawContours(output_img, [contour], -1, (0, 255, 0), 2) # 绿色轮廓
            cv2.putText(output_img, f"A:{area:.0f} P:{perimeter:.0f}", (x_rect, y_rect - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)
            cv2.putText(output_img, f"Circ:{circularity:.2f} Conv:{convexity:.2f}", (x_rect, y_rect - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)
            if len(contour) >= 5:
                cv2.putText(output_img, f"Anis:{anisotropy:.2f}", (x_rect, y_rect + h_rect + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)


            print("-" * 20)


        # 显示结果图像 (可选)
        # cv2.imshow("Original Image with Analysis", output_img)
        # cv2.imshow("Red Mask", red_mask)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # 如果你需要保存带有标记的图片
        # output_image_path = "analyzed_" + image_path.split('/')[-1] # e.g., analyzed_2025-04-01horizontal2.png
        # cv2.imwrite(output_image_path, output_img)
        # print(f"分析结果图像已保存为: {output_image_path}")


    except Exception as e:
        import traceback
        print(f"处理图片时发生错误: {e}")
        print(traceback.format_exc())


# --- 使用示例 ---
image_file_path = '2025-04-01horizontal2.png' # 请确保图片在此路径
analyze_red_area_detailed(image_file_path)