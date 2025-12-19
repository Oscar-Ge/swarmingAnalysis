import os
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
import re
import sys

# --- 配置 ---
# 输出的CSV文件名
PLATE_CSV_FILE = 'plates_data.csv'
COLONY_CSV_FILE = 'colonies_data.csv'
# 根目录，'.' 代表当前目录
ROOT_DIRECTORY = '.'

def calculate_ellipticity(mask_path):
    """
    加载一个mask图片，找到最大的轮廓，拟合一个椭圆，并计算其椭圆度。
    椭圆度定义为：长轴 / 短轴。一个完美的圆椭圆度为1。
    
    参数:
        mask_path (str): mask图片的文件路径。
        
    返回:
        float: 计算出的椭圆度，如果无法计算则返回None。
    """
    try:
        # 以灰度模式读取图像
        image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"    - Warning: Could not read image {mask_path}")
            return None

        # 查找轮廓
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            print(f"    - Warning: No contours found in {mask_path}")
            return None

        # 找到面积最大的轮廓
        main_contour = max(contours, key=cv2.contourArea)

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
            return np.inf

        ellipticity = major_axis / minor_axis
        return ellipticity

    except Exception as e:
        print(f"    - Error calculating ellipticity for {mask_path}: {e}")
        return None

def display_and_get_selection(image_folder, metadata_df):
    """
    在一个窗口中显示所有mask图片，并让用户在命令行中选择。
    
    参数:
        image_folder (str): 包含图片和metadata.csv的文件夹路径。
        metadata_df (pd.DataFrame): 加载后的metadata数据。
        
    返回:
        tuple: (plate_id, colony_id)，如果用户跳过，则值为None。
    """
    image_files = sorted(
        [f for f in os.listdir(image_folder) if f.endswith('.png')],
        key=lambda x: int(os.path.splitext(x)[0])
    )
    
    if not image_files:
        print("    - No PNG files found to display.")
        return None, None

    num_images = len(image_files)
    cols = int(np.ceil(np.sqrt(num_images)))
    rows = int(np.ceil(num_images / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, 10))
    fig.suptitle(f"Select Plate and Colony for: {image_folder}", fontsize=16)
    
    # 将axes展平以便于索引
    axes = axes.flatten()

    for i, img_file in enumerate(image_files):
        img_id = os.path.splitext(img_file)[0]
        try:
            img = plt.imread(os.path.join(image_folder, img_file))
            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(f"ID: {img_id}")
        except Exception as e:
            axes[i].set_title(f"Error ID: {img_id}")
            print(f"    - Warning: Could not load image {img_file}: {e}")
        finally:
            axes[i].axis('off')

    # 隐藏多余的子图
    for j in range(num_images, len(axes)):
        axes[j].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show(block=False)

    # 在命令行获取用户输入
    print("\n    --- User Input Required ---")
    print(f"    Displaying {num_images} masks from folder: {image_folder}")
    print("    Review the window with images.")
    
    valid_ids = [str(id_val) for id_val in metadata_df['id'].tolist()]
    
    plate_id_str = input("    > Enter the ID for the AGAR PLATE (or press Enter to skip): ")
    colony_id_str = input("    > Enter the ID for the COLONY (or press Enter to skip): ")
    
    plt.close(fig) # 关闭图片窗口

    # 验证输入
    plate_id = int(plate_id_str) if plate_id_str.isdigit() and plate_id_str in valid_ids else None
    colony_id = int(colony_id_str) if colony_id_str.isdigit() and colony_id_str in valid_ids else None
    
    if plate_id is None and plate_id_str not in ['', 's']:
        print(f"    - Warning: Invalid Plate ID '{plate_id_str}'. Skipping plate.")
    if colony_id is None and colony_id_str not in ['', 's']:
        print(f"    - Warning: Invalid Colony ID '{colony_id_str}'. Skipping colony.")
        
    return plate_id, colony_id


def process_directory(root_dir):
    """
    遍历目录结构，为每个实验文件夹调用处理函数。
    """
    # 检查根目录是否存在
    if not os.path.isdir(root_dir):
        print(f"Error: Root directory '{root_dir}' not found.")
        sys.exit(1)
        
    # 在开始前加载已有的数据，以避免重复处理
    processed_paths = set()
    if os.path.exists(PLATE_CSV_FILE):
        try:
            df_plates_old = pd.read_csv(PLATE_CSV_FILE)
            if 'mask_path' in df_plates_old.columns:
                processed_paths.update(df_plates_old['mask_path'])
        except pd.errors.EmptyDataError:
            pass # 文件为空
            
    if os.path.exists(COLONY_CSV_FILE):
        try:
            df_colonies_old = pd.read_csv(COLONY_CSV_FILE)
            if 'mask_path' in df_colonies_old.columns:
                processed_paths.update(df_colonies_old['mask_path'])
        except pd.errors.EmptyDataError:
            pass # 文件为空

    if processed_paths:
        print(f"Found {len(processed_paths)} already processed records. They will be skipped.")

    for dirpath, dirnames, filenames in os.walk(root_dir):
        # 寻找包含 metadata.csv 的目标文件夹
        if 'metadata.csv' in filenames:
            
            # --- 1. 从路径中解析信息 (已更新此处的逻辑) ---
            # 正则表达式来匹配 '.../YYYY-MM-DD/direction/experiment_id'
            match = re.search(r'(\d{4}-\d{2}-\d{2})[/\\]+(\w+)[/\\]+(\d+)$', dirpath)
            if not match:
                print(f"\n- Skipping folder (could not parse info): {dirpath}")
                continue
            
            date, direction, experiment_id = match.groups()
            print(f"\n--- Processing Folder ---")
            print(f"  Date: {date}, Direction: {direction}, Experiment: {experiment_id}")
            print(f"  Path: {dirpath}")

            # --- 2. 加载元数据 ---
            metadata_path = os.path.join(dirpath, 'metadata.csv')
            try:
                metadata_df = pd.read_csv(metadata_path)
            except Exception as e:
                print(f"  - Error reading metadata.csv: {e}. Skipping folder.")
                continue

            # --- 3. 显示图片并获取用户选择 ---
            plate_id, colony_id = display_and_get_selection(dirpath, metadata_df)
            
            if plate_id is None and colony_id is None:
                print("  - User skipped this folder. Moving to the next one.")
                continue

            # --- 4. 处理并保存数据 ---
            plate_area = None
            
            # 处理培养皿 (Plate)
            if plate_id is not None:
                plate_mask_path = os.path.join(dirpath, f"{plate_id}.png")
                if plate_mask_path in processed_paths:
                    print(f"  - Plate ID {plate_id} has already been processed. Skipping.")
                else:
                    plate_info = metadata_df[metadata_df['id'] == plate_id]
                    if not plate_info.empty:
                        plate_record = plate_info.iloc[0].to_dict()
                        plate_area = plate_record.get('area')
                        
                        # 添加额外信息
                        plate_record['date'] = date
                        plate_record['direction'] = direction
                        plate_record['experiment_id'] = experiment_id
                        plate_record['mask_path'] = plate_mask_path

                        # 保存到CSV
                        df_plate = pd.DataFrame([plate_record])
                        # 如果文件不存在，则写入header，否则不写
                        header = not os.path.exists(PLATE_CSV_FILE)
                        df_plate.to_csv(PLATE_CSV_FILE, mode='a', header=header, index=False)
                        print(f"  - Saved plate data for ID {plate_id}.")
                        processed_paths.add(plate_mask_path)
                    else:
                        print(f"  - Warning: Plate ID {plate_id} not found in metadata.csv.")

            # 处理菌落 (Colony)
            if colony_id is not None:
                colony_mask_path = os.path.join(dirpath, f"{colony_id}.png")
                if colony_mask_path in processed_paths:
                    print(f"  - Colony ID {colony_id} has already been processed. Skipping.")
                else:
                    colony_info = metadata_df[metadata_df['id'] == colony_id]
                    if not colony_info.empty:
                        colony_record = colony_info.iloc[0].to_dict()
                        colony_area = colony_record.get('area')
                        
                        # 添加额外信息
                        colony_record['date'] = date
                        colony_record['direction'] = direction
                        colony_record['experiment_id'] = experiment_id
                        colony_record['mask_path'] = colony_mask_path

                        # 计算面积比
                        if plate_area is not None and plate_area > 0:
                            colony_record['area_ratio_to_plate'] = colony_area / plate_area
                        else:
                            colony_record['area_ratio_to_plate'] = None
                            if plate_id is not None:
                                print("  - Warning: Cannot calculate area ratio because plate area is zero or unavailable.")
                        
                        # 计算椭圆度
                        colony_record['ellipticity'] = calculate_ellipticity(colony_mask_path)
                        
                        # 保存到CSV
                        df_colony = pd.DataFrame([colony_record])
                        header = not os.path.exists(COLONY_CSV_FILE)
                        df_colony.to_csv(COLONY_CSV_FILE, mode='a', header=header, index=False)
                        print(f"  - Saved colony data for ID {colony_id}.")
                        processed_paths.add(colony_mask_path)
                    else:
                        print(f"  - Warning: Colony ID {colony_id} not found in metadata.csv.")
    
    print("\n--- All folders processed. Script finished. ---")


if __name__ == "__main__":
    process_directory(ROOT_DIRECTORY)