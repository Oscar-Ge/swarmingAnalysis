import os
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
import re
import sys
import shutil
# usage: python update_entry.py 2025-06-05/down/9
# --- 配置 ---
PLATE_CSV_FILE = 'plates_data.csv'
COLONY_CSV_FILE = 'colonies_data.csv'

# --- 从上一个脚本复制过来的辅助函数 (无需修改) ---

def calculate_ellipticity(mask_path):
    """计算椭圆度"""
    try:
        image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if image is None: return None
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return None
        main_contour = max(contours, key=cv2.contourArea)
        if len(main_contour) < 5: return None
        (center, axes, angle) = cv2.fitEllipse(main_contour)
        major_axis, minor_axis = max(axes), min(axes)
        if minor_axis == 0: return np.inf
        return major_axis / minor_axis
    except Exception as e:
        print(f"    - Error calculating ellipticity for {mask_path}: {e}")
        return None

def display_and_get_selection(image_folder, metadata_df):
    """显示图片并获取选择"""
    image_files = sorted(
        [f for f in os.listdir(image_folder) if f.endswith('.png')],
        key=lambda x: int(os.path.splitext(x)[0])
    )
    if not image_files: return None, None
    num_images = len(image_files)
    cols = int(np.ceil(np.sqrt(num_images)))
    rows = int(np.ceil(num_images / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 10))
    fig.suptitle(f"RE-SELECT Plate and Colony for: {image_folder}", fontsize=16)
    axes = axes.flatten()
    for i, img_file in enumerate(image_files):
        img_id = os.path.splitext(img_file)[0]
        try:
            img = plt.imread(os.path.join(image_folder, img_file))
            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(f"ID: {img_id}")
        except Exception:
            axes[i].set_title(f"Error ID: {img_id}")
        finally:
            axes[i].axis('off')
    for j in range(num_images, len(axes)): axes[j].axis('off')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show(block=False)
    print("\n    --- User Input Required for Update ---")
    valid_ids = [str(id_val) for id_val in metadata_df['id'].tolist()]
    plate_id_str = input("    > Enter the NEW ID for the AGAR PLATE (or press Enter to remove entry): ")
    colony_id_str = input("    > Enter the NEW ID for the COLONY (or press Enter to remove entry): ")
    plt.close(fig)
    plate_id = int(plate_id_str) if plate_id_str.isdigit() and plate_id_str in valid_ids else None
    colony_id = int(colony_id_str) if colony_id_str.isdigit() and colony_id_str in valid_ids else None
    return plate_id, colony_id

# --- 主程序逻辑 ---

def main():
    # 1. 检查命令行参数
    if len(sys.argv) != 2:
        print("Usage: python update_entry.py <path_to_folder_to_update>")
        print("Example: python update_entry.py ./2025-06-05/down/9")
        sys.exit(1)

    folder_path = os.path.normpath(sys.argv[1]) # 规范化路径

    # 2. 验证文件夹路径
    metadata_path = os.path.join(folder_path, 'metadata.csv')
    if not os.path.isdir(folder_path) or not os.path.exists(metadata_path):
        print(f"Error: Path '{folder_path}' is not a valid experiment folder with a metadata.csv file.")
        sys.exit(1)

    print(f"--- Starting Update Process for: {folder_path} ---")

    # 3. 加载现有的CSV文件
    try:
        plate_df = pd.read_csv(PLATE_CSV_FILE)
        print(f"Loaded '{PLATE_CSV_FILE}' with {len(plate_df)} rows.")
    except FileNotFoundError:
        print(f"Warning: '{PLATE_CSV_FILE}' not found. Will create a new one.")
        plate_df = pd.DataFrame()

    try:
        colony_df = pd.read_csv(COLONY_CSV_FILE)
        print(f"Loaded '{COLONY_CSV_FILE}' with {len(colony_df)} rows.")
    except FileNotFoundError:
        print(f"Warning: '{COLONY_CSV_FILE}' not found. Will create a new one.")
        colony_df = pd.DataFrame()

    # 4. 删除与该文件夹相关的旧条目
    # 我们使用 `mask_path` 来唯一识别条目。
    # `str.startswith` 能确保即使路径格式略有不同也能匹配 (e.g., './' vs '')
    if not plate_df.empty and 'mask_path' in plate_df.columns:
        old_plate_mask = plate_df['mask_path'].str.startswith(folder_path, na=False)
        if old_plate_mask.any():
            print(f"Found and removed {old_plate_mask.sum()} old plate entries for this folder.")
            plate_df = plate_df[~old_plate_mask]

    if not colony_df.empty and 'mask_path' in colony_df.columns:
        old_colony_mask = colony_df['mask_path'].str.startswith(folder_path, na=False)
        if old_colony_mask.any():
            print(f"Found and removed {old_colony_mask.sum()} old colony entries for this folder.")
            colony_df = colony_df[~old_colony_mask]

    # 5. 重新进行选择和数据处理
    metadata_df = pd.read_csv(metadata_path)
    new_plate_id, new_colony_id = display_and_get_selection(folder_path, metadata_df)

    if new_plate_id is None and new_colony_id is None:
        print("User did not select any new items. The old entries have been removed (if they existed).")
    else:
        # 从路径中解析信息
        match = re.search(r'(\d{4}-\d{2}-\d{2})[/\\]+(\w+)[/\\]+(\d+)$', folder_path)
        if not match:
            print(f"Error: Could not parse info from path '{folder_path}'. Cannot proceed.")
            sys.exit(1)
        date, direction, experiment_id = match.groups()

        new_plate_record = None
        new_colony_record = None
        plate_area = None

        # 准备新的培养皿记录
        if new_plate_id is not None:
            plate_info = metadata_df[metadata_df['id'] == new_plate_id].iloc[0]
            new_plate_record = plate_info.to_dict()
            plate_area = new_plate_record.get('area')
            new_plate_record.update({
                'date': date, 'direction': direction, 'experiment_id': experiment_id,
                'mask_path': os.path.join(folder_path, f"{new_plate_id}.png")
            })
            print(f"Prepared new plate record for ID: {new_plate_id}")
        
        # 准备新的菌落记录
        if new_colony_id is not None:
            colony_info = metadata_df[metadata_df['id'] == new_colony_id].iloc[0]
            new_colony_record = colony_info.to_dict()
            colony_area = new_colony_record.get('area')
            mask_path = os.path.join(folder_path, f"{new_colony_id}.png")
            
            new_colony_record.update({
                'date': date, 'direction': direction, 'experiment_id': experiment_id,
                'mask_path': mask_path
            })

            if plate_area is not None and plate_area > 0:
                new_colony_record['area_ratio_to_plate'] = colony_area / plate_area
            
            new_colony_record['ellipticity'] = calculate_ellipticity(mask_path)
            print(f"Prepared new colony record for ID: {new_colony_id}")

        # 6. 将新记录添加到DataFrame中
        if new_plate_record:
            plate_df = pd.concat([plate_df, pd.DataFrame([new_plate_record])], ignore_index=True)
        if new_colony_record:
            colony_df = pd.concat([colony_df, pd.DataFrame([new_colony_record])], ignore_index=True)

    # 7. 备份并保存更新后的CSV文件
    print("\n--- Saving Results ---")
    try:
        # 备份旧文件
        if os.path.exists(PLATE_CSV_FILE):
            shutil.copy(PLATE_CSV_FILE, PLATE_CSV_FILE + '.bak')
            print(f"Backup created: {PLATE_CSV_FILE}.bak")
        if os.path.exists(COLONY_CSV_FILE):
            shutil.copy(COLONY_CSV_FILE, COLONY_CSV_FILE + '.bak')
            print(f"Backup created: {COLONY_CSV_FILE}.bak")

        # 保存新文件
        plate_df.to_csv(PLATE_CSV_FILE, index=False)
        print(f"Successfully updated and saved '{PLATE_CSV_FILE}'.")
        colony_df.to_csv(COLONY_CSV_FILE, index=False)
        print(f"Successfully updated and saved '{COLONY_CSV_FILE}'.")

    except Exception as e:
        print(f"An error occurred during file saving: {e}")

    print("\n--- Update Process Finished ---")

if __name__ == "__main__":
    main()