# SAM Colony Detection Visualization Tool

这个工具用于可视化 Segment Anything Model (SAM) 对菌群识别的效果，适合用于论文展示。

## 工作流程

1. **amg.py** - 使用SAM自动生成所有可能的mask
   - 输入：原始培养皿图片 (BMP格式)
   - 输出：多个mask PNG文件 + metadata.csv

2. **analysis.py** - 人工选择正确的plate和colony mask
   - 输入：amg.py生成的mask文件夹
   - 输出：plates_data.csv 和 colonies_data.csv

3. **visualize_sam_results.py** - 生成论文用的可视化图片
   - 输入：CSV数据文件
   - 输出：高质量可视化图片（300 DPI）

## 目录结构

```
images/
  └── 2025-06-05/
      ├── up/
      │   ├── 1.bmp          # 原始图片
      │   ├── 2.bmp
      │   └── ...
      ├── down/
      └── vertical/

images/analysis/
  └── 2025-06-05/
      ├── up/
      │   ├── 1/              # 实验1的所有masks
      │   │   ├── 0.png
      │   │   ├── 1.png
      │   │   ├── 4.png
      │   │   └── metadata.csv
      │   └── 2/
      ├── down/
      └── vertical/
  ├── colonies_data.csv       # 选中的菌落数据
  └── plates_data.csv         # 选中的培养皿数据
```

## 使用方法

### 方法1：生成单个样本（交互式）

```bash
python visualize_sam_results.py
```

脚本会自动选择一个高质量的样本并生成可视化，同时显示图片窗口。

### 方法2：批量生成多个样本

```bash
# 生成前5个高质量样本
python visualize_sam_results.py batch 5

# 生成前10个高质量样本
python visualize_sam_results.py batch 10
```

### 样本筛选标准

脚本会自动筛选高质量的样本：
- 面积：10万到100万像素之间（大小适中）
- 椭圆度 < 2.0（形状比较规则）
- Predicted IOU > 0.98（SAM模型预测质量高）

## 输出说明

生成的图片包含三部分：
1. **Original Image**: 原始培养皿图片
2. **Segmentation Overlay**: mask叠加在原图上（黄色区域 + 红色边界框）
3. **Segmentation Mask**: 单独的mask图（黑白二值图）

图片还包含元数据标注：
- Date（日期）、Direction（方向）、Experiment ID（实验编号）
- Area（面积）、Ellipticity（椭圆度）、IOU（预测质量分数）

## 输出文件

所有生成的图片保存在 `visualization_output/` 目录下：

```
visualization_output/
  ├── sam_viz_1_2025-06-17_down_exp9.png
  ├── sam_viz_2_2025-06-17_up_exp1.png
  └── ...
```

文件命名格式：`sam_viz_{序号}_{日期}_{方向}_exp{实验ID}.png`

## 图片参数

- 分辨率：300 DPI（适合论文发表）
- 尺寸：18x6 英寸
- 格式：PNG（无损压缩）
- 文件大小：约6-7 MB/张

## 自定义筛选条件

如果需要修改样本筛选条件，可以编辑 `visualize_sam_results.py` 中的这部分代码：

```python
filtered_df = df[
    (df['area'] > 100000) &      # 最小面积
    (df['area'] < 1000000) &     # 最大面积
    (df['ellipticity'] < 2.0) &  # 最大椭圆度
    (df['predicted_iou'] > 0.98) # 最小IOU分数
].copy()
```

## 依赖库

- pandas
- opencv-python (cv2)
- numpy
- matplotlib

## 论文使用建议

生成的图片可以直接用于论文的以下部分：
- Methods部分：展示SAM模型的分割效果
- Results部分：展示菌落检测的准确性
- Supplementary Materials：展示多个样本的分割效果

建议选择2-3张效果最好的图片用于主文，其余作为补充材料。
