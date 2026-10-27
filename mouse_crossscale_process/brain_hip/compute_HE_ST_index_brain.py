import cv2
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


class HippocampusMask:
    def __init__(self, mask_path):
        """
        初始化：加载mask并提取轮廓
        注意：原始mask图像中，mask内部用黑色(0)涂抹，mask外部用白色(255)背景
        统一转换为：mask内部=255，外部=0
        """
        # 读取原始mask图像（灰度）
        img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # 反转黑白：将内部黑色(0)变为白色(255)，外部白色(255)变为黑色(0)
        mask_internal_white = cv2.bitwise_not(img)

        # 提取轮廓（基于内部=255的区域）
        _, binary = cv2.threshold(mask_internal_white, 127, 255, cv2.THRESH_BINARY)
        self.contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 存储统一版本的mask（内部=255，外部=0）
        self.mask_shape = mask_internal_white.shape
        self.mask_img = mask_internal_white

        # 计算海马区面积（白色像素点数，即内部区域）
        self.area_pixels = np.sum(mask_internal_white == 255)

        print(f"加载完成，共 {len(self.contours)} 个海马区轮廓，面积：{self.area_pixels} 像素")

    def contains(self, x, y):
        """判断点是否在mask内（基于轮廓）"""
        pt = (float(x), float(y))
        return any(cv2.pointPolygonTest(cnt, pt, False) >= 0 for cnt in self.contours)

    def contains_pixel(self, x, y):
        """像素级判断（更精确）：判断点是否在mask内部（白色区域）"""
        h, w = self.mask_shape
        return 0 <= int(x) < w and 0 <= int(y) < h and self.mask_img[int(y), int(x)] == 255


def filter_celltypes(celltype_csv_path, mouse_id):
    """
    功能1: 筛选目标细胞类型的barcode
    """
    df = pd.read_csv(celltype_csv_path)
    target_types = ['CPECs', 'Endothelial', 'Astrocyte_Resting', 'Microglia_Homeostatic']
    filtered = df[(df['orig.ident'] == mouse_id) & (df['CellAnnotation'].isin(target_types))]
    result = {ct: filtered[filtered['CellAnnotation'] == ct]['barcode'].tolist() for ct in target_types}
    for ct, bcs in result.items():
        print(f"  {ct}: {len(bcs)}个细胞")
    return result


def transform_coords(cell_locations_path, barcode_list, he_img_path):
    """
    功能2: 转换坐标到图像坐标系（翻转Y轴）
    返回: DataFrame包含 Cell_Barcode, X, Y
    """
    coord_df = pd.read_csv(cell_locations_path, sep='\t')
    coord_df = coord_df[coord_df['Cell_Barcode'].isin(barcode_list)]

    if len(coord_df) == 0:
        return pd.DataFrame(columns=['Cell_Barcode', 'X', 'Y'])

    img = np.array(Image.open(he_img_path))
    h, w = img.shape[:2]
    # 缩放比例（固定值）
    scale_ratio_x, scale_ratio_y = 56500 / w, 20434 / h
    # 坐标转换：缩放 + Y轴翻转
    coord_df['X'] = coord_df['X'] / scale_ratio_x
    coord_df['Y'] = h - (coord_df['Y'] / scale_ratio_y)
    return coord_df[['Cell_Barcode', 'X', 'Y']]


def analyze_cells_in_mask(mask_obj, celltype_barcodes, cell_locations_path, he_img_path, gene_exp_path):
    """
    功能3: 统计mask内所有细胞数量、各类型细胞密度和占比，并计算基因表达均值
    返回: 结果字典包含：
        - area: 海马区面积
        - total_cells_in_mask: mask内所有细胞数
        - celltype_results: 各类型细胞的详细结果（密度、占比、基因表达均值）
        - cell_coords: 各细胞的坐标和类型（用于可视化）
    """
    # 标记基因定义
    marker_genes = {
        'CPECs': ['Ttr', 'Atp1a1', 'Folr1', 'Slco1c1', 'Clic6'],
        'Endothelial': ['Cldn5', 'H2-D1', 'Thra', 'Cxcl12', 'Cdh5', 'Ctnnb1', 'Flt1', 'H2-K1', 'B2m', 'Pecam1'],
        'Astrocyte_Resting': ['Sox9', 'Cdh4', 'Kcnj10', 'Grm3', 'Slc1a2', 'Atp1a2', 'Fasn', 'Sorcs2', 'Gfap', 'Slco1c1'],
        'Microglia_Homeostatic': ['Smad3', 'Tgfbr2', 'Lrp1', 'Apoe', 'Spred2', 'Elmo1', 'Dock4', 'Mertk', 'P2ry12', 'Cx3cr1']
    }

    gene_df = pd.read_csv(gene_exp_path)

    # 获取所有细胞的barcode并统计mask内分布
    all_barcodes = gene_df['barcode'].tolist()
    all_coords = transform_coords(cell_locations_path, all_barcodes, he_img_path)

    # 判断所有细胞是否在mask内
    all_coords['in_mask'] = all_coords.apply(lambda r: mask_obj.contains_pixel(r['X'], r['Y']), axis=1)
    in_mask_all_barcodes = all_coords[all_coords['in_mask']]['Cell_Barcode'].tolist()
    total_cells_in_mask = len(in_mask_all_barcodes)

    # 获取海马区面积（像素）
    area = mask_obj.area_pixels

    print(f"\n海马区总面积: {area} 像素")
    print(f"海马区内总细胞数: {total_cells_in_mask}")

    celltype_results = {}

    # 记录所有mask内细胞的坐标和类型（用于可视化）
    cell_coords = []

    # 构建barcode到细胞类型的映射
    barcode_to_type = {}
    for ct, bcs in celltype_barcodes.items():
        for bc in bcs:
            barcode_to_type[bc] = ct

    for celltype, barcodes in celltype_barcodes.items():
        if not barcodes:
            celltype_results[celltype] = {
                'count': 0,
                'density': 0,
                'percentage': 0,
                'gene_means': {gene: np.nan for gene in marker_genes[celltype]}
            }
            continue

        # 获取转换后的坐标
        coord_df = transform_coords(cell_locations_path, barcodes, he_img_path)

        # 判断是否在mask内
        coord_df['in_mask'] = coord_df.apply(lambda r: mask_obj.contains_pixel(r['X'], r['Y']), axis=1)
        in_mask_barcodes = coord_df[coord_df['in_mask']]['Cell_Barcode'].tolist()
        cell_count_in_mask = len(in_mask_barcodes)

        # 记录坐标信息
        for _, row in coord_df[coord_df['in_mask']].iterrows():
            cell_coords.append((row['X'], row['Y'], celltype))

        # 计算密度（细胞数/面积）和占比（该类型细胞数/总细胞数）
        density = cell_count_in_mask / area if area > 0 else 0
        percentage = (cell_count_in_mask / total_cells_in_mask * 100) if total_cells_in_mask > 0 else 0

        # 计算基因表达均值（仅mask内细胞）
        exp_subset = gene_df[gene_df['barcode'].isin(in_mask_barcodes)]
        gene_means = {}
        for gene in marker_genes[celltype]:
            if gene in exp_subset.columns:
                gene_means[gene] = exp_subset[gene].mean() if len(exp_subset) > 0 else np.nan
            else:
                gene_means[gene] = np.nan

        celltype_results[celltype] = {
            'count': cell_count_in_mask,
            'density': density,
            'percentage': percentage,
            'gene_means': gene_means
        }

        print(f"  {celltype}: {cell_count_in_mask}个细胞, "
              f"密度={density:.6f}, 占比={percentage:.2f}%")

    # 添加不属于四种类型的其他细胞
    four_types_barcodes = set()
    for bcs in celltype_barcodes.values():
        four_types_barcodes.update(bcs)
    other_barcodes = [bc for bc in in_mask_all_barcodes if bc not in four_types_barcodes]

    # 获取其他细胞的坐标
    other_coords_df = transform_coords(cell_locations_path, other_barcodes, he_img_path)
    for _, row in other_coords_df.iterrows():
        if mask_obj.contains_pixel(row['X'], row['Y']):
            cell_coords.append((row['X'], row['Y'], 'Other'))

    # 汇总结果
    results = {
        'area': area,
        'total_cells_in_mask': total_cells_in_mask,
        'celltype_results': celltype_results,
        'cell_coords': cell_coords
    }

    return results


def visualize_cells_on_he(he_img_path, mask_obj, cell_coords, output_path=None):
    """
    功能5: 在HE图像上显示mask轮廓和细胞位置
    参数:
        he_img_path: HE图像路径
        mask_obj: HippocampusMask对象
        cell_coords: 细胞坐标列表，每个元素为 (x, y, celltype)
        output_path: 输出图像保存路径（可选）
    """
    # 读取HE图像（彩色）
    he_img = cv2.imread(he_img_path)
    he_img = cv2.cvtColor(he_img, cv2.COLOR_BGR2RGB)  # 转换为RGB用于matplotlib

    # 创建图像副本用于绘制
    vis_img = he_img.copy()

    # 定义细胞类型对应的颜色
    color_map = {
        'CPECs': (255, 0, 0),  # 红色
        'Endothelial': (0, 255, 0),  # 绿色
        'Astrocyte_Resting': (0, 0, 255),  # 蓝色
        'Microglia_Homeostatic': (255, 255, 0),  # 青色
        'Other': (128, 128, 128)  # 灰色
    }

    # 绘制mask轮廓（注意：mask_obj.contours已经是内部区域轮廓）
    # 使用绿色轮廓以便清晰显示
    cv2.drawContours(vis_img, mask_obj.contours, -1, (0, 255, 188), 10)

    # 绘制细胞位置
    for x, y, celltype in cell_coords:
        color = color_map.get(celltype, (200, 200, 200))
        # 绘制点（圆形）
        cv2.circle(vis_img, (int(x), int(y)), 6, color, -1)

    # 创建图例并显示
    plt.figure(figsize=(14, 10))
    plt.imshow(vis_img)
    plt.axis('off')
    plt.title('Hippocampus Mask and Cell Distribution', fontsize=14)

    # 添加图例
    legend_elements = [Patch(facecolor=np.array(color) / 255, label=ct, edgecolor='black')
                       for ct, color in color_map.items()]
    # 添加mask轮廓的图例项
    legend_elements.insert(0, Patch(facecolor='none', edgecolor='green', label='Hippocampus Mask', linewidth=2))
    plt.legend(handles=legend_elements, loc='upper right', fontsize=10)

    # 保存或显示
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"图像已保存至: {output_path}")
    plt.show()

    # 统计可视化中的细胞数量
    type_counts = {}
    for _, _, ct in cell_coords:
        type_counts[ct] = type_counts.get(ct, 0) + 1
    print("\n可视化细胞统计:")
    for ct, count in type_counts.items():
        print(f"  {ct}: {count}个")


def print_results(results):
    """功能4: 打印分析结果（包含密度和占比）"""
    print("\n" + "=" * 60)
    print("分析结果汇总:")
    print("=" * 60)

    area = results['area']
    total_cells = results['total_cells_in_mask']
    print(f"\n海马区总面积: {area} 像素")
    print(f"海马区内总细胞数: {total_cells}")

    print("\n各细胞类型在海马区内的分布:")
    print("-" * 60)

    for ct, data in results['celltype_results'].items():
        print(f"\n【{ct}】")
        print(f"  细胞数量: {data['count']}")
        print(f"  细胞密度: {data['density'] * 100000:.2f} 个/像素")
        print(f"  细胞占比: {data['percentage']:.2f}% (占海马区内所有细胞)")
        print(f"  基因表达均值:")
        for gene, val in data['gene_means'].items():
            if not np.isnan(val):
                print(f"    {gene}: {val:.2f}")
            else:
                print(f"    {gene}: N/A")

    # 计算四种类型细胞的总数和占比
    total_four_types = sum(data['count'] for data in results['celltype_results'].values())
    percentage_four_types = (total_four_types / total_cells * 100) if total_cells > 0 else 0
    print(f"\n四种目标细胞类型总计:")
    print(f"  总数量: {total_four_types}/{total_cells}")
    print(f"  总占比: {percentage_four_types:.2f}%")


# ========== 主程序 ==========
if __name__ == "__main__":
    # 路径配置
    celltype_csv = "H:/Cross-scale_mouse_data/ST_data/all_samples_celltype.csv"

    # 设置路径参数
    mask_path = "H:/Cross-scale_mouse_data/brain_HE_image/brain_HE_mask/Bla_brain_1_aligned_HE_mask.png"
    cell_locations = "H:/Cross-scale_mouse_data/ST_data/blank_1_brain/filtered_feature_bc_matrix/cell_locations.tsv"
    he_img_path = "H:/Cross-scale_mouse_data/ST_data/blank_1_brain/Bla_brain_1_aligned_HE.png"
    gene_exp_path = "H:/Cross-scale_mouse_data/ST_data/blank_1_brain/blank_1_brain_target_gene_expression.csv"
    output_vis_path = "H:/Cross-scale_mouse_data/ST_data/blank_1_brain/cell_visualization.png"  # 输出图像路径

    # 子函数串联执行
    print("1. 加载海马区mask...")
    mask = HippocampusMask(mask_path)

    print("\n2. 筛选目标细胞类型...")
    celltype_barcodes = filter_celltypes(celltype_csv, mouse_id='bla_brain')  # 设置小鼠id

    print("\n3. 分析细胞分布和基因表达...")
    results = analyze_cells_in_mask(mask, celltype_barcodes, cell_locations, he_img_path, gene_exp_path)

    print("\n4. 输出结果...")
    print_results(results)

    print("\n5. 生成可视化图像...")
    visualize_cells_on_he(he_img_path, mask, results['cell_coords'], output_vis_path)
