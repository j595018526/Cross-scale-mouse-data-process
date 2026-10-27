import pandas as pd
from scipy.stats import mannwhitneyu
import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
import os


# ============================================================
# 模块1：统计分析（打印结果）
# ============================================================
def perform_statistical_analysis(file_path):
    """
    对average_cell_expression工作表进行Mann-Whitney U检验，
    仅打印统计分析结果，不生成表格

    参数:
        file_path: Excel文件路径
    """
    print("=" * 60)
    print("模块1：统计分析（仅打印结果，供手动记录）")
    print("=" * 60)

    # 读取数据
    df_avg = pd.read_excel(file_path, sheet_name='average_cell_expression')

    # 提取基因列（第二列到最后一列）
    gene_columns = df_avg.columns[1:].tolist()
    print(f"\n检测到 {len(gene_columns)} 个基因")

    # 分组信息
    types = ['niche', 'tumor', 'normal']

    # 定义显著性标记函数
    def get_significance_marker(p_value):
        if p_value < 0.001:
            return '***'
        elif p_value < 0.01:
            return '**'
        elif p_value < 0.05:
            return '*'
        else:
            return 'ns'

    # 存储统计结果摘要（仅用于打印汇总）
    summary_results = []

    # 对每个基因进行统计分析
    for gene in gene_columns:
        print(f"\n{'=' * 50}")
        print(f"基因: {gene}")
        if gene.startswith('mm10___'):
            gene_short_name = gene.replace('mm10___', '')
        else:
            gene_short_name = gene
        print(f"基因短名: {gene_short_name}")
        print("-" * 40)

        # 提取三组数据
        data_groups = {}
        for t in types:
            data_groups[t] = df_avg[df_avg['type'] == t][gene].dropna().values

        # 打印各组均值和标准差
        for t in types:
            mean_val = data_groups[t].mean()
            std_val = data_groups[t].std()
            print(f"{t}组: 均值={mean_val:.2f}, 标准差={std_val:.2f}")

        # 检查样本数是否足够
        if any(len(data_groups[t]) < 2 for t in types):
            print("警告: 某组样本数不足2，跳过统计检验")
            summary_results.append({
                'gene': gene_short_name,
                'niche_vs_tumor': '样本不足',
                'tumor_vs_normal': '样本不足',
                'niche_vs_normal': '样本不足'
            })
            continue

        print("\n统计检验结果 (Mann-Whitney U检验):")

        # 两两比较
        comparisons = [
            ('niche vs tumor', 'niche', 'tumor'),
            ('tumor vs normal', 'tumor', 'normal'),
            ('niche vs normal', 'niche', 'normal')
        ]

        result_dict = {'gene': gene_short_name}

        for comp_name, group1, group2 in comparisons:
            # Mann-Whitney U检验
            stat, p_value = mannwhitneyu(data_groups[group1], data_groups[group2], alternative='two-sided')

            # 计算效应量 Cliff's delta
            n1 = len(data_groups[group1])
            n2 = len(data_groups[group2])
            cliff_delta = (stat - (n1 * n2 / 2)) / (n1 * n2)

            # 获取显著性标记
            marker = get_significance_marker(p_value)

            # 存储结果
            result_dict[comp_name] = marker

            # 打印结果
            print(f"  {comp_name}:")
            print(f"    p-value = {p_value:.2f} ({marker})")
            print(f"    Cliff's delta = {cliff_delta:.2f}")

        summary_results.append(result_dict)
        print("-" * 40)



# ============================================================
# 模块2：绘制柱状图（从statistics工作表读取显著性标签）
# ============================================================
def plot_barplots_with_significance(file_path, save_dir, gene_columns=None):
    """
    读取统计结果和表达数据，绘制带显著性标签的柱状图

    参数:
        file_path: Excel文件路径（包含average_cell_expression和statistics工作表）
        save_dir: 图片保存目录
        gene_columns: 指定要绘图的基因列表（如果为None，则使用所有基因）
    """
    print("\n" + "=" * 60)
    print("模块2：绘制柱状图（从statistics工作表读取显著性标签）")
    print("=" * 60)

    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 读取数据
    df_avg = pd.read_excel(file_path, sheet_name='average_cell_expression')
    df_stat = pd.read_excel(file_path, sheet_name='statistics_gene_avg')

    # 清理列名
    df_stat.columns = df_stat.columns.str.strip()

    # 确定基因列表
    if gene_columns is None:
        gene_columns = df_avg.columns[1:].tolist()

    print(f"\n检测到 {len(gene_columns)} 个基因需要绘图")

    # 分组信息
    types = ['niche', 'tumor', 'normal']
    type_names = ['Niche', 'Tumor', 'Normal']
    colors = ['#c44e52', '#F1C40F', '#55a868']

    # 构建显著性字典
    sig_dict = {}
    for _, row in df_stat.iterrows():
        gene = row['genes']
        sig_dict[gene] = {
            'niche_vs_tumor': str(row.get('niche vs tumor', '')) if pd.notna(row.get('niche vs tumor', '')) else '',
            'tumor_vs_normal': str(row.get('tumor vs normal', '')) if pd.notna(row.get('tumor vs normal', '')) else '',
            'niche_vs_normal': str(row.get('niche vs normal', '')) if pd.notna(row.get('niche vs normal', '')) else ''
        }

    print(f"从statistics工作表读取了 {len(sig_dict)} 个基因的显著性标记")

    def add_significance_bar(ax, x1, x2, y_height, marker):
        """添加T型统计标注"""
        y_min, y_max = ax.get_ylim()
        y_range = y_max - y_min
        short_bar_length = y_range * 0.015

        # 水平线
        ax.plot([x1, x2], [y_height, y_height], color='black', linewidth=1.5)
        # 左右短竖杠
        ax.plot([x1, x1], [y_height - short_bar_length / 2, y_height], color='black', linewidth=1.5)
        ax.plot([x2, x2], [y_height - short_bar_length / 2, y_height], color='black', linewidth=1.5)
        # 星号标记
        x_center = (x1 + x2) / 2
        star_height = y_height + short_bar_length * 0.05
        ax.text(x_center, star_height, marker, ha='center', va='bottom', fontsize=18, fontweight='bold')

    # 绘制每个基因的柱状图
    plotted_count = 0
    for gene in gene_columns:
        print(f"\n绘制基因: {gene}")

        # 提取基因短名
        if gene.startswith('mm10___'):
            gene_short_name = gene.replace('mm10___', '')
        else:
            gene_short_name = gene

        # 提取数据
        data_groups = {}
        for t in types:
            data_groups[t] = df_avg[df_avg['type'] == t][gene].dropna().values

        # 检查数据有效性
        if any(len(data_groups[t]) == 0 for t in types):
            print(f"  警告: 某组无数据，跳过")
            continue

        # 计算均值和标准差
        means = [data_groups[t].mean() for t in types]
        stds = [data_groups[t].std() for t in types]
        x_pos = np.arange(len(types))

        # 创建图形
        plt.figure(figsize=(8, 6))

        # 绘制柱状图
        bars = plt.bar(x_pos, means, color=colors, edgecolor='black', linewidth=1.5, width=0.6, alpha=0.8)
        plt.errorbar(x_pos, means, yerr=stds, fmt='none', ecolor='black', capsize=8, capthick=2, linewidth=1.5)

        # 添加标签
        plt.xticks(x_pos, type_names, fontsize=12)
        plt.ylabel(gene_short_name, fontsize=18, fontweight='bold')
        plt.xlabel('Region', fontsize=12)

        # 获取显著性标记
        sig = sig_dict.get(gene, {})

        # 设置y轴范围
        y_max = max(means[i] + stds[i] for i in range(len(means))) if len(means) > 0 else 1
        plt.ylim(0, y_max * 1.2)

        # 添加显著性标注（只添加非空且非ns的标记）
        added_count = 0
        if sig.get('niche_vs_tumor') and sig['niche_vs_tumor'] not in ['', 'nan', 'NaN', 'ns']:
            add_significance_bar(plt.gca(), 0, 1, y_max * 1.03, sig['niche_vs_tumor'])
            print(f"  添加 niche vs tumor: {sig['niche_vs_tumor']}")
            added_count += 1

        if sig.get('tumor_vs_normal') and sig['tumor_vs_normal'] not in ['', 'nan', 'NaN', 'ns']:
            add_significance_bar(plt.gca(), 1, 2, y_max * 1.03, sig['tumor_vs_normal'])
            print(f"  添加 tumor vs normal: {sig['tumor_vs_normal']}")
            added_count += 1

        if sig.get('niche_vs_normal') and sig['niche_vs_normal'] not in ['', 'nan', 'NaN', 'ns']:
            add_significance_bar(plt.gca(), 0, 2, y_max * 1.11, sig['niche_vs_normal'])
            print(f"  添加 niche vs normal: {sig['niche_vs_normal']}")
            added_count += 1

        if added_count == 0:
            print(f"  无显著差异标记")

        # 美化图形
        plt.grid(axis='y', alpha=0.3, linestyle='--')
        ax = plt.gca()
        # 保存图片
        plt.tight_layout()
        save_path = os.path.join(save_dir, f'{gene_short_name}_barplot.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  已保存: {save_path}")
        plotted_count += 1

    print(f"\n绘图完成！共绘制 {plotted_count}/{len(gene_columns)} 张图片，保存至: {save_dir}")


def perform_cell_count_analysis(file_path):
    """
    读取cell_count工作表，对每种细胞类型进行Mann-Whitney U检验，
    比较niche vs tumor, tumor vs normal, niche vs normal
    """
    print("\n" + "=" * 60)
    print("模块3：细胞数量统计分析（Mann-Whitney U检验）")
    print("=" * 60)

    # 读取数据
    df_cell = pd.read_excel(file_path, sheet_name='cell_count')

    # 第一列为type，第三列到最后一列为细胞类型
    types = df_cell.iloc[:, 0].values  # 第一列
    cell_type_columns = df_cell.columns[2:]  # 第三列到最后一列

    print(f"\n检测到 {len(cell_type_columns)} 种细胞类型")

    # 定义显著性标记函数
    def get_marker(p):
        if p < 0.001:
            return '***'
        elif p < 0.01:
            return '**'
        elif p < 0.05:
            return '*'
        else:
            return 'ns'

    # 分组提取函数
    def get_group_data(group_name):
        return df_cell[df_cell.iloc[:, 0] == group_name][cell_type_columns].values.flatten()

    # 存储结果
    results = []

    for cell_type in cell_type_columns:
        print(f"\n{'=' * 50}")
        print(f"细胞类型: {cell_type}")

        # 提取三组数据
        niche_vals = df_cell[df_cell.iloc[:, 0] == 'niche'][cell_type].dropna().values
        tumor_vals = df_cell[df_cell.iloc[:, 0] == 'tumor'][cell_type].dropna().values
        normal_vals = df_cell[df_cell.iloc[:, 0] == 'normal'][cell_type].dropna().values

        # 打印均值和标准差
        print(f"  niche组: 均值={niche_vals.mean():.2f}, 标准差={niche_vals.std():.2f}")
        print(f"  tumor组: 均值={tumor_vals.mean():.2f}, 标准差={tumor_vals.std():.2f}")
        print(f"  normal组: 均值={normal_vals.mean():.2f}, 标准差={normal_vals.std():.2f}")

        # 两两比较
        comparisons = [
            ('niche vs tumor', niche_vals, tumor_vals),
            ('tumor vs normal', tumor_vals, normal_vals),
            ('niche vs normal', niche_vals, normal_vals)
        ]

        for name, g1, g2 in comparisons:
            if len(g1) < 2 or len(g2) < 2:
                print(f"  {name}: 样本数不足，跳过")
                continue
            stat, p = mannwhitneyu(g1, g2, alternative='two-sided')
            n1, n2 = len(g1), len(g2)
            cliff_delta = (stat - (n1 * n2 / 2)) / (n1 * n2)
            marker = get_marker(p)
            print(f"  {name}: p={p:.4f} ({marker}), Cliff's delta={cliff_delta:.3f}")

        results.append({'cell_type': cell_type, 'niche_vals': niche_vals,
                        'tumor_vals': tumor_vals, 'normal_vals': normal_vals})


def plot_cell_count_barplots(file_path, save_dir):
    """
    读取cell_count工作表和statistics_cell工作表，绘制每种细胞类型的柱状图
    参数:
        file_path: Excel文件路径
        save_dir: 图片保存目录
    """
    print("\n" + "=" * 60)
    print("模块4：绘制细胞数量柱状图")
    print("=" * 60)

    os.makedirs(save_dir, exist_ok=True)

    # 读取数据
    df_cell = pd.read_excel(file_path, sheet_name='cell_count')
    df_stat = pd.read_excel(file_path, sheet_name='statistics_cell')

    # 清理列名
    df_stat.columns = df_stat.columns.str.strip()

    # 分组信息
    types = ['niche', 'tumor', 'normal']
    type_names = ['Niche', 'Tumor', 'Normal']
    colors = ['#c44e52', '#F1C40F', '#55a868']

    # 构建显著性字典
    sig_dict = {}
    for _, row in df_stat.iterrows():
        cell_type = row['cell_type']
        sig_dict[cell_type] = {
            'niche_vs_tumor': str(row.get('niche vs tumor', '')) if pd.notna(row.get('niche vs tumor', '')) else '',
            'tumor_vs_normal': str(row.get('tumor vs normal', '')) if pd.notna(row.get('tumor vs normal', '')) else '',
            'niche_vs_normal': str(row.get('niche vs normal', '')) if pd.notna(row.get('niche vs normal', '')) else ''
        }

    print(f"从statistics_cell工作表读取了 {len(sig_dict)} 种细胞类型的显著性标记")

    # 获取所有细胞类型（第三列到最后一列）
    cell_types = df_cell.columns[2:].tolist()

    def add_significance_bar(ax, x1, x2, y_height, marker):
        """添加T型统计标注"""
        y_min, y_max = ax.get_ylim()
        y_range = y_max - y_min
        short_bar_length = y_range * 0.015

        ax.plot([x1, x2], [y_height, y_height], color='black', linewidth=1.5)
        ax.plot([x1, x1], [y_height - short_bar_length / 2, y_height], color='black', linewidth=1.5)
        ax.plot([x2, x2], [y_height - short_bar_length / 2, y_height], color='black', linewidth=1.5)

        x_center = (x1 + x2) / 2
        star_height = y_height + short_bar_length * 0.05
        ax.text(x_center, star_height, marker, ha='center', va='bottom', fontsize=18, fontweight='bold')

    # 绘制每种细胞类型
    plotted_count = 0
    for cell_type in cell_types:
        print(f"\n绘制细胞类型: {cell_type}")

        # 提取三组数据
        niche_vals = df_cell[df_cell.iloc[:, 0] == 'niche'][cell_type].dropna().values
        tumor_vals = df_cell[df_cell.iloc[:, 0] == 'tumor'][cell_type].dropna().values
        normal_vals = df_cell[df_cell.iloc[:, 0] == 'normal'][cell_type].dropna().values

        # 检查数据有效性
        if any(len(vals) == 0 for vals in [niche_vals, tumor_vals, normal_vals]):
            print(f"  警告: 某组无数据，跳过")
            continue

        # 计算均值和标准差
        means = [niche_vals.mean(), tumor_vals.mean(), normal_vals.mean()]
        stds = [niche_vals.std(), tumor_vals.std(), normal_vals.std()]
        x_pos = np.arange(len(types))

        # 创建图形
        plt.figure(figsize=(8, 6))
        bars = plt.bar(x_pos, means, color=colors, edgecolor='black', linewidth=1.5, width=0.6, alpha=0.8)
        plt.errorbar(x_pos, means, yerr=stds, fmt='none', ecolor='black', capsize=8, capthick=2, linewidth=1.5)

        # 标签
        plt.xticks(x_pos, type_names, fontsize=12)
        plt.ylabel(cell_type, fontsize=18, fontweight='bold')
        plt.xlabel('Region', fontsize=12)

        # 获取显著性标记
        sig = sig_dict.get(cell_type, {})

        # 设置y轴范围
        y_max = max(means[i] + stds[i] for i in range(len(means))) if len(means) > 0 else 1
        plt.ylim(0, y_max * 1.2)

        # 添加显著性标注
        added_count = 0
        if sig.get('niche_vs_tumor') and sig['niche_vs_tumor'] not in ['', 'nan', 'NaN', 'ns']:
            add_significance_bar(plt.gca(), 0, 1, y_max * 1.03, sig['niche_vs_tumor'])
            print(f"  添加 niche vs tumor: {sig['niche_vs_tumor']}")
            added_count += 1

        if sig.get('tumor_vs_normal') and sig['tumor_vs_normal'] not in ['', 'nan', 'NaN', 'ns']:
            add_significance_bar(plt.gca(), 1, 2, y_max * 1.03, sig['tumor_vs_normal'])
            print(f"  添加 tumor vs normal: {sig['tumor_vs_normal']}")
            added_count += 1

        if sig.get('niche_vs_normal') and sig['niche_vs_normal'] not in ['', 'nan', 'NaN', 'ns']:
            add_significance_bar(plt.gca(), 0, 2, y_max * 1.11, sig['niche_vs_normal'])
            print(f"  添加 niche vs normal: {sig['niche_vs_normal']}")
            added_count += 1

        if added_count == 0:
            print(f"  无显著差异标记")

        # 美化
        plt.grid(axis='y', alpha=0.3, linestyle='--')
        ax = plt.gca()
        # 保存
        plt.tight_layout()
        save_path = os.path.join(save_dir, f'{cell_type}_cell_count_barplot.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  已保存: {save_path}")
        plotted_count += 1

    print(f"\n绘图完成！共绘制 {plotted_count}/{len(cell_types)} 张图片，保存至: {save_dir}")


def compare_groups(file_path, sheet_name):
    df = pd.read_excel(file_path, sheet_name, usecols=['type', 'rT2 value', 'Gray standard deviation']).dropna()

    for metric in ['rT2 value', 'Gray standard deviation']:
        print(f"\n{metric}:")
        g = {t: df[df['type'] == t][metric] for t in ['niche', 'tumor', 'normal']}
        for a, b in [('niche', 'tumor'), ('tumor', 'normal'), ('niche', 'normal')]:
            u, p = mannwhitneyu(g[a], g[b], alternative='two-sided')
            es = u / (len(g[a]) * len(g[b]))
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
            print(f"  {a} vs {b}: p={p:.4f} {sig}, effect={es:.3f}")


def plot_MRI_barplot(file_path, save_dir):
    """
    绘制MRI指标的柱状图，从statistics工作表读取显著性标签
    参考plot_cell_count_barplots的实现方式

    参数:
        file_path: Excel文件路径
        save_dir: 图片保存目录
    """
    print("\n" + "=" * 60)
    print("模块5：绘制MRI指标柱状图")
    print("=" * 60)

    os.makedirs(save_dir, exist_ok=True)

    # 读取数据
    df_data = pd.read_excel(file_path, sheet_name='mouse8')
    df_stat = pd.read_excel(file_path, sheet_name='statics_m8')

    # 清理列名
    df_stat.columns = df_stat.columns.str.strip()

    # 分组信息
    types = ['niche', 'tumor', 'normal']
    type_names = ['Niche', 'Tumor', 'Normal']
    colors = ['#c44e52', '#F1C40F', '#55a868']

    # 指标列表
    metrics = ['rT2 value', 'Gray standard deviation']

    # 构建显著性字典
    sig_dict = {}
    for _, row in df_stat.iterrows():
        metric = row['metric']
        sig_dict[metric] = {
            'niche_vs_tumor': str(row.get('niche vs tumor', '')) if pd.notna(row.get('niche vs tumor', '')) else '',
            'tumor_vs_normal': str(row.get('tumor vs normal', '')) if pd.notna(row.get('tumor vs normal', '')) else '',
            'niche_vs_normal': str(row.get('niche vs normal', '')) if pd.notna(row.get('niche vs normal', '')) else ''
        }

    print(f"从statics_m3工作表读取了 {len(sig_dict)} 个指标的显著性标记")

    def add_significance_bar(ax, x1, x2, y_height, marker):
        """添加T型统计标注"""
        if marker == 'ns' or marker == '':
            return
        y_min, y_max = ax.get_ylim()
        y_range = y_max - y_min
        short_bar_length = y_range * 0.015

        ax.plot([x1, x2], [y_height, y_height], color='black', linewidth=1.5)
        ax.plot([x1, x1], [y_height - short_bar_length / 2, y_height], color='black', linewidth=1.5)
        ax.plot([x2, x2], [y_height - short_bar_length / 2, y_height], color='black', linewidth=1.5)

        x_center = (x1 + x2) / 2
        star_height = y_height + short_bar_length * 0.05
        ax.text(x_center, star_height, marker, ha='center', va='bottom', fontsize=18, fontweight='bold')

    # 绘制每个指标
    plotted_count = 0
    for metric in metrics:
        print(f"\n绘制指标: {metric}")

        # 提取三组数据
        data_groups = {}
        for t in types:
            data_groups[t] = df_data[df_data['type'] == t][metric].dropna().values

        # 检查数据有效性
        if any(len(vals) == 0 for vals in data_groups.values()):
            print(f"  警告: 某组无数据，跳过")
            continue

        # 计算均值和标准差
        means = [data_groups[t].mean() for t in types]
        stds = [data_groups[t].std() for t in types]
        x_pos = np.arange(len(types))

        # 创建图形
        plt.figure(figsize=(8, 6))
        bars = plt.bar(x_pos, means, color=colors, edgecolor='black', linewidth=1.5, width=0.6, alpha=0.8)
        plt.errorbar(x_pos, means, yerr=stds, fmt='none', ecolor='black', capsize=8, capthick=2, linewidth=1.5)

        # 标签
        plt.xticks(x_pos, type_names, fontsize=12)
        plt.ylabel(metric, fontsize=18, fontweight='bold')
        plt.xlabel('Region', fontsize=12)

        # 获取显著性标记
        sig = sig_dict.get(metric, {})

        # 设置y轴范围
        y_max = max(means[i] + stds[i] for i in range(len(means))) if len(means) > 0 else 1
        plt.ylim(0, y_max * 1.2)

        # 添加显著性标注
        added_count = 0
        if sig.get('niche_vs_tumor') and sig['niche_vs_tumor'] not in ['', 'nan', 'NaN', 'ns']:
            add_significance_bar(plt.gca(), 0, 1, y_max * 1.03, sig['niche_vs_tumor'])
            print(f"  添加 niche vs tumor: {sig['niche_vs_tumor']}")
            added_count += 1

        if sig.get('tumor_vs_normal') and sig['tumor_vs_normal'] not in ['', 'nan', 'NaN', 'ns']:
            add_significance_bar(plt.gca(), 1, 2, y_max * 1.03, sig['tumor_vs_normal'])
            print(f"  添加 tumor vs normal: {sig['tumor_vs_normal']}")
            added_count += 1

        if sig.get('niche_vs_normal') and sig['niche_vs_normal'] not in ['', 'nan', 'NaN', 'ns']:
            add_significance_bar(plt.gca(), 0, 2, y_max * 1.11, sig['niche_vs_normal'])
            print(f"  添加 niche vs normal: {sig['niche_vs_normal']}")
            added_count += 1

        if added_count == 0:
            print(f"  无显著差异标记")

        # 美化
        plt.grid(axis='y', alpha=0.3, linestyle='--')
        # 保存
        plt.tight_layout()
        save_path = os.path.join(save_dir, f'{metric}_barplot.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  已保存: {save_path}")
        plotted_count += 1

    print(f"\n绘图完成！共绘制 {plotted_count}/{len(metrics)} 张图片，保存至: {save_dir}")






if __name__ == "__main__":
    # 文件路径
    file_path = r'H:\Cross-scale_mouse_data\ST_data\tumor_8_liver\tumor_8_liver_gene_index.xlsx'
    save_dir = r'H:\Cross-scale_mouse_data\ST_data\tumor_8_liver\figures\cell_count'

    # # 模块1：基因表达统计分析（仅打印结果，供手动记录到statistics工作表）
    # summary_results = perform_statistical_analysis(file_path)

    # # 模块2：基因表达绘制柱状图（从statistics工作表读取显著性标签）
    # plot_barplots_with_significance(file_path, save_dir)

    # # 模块3：细胞数量统计分析
    # perform_cell_count_analysis(file_path)

    # 模块4：细胞数量柱状图
    plot_cell_count_barplots(file_path, save_dir)

    # compare_groups(r'H:/MRI_index_liver_niche.xlsx', 'mouse8')

    # plot_MRI_barplot(file_path=r'H:/MRI_index_liver_niche.xlsx', save_dir=save_dir)
