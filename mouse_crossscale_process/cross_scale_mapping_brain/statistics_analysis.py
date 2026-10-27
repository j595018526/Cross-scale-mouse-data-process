"""
对细胞和基因指标进行统计分析，并绘制带显著性标记的柱状图
"""
import pandas as pd
from scipy.stats import mannwhitneyu
import numpy as np
import matplotlib.pyplot as plt
import os


def perform_statistical_analysis(file_path):
    """
    读取features工作表数据，进行Mann-Whitney U检验，仅打印结果供手动记录
    """
    # 读取数据
    df = pd.read_excel(file_path, sheet_name='features')

    print("=" * 60)
    print("统计分析（Mann-Whitney U检验）")
    print("=" * 60)

    # 定义分组
    groups = ['hip', 'tha', 'white']
    group_labels = {'hip': 'Hip', 'tha': 'Tha', 'white': 'White'}

    # 获取特征列（MRI + cell + 10个基因列）
    gene_columns = ['T2 value', 'All cells', 'Camk2a', 'Dlg4', 'Homer1', 'Map2', 'Grin2b', 'Col4a2', 'Cx3cl1', 'Mapt', 'Fgf12', 'Vsnl1']

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

    # 对每个特征进行统计分析
    for col in gene_columns:
        print(f"\n{'=' * 50}")
        print(f"特征: {col}")

        # 提取三组数据
        data = {g: df[df['type'] == g][col].dropna().values for g in groups}

        # 打印均值和标准差
        for g in groups:
            print(f"  {group_labels[g]}组: 均值={data[g].mean():.3f}, 标准差={data[g].std():.3f}")

        # 两两比较
        comparisons = [('hip_vs_tha', 'hip', 'tha'),
                      ('tha_vs_white', 'tha', 'white'),
                      ('hip_vs_white', 'hip', 'white')]

        for name, g1, g2 in comparisons:
            if len(data[g1]) < 2 or len(data[g2]) < 2:
                print(f"  {name}: 样本数不足，跳过")
                continue

            stat, p = mannwhitneyu(data[g1], data[g2], alternative='two-sided')
            marker = get_marker(p)
            print(f"  {name}: p={p:.4f} ({marker})")

        print("-" * 40)

    print(f"\n统计分析完成！共分析 {len(gene_columns)} 个特征")
    print("\n请手动将上述统计结果记录到 Excel 的 statistics 工作表中")


def plot_barplots_with_significance(file_path, save_dir):
    """
    从statistics工作表读取显著性标记，绘制带显著性标记的柱状图
    """
    print("\n" + "=" * 60)
    print("绘制带显著性标记的柱状图（从statistics工作表读取）")
    print("=" * 60)

    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 读取数据
    df = pd.read_excel(file_path, sheet_name='features')
    df_stat = pd.read_excel(file_path, sheet_name='statistics')

    # 分组信息
    groups = ['hip', 'tha', 'white']
    group_labels = ['Hip', 'Tha', 'White']
    colors = ['#c44e52', '#F1C40F', '#55a868']

    # MRI + cell + 10个基因列
    gene_columns = ['T2 value', 'All cells', 'Camk2a', 'Dlg4', 'Homer1', 'Map2', 'Grin2b', 'Col4a2', 'Cx3cl1', 'Mapt', 'Fgf12', 'Vsnl1']

    # 构建显著性字典
    sig_dict = {}
    for _, row in df_stat.iterrows():
        feature = row['feature']
        sig_dict[feature] = {
            'hip_vs_tha': str(row.get('hip_vs_tha', 'ns')) if pd.notna(row.get('hip_vs_tha', 'ns')) else 'ns',
            'tha_vs_white': str(row.get('tha_vs_white', 'ns')) if pd.notna(row.get('tha_vs_white', 'ns')) else 'ns',
            'hip_vs_white': str(row.get('hip_vs_white', 'ns')) if pd.notna(row.get('hip_vs_white', 'ns')) else 'ns'
        }

    print(f"从statistics工作表读取了 {len(sig_dict)} 个特征的显著性标记")

    def add_significance_bar(ax, x1, x2, y_height, marker):
        """添加T型统计标注"""
        if not marker or marker == 'ns' or marker == 'nan' or marker == '':
            return

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
        ax.text(x_center, star_height, marker, ha='center', va='bottom', fontsize=12, fontweight='bold')

    # 绘制每个特征的柱状图
    plotted_count = 0
    for feature in gene_columns:
        print(f"\n绘制特征: {feature}")

        # 提取三组数据
        data = {g: df[df['type'] == g][feature].dropna().values for g in groups}

        # 检查数据有效性
        if any(len(data[g]) == 0 for g in groups):
            print(f"  警告: 某组无数据，跳过")
            continue

        # 计算均值和标准差
        means = [data[g].mean() for g in groups]
        stds = [data[g].std() for g in groups]
        x_pos = np.arange(len(groups))

        # 创建图形
        plt.figure(figsize=(8, 6))
        bars = plt.bar(x_pos, means, color=colors, edgecolor='black', linewidth=1.5, width=0.6, alpha=0.8)
        plt.errorbar(x_pos, means, yerr=stds, fmt='none', ecolor='black', capsize=8, capthick=2, linewidth=1.5)

        # 设置标签
        plt.xticks(x_pos, group_labels, fontsize=12)
        plt.ylabel(feature, fontsize=14, fontweight='bold')
        plt.xlabel('Region', fontsize=12)

        # 设置y轴范围
        y_max = max(means[i] + stds[i] for i in range(len(means))) if len(means) > 0 else 1
        plt.ylim(0, y_max * 1.25)

        # 获取显著性标记
        sig = sig_dict.get(feature, {})

        # 添加显著性标注
        ax = plt.gca()
        added_count = 0

        # hip vs tha (位置0和1)
        if sig.get('hip_vs_tha') and sig['hip_vs_tha'] not in ['', 'ns', 'nan']:
            add_significance_bar(ax, 0, 1, y_max * 1.03, sig['hip_vs_tha'])
            print(f"  添加 hip vs tha: {sig['hip_vs_tha']}")
            added_count += 1

        # tha vs white (位置1和2)
        if sig.get('tha_vs_white') and sig['tha_vs_white'] not in ['', 'ns', 'nan']:
            add_significance_bar(ax, 1, 2, y_max * 1.03, sig['tha_vs_white'])
            print(f"  添加 tha vs white: {sig['tha_vs_white']}")
            added_count += 1

        # hip vs white (位置0和2，高度稍高)
        if sig.get('hip_vs_white') and sig['hip_vs_white'] not in ['', 'ns', 'nan']:
            add_significance_bar(ax, 0, 2, y_max * 1.11, sig['hip_vs_white'])
            print(f"  添加 hip vs white: {sig['hip_vs_white']}")
            added_count += 1

        if added_count == 0:
            print(f"  无显著差异标记")

        # 美化图形
        plt.grid(axis='y', alpha=0.3, linestyle='--')
        plt.tight_layout()

        # 保存图片
        save_path = os.path.join(save_dir, f'{feature}_barplot.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  已保存: {save_path}")
        plotted_count += 1

    print(f"\n绘图完成！共绘制 {plotted_count}/{len(gene_columns)} 张图片，保存至: {save_dir}")


if __name__ == "__main__":
    # 文件路径
    file_path = r'./brain_data/Box_features_b1.xlsx'
    save_dir = r'D:\PythonCode\mouse_process\image_brain\statistics\blank1'

    # # 1. 统计分析（仅打印结果，供手动记录到statistics工作表）
    # perform_statistical_analysis(file_path)

    # 2. 绘制柱状图（从statistics工作表读取显著性标记）
    plot_barplots_with_significance(file_path, save_dir)
