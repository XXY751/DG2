# analyze_dataset.py (v5_fixed - 规范化和错误处理优化)
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
from tqdm import tqdm  # 用于显示加载进度条
import random
import math

# --- 绘图依赖 ---
# 尝试导入，以便在缺失时提供明确的错误信息
try:
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    from sklearn.manifold import TSNE

    # 启用 matplotlib 的 Agg 后端，这样可以在没有显示器的服务器上运行
    plt.switch_backend('Agg')
    _PLOT_LIBS_INSTALLED = True
except ImportError:
    _PLOT_LIBS_INSTALLED = False
    print("[警告] 缺少绘图库 (matplotlib, scikit-learn)。")
    print("[警告] 所有 t-SNE 绘图功能将不可用。")
    print("[警告] 请运行: pip install matplotlib scikit-learn")


# --- 结束依赖 ---

def run_tsne_and_plot(X_data, y_data, plot_limit, title, save_path):
    """
    辅助函数：对给定的数据运行 t-SNE 并保存图像。

    Args:
        X_data (np.array): 特征数据 (N, D)
        y_data (np.array): 标签数据 (N,)
        plot_limit (int): 采样点数上限
        title (str): 图像标题
        save_path (Path): 图像保存路径
    """
    total_points = X_data.shape[0]
    print(f"      - {title}: 运行 t-SNE... (数据点: {total_points})")

    # 1. 采样
    if total_points > plot_limit:
        print(f"      - {title}: 数据点过多 ({total_points})，随机采样至 {plot_limit} 个点。")
        indices = np.random.permutation(total_points)[:plot_limit]
        X_sample = X_data[indices]
        y_sample = y_data[indices]
    else:
        X_sample = X_data
        y_sample = y_data

    # 2. 运行 t-SNE
    try:
        tsne = TSNE(
            n_components=2,
            random_state=42,
            perplexity=30,  # 经典值
            n_iter=1000,
            init='pca',  # 使用 PCA 初始化，更快更稳定
            learning_rate='auto',
            n_jobs=-1  # 使用所有 CPU 核心
        )
        X_embedded = tsne.fit_transform(X_sample)
        print(f"      - {title}: t-SNE 完成。正在生成图像...")

        # 3. 绘图
        fig, ax = plt.subplots(figsize=(10, 8))
        colors = ['yellow', 'red', 'blue', 'pink', 'green']
        labels_text = ['Wake', 'N1', 'N2', 'N3', 'REM']

        for i in range(5):
            indices = (y_sample == i)
            if np.sum(indices) > 0:
                ax.scatter(
                    X_embedded[indices, 0],
                    X_embedded[indices, 1],
                    c=colors[i],
                    label=labels_text[i],
                    s=5,  # 点更小
                    alpha=0.6  # 轻微透明
                )

        # 4. 设置图例和标题
        ax.legend(
            loc='upper center',
            bbox_to_anchor=(0.5, 1.15),
            ncol=5,
            fancybox=True,
            shadow=False,
            markerscale=3  # 增大图例标记
        )

        ax.set_title(title, pad=30)  # 增加标题和图例的间距
        ax.set_xticks([])
        ax.set_yticks([])

        # 5. 保存
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"      [√] t-SNE 图像已保存到: {save_path}")
        plt.close(fig)  # 释放内存
        return True

    except Exception as e:
        print(f"      [错误] t-SNE 运行或绘图失败 ({title}): {e}")
        if 'fig' in locals():
            plt.close(fig)  # 确保即使出错也关闭图像
        return False


def combine_plots(plot_info, output_path, base_dir):
    """
    辅助函数：将多个 t-SNE 绘图结果合并(拼接)到一个网格图像中。

    Args:
        plot_info (list): 包含 {"path": Path, "title": str} 的字典列表
        output_path (Path): 最终合并图像的保存路径
        base_dir (Path): 数据集根目录，用于解析相对路径
    """
    try:
        n_plots = len(plot_info)
        if n_plots == 0:
            print("  [警告] 没有可供合并的图像。")
            return

        # --- 决定网格布局 ---
        # 优先单行
        if n_plots <= 4:
            n_rows, n_cols = 1, n_plots
        # 变为 2xN 网格
        elif n_plots <= 8:
            n_cols = math.ceil(n_plots / 2)
            n_rows = 2
        # 变为 3xN 网格
        else:
            n_cols = math.ceil(n_plots / 3)
            n_rows = 3
        # --- 结束布局 ---

        fig_width = n_cols * 8  # 8 英寸/子图宽度
        fig_height = n_rows * 8  # 8 英寸/子图高度

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), squeeze=False)
        axes = axes.flatten()  # 将 2D 数组变为 1D，方便迭代

        # 生成 (a) Title, (b) Title ...
        plot_labels = [f"({chr(97 + i)}) {info['title']}" for i, info in enumerate(plot_info)]

        for i in range(n_plots):
            try:
                img_path = plot_info[i]['path']
                # 确保路径是绝对的
                if not img_path.is_absolute():
                    img_path = base_dir / img_path

                img = mpimg.imread(img_path)
                axes[i].imshow(img)
                axes[i].set_title(plot_labels[i], fontsize=20, pad=10)
                axes[i].axis('off')
            except Exception as e:
                print(f"  [警告] 无法加载图像 {plot_info[i]['path']} 进行合并: {e}")
                axes[i].set_title(f"Error loading {plot_info[i]['title']}", color='red')
                axes[i].axis('off')

        # 隐藏未使用的子图
        for j in range(n_plots, len(axes)):
            axes[j].axis('off')

        plt.tight_layout(pad=2.0)
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        print(f"\n[√] 合并后的 t-SNE 网格图已保存到: {output_path}")

    except Exception as e:
        print(f"\n[错误] 无法合并 t-SNE 图像: {e}")


def analyze_dataset(
        datasets_dir: str,
        dataset_name: str = None,
        plot_individual: bool = False,
        plot_comparison_grid: bool = False,
        plot_combined_data: bool = False,
        plot_limit: int = 5000
):
    """
    主函数：
    检查并分析一个或多个数据集的文件完整性、样本量、总长度、标签分布，
    并可选择生成 t-SNE 特征分布图。
    """

    # 1. 检查绘图库是否安装
    plotting_requested = plot_individual or plot_comparison_grid or plot_combined_data
    if plotting_requested and not _PLOT_LIBS_INSTALLED:
        print("[错误] 请求了 t-SNE 绘图，但缺少必要的库 (matplotlib, scikit-learn)。")
        print("[错误] 所有绘图功能已禁用。")
        plot_individual = plot_comparison_grid = plot_combined_data = False

    # 2. 检查绘图逻辑
    if plot_comparison_grid and not plot_individual:
        print("[警告] --plot_comparison_grid (对比网格) 依赖 --plot_individual (单独绘制) 来生成子图。")
        print("[警告] 请同时指定 --plot_individual 和 --plot_comparison_grid。")
        plot_comparison_grid = False

    # 3. 检查根目录
    base_dir = Path(datasets_dir)
    if not base_dir.is_dir():
        print(f"[错误] 数据集根目录不存在: {datasets_dir}")
        return

    # 4. 确定要检查的数据集
    datasets_to_check = []
    if dataset_name:
        if (base_dir / dataset_name).is_dir():
            datasets_to_check = [dataset_name]
        else:
            print(f"[错误] 指定的数据集 '{dataset_name}' 在 {datasets_dir} 中未找到。")
            return
    else:
        # 检查所有子目录
        datasets_to_check = [d.name for d in base_dir.iterdir() if d.is_dir()]
        print(f"[信息] 未指定特定数据集，将检查所有找到的目录: {datasets_to_check}")

    print("\n" + "=" * 80)
    print(f"开始检查并分析数据集...")
    print("=" * 80)

    # --- 汇总变量 ---
    overall_summary = []
    generated_plot_info = []  # 存储 {"path": Path, "title": str}
    overall_features_list = []  # 存储所有特征 (用于组合图)
    overall_labels_list = []  # 存储所有标签 (用于组合图)

    # 5. 遍历每个数据集
    for d_name in datasets_to_check:
        print(f"\n--- 正在分析数据集: {d_name} ---")
        dataset_dir = base_dir / d_name
        seq_base_path = dataset_dir / 'seq'
        label_base_path = dataset_dir / 'labels'

        if not seq_base_path.is_dir():
            print(f"  [跳过] 未找到 'seq' 目录: {seq_base_path}")
            continue
        if not label_base_path.is_dir():
            print(f"  [跳过] 未找到 'labels' 目录: {label_base_path}")
            continue

        # --- 5.1 扫描文件 ---
        seq_files_map = defaultdict(dict)
        label_files_map = defaultdict(dict)

        print(f"  [1] 扫描 'seq' 目录...")
        dataset_seq_count = 0
        try:
            sub_dirs = sorted([d.name for d in seq_base_path.iterdir() if d.is_dir()])
            if not sub_dirs:
                sub_dirs = ["."]  # 如果没有子目录，就直接在 seq/ 目录下查找
        except Exception as e:
            print(f"    [错误] 访问 {seq_base_path} 出错: {e}")
            continue

        for sub_dir in sub_dirs:
            current_seq_dir = seq_base_path / sub_dir
            try:
                for f in current_seq_dir.glob('*.npy'):
                    seq_files_map[sub_dir][f.name] = f
                    dataset_seq_count += 1
            except Exception as e:
                print(f"    [错误] 访问 {current_seq_dir} 出错: {e}")
        print(f"      找到 {dataset_seq_count} 个 'seq' 文件。")

        print(f"  [2] 扫描 'labels' 目录...")
        dataset_label_count = 0
        for sub_dir in sub_dirs:  # 使用从 seq 目录扫描到的相同子目录结构
            current_label_dir = label_base_path / sub_dir
            if not current_label_dir.is_dir():
                continue
            try:
                for f in current_label_dir.glob('*.npy'):
                    label_files_map[sub_dir][f.name] = f
                    dataset_label_count += 1
            except Exception as e:
                print(f"    [错误] 访问 {current_label_dir} 出错: {e}")
        print(f"      找到 {dataset_label_count} 个 'label' 文件。")

        # --- 5.2 对比分析与统计 ---
        print(f"  [3] 对比分析与统计 (数据集: {d_name})")
        matched_count = 0
        seq_missing_labels = []
        labels_missing_seq = []
        matched_pairs = []  # (seq_path, label_path)

        all_sub_dirs_keys = set(seq_files_map.keys()) | set(label_files_map.keys())

        for sub_dir in sorted(list(all_sub_dirs_keys)):
            seqs_in_subdir = seq_files_map.get(sub_dir, {})
            labels_in_subdir = label_files_map.get(sub_dir, {})
            all_files_in_subdir = set(seqs_in_subdir.keys()) | set(labels_in_subdir.keys())

            for file_name in sorted(list(all_files_in_subdir)):
                has_seq = file_name in seqs_in_subdir
                has_label = file_name in labels_in_subdir

                if has_seq and has_label:
                    matched_count += 1
                    matched_pairs.append((seqs_in_subdir[file_name], labels_in_subdir[file_name]))
                elif has_seq and not has_label:
                    seq_missing_labels.append(f"{sub_dir}/{file_name}")
                elif not has_seq and has_label:
                    labels_missing_seq.append(f"{sub_dir}/{file_name}")

        # --- 5.3 打印文件完整性报告 ---
        print(f"\n  --- 文件完整性报告: {d_name} ---")
        print(f"    总 'seq' 文件数: {dataset_seq_count}")
        print(f"    总 'label' 文件数: {dataset_label_count}")
        print(f"    [√] 成功匹配 {matched_count} 对文件 (样本量)")

        if seq_missing_labels:
            print(f"    [!!] 缺失 'label' 的 'seq' 文件 ({len(seq_missing_labels)} 个):")
            for i, missing in enumerate(seq_missing_labels[:10]):  # 最多显示 10 个
                print(f"      - {missing}")
            if len(seq_missing_labels) > 10:
                print(f"      ... (及其他 {len(seq_missing_labels) - 10} 个)")

        if labels_missing_seq:
            print(f"    [?] 缺失 'seq' 的 'label' 文件 ({len(labels_missing_seq)} 个):")
            for i, missing in enumerate(labels_missing_seq[:10]):
                print(f"      - {missing}")
            if len(labels_missing_seq) > 10:
                print(f"      ... (及其他 {len(labels_missing_seq) - 10} 个)")

        # --- 5.4 统计数据内容 ---
        print(f"\n  --- 数据内容分析: {d_name} ---")
        if matched_count == 0:
            print("    没有匹配的文件，无法进行内容分析。")
            overall_summary.append({
                "Dataset": d_name, "Samples (Files)": 0, "Epochs/Sample": "N/A", "Total Epochs": 0,
                "Class 0 (W)": 0, "Class 1 (N1)": 0, "Class 2 (N2)": 0, "Class 3 (N3)": 0, "Class 4 (R)": 0
            })
            continue

        # 1. 获取数据形状 (仅加载第一个样本)
        epochs_per_sample = "N/A"
        try:
            first_seq_path, first_label_path = matched_pairs[0]
            first_seq = np.load(first_seq_path)
            first_label = np.load(first_label_path)

            seq_shape = first_seq.shape
            label_shape = first_label.shape

            epochs_per_sample = label_shape[0] if len(label_shape) == 1 else -1

            print(f"    样本形状 (基于第一个文件):")
            if len(seq_shape) == 3:
                print(f"      - Seq Shape   : {seq_shape} (推断: N={seq_shape[0]}, C={seq_shape[1]}, L={seq_shape[2]})")
            elif len(seq_shape) == 2:
                print(f"      - Seq Shape   : {seq_shape} (推断: N={seq_shape[0]}, Features={seq_shape[1]})")
            else:
                print(f"      - Seq Shape   : {seq_shape}")

            print(f"      - Label Shape : {label_shape} (推断: Epochs/Sample = {epochs_per_sample})")
            print(f"    样本量 (文件数): {matched_count}")

        except Exception as e:
            print(f"    [错误] 加载第一个样本文件失败: {e}")
            print("    无法推断形状和总长度。")
            epochs_per_sample = -1  # 标记为错误

        # 2. 统计标签分布 (加载所有标签)
        total_class_counts = np.zeros(5, dtype=np.int64)

        print(f"    正在加载所有 {matched_count} 个标签文件以统计分布...")
        valid_labels_loaded = 0
        total_epochs_from_labels = 0
        label_range_warning_printed = False  # 优化：防止警告刷屏

        expected_epochs = epochs_per_sample if epochs_per_sample != -1 and epochs_per_sample != "N/A" else None

        for _, label_path in tqdm(matched_pairs, desc="加载标签"):
            try:
                label_array = np.load(label_path).astype(np.int64)

                # 如果是第一次成功加载，则设置期望的 epoch 数量
                if expected_epochs is None:
                    if len(label_array.shape) == 1:
                        expected_epochs = label_array.shape[0]
                        print(f"      [信息] 从 {label_path} 推断 Epochs/Sample = {expected_epochs}")
                    else:
                        print(f"      [警告] 标签形状无效: {label_path} 形状为 {label_array.shape}, 跳过。")
                        continue

                # 检查形状是否一致
                if label_array.shape != (expected_epochs,):
                    print(
                        f"      [警告] 标签形状不一致: {label_path} 的形状为 {label_array.shape}, 预期为 {(expected_epochs,)}")
                    continue

                # 确保标签值在 0-4 范围内
                valid_mask = (label_array >= 0) & (label_array <= 4)
                if np.sum(valid_mask) != len(label_array):
                    if not label_range_warning_printed:
                        print(
                            f"      [警告] 标签值超出范围 [0, 4]: {label_path} (min: {np.min(label_array)}, max: {np.max(label_array)})")
                        print("      [警告] ... (后续同类警告将在此数据集中被抑制)")
                        label_range_warning_printed = True

                label_array_valid = label_array[valid_mask]

                if len(label_array_valid) > 0:
                    counts = np.bincount(label_array_valid, minlength=5)
                    total_class_counts += counts[:5]  # 只取前 5 个
                    total_epochs_from_labels += len(label_array_valid)  # 仅统计有效 epoch

                valid_labels_loaded += 1

            except Exception as e:
                print(f"      [错误] 加载标签文件失败: {label_path}. 错误: {e}")

        print(
            f"    标签分布 (基于 {valid_labels_loaded} 个成功加载的样本, 共 {total_epochs_from_labels} 个有效 Epochs):")
        if total_epochs_from_labels == 0:
            print("      无法计算分布（没有有效的 Epochs）。")
        else:
            labels = ["Class 0 (Wake)", "Class 1 (N1)", "Class 2 (N2)", "Class 3 (N3)", "Class 4 (REM)"]
            print(f"      {'阶段':<15} | {'Epochs 数量':<15} | {'百分比':<10}")
            print("      " + "-" * 44)
            for i in range(5):
                count = total_class_counts[i]
                percent = (count / total_epochs_from_labels) * 100
                print(f"      {labels[i]:<15} | {count:<15} | {percent: <10.2f}%")

        # 3. 存储汇总信息
        overall_summary.append({
            "Dataset": d_name,
            "Samples (Files)": matched_count,
            "Epochs/Sample": expected_epochs if expected_epochs is not None else "N/A",
            "Total Epochs": total_epochs_from_labels,
            "Class 0 (W)": total_class_counts[0],
            "Class 1 (N1)": total_class_counts[1],
            "Class 2 (N2)": total_class_counts[2],
            "Class 3 (N3)": total_class_counts[3],
            "Class 4 (R)": total_class_counts[4]
        })

        # --- 5.5 t-SNE 绘图 (v5 逻辑) ---
        if plot_individual or plot_combined_data:
            print(f"\n  [4] t-SNE 数据加载 (数据集: {d_name})")

            dataset_features = []
            dataset_labels = []
            tsne_shape_warning_printed = False  # 优化：防止警告刷屏
            tsne_mismatch_warning_printed = False  # 优化：防止警告刷屏

            print(f"    正在加载 {matched_count} 个 'seq' 和 'label' 文件用于 t-SNE...")
            for seq_path, label_path in tqdm(matched_pairs, desc="加载 t-SNE 数据"):
                try:
                    label_array = np.load(label_path).astype(np.int64)
                    seq_array = np.load(seq_path)

                    # 预处理特征：将其变为 (N, Features)
                    features = None
                    if len(seq_array.shape) == 3:  # (N, C, L)
                        n_epochs, c, l = seq_array.shape
                        features = seq_array.reshape((n_epochs, c * l))
                    elif len(seq_array.shape) == 2:  # 已经是 (N, Features)
                        features = seq_array
                    else:
                        if not tsne_shape_warning_printed:
                            print(f"      [警告] 不支持的 seq 形状: {seq_array.shape} in {seq_path}, 跳过此文件。")
                            print("      [警告] ... (后续同类警告将在此数据集中被抑制)")
                            tsne_shape_warning_printed = True
                        continue

                    # 检查 seq 和 label 长度是否匹配
                    if features.shape[0] != label_array.shape[0]:
                        if not tsne_mismatch_warning_printed:
                            print(
                                f"      [警告] seq 和 label 长度不匹配: {features.shape[0]} vs {label_array.shape[0]} in {seq_path}, 跳过此文件。")
                            print("      [警告] ... (后续同类警告将在此数据集中被抑制)")
                            tsne_mismatch_warning_printed = True
                        continue

                    # 过滤无效标签 (值 < 0 或 > 4)
                    valid_mask = (label_array >= 0) & (label_array <= 4)
                    if np.sum(valid_mask) == 0:
                        continue

                    dataset_features.append(features[valid_mask])
                    dataset_labels.append(label_array[valid_mask])

                except Exception as e:
                    print(f"      [错误] 加载 t-SNE 数据失败: {seq_path} 或 {label_path}. 错误: {e}")

            if not dataset_features:
                print("    [错误] 未能加载任何有效的 t-SNE 数据。跳过此数据集的绘图。")
                continue

            # 合并当前数据集的数据
            X_current = np.vstack(dataset_features)
            y_current = np.concatenate(dataset_labels)

            # --- 选项 1：单独绘制 ---
            if plot_individual:
                plot_filename = base_dir / f"{d_name}_tsne_visualization.png"
                plot_title = f't-SNE for {d_name} (Sampled from {X_current.shape[0]} points)'

                success = run_tsne_and_plot(X_current, y_current, plot_limit, plot_title, plot_filename)

                if success and plot_comparison_grid:  # 仅在成功时才添加到网格
                    generated_plot_info.append({"path": plot_filename, "title": d_name})

            # --- 选项 2：为组合图存储数据 ---
            if plot_combined_data:
                overall_features_list.append(X_current)
                overall_labels_list.append(y_current)
                print(f"      - {d_name}: 已存储 {X_current.shape[0]} 个点用于组合图。")
        # --- 结束 t-SNE ---

    # 6. 打印最终汇总表格
    print("\n" + "=" * 80)
    print("分析完成：总体汇总表")
    print("=" * 80)

    if not overall_summary:
        print("未分析任何数据集。")
        return

    # 打印表头
    headers = ["Dataset", "Samples (Files)", "Epochs/Sample", "Total Epochs", "Class 0 (W)", "Class 1 (N1)",
               "Class 2 (N2)", "Class 3 (N3)", "Class 4 (R)"]
    # 计算每列宽度
    col_widths = {h: len(h) for h in headers}
    for row in overall_summary:
        for h in headers:
            col_widths[h] = max(col_widths[h], len(str(row.get(h, 'N/A'))))

    # --- 计算总计 ---
    total_row = {h: 0 for h in headers if h not in ["Dataset", "Epochs/Sample"]}
    total_row["Dataset"] = "OVERALL TOTAL"
    total_row["Epochs/Sample"] = "N/A"

    for row in overall_summary:
        total_row["Samples (Files)"] += row.get("Samples (Files)", 0)
        total_row["Total Epochs"] += row.get("Total Epochs", 0)
        total_row["Class 0 (W)"] += row.get("Class 0 (W)", 0)
        total_row["Class 1 (N1)"] += row.get("Class 1 (N1)", 0)
        total_row["Class 2 (N2)"] += row.get("Class 2 (N2)", 0)
        total_row["Class 3 (N3)"] += row.get("Class 3 (N3)", 0)
        total_row["Class 4 (R)"] += row.get("Class 4 (R)", 0)

    # 确保总计行的宽度也被计算在内
    for h in headers:
        col_widths[h] = max(col_widths[h], len(str(total_row.get(h, 'N/A'))))
    # --- 结束总计 ---

    # 打印表头
    header_line = " | ".join(f"{h:<{col_widths[h]}}" for h in headers)
    print(header_line)
    print("-" * len(header_line))

    # 打印数据行
    for row in overall_summary:
        print(" | ".join(f"{str(row.get(h, 'N/A')):<{col_widths[h]}}" for h in headers))

    # --- 打印总计行 ---
    print("-" * len(header_line))
    print(" | ".join(f"{str(total_row.get(h, 'N/A')):<{col_widths[h]}}" for h in headers))

    # --- 打印整体分布百分比 ---
    print("\n" + "-" * 80)
    print("整体分布百分比 (OVERALL TOTAL)")
    total_epochs_all = total_row["Total Epochs"]
    if total_epochs_all == 0:
        print("  没有有效的 Epochs，无法计算整体分布。")
    else:
        labels = ["Class 0 (Wake)", "Class 1 (N1)", "Class 2 (N2)", "Class 3 (N3)", "Class 4 (REM)"]
        counts = [total_row["Class 0 (W)"], total_row["Class 1 (N1)"], total_row["Class 2 (N2)"],
                  total_row["Class 3 (N3)"], total_row["Class 4 (R)"]]
        print(f"  {'阶段':<15} | {'总 Epochs 数量':<15} | {'百分比':<10}")
        print("  " + "-" * 46)
        for i in range(5):
            count = counts[i]
            percent = (count / total_epochs_all) * 100
            print(f"  {labels[i]:<15} | {count:<15} | {percent: <10.2f}%")

    print("\n" + "=" * 80)

    # 7. 分析结束后的最终绘图步骤
    print("\n" + "=" * 80)
    print("开始最终绘图... (如果已请求)")

    # --- 选项 2：绘制对比网格 ---
    if plot_comparison_grid:
        print(f"[绘图] 正在合并 {len(generated_plot_info)} 个单独的 t-SNE 图像...")
        if len(generated_plot_info) > 0:
            combine_plots(
                generated_plot_info,
                base_dir / "OVERALL_comparison_grid.png",
                base_dir
            )
        else:
            print("[警告] 请求了对比网格，但没有成功生成的单独图像可供合并。")

    # --- 选项 3：绘制组合数据图 ---
    if plot_combined_data:
        print(f"[绘图] 正在组合来自 {len(overall_features_list)} 个数据集的所有数据...")
        if len(overall_features_list) > 0:
            X_combined = np.vstack(overall_features_list)
            y_combined = np.concatenate(overall_labels_list)

            plot_filename = base_dir / "OVERALL_combined_data_tsne.png"
            plot_title = f't-SNE for All Combined Data (Sampled from {X_combined.shape[0]} points)'

            run_tsne_and_plot(X_combined, y_combined, plot_limit, plot_title, plot_filename)
        else:
            print("[警告] 请求了组合数据图，但没有加载任何有效数据。")

    print("分析和绘图完成。")
    print("=" * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='分析数据集的文件完整性、样本量、总长度和标签分布，并可选择绘制 t-SNE 图。',
        formatter_class=argparse.RawTextHelpFormatter  # 保持换行
    )

    # --- 必需参数 ---
    parser.add_argument(
        '--datasets_dir',
        type=str,
        required=True,
        help='数据集的根目录 (例如: /data/sleep/datasets_dir)'
    )

    # --- 可选参数 ---
    parser.add_argument(
        '--dataset',
        type=str,
        default=None,
        help='(可选) 要分析的特定数据集名称 (例如: ABC)。\n如果省略，则分析根目录下的所有数据集。'
    )

    # --- 绘图参数 ---
    plot_group = parser.add_argument_group('t-SNE 绘图选项 (需要 matplotlib 和 scikit-learn)')
    plot_group.add_argument(
        '--plot_individual',
        action='store_true',
        help='(可选, 对应"是否单独绘制") \n为每个数据集生成一个单独的 t-SNE 图像。'
    )
    plot_group.add_argument(
        '--plot_comparison_grid',
        action='store_true',
        help='(可选, 对应"大的对比图" - 方式1) \n将 --plot_individual 生成的图像合并(拼接)成一个网格对比图。\n(依赖 --plot_individual)'
    )
    plot_group.add_argument(
        '--plot_combined_data',
        action='store_true',
        help='(可选, 对应"大的对比图" - 方式2) \n将 *所有* 数据集的数据点合并后，运行一次 t-SNE。'
    )
    plot_group.add_argument(
        '--plot_limit',
        type=int,
        default=5000,
        help='(可选) t-SNE 绘图时使用的最大数据点数（Epochs），\n用于加速计算 (默认: 5000)'
    )

    args = parser.parse_args()

    # 传递参数
    analyze_dataset(
        args.datasets_dir,
        args.dataset,
        args.plot_individual,
        args.plot_comparison_grid,
        args.plot_combined_data,
        args.plot_limit
    )