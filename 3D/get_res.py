import os
import glob

TARGET_METRICS = [
    'val_mse',
    'val_mae',
    'pcc_gene_mean_all',
    'pcc_gene_mean_top10',
    'pcc_gene_mean_top20',
    'pcc_gene_mean_top50',
    'pcc_gene_mean_top100',
    'pcc_spot_mean'
]

def analyze_log_file(filepath):
    print(f"--- 正在分析文件: {os.path.basename(filepath)} ---")

    collected_values = {metric: [] for metric in TARGET_METRICS}

    start_collecting = False

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if 'Training:' in line:
                    start_collecting = True
                    continue

                if not start_collecting:
                    continue

                for metric_name in TARGET_METRICS:
                    if line.strip().startswith(metric_name + ' :'):
                        try:
                            value_str = line.split(':')[1].strip()
                            value = float(value_str)
                            collected_values[metric_name].append(value)
                        except (ValueError, IndexError):
                            print(f"警告: 无法解析行 '{line.strip()}'")
                        break
    except Exception as e:
        print(f"错误: 读取文件 {filepath} 时发生错误: {e}")
        return

    final_results = {}
    for metric_name, values in collected_values.items():
        if not values:
            final_results[metric_name] = "未找到"
            continue

        if 'mae' in metric_name or 'mse' in metric_name:
            final_results[metric_name] = min(values)
        elif 'pcc' in metric_name:
            final_results[metric_name] = max(values)
        else:
            final_results[metric_name] = max(values)

    print("最终提取的最佳指标如下:")
    has_results = False
    for metric_name, best_value in final_results.items():
        if best_value != "未找到":
            print(f"  {metric_name.ljust(25)}: {best_value}")
            has_results = True
    
    if not has_results:
        print("  在该文件的训练阶段没有找到任何有效指标。")
        
    print("-" * (30 + len(os.path.basename(filepath))) + "\n")


if __name__ == "__main__":
    log_files = glob.glob('*.txt')

    if not log_files:
        print("在当前目录下没有找到任何 .txt 文件。")
    else:
        print(f"总共找到了 {len(log_files)} 个 .txt 文件，开始处理...\n")
        for file in log_files:
            analyze_log_file(file)
