import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def draw_roofline_from_excel(excel_path: str = 'stack_profile_comparison.xlsx'):
    print('📊 Đang vẽ biểu đồ Roofline từ file Excel so sánh profiler...')

    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(script_dir, excel_path)
    if not os.path.exists(full_path):
        print(f'❌ LỖI: Không tìm thấy file {full_path}')
        return

    df = pd.read_excel(full_path)

    PE = 256
    FREQ_MHZ = 200
    BW_GBPS = 10

    peak_compute = (PE * FREQ_MHZ) / 1000.0
    peak_bandwidth = BW_GBPS
    ridge_point = peak_compute / peak_bandwidth

    plt.figure(figsize=(12, 7), dpi=150)
    x_roof = np.logspace(-1, 3, 500)
    y_roof = np.minimum(peak_compute, peak_bandwidth * x_roof)
    plt.plot(x_roof, y_roof, color='black', linewidth=2.5, label='Hardware Roofline Limit')

    stack_colors = {
        0: '#1f77b4',
        1: '#ff7f0e',
        2: '#2ca02c',
        3: '#d62728',
        4: '#9467bd',
        5: '#8c564b',
    }
    mode_colors = {
        'tiled': '#ff3b30',
        'full_matrix': '#007aff',
        'naive': '#34c759',
    }
    markers = {
        'tiled': 'o',
        'full_matrix': 's',
        'naive': '^',
    }

    for stack_id in sorted(df['Stack_ID'].unique()):
        stack_df = df[df['Stack_ID'] == stack_id]
        color = stack_colors.get(stack_id, '#777777')
        for mode in ['tiled', 'full_matrix', 'naive']:
            row = stack_df[stack_df['Mode'] == mode]
            if row.empty:
                continue
            row = row.iloc[0]
            intensity = row['Intensity_MAC_Byte']
            perf_gmacs = row['Perf_GMACs_s']
            x_offset = 0.0
            y_offset = 0.0
            if mode == 'full_matrix':
                x_offset = 0.02 * max(intensity, 1.0)
                y_offset = 0.03 * max(perf_gmacs, 1.0)
            elif mode == 'naive':
                x_offset = 0.04 * max(intensity, 1.0)
                y_offset = 0.06 * max(perf_gmacs, 1.0)
            plt.scatter(
                intensity + x_offset,
                perf_gmacs + y_offset,
                s=140,
                color=mode_colors[mode],
                marker=markers[mode],
                edgecolors=color,
                linewidths=1.2,
                zorder=5,
                alpha=0.95,
            )
            plt.annotate(
                f'S{stack_id}',
                (intensity + x_offset, perf_gmacs + y_offset),
                xytext=(3, 3),
                textcoords='offset points',
                fontsize=7,
                color=color,
            )

    plt.xscale('log', base=2)
    plt.yscale('log', base=2)
    plt.axvline(x=ridge_point, color='gray', linestyle='--', linewidth=1)
    plt.text(ridge_point * 0.5, peak_compute * 1.5, 'Memory Bound\n(Nghẽn bộ nhớ)', horizontalalignment='center', color='gray', fontweight='bold')
    plt.text(ridge_point * 3, peak_compute * 1.5, 'Compute Bound\n(Nghẽn tính toán)', horizontalalignment='center', color='gray', fontweight='bold')

    plt.title('Zipformer Accelerator Roofline Model (Tiled vs Full-Matrix vs Naive)', fontsize=14, fontweight='bold')
    plt.xlabel('Arithmetic Intensity (MACs / Byte)', fontsize=12)
    plt.ylabel('Actual Performance (GMACs / Second)', fontsize=12)
    plt.grid(True, which='both', ls='--', alpha=0.3)
    plt.legend(loc='lower right', fontsize=8)
    plt.xlim(0.5, 128)
    plt.ylim(0.5, peak_compute * 3)
    plt.tight_layout()

    out_path = os.path.join(script_dir, 'zipformer_roofline_comparison.png')
    plt.savefig(out_path, bbox_inches='tight')
    print(f'✅ Đã vẽ xong! Biểu đồ được lưu tại: {out_path}')
    plt.show()


if __name__ == '__main__':
    draw_roofline_from_excel()
