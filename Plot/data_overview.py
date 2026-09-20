"""绘制数据全貌相关图片"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from _basic_set import plt_set, save_fig, BASE_DIR, RESULT_DIR
import time


def distribution_of_order_quantities_on_different_dates(save_name='None'):
    """绘制订单数量分布柱状图"""
    dates = ['2022.10.17', '2022.10.18', '2022.10.19', '2022.10.20',
             '2022.10.21', '2022.10.22', '2022.10.23', '2022.10.24']
    values = [66887, 69105, 68838, 68835, 72191, 77640, 77717, 67370]

    fig, ax = plt.subplots()
    bars = ax.bar(dates, values, width=0.5)

    # 在柱子上方标注数值
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.01,
                str(val), ha="center", va="bottom")

    ax.set_xlabel("日期")
    ax.set_xticks(dates)
    # ax.set_xticklabels(dates)
    ax.set_ylabel("数量")
    # 设置 y 轴范围，为顶部标注留出空间
    ax.set_ylim(bottom=60000)

    # 保存图表
    save_fig(save_name)


def distribution_of_order_and_rider_quantities_on_different_hours(save_name='None'):
    """绘制订单和骑手数量折线图"""
    hours = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09',
             '10', '11', '12', '13', '14', '15', '16', '17', '18', '19',
             '20', '21', '22', '23']
    order_values = [651, 376, 252, 171, 120, 156, 424, 743, 1150, 1377, 3718,
                    11881, 7058, 3429, 2429, 2217, 2626, 5619, 7945, 5903,
                    3827, 2442, 1629, 989]
    rider_values = [241, 157, 89, 64, 51, 77, 181, 359, 515, 721, 1640,
                    2628, 2382, 1524, 1084, 987, 1202, 1943, 2244, 1962,
                    1400, 908, 623, 368]

    # x = list(range(len(hours)))

    fig, ax = plt.subplots()
    ax.plot(hours, order_values, marker="o", linestyle="-", linewidth=1.5, markersize=3, label="订单数量")
    ax.plot(hours, rider_values, marker="^", linestyle="--", linewidth=1.5, markersize=3, label="骑手数量")

    ax.set_xlabel("小时")
    ax.set_xticks(hours)
    # ax.set_xticklabels(hours)
    ax.set_ylabel("数量")
    ax.set_ylim(bottom=0, top=max(order_values) * 1.08)
    ax.legend(loc="upper right", facecolor="white", edgecolor="#CCCCCC", framealpha=1)

    # 保存图表
    save_fig(save_name)


def distribution_of_merchant_and_user_locations_for_orders(save_name='None'):
    """绘制发单与收单位置的散点图（dt=20221017, dispatch_time 09:00-10:00）"""
    # 读取数据
    data_path = os.path.join(os.path.dirname(BASE_DIR), r'Plot\result', 'all_waybill_info.csv')

    cols = ['dt', 'dispatch_time', 'sender_lng', 'sender_lat', 'recipient_lng', 'recipient_lat']
    df = pd.read_csv(data_path, usecols=cols)
    df = df[df['dt'] == 20221017].copy()

    # 筛选 dispatch_time 09:00-10:00（CST）
    t_start, t_end = 1665968400, 1665972000
    df = df[(df['dispatch_time'] >= t_start) & (df['dispatch_time'] < t_end)].copy()

    # 坐标从整数转成度
    df['slng'] = df['sender_lng'] / 1e6
    df['slat'] = df['sender_lat'] / 1e6
    df['rlng'] = df['recipient_lng'] / 1e6
    df['rlat'] = df['recipient_lat'] / 1e6

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(df['slng'], df['slat'], marker='o', s=8, alpha=0.5, edgecolors='none',
               label='商家位置')
    ax.scatter(df['rlng'], df['rlat'], marker='^', s=8, alpha=0.5, edgecolors='none',
               label='用户位置')

    ax.set_xlabel("经度")
    ax.set_ylabel("纬度")
    ax.legend(loc="upper left", markerscale=2, facecolor="white", edgecolor="#CCCCCC", framealpha=1)

    save_fig(save_name)


def distribution_of_rejection_rate_on_different_hours(save_name='None'):
    """绘制拒单率折线图"""
    hours = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09',
             '10', '11', '12', '13', '14', '15', '16', '17', '18', '19',
             '20', '21', '22', '23']
    order_values = [651, 376, 252, 171, 120, 156, 424, 743, 1150, 1377, 3718,
                    11881, 7058, 3429, 2429, 2217, 2626, 5619, 7945, 5903,
                    3827, 2442, 1629, 989]
    rejection_rates = [8.96, 7.94, 6.31, 4.12, 2.85, 3.78, 4.12, 2.65, 3.48, 3.77,
              10.95, 20.47, 11.99, 7.44, 7.41, 6.68, 8.39, 13.36, 16.14,
              14.59, 11.72, 12.47, 11.66, 10.10]

    # x = list(range(len(hours)))

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax2.grid(False)  # 关闭重复网格

    colors = [plt.cm.Set2(i) for i in range(8)]
    ax1.plot(hours, order_values, marker="o", linestyle="-", linewidth=1.5, markersize=3, label="订单数量",
             color=colors[0])
    ax2.plot(hours, rejection_rates, marker="^", linestyle="--", linewidth=1.5, markersize=3, label="拒单率(%)",
             color=colors[1])

    ax1.set_xlabel("小时")
    ax1.set_xticks(hours)
    ax1.set_ylabel("订单数量")
    ax2.set_ylabel("拒单率(%)")
    ax1.set_ylim(bottom=0, top=max(order_values) * 1.08)
    ax2.set_ylim(bottom=0, top=100)

    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right",
               facecolor="white", edgecolor="#CCCCCC", framealpha=1)

    # 保存图表
    save_fig(save_name)


def distribution_of_rejection_rate_on_different_distance_left(save_name='None'):
    """绘制拒单率折线图"""
    distances = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    rejection_rates_dis = [12.14, 13.94, 14.33, 22.69, 41.38, 45.72, 50.68, 84.94, 92.47, 100.00, 100.00]
    # rejection_rates_del = [9.43, 13.26, 15.10, 14.76, 10.72, 8.85, 7.29, 5.46, 5.59, 6.02, 2.97, 4.86]

    # x = list(range(len(distances)))

    colors = [plt.cm.Set2(i) for i in range(8)]
    fig, ax = plt.subplots(figsize=(4, 5))
    ax.plot(distances, rejection_rates_dis, marker="o", linestyle="-", linewidth=1.5, markersize=3, label="接单距离",
             color=colors[0])
    # ax.plot(distances, rejection_rates_del, marker="^", linestyle="--", linewidth=1.5, markersize=3, label="配送距离",
    #          color=colors[1])

    ax.set_xlabel("接单距离(km)")
    ax.set_xticks(distances)
    # ax.set_xticklabels(distances)
    ax.set_ylabel("拒单率(%)")
    ax.set_ylim(bottom=0, top=105)
    # ax.legend(loc="upper right", facecolor="white", edgecolor="#CCCCCC", framealpha=1)

    # 子图编号标注（图左上角外侧）
    fig.text(-0.005, 0.95, "(a)", fontsize=12, va='top', ha='left')

    # 保存图表
    save_fig(save_name)


def distribution_of_rejection_rate_on_different_distance_right(save_name='None'):
    """绘制拒单率折线图"""
    distances = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # rejection_rates_dis = [11.93, 13.96, 14.37, 22.93, 42.05, 46.78, 53.37, 85.98, 94.51, 100.00, 100.00, 100.00]
    rejection_rates_del = [9.50, 13.36, 15.24, 14.93, 10.92, 9.17, 7.66, 5.82, 5.72, 6.34, 3.16]
    # x = list(range(len(distances)))

    colors = [plt.cm.Set2(i) for i in range(8)]
    fig, ax = plt.subplots(figsize=(4, 5))
    # ax.plot(distances, rejection_rates_dis, marker="o", linestyle="-", linewidth=1.5, markersize=3, label="接单距离",
            #  color=colors[0])
    ax.plot(distances, rejection_rates_del, marker="o", linestyle="-", linewidth=1.5, markersize=3, label="配送距离",
             color=colors[1])

    ax.set_xlabel("配送距离(km)")
    ax.set_xticks(distances)
    # ax.set_xticklabels(distances)
    ax.set_ylabel("拒单率(%)")
    ax.set_ylim(bottom=0, top=105)
    # ax.legend(loc="upper right", facecolor="white", edgecolor="#CCCCCC", framealpha=1)

    # 子图编号标注（图左上角外侧）
    fig.text(-0.005, 0.95, "(b)", fontsize=12, va='top', ha='left')

    # 保存图表
    save_fig(save_name)


def descriptive_statistics_of_key_variables():
    """描述性统计"""
    data_path = os.path.join(os.path.dirname(BASE_DIR), r'Plot\result', 'all_waybill_info.csv')

    cols = ['dispatch_to_sender_delivery_distance', 'sender_to_recipient_delivery_distance',
            'delivery_time_difference', 'available_time_difference', 'is_courier_grabbed']
    df = pd.read_csv(data_path, usecols=cols)
    # 转为数值再筛选（避免CSV中字符串'1'无法匹配数值1）
    df['is_courier_grabbed'] = pd.to_numeric(df['is_courier_grabbed'], errors='coerce')
    df = df[df['is_courier_grabbed'] == 1].drop(columns=['is_courier_grabbed'])
    print(df.describe())
    return df.describe()


def save_fig_jpg600(save_name):
    """以 jpg 格式、600 dpi 保存当前图并关闭窗口"""
    os.makedirs(RESULT_DIR, exist_ok=True)
    out_path = os.path.join(RESULT_DIR, save_name)
    plt.savefig(out_path, bbox_inches="tight", dpi=600,
                facecolor="white", pad_inches=0.02)
    plt.close()
    print(f"图片保存：{out_path}")


def distribution_of_rejection_rate_on_different_hours_en(save_name='None'):
    """Plot rejection rate line chart (English version)"""
    hours = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09',
             '10', '11', '12', '13', '14', '15', '16', '17', '18', '19',
             '20', '21', '22', '23']
    order_values = [651, 376, 252, 171, 120, 156, 424, 743, 1150, 1377, 3718,
                    11881, 7058, 3429, 2429, 2217, 2626, 5619, 7945, 5903,
                    3827, 2442, 1629, 989]
    rejection_rates = [8.96, 7.94, 6.31, 4.12, 2.85, 3.78, 4.12, 2.65, 3.48, 3.77,
              10.95, 20.47, 11.99, 7.44, 7.41, 6.68, 8.39, 13.36, 16.14,
              14.59, 11.72, 12.47, 11.66, 10.10]

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax2.grid(False)  # 关闭重复网格

    colors = [plt.cm.Set2(i) for i in range(8)]
    ax1.plot(hours, order_values, marker="o", linestyle="-", linewidth=1.5, markersize=3, label="Order Quantity",
             color=colors[0])
    ax2.plot(hours, rejection_rates, marker="^", linestyle="--", linewidth=1.5, markersize=3, label="Rejection Rate (%)",
             color=colors[1])

    ax1.set_xlabel("Hour")
    ax1.set_xticks(hours)
    ax1.set_ylabel("Order Quantity")
    ax2.set_ylabel("Rejection Rate (%)")
    ax1.set_ylim(bottom=0, top=max(order_values) * 1.08)
    ax2.set_ylim(bottom=0, top=100)

    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right",
               facecolor="white", edgecolor="#CCCCCC", framealpha=1)

    # 保存图表
    save_fig_jpg600(save_name)


def distribution_of_rejection_rate_on_different_distance_left_en(save_name='None'):
    """Plot rejection rate line chart (English version)"""
    distances = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    rejection_rates_dis = [12.14, 13.94, 14.33, 22.69, 41.38, 45.72, 50.68, 84.94, 92.47, 100.00, 100.00]

    colors = [plt.cm.Set2(i) for i in range(8)]
    fig, ax = plt.subplots(figsize=(4, 5))
    ax.plot(distances, rejection_rates_dis, marker="o", linestyle="-", linewidth=1.5, markersize=3, label="Pickup Distance",
             color=colors[0])

    ax.set_xlabel("Pickup Distance (km)")
    ax.set_xticks(distances)
    ax.set_ylabel("Rejection Rate (%)")
    ax.set_ylim(bottom=0, top=105)

    # 子图编号标注（图左上角外侧）
    fig.text(-0.005, 0.95, "(a)", fontsize=12, va='top', ha='left')

    # 保存图表
    save_fig_jpg600(save_name)


def distribution_of_rejection_rate_on_different_distance_right_en(save_name='None'):
    """Plot rejection rate line chart (English version)"""
    distances = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    rejection_rates_del = [9.50, 13.36, 15.24, 14.93, 10.92, 9.17, 7.66, 5.82, 5.72, 6.34, 3.16]

    colors = [plt.cm.Set2(i) for i in range(8)]
    fig, ax = plt.subplots(figsize=(4, 5))
    ax.plot(distances, rejection_rates_del, marker="o", linestyle="-", linewidth=1.5, markersize=3, label="Delivery Distance",
             color=colors[1])

    ax.set_xlabel("Delivery Distance (km)")
    ax.set_xticks(distances)
    ax.set_ylabel("Rejection Rate (%)")
    ax.set_ylim(bottom=0, top=105)

    # 子图编号标注（图左上角外侧）
    fig.text(-0.005, 0.95, "(b)", fontsize=12, va='top', ha='left')

    # 保存图表
    save_fig_jpg600(save_name)


def main():
    plt_set()
    distribution_of_order_quantities_on_different_dates(save_name='不同日期下的订单数量分布.svg')
    distribution_of_order_and_rider_quantities_on_different_hours(save_name='不同时间下的订单和骑手数量分布.svg')
    distribution_of_merchant_and_user_locations_for_orders(save_name='订单的商家与用户位置分布.svg')
    distribution_of_rejection_rate_on_different_hours(save_name='不同时间下的拒单率分布.svg')
    distribution_of_rejection_rate_on_different_distance_left(save_name='不同接单距离下的拒单率分布.svg')
    distribution_of_rejection_rate_on_different_distance_right(save_name='不同配送距离下的拒单率分布.svg')
    descriptive_statistics_of_key_variables()
    # 英文版（jpg, 600 dpi）
    distribution_of_rejection_rate_on_different_hours_en(save_name='不同时间下的拒单率分布_en.jpg')
    distribution_of_rejection_rate_on_different_distance_left_en(save_name='不同接单距离下的拒单率分布_en.jpg')
    distribution_of_rejection_rate_on_different_distance_right_en(save_name='不同配送距离下的拒单率分布_en.jpg')

if __name__ == "__main__":
    main()
