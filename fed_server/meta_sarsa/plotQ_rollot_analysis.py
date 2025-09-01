import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
from plant_analysis import RL_Debugger
# 设置中文字体和图表样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
plt.style.use('ggplot')


def analyze_rollout_results(csv_file_path):
    """
    分析rollout测试结果的CSV文件并生成图表
    """
    # 读取CSV文件
    try:
        df = pd.read_csv(csv_file_path)
        print(f"成功读取文件: {csv_file_path}")
        print(f"数据形状: {df.shape}")
        print("\n前5行数据:")
        print(df.head())
        print("\n数据基本信息:")
        print(df.info())
        print("\n描述性统计:")
        print(df.describe())
    except FileNotFoundError:
        print(f"文件 {csv_file_path} 未找到")
        return
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return

    # 创建图表
    fig = plt.figure(figsize=(20, 16))

    # 1. 奖励随时间变化
    plt.subplot(3, 3, 1)
    plt.plot(df['time_step'], df['reward'], 'b-', linewidth=2, marker='o', markersize=4)
    plt.title('Rewards as time', fontsize=14, fontweight='bold')
    plt.xlabel('time s')
    plt.ylabel('reward')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=df['reward'].mean(), color='r', linestyle='--', alpha=0.7,
                label=f'avg reward: {df["reward"].mean():.3f}')
    plt.legend()

    # 2. Q值最大值变化
    plt.subplot(3, 3, 2)
    plt.plot(df['time_step'], df['q_value_max'], 'g-', linewidth=2, marker='s', markersize=4)
    plt.title('Q max difference', fontsize=14, fontweight='bold')
    plt.xlabel('time step')
    plt.ylabel('MaxQ')
    plt.grid(True, alpha=0.3)

    # 3. 动作分布
    plt.subplot(3, 3, 3)
    action_counts = df['action'].value_counts().sort_index()
    colors = plt.cm.Set3(np.linspace(0, 1, len(action_counts)))
    bars = plt.bar(range(len(action_counts)), action_counts.values, color=colors)
    plt.title('actions distribution', fontsize=14, fontweight='bold')
    plt.xlabel('action index')
    plt.ylabel('appearance count')
    plt.xticks(range(len(action_counts)), action_counts.index)
    # 在柱子上添加数值标签
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                 f'{int(height)}', ha='center', va='bottom')

    # 4. 设备开关状态随时间变化
    plt.subplot(3, 3, 4)
    devices = ['ac', 'heater', 'dehum', 'hum']
    colors = ['red', 'blue', 'green', 'orange']
    for i, device in enumerate(devices):
        plt.step(df['time_step'], df[device], where='post', label=device.upper(), linewidth=2, color=colors[i])
    plt.title('switch state ', fontsize=14, fontweight='bold')
    plt.xlabel('time step')
    plt.ylabel('state (0=open, 1=close)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.1, 1.1)

    # 5. 环境参数变化
    plt.subplot(3, 3, 5)
    if 'temp' in df.columns and 'humid' in df.columns and 'light' in df.columns:
        # 标准化显示
        temp_norm = (df['temp'] - df['temp'].min()) / (df['temp'].max() - df['temp'].min())
        humid_norm = (df['humid'] - df['humid'].min()) / (df['humid'].max() - df['humid'].min())
        light_norm = (df['light'] - df['light'].min()) / (df['light'].max() - df['light'].min())

        plt.plot(df['time_step'], temp_norm, 'r-', label='温度', linewidth=2)
        plt.plot(df['time_step'], humid_norm, 'b-', label='湿度', linewidth=2)
        plt.plot(df['time_step'], light_norm, 'g-', label='光照', linewidth=2)
        plt.title('env param variance (std)', fontsize=14, fontweight='bold')
        plt.xlabel('time step')
        plt.ylabel('standard deviation')
        plt.legend()
        plt.grid(True, alpha=0.3)

    # 6. 奖励分布直方图
    plt.subplot(3, 3, 6)
    plt.hist(df['reward'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title('reward histogram', fontsize=14, fontweight='bold')
    plt.xlabel('reward')
    plt.ylabel('frequency')
    plt.axvline(df['reward'].mean(), color='red', linestyle='--', linewidth=2, label=f'avg: {df["reward"].mean():.3f}')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 7. 动作与奖励的关系
    plt.subplot(3, 3, 7)
    action_reward = df.groupby('action')['reward'].mean()
    plt.bar(action_reward.index, action_reward.values, alpha=0.7, color='lightcoral')
    plt.title('per action reward', fontsize=14, fontweight='bold')
    plt.xlabel('action index')
    plt.ylabel('average reward')
    plt.grid(True, alpha=0.3)

    # 8. 累计奖励
    plt.subplot(3, 3, 8)
    cumulative_reward = df['reward'].cumsum()
    plt.plot(df['time_step'], cumulative_reward, 'purple', linewidth=3)
    plt.title('reward acc', fontsize=14, fontweight='bold')
    plt.xlabel('time step')
    plt.ylabel('reward acc')
    plt.grid(True, alpha=0.3)

    # 9. 相关性热力图
    plt.subplot(3, 3, 9)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlation = df[numeric_cols].corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title('heat map', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.show()

    # 打印详细统计信息
    print("\n" + "=" * 50)
    print("详细统计信息")
    print("=" * 50)

    print(f"\n总奖励: {df['reward'].sum():.3f}")
    print(f"平均奖励: {df['reward'].mean():.3f}")
    print(f"奖励标准差: {df['reward'].std():.3f}")
    print(f"最大奖励: {df['reward'].max():.3f}")
    print(f"最小奖励: {df['reward'].min():.3f}")

    print(f"\n平均Q值最大值: {df['q_value_max'].mean():.3f}")

    print(f"\n动作统计:")
    action_stats = df['action'].value_counts().sort_index()
    for action, count in action_stats.items():
        print(f"动作 {action}: {count}次 ({count / len(df) * 100:.1f}%)")

    print(f"\n设备使用统计:")
    for device in devices:
        if device in df.columns:
            usage = df[device].mean() * 100
            print(f"{device.upper()}: {usage:.1f}% 时间开启")

    # 保存图表
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"rollout_analysis_{timestamp}.png"
    fig.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"\n分析图表已保存为: {output_filename}")


def advanced_analysis(csv_file_path):
    """
    高级分析：包含时间序列分析和模式识别
    """
    df = pd.read_csv(csv_file_path)

    # 创建高级分析图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. 奖励的移动平均
    df['reward_ma'] = df['reward'].rolling(window=5, min_periods=1).mean()
    axes[0, 0].plot(df['time_step'], df['reward'], 'b-', alpha=0.5, label='start reward')
    axes[0, 0].plot(df['time_step'], df['reward_ma'], 'r-', linewidth=2, label='5 average reward')
    axes[0, 0].set_title('reward moving average')
    axes[0, 0].set_xlabel('step')
    axes[0, 0].set_ylabel('reward')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. 动作转换矩阵（简化版）
    action_changes = []
    for i in range(1, len(df)):
        if df['action'].iloc[i] != df['action'].iloc[i - 1]:
            action_changes.append(1)
        else:
            action_changes.append(0)

    axes[0, 1].plot(range(1, len(df)), action_changes, 'go-', markersize=4)
    axes[0, 1].set_title('action change point (1=v, 0=nv)')
    axes[0, 1].set_xlabel('time step')
    axes[0, 1].set_ylabel('action change')
    axes[0, 1].grid(True, alpha=0.3)

    # 3. 设备组合使用情况
    if all(col in df.columns for col in ['ac', 'heater', 'dehum', 'hum']):
        device_combinations = df.apply(lambda x: f"{int(x['ac'])}{int(x['heater'])}{int(x['dehum'])}{int(x['hum'])}",
                                       axis=1)
        combo_counts = device_combinations.value_counts().head(8)  # 显示前8种组合

        axes[1, 0].bar(range(len(combo_counts)), combo_counts.values)
        axes[1, 0].set_title('facility combinations')
        axes[1, 0].set_xlabel('facility combinations (AC,Heater,Dehum,Hum)')
        axes[1, 0].set_ylabel('occurance count')
        axes[1, 0].set_xticks(range(len(combo_counts)))
        axes[1, 0].set_xticklabels(combo_counts.index, rotation=45)

    # 4. 奖励与Q值的关系
    axes[1, 1].scatter(df['q_value_max'], df['reward'], alpha=0.6)
    axes[1, 1].set_title('Qmax vs Reward')
    axes[1, 1].set_xlabel('Qmax')
    axes[1, 1].set_ylabel('Reward')
    axes[1, 1].grid(True, alpha=0.3)

    # 计算相关系数
    correlation = df['q_value_max'].corr(df['reward'])
    axes[1, 1].text(0.05, 0.95, f'相关系数: {correlation:.3f}',
                    transform=axes[1, 1].transAxes, fontsize=12,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

    plt.tight_layout()
    plt.show()

    # 打印高级统计
    print("\n高级分析结果:")
    print(f"动作变化频率: {sum(action_changes) / len(action_changes) * 100:.1f}%")
    print(f"奖励与Q值的相关性: {correlation:.3f}")


# 使用高级分析
# advanced_analysis("rollout_results.csv")
# 使用示例
if __name__ == "__main__":
    # 替换为你的CSV文件路径
    csv_file = "rollout_results.csv"  # 或者 "detailed_rollout_results.csv"

    # 进行分析
    #analyze_rollout_results(csv_file)
    advanced_analysis(csv_file)
