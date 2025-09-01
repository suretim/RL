import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime

from sklearn.preprocessing import StandardScaler




def check_value_estimation(qmodel, encoder, files, num_samples=100):
    """
    检查值函数估计的质量
    """
    print("\n" + "=" * 60)
    print("📊 值函数估计质量检查")
    print("=" * 60)

    results = {
        'bellman_errors': [],
        'td_errors': [],
        'q_value_consistency': [],
        'value_drift': [],
        'estimation_bias': []
    }

    # 1. 检查Bellman一致性
    print("1. 🔄 Bellman一致性检查:")
    bellman_results = check_bellman_consistency(qmodel, encoder, files, num_samples)
    results['bellman_errors'].extend(bellman_results['errors'])

    # 2. 检查Q值稳定性
    print("\n2. 📈 Q值稳定性检查:")
    stability_results = check_q_value_stability(qmodel, encoder, files)
    results['q_value_consistency'].extend(stability_results['consistency_scores'])

    # 3. 检查值漂移
    print("\n3. 🎯 值漂移检查:")
    drift_results = check_value_drift(qmodel, encoder, files)
    results['value_drift'].extend(drift_results['drift_values'])

    # 4. 检查估计偏差
    print("\n4. ⚖️ 估计偏差检查:")
    bias_results = check_estimation_bias(qmodel, encoder, files)
    results['estimation_bias'].extend(bias_results['bias_scores'])

    # 5. 综合分析
    print("\n5. 📋 综合分析结果:")
    analyze_estimation_quality(results)

    return results
def standardize_row(row):
    mean = np.mean(row)
    std = np.std(row)
    if std == 0:
        return np.zeros_like(row) # Avoid division by zero
    return (row - mean) / std

def normalize_row(row):
    norm = np.linalg.norm(row) # L2 norm by default
    if norm == 0:
        return row # Avoid division by zero
    return row / norm
def check_bellman_consistency(qmodel, encoder, files, num_samples=50):
    """
    检查Bellman方程的一致性
    """
    print("  计算Bellman误差...")

    errors = []
    td_errors = []

    # 随机选择文件和数据点
    for _ in range(num_samples):
        df = pd.read_csv(np.random.choice(files))
        if len(df) < 12:  # 需要至少12个点（当前+下一个+10窗口）
            continue

        # 随机选择起始点
        start_idx = np.random.randint(0, len(df) - 11)

        # 构建当前状态（10个时间步的窗口）
        current_states = []
        for i in range(10):
            row = df.iloc[start_idx + i][["temp", "humid", "light", "ac", "heater", "dehum", "hum"]].values
            current_states.append(normalize_row(row))
        current_state = np.array(current_states)

        # 编码当前状态
        current_encoded = encoder(current_state[np.newaxis, :, :]).numpy()[0]

        # 获取当前Q值
        current_q_values = qmodel.q_net.predict(current_encoded.reshape(1, -1), verbose=0)[0]
        current_action = np.argmax(current_q_values)
        current_max_q = np.max(current_q_values)

        # 获取下一个状态
        next_idx = start_idx + 10
        next_row = df.iloc[next_idx][["temp", "humid", "light", "ac", "heater", "dehum", "hum"]].values
        next_row = normalize_row(next_row)

        # 用当前动作替换开关状态
        action_bits = qmodel.action_int_to_bits(current_action)
        next_row[3:7] = action_bits

        # 构建下一个状态窗口（移除最旧，添加最新）
        next_states = np.vstack([current_state[1:], next_row])
        next_encoded = encoder(next_states[np.newaxis, :, :]).numpy()[0]

        # 获取下一个状态的Q值
        next_q_values = qmodel.q_net.predict(next_encoded.reshape(1, -1), verbose=0)[0]
        next_max_q = np.max(next_q_values)

        # 计算奖励
        label = df.iloc[next_idx]["label"] if "label" in df.columns else 0
        reward = qmodel.compute_reward_batch(np.array([label]), np.array([action_bits]))[0]

        # 计算Bellman误差
        bellman_estimate = reward + qmodel.gamma * next_max_q
        bellman_error = abs(current_max_q - bellman_estimate)
        td_error = current_max_q - bellman_estimate

        errors.append(bellman_error)
        td_errors.append(td_error)

    print(f"  平均Bellman误差: {np.mean(errors):.4f} ± {np.std(errors):.4f}")
    print(f"  平均TD误差: {np.mean(td_errors):.4f} ± {np.std(td_errors):.4f}")

    return {'errors': errors, 'td_errors': td_errors}


def check_q_value_stability(qmodel, encoder, files, num_tests=20):
    """
    检查Q值的稳定性（多次估计的一致性）
    """
    print("  检查Q值估计稳定性...")

    consistency_scores = []

    for _ in range(num_tests):
        df = pd.read_csv(np.random.choice(files))
        if len(df) < 10:
            continue

        # 选择随机状态
        start_idx = np.random.randint(0, len(df) - 9)
        states = []
        for i in range(10):
            row = df.iloc[start_idx + i][["temp", "humid", "light", "ac", "heater", "dehum", "hum"]].values
            states.append(normalize_row(row))

        state = np.array(states)
        state_encoded = encoder(state[np.newaxis, :, :]).numpy()[0]

        # 多次预测同一个状态，检查一致性
        predictions = []
        for _ in range(5):  # 预测5次
            q_values = qmodel.q_net.predict(state_encoded.reshape(1, -1), verbose=0)[0]
            predictions.append(q_values)

        # 计算预测之间的标准差
        pred_array = np.array(predictions)
        consistency = np.mean(np.std(pred_array, axis=0))  # 平均标准差
        consistency_scores.append(consistency)

    avg_consistency = np.mean(consistency_scores)
    print(f"  Q值估计稳定性: {avg_consistency:.6f}")

    if avg_consistency > 0.1:
        print("  ⚠️  警告: Q值估计不稳定")
    else:
        print("  ✅ Q值估计稳定")

    return {'consistency_scores': consistency_scores}


def check_value_drift(qmodel, encoder, files, num_sequences=10):
    """
    检查值函数漂移（随时间的变化）
    """
    print("  检查值函数漂移...")

    drift_values = []

    for _ in range(num_sequences):
        df = pd.read_csv(np.random.choice(files))
        if len(df) < 50:
            continue

        # 跟踪一个序列中的Q值变化
        q_values_over_time = []

        # 创建初始状态窗口
        states = []
        for i in range(10):
            row = df.iloc[i][["temp", "humid", "light", "ac", "heater", "dehum", "hum"]].values
            states.append(normalize_row(row))

        for t in range(10, min(50, len(df))):
            # 编码当前状态
            current_state = np.array(states)
            state_encoded = encoder(current_state[np.newaxis, :, :]).numpy()[0]

            # 获取Q值
            q_values = qmodel.q_net.predict(state_encoded.reshape(1, -1), verbose=0)[0]
            max_q = np.max(q_values)
            q_values_over_time.append(max_q)

            # 选择动作并更新状态
            action = np.argmax(q_values)
            action_bits = qmodel.action_int_to_bits(action)

            # 获取下一个观测
            next_row = df.iloc[t][["temp", "humid", "light", "ac", "heater", "dehum", "hum"]].values
            next_row = normalize_row(next_row)
            next_row[3:7] = action_bits

            # 更新状态窗口
            states = states[1:] + [next_row]

        # 计算漂移（开始和结束的差异）
        if len(q_values_over_time) > 1:
            drift = abs(q_values_over_time[-1] - q_values_over_time[0])
            drift_values.append(drift)

    avg_drift = np.mean(drift_values) if drift_values else 0
    print(f"  平均值漂移: {avg_drift:.4f}")

    return {'drift_values': drift_values}


def check_estimation_bias(qmodel, encoder, files, num_comparisons=30):
    """
    检查估计偏差（与理想值的比较）
    """
    print("  检查估计偏差...")

    bias_scores = []

    # 这里我们需要一些"理想"的参考值
    # 由于没有真实值，我们使用一些启发式方法

    for _ in range(num_comparisons):
        df = pd.read_csv(np.random.choice(files))
        if len(df) < 20:
            continue

        # 选择状态
        idx = np.random.randint(10, len(df) - 10)
        states = []
        for i in range(10):
            row = df.iloc[idx - 9 + i][["temp", "humid", "light", "ac", "heater", "dehum", "hum"]].values
            states.append(normalize_row(row))

        state = np.array(states)
        state_encoded = encoder(state[np.newaxis, :, :]).numpy()[0]

        # 模型预测的Q值
        model_q_values = qmodel.q_net.predict(state_encoded.reshape(1, -1), verbose=0)[0]
        model_max_q = np.max(model_q_values)

        # 启发式参考值（基于状态特征）
        # 这里使用简单的启发式：温度、湿度越接近0.5，价值越高
        avg_temp = np.mean(state[:, 0])
        avg_humid = np.mean(state[:, 1])

        # 理想值（0.5是最佳状态）
        ideal_value = 1.0 - (abs(avg_temp - 0.5) + abs(avg_humid - 0.5)) / 2.0

        # 计算偏差
        bias = abs(model_max_q - ideal_value)
        bias_scores.append(bias)

    avg_bias = np.mean(bias_scores)
    print(f"  平均估计偏差: {avg_bias:.4f}")

    if avg_bias > 0.3:
        print("  ⚠️  警告: 估计偏差较大")
    elif avg_bias > 0.1:
        print("  ℹ️  估计偏差中等")
    else:
        print("  ✅ 估计偏差较小")

    return {'bias_scores': bias_scores}


def analyze_estimation_quality(results):
    """
    综合分析值函数估计质量
    """
    bellman_errors = results['bellman_errors']
    consistency_scores = results['q_value_consistency']
    drift_values = results['value_drift']
    bias_scores = results['estimation_bias']

    print("  📊 值函数估计质量报告:")
    print("  " + "-" * 40)

    if bellman_errors:
        avg_bellman = np.mean(bellman_errors)
        print(f"  Bellman一致性: {avg_bellman:.4f}")
        if avg_bellman > 0.5:
            print("    ❌ Bellman误差过大 - 值函数估计不准确")
        elif avg_bellman > 0.2:
            print("    ⚠️  Bellman误差中等 - 需要改进")
        else:
            print("    ✅ Bellman一致性良好")

    if consistency_scores:
        avg_consistency = np.mean(consistency_scores)
        print(f"  估计稳定性: {avg_consistency:.6f}")
        if avg_consistency > 0.05:
            print("    ❌ 估计不稳定 - 网络训练有问题")
        elif avg_consistency > 0.01:
            print("    ⚠️  稳定性一般 - 可能有波动")
        else:
            print("    ✅ 估计稳定")

    if drift_values:
        avg_drift = np.mean(drift_values)
        print(f"  值漂移: {avg_drift:.4f}")
        if avg_drift > 0.3:
            print("    ❌ 值漂移严重 - 学习过程不稳定")
        elif avg_drift > 0.1:
            print("    ⚠️  中等值漂移")
        else:
            print("    ✅ 值漂移控制良好")

    if bias_scores:
        avg_bias = np.mean(bias_scores)
        print(f"  估计偏差: {avg_bias:.4f}")
        if avg_bias > 0.3:
            print("    ❌ 估计偏差过大")
        elif avg_bias > 0.1:
            print("    ⚠️  中等估计偏差")
        else:
            print("    ✅ 估计偏差较小")

    # 总体评估
    overall_score = 0
    max_score = 4

    if bellman_errors and np.mean(bellman_errors) < 0.2:
        overall_score += 1
    if consistency_scores and np.mean(consistency_scores) < 0.01:
        overall_score += 1
    if drift_values and np.mean(drift_values) < 0.1:
        overall_score += 1
    if bias_scores and np.mean(bias_scores) < 0.1:
        overall_score += 1

    quality_rating = overall_score / max_score
    print(f"  🎯 总体质量评分: {quality_rating:.1%}")

    if quality_rating >= 0.75:
        print("    ✅ 值函数估计质量优秀")
    elif quality_rating >= 0.5:
        print("    ⚠️  值函数估计质量一般")
    else:
        print("    ❌ 值函数估计质量较差")


# 使用示例
def run_value_estimation_check(qmodel, encoder, files):
    """
    运行值函数估计检查
    """
    # 假设你已经有了这些组件
    # qmodel = YourDQNAgent(...)
    # encoder = YourEncoder(...)
    # files = ["data1.csv", "data2.csv"]

    print("开始值函数估计检查...")
    results = check_value_estimation(qmodel, encoder, files, num_samples=50)

    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_df = pd.DataFrame({
        'bellman_errors': results['bellman_errors'],
        'consistency_scores': results['q_value_consistency'],
        'value_drift': results['value_drift'],
        'estimation_bias': results['estimation_bias']
    })
    results_df.to_csv(f'value_estimation_analysis_{timestamp}.csv', index=False)

    return results

# 运行检查
# value_estimation_results = run_value_estimation_check()