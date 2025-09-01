import os
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional


class PlantGrowthEnv:
    def __init__(self, seq_len: int = 10, noise_std: float = 1.0,
                 insect_prob: float = 0.5, equip_fail_prob: float = 0.5):
        """
        植物生长环境模拟器

        参数:
        - seq_len: 序列长度
        - noise_std: 噪声标准差
        - insect_prob: 虫害发生概率
        - equip_fail_prob: 设备故障概率
        """
        self.seq_len = seq_len
        self.noise_std = noise_std
        self.insect_prob = insect_prob
        self.equip_fail_prob = equip_fail_prob

        # 环境状态
        self.current_step = 0
        self.data = None
        self.insect_event = False
        self.equip_fail_event = False
        self.insect_start = -1
        self.insect_end = -1
        self.fail_type = None
        self.fail_start = -1
        self.fail_end = -1

        # 重置环境
        self.reset()

    def reset(self) -> Dict[str, float]:
        """重置环境到初始状态"""
        self.current_step = 0

        # 随机虫害事件
        self.insect_event = np.random.rand() < self.insect_prob
        self.insect_start = np.random.randint(300, 800) if self.insect_event else -1
        self.insect_end = self.insect_start + np.random.randint(50, 150) if self.insect_event else -1

        # 随机设备异常事件
        self.equip_fail_event = np.random.rand() < self.equip_fail_prob
        self.fail_type = None
        self.fail_start, self.fail_end = -1, -1

        if self.equip_fail_event:
            self.fail_type = np.random.choice(["humidifier_fail", "dehumidifier_fail", "heater_fail", "ac_fail"])
            self.fail_start = np.random.randint(200, 700)
            self.fail_end = self.fail_start + np.random.randint(80, 200)

        # 生成完整序列数据
        self.data = self._generate_data()

        return self._get_observation(0)

    def step(self, action=None):
        self.current_step += 1
        done = self.current_step >= self.seq_len

        observation = self._get_observation(self.current_step)
        label = self.data['label'].iloc[self.current_step] if not done else 0

        # 檢查 observation 是否有效
        if observation and 'temp' in observation:  # 檢查是否包含必要鍵
            reward = self._calculate_reward(observation, label, action)
        else:
            reward = 0  # 如果觀察值無效，給默認獎勵

        if action is not None:
            self._apply_action(action, observation)

        info = {'label': label, 'step': self.current_step}

        return observation, reward, done, info

    def _calculate_reward(self, observation, label, action):
        """
        計算獎勵值
        根據您的具體需求實現獎勵函數
        """
        # 示例獎勵函數：
        base_reward = 0

        # 根據健康狀態給予獎勵
        if label == 0:  # 健康
            base_reward = 1.0
        elif label == 1:  # 非植物狀態
            base_reward = -2.0
        elif label == 2:  # 不健康
            base_reward = -1.0

        # 根據環境條件調整獎勵
        temp = observation.get('temp', 25.0)
        humid = observation.get('humid', 60.0)
        light = observation.get('light', 300.0)

        # 理想範圍獎勵
        if 20 <= temp <= 28:
            base_reward += 0.2
        if 40 <= humid <= 70:
            base_reward += 0.2
        if 200 <= light <= 600:
            base_reward += 0.2

        # 懲罰頻繁動作切換（可選）
        # if action != self.last_action:
        #     base_reward -= 0.1

        return base_reward

    def _apply_action(self, action, observation):
            """
            根據動作調整環境狀態
            根據您的具體需求實現這個方法
            """
            # 示例：動作可能控制設備
            if action == 0:  # 開啟空調
                observation['temp'] = max(15, observation['temp'] - 2)
            elif action == 1:  # 開啟加熱器
                observation['temp'] = min(35, observation['temp'] + 2)
            elif action == 2:  # 開啟加濕器
                observation['humid'] = min(85, observation['humid'] + 5)
            elif action == 3:  # 開啟除濕器
                observation['humid'] = max(25, observation['humid'] - 5)

    def _get_observation(self, step: int) -> Dict[str, float]:
        """確保總是返回包含所有特徵的有效字典"""
        if step >= len(self.data) or step < 0:
            # 返回默認值而不是空字典
            return {
                'temp': 25.0,
                'humid': 60.0,
                'light': 300.0,
                'ac': 0,
                'heater': 0,
                'dehum': 0,
                'hum': 0,
                'label': 0
            }

        try:
            return {
                'temp': float(self.data['temp'].iloc[step]),
                'humid': float(self.data['humid'].iloc[step]),
                'light': float(self.data['light'].iloc[step]),
                'ac': int(self.data['ac'].iloc[step]),
                'heater': int(self.data['heater'].iloc[step]),
                'dehum': int(self.data['dehum'].iloc[step]),
                'hum': int(self.data['hum'].iloc[step]),
                'label': int(self.data['label'].iloc[step])
            }
        except (IndexError, KeyError):
            # 發生錯誤時返回默認值
            return {
                'temp': 25.0,
                'humid': 60.0,
                'light': 300.0,
                'ac': 0,
                'heater': 0,
                'dehum': 0,
                'hum': 0,
                'label': 0
            }
    def _generate_data(self) -> pd.DataFrame:
        """生成植物生长数据"""
        # 初始化数据列表
        temperature, humidity, light, labels = [], [], [], []
        ac, heater, dehum, humidifier = [], [], [], []

        # 设备状态翻转概率
        flip_prob = 0.05

        for step in range(self.seq_len):
            # ========= 生命周期阶段 =========
            if step < 200:  # 育苗期
                base_t, base_h, base_l = 22, 65, 250
            elif step < 600:  # 生长期
                base_t, base_h, base_l = 25, 58, 400
            else:  # 开花期
                base_t, base_h, base_l = 28, 48, 600

            # 基础波动 + 噪声
            ti = base_t + np.sin(step / 50) + np.random.randn() * self.noise_std
            hi = base_h + np.cos(step / 70) + np.random.randn() * self.noise_std
            li = base_l + np.sin(step / 100) * 20 + np.random.randn() * self.noise_std * 5

            # 初始化标签和HVAC状态
            label = 0  # 默认为健康
            ac_state, heater_state, dehum_state, hum_state = 0, 0, 0, 0

            # ========= 虫害事件 =========
            if self.insect_event and self.insect_start <= step <= self.insect_end:
                li *= np.random.uniform(0.6, 0.8)  # 光照下降
                hi += np.random.uniform(-5, 5)  # 湿度异常波动
                label = 2  # 标记为不健康

            # ========= 设备异常事件 =========
            if self.equip_fail_event and self.fail_start <= step <= self.fail_end:
                if self.fail_type == "humidifier_fail":
                    hum_state = 1  # 一直开
                    hi += np.random.uniform(5, 15)  # 湿度过重
                    label = 2
                elif self.fail_type == "dehumidifier_fail":
                    dehum_state = 1
                    hi -= np.random.uniform(5, 15)  # 湿度过低
                    label = 2
                elif self.fail_type == "heater_fail":
                    heater_state = 1
                    ti += np.random.uniform(5, 10)  # 过热
                    label = 2
                elif self.fail_type == "ac_fail":
                    ac_state = 1
                    ti -= np.random.uniform(5, 10)  # 过冷
                    label = 2

            # ========= 健康状态检查 =========
            if label == 0:
                if (ti < 10) or (li < 100):
                    label = 1  # 非植物状态
                elif (ti < 15) or (ti > 35) or (hi < 30) or (hi > 80) or (li > 800):
                    label = 2  # 不健康

            # ========= HVAC 正常控制逻辑 =========
            if not (self.equip_fail_event and self.fail_start <= step <= self.fail_end):
                ac_state = 1 if ti > 26 else 0
                heater_state = 1 if ti < 20 else 0
                dehum_state = 1 if hi > 70 else 0
                hum_state = 1 if hi < 40 else 0

            # ========= 随机扰动 =========
            if label == 2:
                if np.random.rand() < 0.15:
                    ac_state = 1 - ac_state
                if np.random.rand() < 0.15:
                    heater_state = 1 - heater_state
                if np.random.rand() < 0.15:
                    dehum_state = 1 - dehum_state
                if np.random.rand() < 0.15:
                    hum_state = 1 - hum_state
            else:
                ac_state = ac_state ^ (np.random.rand() < flip_prob)
                heater_state = heater_state ^ (np.random.rand() < flip_prob)
                hum_state = hum_state ^ (np.random.rand() < flip_prob)
                dehum_state = dehum_state ^ (np.random.rand() < flip_prob)

            # 存储数据
            temperature.append(ti)
            humidity.append(hi)
            light.append(li)
            labels.append(label)
            ac.append(ac_state)
            heater.append(heater_state)
            dehum.append(dehum_state)
            humidifier.append(hum_state)

        # 创建DataFrame
        df = pd.DataFrame({
            "temp": temperature,
            "humid": humidity,
            "light": light,
            "ac": ac,
            "heater": heater,
            "dehum": dehum,
            "hum": humidifier,
            "label": labels
        })

        return df

    def get_event_info(self) -> Dict[str, any]:
        """获取当前环境的事件信息"""
        return {
            'insect_event': self.insect_event,
            'insect_start': self.insect_start,
            'insect_end': self.insect_end,
            'equip_fail_event': self.equip_fail_event,
            'fail_type': self.fail_type,
            'fail_start': self.fail_start,
            'fail_end': self.fail_end
        }


# 使用示例
if __name__ == "__main__":
    # 创建环境
    env = PlantGrowthEnv(seq_len=1000, noise_std=1.0)

    # 重置环境
    obs = env.reset()
    print("初始观测:", obs)

    # 获取事件信息
    events = env.get_event_info()
    print("环境事件:", events)

    # 运行环境
    done = False
    while not done:
        obs, label, done = env.step()
        if label == 2:  # 检测到异常
            print(f"步骤 {env.current_step}: 检测到异常 - {obs}")

    print("模拟结束")