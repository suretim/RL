"""
PlantNannyEnv - a Growlink-style Gym environment for moisture-based irrigation control.

Features:
- Discrete action space (0: no irrigation, 1: small, 2: medium, 3: large)
- Continuous observation: [moisture (0-1), time_of_day (0-24), phase_index (0..P-1), cumulative_growth]
- Phase-based schedule (P1/P2/P3) configurable with different target VWC ranges
- Simple physical dynamics: irrigation increases moisture, evaporation decreases it
- Rewards: survival reward for staying within target band, milestone growth rewards, penalties for over/under watering
- Simple `render()` (text) and optional matplotlib plotting helper

Usage:
- Save as `plantnanny_env.py` and import into your RL training script
- Example usage included at bottom (random agent). If you want to use SB3/PPO, see commented snippet.

This implementation uses classic `gym` API. Tested with gym 0.26+ (should also work with older gym versions).
"""

import math
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class PlantNannyEnv(gym.Env):
    """A simplified Growlink-style irrigation environment.

    Observation (float32 vector):
      0: substrate moisture (0.0 - 1.0)
      1: time_of_day (0.0 - 24.0)
      2: phase_index (0 - n_phases-1)
      3: cumulative_growth (>=0, normalized)

    Action (Discrete):
      0: no irrigation
      1: small irrigation
      2: medium irrigation
      3: large irrigation

    Episode: sequence of time steps (minutes) for a configurable number of simulated days.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, config=None):
        super().__init__()
        # default configuration
        cfg = {
            "dt_minutes": 20,  # time step length in minutes
            "episode_days": 2,
            "initial_moisture": 0.45,
            "min_moisture": 0.0,
            "max_moisture": 1.0,
            # irrigation amounts per action (normalized moisture units)
            "irrigation_amounts": [0.0, 0.06, 0.12, 0.2],
            # evaporation base rate per dt (moisture units)
            "evap_base": 0.01,
            # diurnal evaporation multiplier (peaks midday)
            "evap_amp": 0.02,
            # phases definition: tuple(start_hour, end_hour, target_min, target_max)
            # P1: ramp up, P2: maintenance, P3: dry down
            "phases": [
                (6, 9, 0.5, 0.8),   # P1
                (9, 13, 0.55, 0.65), # P2
                (13, 24, 0.35, 0.6), # P3
            ],
            # reward coefficients
            "survival_reward": 0.1,
            "milestone_reward": 5.0,
            "overwater_penalty": -1.0,
            "underwater_penalty": -0.5,
            # milestone growth threshold (cumulative growth to trigger reward)
            "growth_milestone": 0.25,
        }
        if config:
            cfg.update(config)
        self.cfg = cfg

        # Derived
        self.steps_per_day = int((24 * 60) / self.cfg["dt_minutes"])  # integer
        self.max_steps = self.steps_per_day * self.cfg["episode_days"]
        self.n_phases = len(self.cfg["phases"])

        # Spaces
        obs_low = np.array([
            self.cfg["min_moisture"],  # moisture
            0.0,  # time
            0.0,  # phase
            0.0,  # cumulative growth
        ], dtype=np.float32)
        obs_high = np.array([
            self.cfg["max_moisture"],
            24.0,
            float(self.n_phases - 1),
            1.0,  # normalize cumulative growth to [0,1] for observation
        ], dtype=np.float32)

        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        self.action_space = spaces.Discrete(len(self.cfg["irrigation_amounts"]))

        # state
        self.state = None
        self.current_step = 0
        self._seed = None
        # bookkeeping
        self._growth_accum = 0
        self._milestone_achieved = False
        self.history = {"moisture": [], "time": [], "reward": []}

        self.reset()

    def seed(self, seed=None):
        self._seed = seed
        np.random.seed(seed)

    def _get_time_of_day(self):
        # returns hour of day based on current_step
        minutes = (self.current_step * self.cfg["dt_minutes"]) % (24 * 60)
        return minutes / 60.0

    def _get_phase_index(self, hour):
        for i, (s, e, _, _) in enumerate(self.cfg["phases"]):
            if s <= hour < e:
                return i
        return self.n_phases - 1

    def _evaporation(self, hour):
        # simple diurnal model: evap_base + evap_amp * gaussian around midday (13:00)
        midday = 13.0
        sigma = 3.0
        diurnal = math.exp(-0.5 * ((hour - midday) / sigma) ** 2)
        evap = self.cfg["evap_base"] + self.cfg["evap_amp"] * diurnal
        return evap

    def xreset(self):
        self.current_step = 0
        moisture = float(self.cfg["initial_moisture"]) + np.random.randn() * 0.01
        moisture = np.clip(moisture, self.cfg["min_moisture"], self.cfg["max_moisture"])
        hour = self._get_time_of_day()
        phase = float(self._get_phase_index(hour))
        self._growth_accum = 0.0
        self._milestone_achieved = False
        self.state = np.array([moisture, hour, phase, 0.0], dtype=np.float32)
        self.history = {"moisture": [moisture], "time": [hour], "reward": []}
        return self.state.copy()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.array([
            np.random.uniform(20, 30),  # 溫度
            np.random.uniform(40, 60),  # 濕度
            np.random.uniform(400, 600),  # 光照
            0.5  # 健康度
        ], dtype=np.float32)
        self.step_count = 0
        return self.state, {}

    def xstep(self, action):
        temp, hum, light, health = self.state
        water_action, light_action = action

        # 模擬植物的物理反應
        temp += np.random.normal(0, 0.3)
        hum += water_action * 5.0 + np.random.normal(0, 1.0)
        light += light_action * 50.0 + np.random.normal(0, 10.0)

        # 限制範圍
        hum = np.clip(hum, 0, 100)
        light = np.clip(light, 0, 1000)

        # 健康度根據最適區域計算
        optimal_temp = 26
        optimal_hum = 60
        optimal_light = 700

        health -= 0.01  # 隨時間衰退
        penalty = (
                abs(temp - optimal_temp) * 0.01 +
                abs(hum - optimal_hum) * 0.005 +
                abs(light - optimal_light) * 0.0005
        )
        health = np.clip(health - penalty, 0, 1)

        # reward: 維持高健康度
        reward = health - penalty

        self.state = np.array([temp, hum, light, health], dtype=np.float32)
        self.step_count += 1
        terminated = health <= 0.1
        truncated = self.step_count >= self.max_steps

        return self.state, reward, terminated, truncated, {}
    def step(self, action):
        assert self.action_space.contains(action), f"invalid action: {action}"

        moisture, hour, phase, growth_norm = self.state

        # apply irrigation
        irrig = float(self.cfg["irrigation_amounts"][action])
        # random absorption efficiency
        eff = 0.9 + 0.1 * (np.random.rand() - 0.5)
        moisture += irrig * eff

        # evaporation
        evap = self._evaporation(hour)
        moisture -= evap

        # clamp
        moisture = float(np.clip(moisture, self.cfg["min_moisture"], self.cfg["max_moisture"]))

        # update time
        self.current_step += 1
        next_hour = self._get_time_of_day()
        next_phase = float(self._get_phase_index(next_hour))

        # growth model: if moisture inside phase target band, plant accumulates growth
        p_min, p_max = self._get_phase_target_band(next_phase)
        in_band = (p_min <= moisture <= p_max)
        # growth increment proportional to how close to center of band
        center = 0.5 * (p_min + p_max)
        growth_inc = 0.0
        if in_band:
            # normalized closeness
            closeness = 1.0 - abs(moisture - center) / (p_max - p_min + 1e-6)
            growth_inc = 0.01 * closeness  # per-step growth contribution
        else:
            # penalty if too dry/wet reduces growth
            growth_inc = -0.005

        self._growth_accum = max(0.0, self._growth_accum + growth_inc)
        # normalize growth for observation (simple scaling)
        growth_norm = float(np.clip(self._growth_accum / 1.0, 0.0, 1.0))

        # reward design
        reward = 0.0
        # survival reward: small positive if inside target band
        if in_band:
            reward += self.cfg["survival_reward"]
        # over/under watering
        if moisture > p_max:
            reward += self.cfg["overwater_penalty"]
        if moisture < p_min:
            reward += self.cfg["underwater_penalty"]

        # milestone reward
        if (not self._milestone_achieved) and (self._growth_accum >= self.cfg["growth_milestone"]):
            reward += self.cfg["milestone_reward"]
            self._milestone_achieved = True

        # done condition
        done = False
        if self.current_step >= self.max_steps:
            done = True
        # optional early termination if plant dies (moisture extreme for long time)
        if moisture <= 0.02 or moisture >= 0.99:
            # heavy penalty and end
            reward -= 5.0
            done = True

        # next state
        self.state = np.array([moisture, next_hour, next_phase, growth_norm], dtype=np.float32)

        # bookkeeping
        self.history["moisture"].append(moisture)
        self.history["time"].append(next_hour)
        self.history["reward"].append(reward)

        info = {
            "growth": self._growth_accum,
            "phase": int(next_phase),
            "in_band": bool(in_band),
            "target_band": (p_min, p_max),
        }

        return self.state.copy(), float(reward), bool(done), info

    def _get_phase_target_band(self, phase_index):
        # returns (min, max) for given phase index
        i = int(phase_index)
        _, _, tmin, tmax = self.cfg["phases"][i]
        return float(tmin), float(tmax)

    def render(self, mode="human"):
        moisture, hour, phase, growth_norm = self.state
        pmin, pmax = self._get_phase_target_band(int(phase))
        s = (
            f"Step:{self.current_step} Hour:{hour:.2f} Phase:{int(phase)} Moisture:{moisture:.3f} "
            f"Target:[{pmin:.2f}-{pmax:.2f}] Growth:{self._growth_accum:.3f}"
        )
        print(s)

    def close(self):
        pass


# --------------------------- Example usage ---------------------------
# If you save this file as `plantnanny_env.py`, you can import the class:
# from plantnanny_env import PlantNannyEnv
# env = PlantNannyEnv()
# obs = env.reset()
# for _ in range(100):
#     action = env.action_space.sample()
#     obs, reward, done, info = env.step(action)
#     env.render()
#     if done:
#         break
# env.close()

# --------------------------- Optional SB3 snippet -----------------------
# To train with Stable Baselines3 PPO (if installed):
#
# from stable_baselines3 import PPO
# env = PlantNannyEnv()
# model = PPO('MlpPolicy', env, verbose=1)
# model.learn(total_timesteps=20000)
# obs = env.reset()
# for _ in range(200):
#     action, _states = model.predict(obs)
#     obs, reward, done, info = env.step(action)
#     env.render()
#     if done:
#         break

# --------------------------- End of file -----------------------------
