# SimplePlantEnv: a small Gymnasium env modeling soil moisture, transpiration, water potential and heat stress.
# Copy this file locally and run after installing gymnasium:
#   pip install gymnasium numpy

import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class SimplePlantEnv(gym.Env):
    """
    Lightweight plant world model for water-potential, transpiration, and heat-stress.
    Observation vector:
      [soil_moisture, leaf_temp, air_temp, rel_humidity, vpd_kpa, water_potential_mpa, transpiration_mm_day]
    Action vector (continuous, 3 dims):
      [irrigation_frac (0..1), shade (-1..1), airflow (-1..1)]
    """
    metadata = {"render_modes": ["human"], "render_fps": 10}
    def __init__(self,
                 max_irrigation_mm = 5.0,
                 soil_depth_mm = 200.0,
                 evap_coef = 4.0,
                 dt_days = 1.0/24.0,
                 seed=None):
        super().__init__()
        self.max_irrigation_mm = float(max_irrigation_mm)
        self.soil_depth_mm = float(soil_depth_mm)
        self.evap_coef = float(evap_coef)
        self.dt_days = float(dt_days)
        low = np.array([0.0, -10.0, -10.0, 0.0, 0.0, -5.0, 0.0], dtype=np.float32)
        high= np.array([1.0, 60.0, 60.0, 1.0, 5.0, 1.0, 50.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = spaces.Box(low=np.array([0.0, -1.0, -1.0], dtype=np.float32),
                                       high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
                                       dtype=np.float32)
        self.rng = np.random.default_rng(seed)
        self._init_state()
    def _init_state(self):
        self.soil_moisture = 0.5
        self.air_temp = 25.0
        self.leaf_temp = 25.0
        self.rel_humidity = 0.6
        self.time_step = 0
        # plant params
        self.psi_dry = -2.5
        self.psi_wet  = -0.1
        self.temp_opt = 25.0
        self.temp_tol = 6.0
        self.LAI = 1.0
    def _saturated_vapor_pressure_kpa(self, T_c):
        return 0.6108 * math.exp(17.27 * T_c / (T_c + 237.3))
    def _vpd_kpa(self, T_c, rh):
        es = self._saturated_vapor_pressure_kpa(T_c)
        return max(0.0, es * (1.0 - rh))
    def _water_potential_mpa(self, soil_moisture):
        sm = np.clip(soil_moisture, 0.0, 1.0)
        x = (sm - 0.2) / 0.15
        sig = 1.0 / (1.0 + math.exp(-x))
        psi = self.psi_dry + (self.psi_wet - self.psi_dry) * sig
        return float(psi)
    def _transpiration_mm_per_day(self, vpd_kpa, shade_factor):
        light_multiplier = 1.0 + 0.7 * np.clip(1.0 - shade_factor, 0.0, 1.0)
        T = self.evap_coef * vpd_kpa * self.LAI * light_multiplier
        return float(max(0.0, T))
    def step(self, action):
        action = np.array(action, dtype=np.float32)
        irrigation_frac = float(np.clip(action[0], 0.0, 1.0))
        shade = float(np.clip(action[1], -1.0, 1.0))
        airflow = float(np.clip(action[2], -1.0, 1.0))
        added_mm = irrigation_frac * self.max_irrigation_mm
        delta_sm_from_irrig = added_mm / self.soil_depth_mm
        vpd = self._vpd_kpa(self.leaf_temp, self.rel_humidity)
        transpiration_day = self._transpiration_mm_per_day(vpd, shade)
        transpiration_step_mm = transpiration_day * self.dt_days
        delta_sm_from_trans = transpiration_step_mm / self.soil_depth_mm
        leakage_coeff = 0.001
        leakage_mm = leakage_coeff * max(0.0, self.soil_moisture - 0.2) * self.soil_depth_mm * self.dt_days
        delta_sm_from_leak = leakage_mm / self.soil_depth_mm
        sm = self.soil_moisture + delta_sm_from_irrig - delta_sm_from_trans - delta_sm_from_leak
        sm += self.rng.normal(0.0, 0.001)
        sm = float(np.clip(sm, 0.0, 1.0))
        psi = self._water_potential_mpa(sm)
        self.air_temp += self.rng.normal(0.0, 0.05)
        self.rel_humidity = float(np.clip(self.rel_humidity + 0.01 * irrigation_frac - 0.002 * (airflow+0.0) + self.rng.normal(0.0,0.002), 0.0, 1.0))
        cooling_per_mm = 0.5
        cooling_effect = cooling_per_mm * transpiration_day * self.dt_days
        leaf_temp = self.leaf_temp + 0.1 * (self.air_temp - self.leaf_temp) - cooling_effect
        sun_factor = np.clip(shade, -1.0, 1.0)
        heating = 0.2 * max(0.0, sun_factor) * self.dt_days * 24.0
        leaf_temp += heating
        leaf_temp += self.rng.normal(0.0, 0.02)
        leaf_temp = float(np.clip(leaf_temp, -10.0, 60.0))
        vpd = self._vpd_kpa(leaf_temp, self.rel_humidity)
        temp_excess = max(0.0, leaf_temp - (self.temp_opt + self.temp_tol))
        heat_stress = float(temp_excess / max(1e-6, self.temp_tol))
        psi_target = -0.5
        psi_penalty = (psi - psi_target)**2
        water_penalty = (added_mm / max(1.0, self.max_irrigation_mm))**1.5
        reward = - (psi_penalty * 2.0 + water_penalty * 1.5 + heat_stress * 3.0)
        self.soil_moisture = sm
        self.leaf_temp = leaf_temp
        obs = np.array([self.soil_moisture, self.leaf_temp, self.air_temp, self.rel_humidity, vpd, psi, transpiration_day], dtype=np.float32)
        self.time_step += 1
        done = bool(psi < -3.8)
        info = {"added_mm": added_mm, "transpiration_mm_day": transpiration_day, "vpd_kpa": vpd, "water_potential_mpa": psi, "heat_stress": heat_stress}
        return obs, float(reward), done, False, info
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._init_state()
        self.soil_moisture = float(np.clip(0.45 + self.rng.normal(0.0, 0.05), 0.05, 0.9))
        self.air_temp = float(20.0 + self.rng.normal(0.0, 3.0))
        self.leaf_temp = float(self.air_temp + self.rng.normal(0.0, 0.5))
        self.rel_humidity = float(np.clip(0.6 + self.rng.normal(0.0, 0.05), 0.2, 0.95))
        obs = np.array([self.soil_moisture, self.leaf_temp, self.air_temp, self.rel_humidity, self._vpd_kpa(self.leaf_temp, self.rel_humidity), self._water_potential_mpa(self.soil_moisture), 0.0], dtype=np.float32)
        return obs, {}
    def render(self, mode="human"):
        txt = (f"t={self.time_step} sm={self.soil_moisture:.3f} leafT={self.leaf_temp:.2f}C airT={self.air_temp:.2f}C "
               f"RH={self.rel_humidity:.2f} vpd={self._vpd_kpa(self.leaf_temp,self.rel_humidity):.3f}kPa "
               f"psi={self._water_potential_mpa(self.soil_moisture):.2f}MPa")
        print(txt)
    def close(self):
        pass

# Example usage:
env = SimplePlantEnv(seed=0)
obs, _ = env.reset(seed=0)
action = np.array([0.2, 0.5, 0.0])   # small irrigation, moderate light, neutral airflow
obs, reward, done, truncated, info = env.step(action)
env.render()
