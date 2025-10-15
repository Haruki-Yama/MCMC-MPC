import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
obs = pd.read_csv('mmg_4dof_wind_data/mmg_4dof_wind_U2.csv')
# obs = pd.read_csv('mmg_4dof_wind_data/mmg_4dof_wind_U4.csv')
# obs = pd.read_csv('mmg_4dof_wind_data/mmg_4dof_wind_U6.csv')
wind_force = pd.read_csv('wind_force_estimate_4dof/wind_force_estimate_4dof_U2.csv')
# wind_force = pd.read_csv('wind_force_estimate_4dof/wind_force_estimate_4dof_U4.csv')
# wind_force = pd.read_csv('wind_force_estimate_4dof/wind_force_estimate_4dof_U6.csv')

ρ_air = 1.225

# KCS model
L_pp = 3.057
B = 0.428  # 船幅[m]
d = 0.144  # 喫水[m]
D = 0.239 # 深さ[m]
A_OD = 0.598 # デッキ上の構造物の側面投影面積[m^2]
H_BR = 0.553 # 喫水からブリッジ主要構造物の最高位[m]

A_F = H_BR * B  # 船体の正面投影面積[m^2]
A_L = (D - d) * L_pp + A_OD # 船体の側面投影面積[m^2]

psi_A = obs['Ψ_A'].values
jump = np.where(np.diff(psi_A) < -100*np.pi/180)[0]  # -100 deg以上の急減を周期境界と仮定
mask = jump > 40  # 40以下のインデックスを除外
jump = jump[mask]  
if len(jump) >= 1:
    start_idx = jump[0] + 1
    end_idx = jump[1] if len(jump) > 1 else len(psi_A)  # 2周期目の始まり or 最後まで
else:
    raise ValueError("Ψ_A に周期的な折返しが見つかりませんでした。")

X_F_Body = wind_force['X_F_Body'].values
Y_F_Body = wind_force['Y_F_Body'].values
N_F_Body = wind_force['N_F_Body'].values
K_F_Body = wind_force['K_F_Body'].values

C_XW = X_F_Body / (0.5 * ρ_air * A_F * obs['U_A'][:400].values**2)
C_YW = Y_F_Body / (0.5 * ρ_air * A_L * obs['U_A'][:400].values**2)
C_NW = N_F_Body / (0.5 * ρ_air * A_L * L_pp * obs['U_A'][:400].values**2)
C_KW = K_F_Body / (0.5 * ρ_air * A_L**2 / L_pp * obs['U_A'][:400].values**2)

spl_C_XW = CubicSpline(obs['Ψ_A'][start_idx:end_idx], C_XW[start_idx:end_idx], extrapolate=True)
spl_C_YW = CubicSpline(obs['Ψ_A'][start_idx:end_idx], C_YW[start_idx:end_idx], extrapolate=True)
spl_C_NW = CubicSpline(obs['Ψ_A'][start_idx:end_idx], C_NW[start_idx:end_idx], extrapolate=True)
spl_C_KW = CubicSpline(obs['Ψ_A'][start_idx:end_idx], C_KW[start_idx:end_idx], extrapolate=True)