import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
obs = pd.read_csv('mmg_4dof_wind.csv')
wind_force = pd.read_csv('wind_force_estimate.csv')

ρ_air = 1.225
# KVLCC2 L7 model
L_pp = 7.00  # 船長Lpp[m]
B = 1.27  # 船幅[m]
d = 0.46  # 喫水[m]
D = 0.6563 # 深さ[m]

# KCS model
L_pp = 3.057
B = 0.428  # 船幅[m]
d = 0.144  # 喫水[m]
D = 0.286 # 深さ[m]

A_F = (D - d) * B  # 船体の正面投影面積[m^2]
A_L = (D - d) * L_pp # 船体の側面投影面積[m^2]

X_F_Body = wind_force['X_F'].values
Y_F_Body = wind_force['Y_F'].values
N_F_Body = wind_force['N_F'].values

C_XW = X_F_Body / (0.5 * ρ_air * A_F * obs['U_A'][:400].values**2)
C_YW = Y_F_Body / (0.5 * ρ_air * A_L * obs['U_A'][:400].values**2)
C_NW = N_F_Body / (0.5 * ρ_air * A_L * L_pp * obs['U_A'][:400].values**2)

spl_C_XW = CubicSpline(obs['Ψ_A'][136:399], C_XW[136:399], extrapolate=True)
spl_C_YW = CubicSpline(obs['Ψ_A'][136:399], C_YW[136:399], extrapolate=True)
spl_C_NW = CubicSpline(obs['Ψ_A'][136:399], C_NW[136:399], extrapolate=True)