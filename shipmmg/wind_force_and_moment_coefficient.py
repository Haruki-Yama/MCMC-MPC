import numpy as np
from scipy.interpolate import CubicSpline



def wind_force_and_moment_coefficients(ψ_A, L_pp, B, A_OD, A_F, A_L, H_BR, H_C, C):
    # 共通係数
    C_CF = 0.404 + 0.368 * A_F / (B * H_BR) + 0.902 * H_BR / L_pp

    if np.deg2rad(0) <= ψ_A <= np.deg2rad(90):
        C_LF = -0.992 + 0.507 * A_L / (L_pp * B) + 1.162 * C / L_pp
        C_XLI = 0.458 + 3.245 * A_L / (L_pp * H_BR) - 2.313 * A_F / (B * H_BR)
        C_ALF = -0.585 - 0.906 * A_OD / A_L + 3.239 * B / L_pp
        C_YLI = np.pi * A_L / L_pp**2 + 0.116 + 3.345 * A_F / (L_pp * B)

        C_X = C_LF * np.cos(ψ_A) + C_XLI * (np.sin(ψ_A) - np.sin(ψ_A) * np.cos(ψ_A)**2 / 2) * np.sin(ψ_A) * np.cos(ψ_A) + C_ALF * np.sin(ψ_A) * np.cos(ψ_A)**3
        C_Y = C_CF * np.sin(ψ_A)**2 + C_YLI * (np.cos(ψ_A) + np.sin(ψ_A)**2 * np.cos(ψ_A) / 2) * np.sin(ψ_A) * np.cos(ψ_A)
        C_N = C_Y * (0.927 * C / L_pp - 0.149 * (ψ_A - np.deg2rad(90)))
        C_K = C_Y * (0.0737 * (H_C / L_pp)**(-0.821)) if H_C / L_pp <= 0.097 else C_Y * 0.500

    elif np.deg2rad(90) < ψ_A <= np.deg2rad(180):
        C_LF = 0.018 - 5.091 * B / L_pp + 10.367 * H_C / L_pp - 3.011 * A_OD / L_pp**2 - 0.341 * A_F / B**2
        C_XLI = -1.901 + 12.727 * A_L / (L_pp * H_BR) + 24.407 * A_F / A_L - 40.310 * B / L_pp - 0.341 * A_F / (B * H_BR)
        C_ALF = -0.314 - 1.117 * A_OD / A_L
        C_YLI = np.pi * A_L / L_pp**2 + 0.446 + 2.192 * A_F / L_pp**2

        C_X = C_LF * np.cos(ψ_A) + C_XLI * (np.sin(ψ_A) - np.sin(ψ_A) * np.cos(ψ_A)**2 / 2) * np.sin(ψ_A) * np.cos(ψ_A) + C_ALF * np.sin(ψ_A) * np.cos(ψ_A)**3
        C_Y = C_CF * np.sin(ψ_A)**2 + C_YLI * (np.cos(ψ_A) + np.sin(ψ_A)**2 * np.cos(ψ_A) / 2) * np.sin(ψ_A) * np.cos(ψ_A)
        C_N = C_Y * (0.927 * C / L_pp - 0.149 * (ψ_A - np.deg2rad(90)))
        C_K = C_Y * (0.0737 * (H_C / L_pp)**(-0.821)) if H_C / L_pp <= 0.097 else C_Y * 0.500

    elif np.deg2rad(180) < ψ_A <= np.deg2rad(270):
        C_LF = 0.018 - 5.091 * B / L_pp + 10.367 * H_C / L_pp - 3.011 * A_OD / L_pp**2 - 0.341 * A_F / B**2
        C_XLI = -1.901 + 12.727 * A_L / (L_pp * H_BR) + 24.407 * A_F / A_L - 40.310 * B / L_pp - 0.341 * A_F / (B * H_BR)
        C_ALF = -0.314 - 1.117 * A_OD / A_L
        C_YLI = np.pi * A_L / L_pp**2 + 0.446 + 2.192 * A_F / L_pp**2
        ψ_A = 2 * np.pi - ψ_A

        C_X = (C_LF * np.cos(ψ_A) + C_XLI * (np.sin(ψ_A) - np.sin(ψ_A) * np.cos(ψ_A)**2 / 2) * np.sin(ψ_A) * np.cos(ψ_A) + C_ALF * np.sin(ψ_A) * np.cos(ψ_A)**3)
        C_Y = -(C_CF * np.sin(ψ_A)**2 + C_YLI * (np.cos(ψ_A) + np.sin(ψ_A)**2 * np.cos(ψ_A) / 2) * np.sin(ψ_A) * np.cos(ψ_A))
        C_N = C_Y * (0.927 * C / L_pp - 0.149 * (ψ_A - np.deg2rad(90)))
        C_K = C_Y * (0.0737 * (H_C / L_pp)**(-0.821)) if H_C / L_pp <= 0.097 else C_Y * 0.500

    elif np.deg2rad(270) < ψ_A <= np.deg2rad(360):
        C_LF = -0.992 + 0.507 * A_L / (L_pp * B) + 1.162 * C / L_pp
        C_XLI = 0.458 + 3.245 * A_L / (L_pp * H_BR) - 2.313 * A_F / (B * H_BR)
        C_ALF = -0.585 - 0.906 * A_OD / A_L + 3.239 * B / L_pp
        C_YLI = np.pi * A_L / L_pp**2 + 0.116 + 3.345 * A_F / (L_pp * B)
        ψ_A = 2 * np.pi - ψ_A

        C_X = (C_LF * np.cos(ψ_A) + C_XLI * (np.sin(ψ_A) - np.sin(ψ_A) * np.cos(ψ_A)**2 / 2) * np.sin(ψ_A) * np.cos(ψ_A) + C_ALF * np.sin(ψ_A) * np.cos(ψ_A)**3)
        C_Y = -(C_CF * np.sin(ψ_A)**2 + C_YLI * (np.cos(ψ_A) + np.sin(ψ_A)**2 * np.cos(ψ_A) / 2) * np.sin(ψ_A) * np.cos(ψ_A))
        C_N = C_Y * (0.927 * C / L_pp - 0.149 * (ψ_A - np.deg2rad(90)))
        C_K = C_Y * (0.0737 * (H_C / L_pp)**(-0.821)) if H_C / L_pp <= 0.097 else C_Y * 0.500

    return C_X, C_Y, C_N, C_K

ψ_A_vec = np.deg2rad(np.arange(0, 361, 10))  # 0〜360度まで10度刻み（ラジアン）
C_X_vec = np.zeros_like(ψ_A_vec)
C_Y_vec = np.zeros_like(ψ_A_vec)
C_N_vec = np.zeros_like(ψ_A_vec)
C_K_vec = np.zeros_like(ψ_A_vec)
# KCS model
L_pp = 3.0464  # 船長Lpp[m]
B = 0.4265  # 船幅[m]
d = 0.1430  # 喫水[m]
D = 0.2389  # 深さ[m]
A_OD = 0.5940  # デッキ上の構造物の側面投影面積[m^2]
H_BR = 0.5150  # 喫水からブリッジ主要構造物の最高位[m]
H_C = 0.1980  # 喫水から側面積中心までの高さ[m]
C = -0.0122  # 船体中心から側面積中心までの前後方向座標(船首方向を正)[m]


A_OD = A_OD # デッキ上の構造物の側面投影面積[m^2]
A_F = H_BR * B  # 船体の正面投影面積[m^2]
A_L = (D - d) * L_pp + A_OD # 船体の側面投影面積[m^2]
# A_F = (D - d) * B  # 船体の正面投影面積[m^2]
# A_L = (D - d) * L_pp # 船体の側面投影面積[m^2]
H_BR = H_BR # 喫水からブリッジ主要構造物の最高位[m]
H_C = H_C # 喫水から側面積中心までの高さ[m]
C = C # 船体中心から側面積中心までの前後方向座標[m]

for i, ψ_A in enumerate(ψ_A_vec):
    C_X, C_Y, C_N, C_K = wind_force_and_moment_coefficients(
        ψ_A,
        L_pp,
        B,
        A_OD,
        A_F,
        A_L,
        H_BR,
        H_C,
        C,
    )
    C_X_vec[i] = C_X
    C_Y_vec[i] = C_Y
    C_N_vec[i] = C_N
    C_K_vec[i] = C_K
    
spl_C_X = CubicSpline(ψ_A_vec, C_X_vec, extrapolate=True)
spl_C_Y = CubicSpline(ψ_A_vec, C_Y_vec, extrapolate=True)
spl_C_N = CubicSpline(ψ_A_vec, C_N_vec, extrapolate=True)
spl_C_K = CubicSpline(ψ_A_vec, C_K_vec, extrapolate=True)
