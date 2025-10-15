import dataclasses
from typing import List

import numpy as np

from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.misc import derivative
from .wind_force_and_moment_coefficient_4dof import spl_C_X, spl_C_Y, spl_C_N, spl_C_K
from .wind_force_and_moment_coefficient_estimate_4dof import spl_C_XW, spl_C_YW, spl_C_NW, spl_C_KW



@dataclasses.dataclass
class Mmg4DofInWindBasicParams:
    
    L_pp: float
    B: float
    d: float
    g: float
    m: float
    x_G: float
    z_G: float
    z_H: float
    m_x: float
    m_y: float
    GM: float
    D_p: float
    A_R: float
    x_R: float
    I_zz: float
    η: float
    f_α: float
    ϵ: float
    t_R: float
    a_H: float
    x_H: float
    γ_R_plus: float
    γ_R_minus: float
    l_P_dash: float
    l_R_dash: float
    z_P_dash: float
    z_R_dash: float
    κ: float
    t_P: float
    w_P0: float
    I_xx: float
    J_xx: float
    J_zz: float
    a: float
    b: float
    α_z: float
    z_R: float
    h_a: float
    z_W: float
    
    
    
@dataclasses.dataclass
class Mmg4DofInWindManeuveringParams:
    
    k_0: float
    k_1: float
    k_2: float
    R_0_dash: float
    X_vv_dash: float
    X_vr_dash: float
    X_rr_dash: float
    X_vvvv_dash: float
    X_vφ_dash: float
    X_rφ_dash: float
    X_φφ_dash: float
    Y_v_dash: float
    Y_r_dash: float
    Y_vvv_dash: float
    Y_vvr_dash: float
    Y_vrr_dash: float
    Y_rrr_dash: float
    Y_φ_dash: float
    Y_vvφ_dash: float
    Y_vφφ_dash: float
    Y_rrφ_dash: float
    Y_rφφ_dash: float
    N_v_dash: float
    N_r_dash: float
    N_vvv_dash: float
    N_vvr_dash: float
    N_vrr_dash: float
    N_rrr_dash: float
    N_φ_dash: float
    N_vvφ_dash: float
    N_vφφ_dash: float
    N_rrφ_dash: float
    N_rφφ_dash: float
    
    
def simulate_mmg_4dof_in_wind(
    basic_params: Mmg4DofInWindBasicParams,
    maneuvering_params: Mmg4DofInWindManeuveringParams,
    time_list: List[float],
    δ_list: List[float],
    nps_list: List[float],
    U_W_list: List[float],
    Ψ_W_list: List[float],
    X_F_list: List[float],
    Y_F_list: List[float],
    N_F_list: List[float],
    K_F_list: List[float],
    u0: float = 0.0,
    v0: float = 0.0,
    r0: float = 0.0,
    p0: float = 0.0,
    x0: float = 0.0,
    y0: float = 0.0,
    ψ0: float = 0.0,
    φ0: float = 0.0,
    ρ: float = 1025.0,
    method: str = "RK45",
    t_eval=None,
    events=None,
    vectorized=False,
    **options
):
    return simulate(
        L_pp=basic_params.L_pp,
        B=basic_params.B,
        d=basic_params.d,
        g=basic_params.g,
        m=basic_params.m,
        x_G=basic_params.x_G,
        z_G=basic_params.z_G,
        z_H=basic_params.z_H,
        m_x=basic_params.m_x,
        m_y=basic_params.m_y,
        GM=basic_params.GM,
        D_p=basic_params.D_p,
        A_R=basic_params.A_R,
        x_R=basic_params.x_R,
        I_zz=basic_params.I_zz,
        η=basic_params.η,
        f_α=basic_params.f_α,
        ϵ=basic_params.ϵ,
        t_R=basic_params.t_R,
        a_H=basic_params.a_H,
        x_H=basic_params.x_H,
        γ_R_plus=basic_params.γ_R_plus,
        γ_R_minus=basic_params.γ_R_minus,
        l_P_dash=basic_params.l_P_dash,
        l_R_dash=basic_params.l_R_dash,
        z_P_dash=basic_params.z_P_dash,
        z_R_dash=basic_params.z_R_dash,
        κ=basic_params.κ,
        t_P=basic_params.t_P,
        w_P0=basic_params.w_P0,
        I_xx=basic_params.I_xx,
        J_xx=basic_params.J_xx,
        J_zz=basic_params.J_zz,
        a=basic_params.a,
        b=basic_params.b,
        α_z=basic_params.α_z,
        z_R=basic_params.z_R,
        h_a=basic_params.h_a,
        z_W=basic_params.z_W,
        k_0=maneuvering_params.k_0,
        k_1=maneuvering_params.k_1,
        k_2=maneuvering_params.k_2,
        R_0_dash=maneuvering_params.R_0_dash,
        X_vv_dash=maneuvering_params.X_vv_dash,
        X_vr_dash=maneuvering_params.X_vr_dash,
        X_rr_dash=maneuvering_params.X_rr_dash,
        X_vvvv_dash=maneuvering_params.X_vvvv_dash,
        X_vφ_dash=maneuvering_params.X_vφ_dash,
        X_rφ_dash=maneuvering_params.X_rφ_dash,
        X_φφ_dash=maneuvering_params.X_φφ_dash,
        Y_v_dash=maneuvering_params.Y_v_dash,
        Y_r_dash=maneuvering_params.Y_r_dash,
        Y_vvv_dash=maneuvering_params.Y_vvv_dash,
        Y_vvr_dash=maneuvering_params.Y_vvr_dash,
        Y_vrr_dash=maneuvering_params.Y_vrr_dash,
        Y_rrr_dash=maneuvering_params.Y_rrr_dash,
        Y_φ_dash=maneuvering_params.Y_φ_dash,
        Y_vvφ_dash=maneuvering_params.Y_vvφ_dash,
        Y_vφφ_dash=maneuvering_params.Y_vφφ_dash,
        Y_rrφ_dash=maneuvering_params.Y_rrφ_dash,
        Y_rφφ_dash=maneuvering_params.Y_rφφ_dash,
        N_v_dash=maneuvering_params.N_v_dash,
        N_r_dash=maneuvering_params.N_r_dash,
        N_vvv_dash=maneuvering_params.N_vvv_dash,
        N_vvr_dash=maneuvering_params.N_vvr_dash,
        N_vrr_dash=maneuvering_params.N_vrr_dash,
        N_rrr_dash=maneuvering_params.N_rrr_dash,
        N_φ_dash=maneuvering_params.N_φ_dash,
        N_vvφ_dash=maneuvering_params.N_vvφ_dash,
        N_vφφ_dash=maneuvering_params.N_vφφ_dash,
        N_rrφ_dash=maneuvering_params.N_rrφ_dash,
        N_rφφ_dash=maneuvering_params.N_rφφ_dash,
        time_list=time_list,
        δ_list=δ_list,
        nps_list=nps_list,
        U_W_list=U_W_list,
        Ψ_W_list=Ψ_W_list,
        X_F_list=X_F_list,
        Y_F_list=Y_F_list,
        N_F_list=N_F_list,
        K_F_list=K_F_list,
        u0=u0,
        v0=v0,
        r0=r0,
        p0=p0,
        x0=x0,
        y0=y0,
        ψ0=ψ0,
        φ0=φ0,
        ρ=ρ,
        method=method,
        t_eval=t_eval,
        events=events,
        vectorized=vectorized,
        **options
    )
    

def simulate(
    L_pp: float,
    B: float,
    d: float,
    g: float,
    m: float,
    x_G: float,
    z_G: float,
    z_H: float,
    m_x: float,
    m_y: float,
    GM: float,
    D_p: float,
    A_R: float,
    x_R: float,
    I_zz: float,
    η: float,
    f_α: float,
    ϵ: float,
    t_R: float,
    a_H: float,
    x_H: float,
    γ_R_plus: float,
    γ_R_minus: float,
    l_P_dash: float,
    l_R_dash: float,
    z_P_dash: float,
    z_R_dash: float,
    κ: float,
    t_P: float,
    w_P0: float,
    I_xx: float,
    J_xx: float,
    J_zz: float,
    a: float,
    b: float,
    α_z: float,
    z_R: float,
    h_a: float,
    z_W: float,
    k_0: float,
    k_1: float,
    k_2: float,
    R_0_dash: float,
    X_vv_dash: float,
    X_vr_dash: float,
    X_rr_dash: float,
    X_vvvv_dash: float,
    X_vφ_dash: float,
    X_rφ_dash: float,
    X_φφ_dash: float,
    Y_v_dash: float,
    Y_r_dash: float,
    Y_vvv_dash: float,
    Y_vvr_dash: float,
    Y_vrr_dash: float,
    Y_rrr_dash: float,
    Y_φ_dash: float,
    Y_vvφ_dash: float,
    Y_vφφ_dash: float,
    Y_rrφ_dash: float,
    Y_rφφ_dash: float,
    N_v_dash: float,
    N_r_dash: float,
    N_vvv_dash: float,
    N_vvr_dash: float,
    N_vrr_dash: float,
    N_rrr_dash: float,
    N_φ_dash: float,
    N_vvφ_dash: float,
    N_vφφ_dash: float,
    N_rrφ_dash: float,
    N_rφφ_dash: float,
    time_list: List[float],
    δ_list: List[float],
    nps_list: List[float],
    U_W_list: List[float],
    Ψ_W_list: List[float],
    X_F_list: List[float],
    Y_F_list: List[float],
    N_F_list: List[float],
    K_F_list: List[float],
    u0: float = 0.0,
    v0: float = 0.0,
    r0: float = 0.0,
    p0: float = 0.0,
    x0: float = 0.0,
    y0: float = 0.0,
    ψ0: float = 0.0,
    φ0: float = 0.0,
    ρ: float = 1025.0,
    method: str = "RK45",
    t_eval=None,
    events=None,
    vectorized=False,
    **options
):
    
    spl_δ = interp1d(time_list, δ_list, "cubic", fill_value="extrapolate")
    spl_nps = interp1d(time_list, nps_list, "cubic", fill_value="extrapolate")
    spl_U_W = interp1d(time_list, U_W_list, "cubic", fill_value="extrapolate")
    spl_Ψ_W = interp1d(time_list, Ψ_W_list, "cubic", fill_value="extrapolate")
    spl_X_F = interp1d(time_list, X_F_list, "linear", fill_value="extrapolate")
    spl_Y_F = interp1d(time_list, Y_F_list, "linear", fill_value="extrapolate")
    spl_N_F = interp1d(time_list, N_F_list, "linear", fill_value="extrapolate")
    spl_K_F = interp1d(time_list, K_F_list, "linear", fill_value="extrapolate")
    
    def mmg_4dof_in_wind_eom_solve_ivp(t, X):
        
        # u, v, r, p, x, y, ψ, φ, δ, nps, U_W, Ψ_W, X_F_l, Y_F_l, N_F_l, K_F_l = X
        u, v, r, p, x, y, ψ, φ, δ, nps, U_W, Ψ_W = X
        
        v_m = v - x_G * r + z_G * p
        v_m = v
        
        U = np.sqrt(u**2 + v_m** 2)
        
        β = 0.0 if u == 0.0 else np.arctan2(-v_m, u)
        v_dash = 0.0 if U == 0.0 else v / U
        r_dash = 0.0 if U == 0.0 else r * L_pp / U
        p_dash = 0.0 if U == 0.0 else p * B / U
        
        β_P = β - l_P_dash * r_dash + z_P_dash * p_dash
        
        w_P = w_P0 * (1 - (1 - np.cos(β_P)**2) * (1 - np.abs(β_P)))
        
        J = 0.0 if nps == 0.0 else u * (1 - w_P) / (nps * D_p)
        K_T = k_0 + k_1 * J + k_2 * J**2
        β_R = β - l_R_dash * r_dash + z_R_dash * p_dash
        γ_R = γ_R_minus if β_R < 0.0 else γ_R_plus
        v_R = U * γ_R * β_R
        u_R = (
            np.sqrt(η * (κ * ϵ * 8.0 * k_0 * nps**2 * D_p**4 / np.pi) ** 2)
            if J == 0.0
            else u
            * (1 - w_P)
            * ϵ
            * np.sqrt(
                η * (1.0 + κ * (np.sqrt(1.0 + 8.0 * K_T / (np.pi * J**2)) - 1)) ** 2
                + (1 - η)
            )
        )
        U_R = np.sqrt(u_R**2 + v_R**2)
        α_R = δ - np.arctan2(v_R, u_R)
        F_N = 0.5 * A_R * ρ * f_α * (U_R**2) * np.sin(α_R)
        
        
        X_P = (1 - t_P) * ρ * nps**2 * D_p**4 * K_T
        X_R = -(1 - t_R) * F_N * np.sin(δ) * np.cos(φ)
        X_H = (
            0.5
            * ρ
            * L_pp
            * d
            * (U**2)
            * (
                -R_0_dash
                + X_vv_dash * v_dash**2
                + X_vr_dash * v_dash * r_dash
                + X_rr_dash * r_dash**2
                + X_vvvv_dash * v_dash**4
                + X_vφ_dash * v_dash * φ
                + X_rφ_dash * r_dash * φ
                + X_φφ_dash * φ**2
            )
        )
        
        Y_R = -(1 + a_H) * F_N * np.cos(δ) * np.cos(φ)
        Y_H = (
            0.5
            * ρ
            * L_pp
            * d
            * (U**2)
            * (
                Y_v_dash * v_dash
                + Y_r_dash * r_dash
                + Y_vvv_dash * v_dash**3
                + Y_vvr_dash * v_dash**2 * r_dash
                + Y_vrr_dash * v_dash * r_dash**2
                + Y_rrr_dash * r_dash**3
                + Y_φ_dash * φ
                + Y_vvφ_dash * v_dash**2 * φ
                + Y_vφφ_dash * v_dash * φ**2
                + Y_rrφ_dash * r_dash**2 * φ
                + Y_rφφ_dash * r_dash * φ**2
            )
        )
        
        N_R = -(x_R + a_H * x_H) * F_N * np.cos(δ) * np.cos(φ)
        N_H = (
            0.5
            * ρ
            * (L_pp**2)
            * d
            * (U**2)
            * (
                N_v_dash * v_dash
                + N_r_dash * r_dash
                + N_vvv_dash * v_dash**3
                + N_vvr_dash * v_dash**2 * r_dash
                + N_vrr_dash * v_dash * r_dash**2
                + N_rrr_dash * r_dash**3
                + N_φ_dash * φ
                + N_vvφ_dash * v_dash**2 * φ
                + N_vφφ_dash * v_dash * φ**2
                + N_rrφ_dash * r_dash**2 * φ
                + N_rφφ_dash * r_dash * φ**2
            )
        )
        
        
        # KCS model
        D = 0.274 # 深さ[m]
        A_OD = 0.7813 # デッキ上の構造物の側面投影面積[m^2]
        H_BR = 0.6324 # 喫水からブリッジ主要構造物の最高位[m]
        H_C = 0.227 # 喫水から側面積中心までの高さ[m]
        C = -0.0139 # 船体中心から側面積中心までの前後方向座標(船首方向を正)[m]
        # D = 0.286 # 深さ[m]
        # A_OD = 0.1239 # デッキ上の構造物の側面投影面積[m^2]
        # H_BR = 0.3712 # 喫水からブリッジ主要構造物の最高位[m]
        # H_C = 0.235 # 喫水から側面積中心までの高さ[m]
        # C = 0.0 # 船体中心から側面積中心までの前後方向座標(船首方向を正)[m]
        # D = 0.286 # 深さ[m]
        # A_OD = 0.284 # デッキ上の構造物の側面投影面積[m^2]
        # H_BR = 0.3712 # 喫水からブリッジ主要構造物の最高位[m]
        # H_C = 0.1026 # 喫水から側面積中心までの高さ[m]
        # C = 0.0 # 船体中心から側面積中心までの前後方向座標(船首方向を正)[m]
        
        
        A_OD = A_OD # デッキ上の構造物の側面投影面積[m^2]
        A_F = H_BR * B  # 船体の正面投影面積[m^2]
        # A_F = (D - d) * B  # 船体の正面投影面積[m^2]
        # A_L = (D - d) * L_pp # 船体の側面投影面積[m^2]
        A_L = (D - d) * L_pp + A_OD # 船体の側面投影面積[m^2]
        
        H_BR = H_BR # 喫水からブリッジ主要構造物の最高位[m]
        H_C = H_C # 喫水から側面積中心までの高さ[m]
        C = C # 船体中心から側面積中心までの前後方向座標[m]
        
        ρ_air = 1.225  # [kg/m^3] (density of air at sea level)
        u_A = u + U_W * np.cos(Ψ_W - ψ)
        v_A = v + U_W * np.sin(Ψ_W - ψ)
        U_A = np.sqrt(u_A**2 + v_A**2)
        Ψ_A = -np.arctan2(v_A, u_A)
        Ψ_A = np.mod(Ψ_A, 2 * np.pi)
        
        # X_wind = ρ_air * A_F * spl_C_XW(Ψ_A) / 2 * U_A**2
        # Y_wind = ρ_air * A_L * spl_C_YW(Ψ_A) / 2 * U_A**2
        # N_wind = ρ_air * A_L * L_pp * spl_C_NW(Ψ_A) / 2 * U_A**2
        # K_wind = ρ_air * A_L**2 / L_pp * spl_C_KW(Ψ_A) / 2 * U_A**2
        X_wind = ρ_air * A_F * spl_C_X(Ψ_A) / 2 * U_A**2
        Y_wind = ρ_air * A_L * spl_C_Y(Ψ_A) / 2 * U_A**2
        N_wind = ρ_air * A_L * L_pp * spl_C_N(Ψ_A) / 2 * U_A**2
        K_wind = ρ_air * A_L**2 / L_pp * spl_C_K(Ψ_A) / 2 * U_A**2
        
        # X_wind = 0
        # Y_wind = 0
        # N_wind = 0
        # K_wind = 0
        
        # X_wind = spl_X_F(t)
        # Y_wind = spl_Y_F(t)
        # N_wind = spl_N_F(t)
        # K_wind = spl_K_F(t)
        # X_wind = X_F_l
        # Y_wind = Y_F_l
        # N_wind = N_F_l
        # K_wind = K_F_l
        
        
        K_p = -2 / np.pi * a * np.sqrt(m * g * GM * (I_xx + J_xx))
        K_pp = -0.75 * b * (180 / np.pi) * (I_xx + J_xx)
        
        X = X_H + X_R + X_P + X_wind
        Y = Y_H + Y_R + Y_wind
        N = N_H + N_R + N_wind
        K = -Y_H * z_H - Y_R * z_R - m * g * GM * φ + K_p * p + K_pp * p * np.abs(p) + K_wind
        # K = -Y_H * z_H - Y_R * z_R - m * g * GM * φ + K_p * p + K_pp * p * np.abs(p) - z_W * Y_wind
        
        A_ = (m + m_y) - (m_y * α_z + m * z_G)**2 / (I_xx + J_xx + m * z_G**2)
        B_ = x_G * m - (m_y * α_z + m * z_G) * m * z_G * x_G / (I_xx + J_xx + m * z_G**2)
        C_ = Y - (m + m_x) * u * r + (m_y * α_z + m * z_G) * (K + m * z_G * u * r) / (I_xx + J_xx + m * z_G**2)
        D_ = m * x_G * (1 - z_G * (m_y * α_z + m * z_G) / (I_xx + J_xx + m * z_G**2))
        E_ = (I_zz + J_zz + m * x_G**2) - m * z_G**2 * x_G / (I_xx + J_xx + m * z_G**2)
        F_ = N + m * x_G * (z_G * (K + m * z_G * u * r) / (I_xx + J_xx + m * z_G**2) - u * r)
        
        d_u = (X + (m + m_y) * v * r + m * x_G * (r**2) - m * z_G * r * p) / (m + m_x)
        d_v = (C_ * E_ - B_ * F_) / (A_ * E_ - B_ * D_)
        d_r = (C_ * D_ - A_ * F_) / (B_ * D_ - A_ * E_)
        d_p = (K + (m_y * α_z + m * z_G) * d_v + m * z_G * (x_G * d_r + u * r)) / (I_xx + J_xx + m * z_G**2)
        
        # 簡略版 --------------
        # d_u = (X + (m + m_y) * v * r) / (m + m_x)
        # d_v = (Y - (m + m_x) * u * r + (m_y * α_z * K) / (I_xx + J_xx)) / (m + m_y - (m_y * α_z)**2 / (I_xx + J_xx))
        # d_r = N / (I_zz + J_zz)
        # d_p = (K + m_y * α_z * d_v) / (I_xx + J_xx)
        # --------------------
        
        d_x = u * np.cos(ψ) - v * np.sin(ψ)
        d_y = u * np.sin(ψ) + v * np.cos(ψ)
        d_ψ = r
        d_φ = p
        d_δ = derivative(spl_δ, t)
        d_nps = derivative(spl_nps, t)
        d_U_W = derivative(spl_U_W, t)
        d_Ψ_W = derivative(spl_Ψ_W, t)
        # d_X_F = (spl_X_F(t) - X_F_l) / 0.1
        # d_Y_F = (spl_Y_F(t) - Y_F_l) / 0.1
        # d_N_F = (spl_N_F(t) - N_F_l) / 0.1
        # d_K_F = (spl_K_F(t) - K_F_l) / 0.1
        
        # return [d_u, d_v, d_r, d_p, d_x, d_y, d_ψ, d_φ, d_δ, d_nps, d_U_W, d_Ψ_W, d_X_F, d_Y_F, d_N_F, d_K_F]
        return [d_u, d_v, d_r, d_p, d_x, d_y, d_ψ, d_φ, d_δ, d_nps, d_U_W, d_Ψ_W]
    
    sol = solve_ivp(
        mmg_4dof_in_wind_eom_solve_ivp,
        [time_list[0], time_list[-1]],
        # [u0, v0, r0, p0, x0, y0, ψ0, φ0, δ_list[0], nps_list[0], U_W_list[0], Ψ_W_list[0], X_F_list[0], Y_F_list[0], N_F_list[0], K_F_list[0]],
        [u0, v0, r0, p0, x0, y0, ψ0, φ0, δ_list[0], nps_list[0], U_W_list[0], Ψ_W_list[0]],
        dense_output=True,
        method=method,
        t_eval=t_eval,
        events=events,
        vectorized=vectorized,
        **options
    )
    return sol
        
        
        
def get_wind_values_from_simulation_result(
    u_list: List[float],
    v_list: List[float],
    r_list: List[float],
    p_list: List[float],
    ψ_list: List[float],
    U_W_list: List[float],
    Ψ_W_list: List[float],
    basic_params: Mmg4DofInWindBasicParams,
    ρ: float = 1025.0,
    ρ_air: float = 1.225,
    return_all_values: bool = False,
):
    u_A_list = list(map(lambda u, U_W, Ψ_W, ψ: u + U_W * np.cos(Ψ_W - ψ), u_list, U_W_list, Ψ_W_list, ψ_list))
    v_A_list = list(map(lambda v, U_W, Ψ_W, ψ: v + U_W * np.sin(Ψ_W - ψ), v_list, U_W_list, Ψ_W_list, ψ_list))
    U_A_list = list(map(lambda u_A, v_A: np.sqrt(u_A**2 + v_A**2), u_A_list, v_A_list))
    Ψ_A_list = list(map(lambda u_A, v_A: -np.arctan2(v_A, u_A), u_A_list, v_A_list))
    Ψ_A_list = list(map(lambda ψ_A: np.mod(ψ_A, 2 * np.pi), Ψ_A_list))
    C_X_list = list(map(lambda Ψ_A: spl_C_X(Ψ_A), Ψ_A_list))
    C_Y_list = list(map(lambda Ψ_A: spl_C_Y(Ψ_A), Ψ_A_list))
    C_N_list = list(map(lambda Ψ_A: spl_C_N(Ψ_A), Ψ_A_list))
    C_K_list = list(map(lambda Ψ_A: spl_C_K(Ψ_A), Ψ_A_list))
    D = 0.274 # 深さ[m]
    A_OD = 0.7813 # デッキ上の構造物の側面投影面積[m^2]
    H_BR = 0.6324 # 喫水からブリッジ主要構造物の最高位[m]
    # D = 0.286 # 深さ[m]
    # A_F = (D - basic_params.d) * basic_params.B  # 船体の正面投影面積[m^2]
    # A_L = (D - basic_params.d) * basic_params.L_pp # 船体の側面投影面積[m^2]
    A_F = H_BR * basic_params.B  # 船体の正面投影面積[m^2]
    A_L = (D - basic_params.d) * basic_params.L_pp + A_OD
    X_wind_list = list(map(lambda C_X, U_A: ρ_air * A_F * C_X / 2 * U_A**2, C_X_list, U_A_list))
    Y_wind_list = list(map(lambda C_Y, U_A: ρ_air * A_L * C_Y / 2 * U_A**2, C_Y_list, U_A_list))
    N_wind_list = list(map(lambda C_N, U_A: ρ_air * A_L * basic_params.L_pp * C_N / 2 * U_A**2, C_N_list, U_A_list))
    K_wind_list = list(map(lambda C_K, U_A: ρ_air * A_L**2 / basic_params.L_pp * C_K / 2 * U_A**2, C_K_list, U_A_list))
    v_m_list = list(
        map(
            lambda v, r, p: v - basic_params.x_G * r + basic_params.z_G * p,
            v_list,
            r_list,
            p_list,
        )
    )
    U_list = list(
        map(
            lambda u, v_m: np.sqrt(u**2 + v_m**2),
            u_list,
            v_m_list,
        )
    )
    
    
    
    if return_all_values:
        return (
            X_wind_list,
            Y_wind_list,
            N_wind_list,
            K_wind_list,
            u_A_list,
            v_A_list,
            U_A_list,
            Ψ_A_list,
            C_X_list,
            C_Y_list,
            C_N_list,
            C_K_list,
            v_m_list,
            U_list,
        )
    else:
        return (
            X_wind_list,
            Y_wind_list,
            N_wind_list,
            K_wind_list,
        )
        

# zigzag
def zigzag_test_mmg_4dof(
    basic_params: Mmg4DofInWindBasicParams,
    maneuvering_params: Mmg4DofInWindManeuveringParams,
    target_δ_rad: float,
    target_ψ_rad_deviation: float,
    time_list: List[float],
    nps_list: List[float],
    U_W_list: List[float],
    Ψ_W_list: List[float],
    X_F_list: List[float],
    Y_F_list: List[float],
    N_F_list: List[float],
    K_F_list: List[float],
    δ0: float = 0.0,
    δ_rad_rate: float = 1.0 * np.pi / 180,
    u0: float = 0.0,
    v0: float = 0.0,
    r0: float = 0.0,
    p0: float = 0.0,
    x0: float = 0.0,
    y0: float = 0.0,
    ψ0: float = 0.0,
    φ0: float = 0.0,
    ρ: float = 1025.0,
    method: str = "RK45",
    t_eval=None,
    events=None,
    vectorized=False,
    **options
):

    target_ψ_rad_deviation = np.abs(target_ψ_rad_deviation)

    final_δ_list = [0.0] * len(time_list)
    final_u_list = [0.0] * len(time_list)
    final_v_list = [0.0] * len(time_list)
    final_r_list = [0.0] * len(time_list)
    final_p_list = [0.0] * len(time_list)
    final_x_list = [0.0] * len(time_list)
    final_y_list = [0.0] * len(time_list)
    final_ψ_list = [0.0] * len(time_list)
    final_φ_list = [0.0] * len(time_list)

    next_stage_index = 0
    target_δ_rad = -target_δ_rad  # for changing in while loop
    ψ = ψ0

    while next_stage_index < len(time_list):
        target_δ_rad = -target_δ_rad
        start_index = next_stage_index

        # Make delta list
        δ_list = [0.0] * (len(time_list) - start_index)
        if start_index == 0:
            δ_list[0] = δ0
            u0 = u0
            v0 = v0
            r0 = r0
            p0 = p0
            x0 = x0
            y0 = y0
            φ0 = φ0
        else:
            δ_list[0] = final_δ_list[start_index - 1]
            u0 = final_u_list[start_index - 1]
            v0 = final_v_list[start_index - 1]
            r0 = final_r_list[start_index - 1]
            p0 = final_p_list[start_index - 1]
            x0 = final_x_list[start_index - 1]
            y0 = final_y_list[start_index - 1]
            φ0 = final_φ_list[start_index - 1]

        for i in range(start_index + 1, len(time_list)):
            Δt = time_list[i] - time_list[i - 1]
            if target_δ_rad > 0:
                δ = δ_list[i - 1 - start_index] + δ_rad_rate * Δt
                if δ >= target_δ_rad:
                    δ = target_δ_rad
                δ_list[i - start_index] = δ
            elif target_δ_rad <= 0:
                δ = δ_list[i - 1 - start_index] - δ_rad_rate * Δt
                if δ <= target_δ_rad:
                    δ = target_δ_rad
                δ_list[i - start_index] = δ

        sol = simulate_mmg_4dof_in_wind(
            basic_params,
            maneuvering_params,
            time_list[start_index:],
            δ_list,
            nps_list[start_index:],
            U_W_list[start_index:],
            Ψ_W_list[start_index:],
            X_F_list[start_index:],
            Y_F_list[start_index:],
            N_F_list[start_index:],
            K_F_list[start_index:],
            u0=u0,
            v0=v0,
            r0=r0,
            p0=p0,
            x0=x0,
            y0=y0,
            ψ0=ψ,
            φ0=φ0,
            ρ=ρ,
            # TODO
        )
        sim_result = sol.sol(time_list[start_index:])
        u_list = sim_result[0]
        v_list = sim_result[1]
        r_list = sim_result[2]
        p_list = sim_result[3]
        x_list = sim_result[4]
        y_list = sim_result[5]
        ψ_list = sim_result[6]
        φ_list = sim_result[7]
        # ship = ShipObj3dof(L=basic_params.L_pp, B=basic_params.B)
        # ship.load_simulation_result(time_list, u_list, v_list, r_list, psi0=ψ)

        # get finish index
        target_ψ_rad = ψ0 + target_ψ_rad_deviation
        if target_δ_rad < 0:
            target_ψ_rad = ψ0 - target_ψ_rad_deviation
        # ψ_list = ship.psi
        bool_ψ_list = [True if ψ < target_ψ_rad else False for ψ in ψ_list]
        if target_δ_rad < 0:
            bool_ψ_list = [True if ψ > target_ψ_rad else False for ψ in ψ_list]
        over_index_list = [i for i, flag in enumerate(bool_ψ_list) if flag is False]
        next_stage_index = len(time_list)
        if len(over_index_list) > 0:
            ψ = ψ_list[over_index_list[0]]
            next_stage_index = over_index_list[0] + start_index
            final_δ_list[start_index:next_stage_index] = δ_list[: over_index_list[0]]
            final_u_list[start_index:next_stage_index] = u_list[: over_index_list[0]]
            final_v_list[start_index:next_stage_index] = v_list[: over_index_list[0]]
            final_r_list[start_index:next_stage_index] = r_list[: over_index_list[0]]
            final_p_list[start_index:next_stage_index] = p_list[: over_index_list[0]]
            final_x_list[start_index:next_stage_index] = x_list[: over_index_list[0]]
            final_y_list[start_index:next_stage_index] = y_list[: over_index_list[0]]
            final_ψ_list[start_index:next_stage_index] = ψ_list[: over_index_list[0]]
            final_φ_list[start_index:next_stage_index] = φ_list[: over_index_list[0]]
        else:
            final_δ_list[start_index:next_stage_index] = δ_list
            final_u_list[start_index:next_stage_index] = u_list
            final_v_list[start_index:next_stage_index] = v_list
            final_r_list[start_index:next_stage_index] = r_list
            final_p_list[start_index:next_stage_index] = p_list
            final_x_list[start_index:next_stage_index] = x_list
            final_y_list[start_index:next_stage_index] = y_list
            final_ψ_list[start_index:next_stage_index] = ψ_list
            final_φ_list[start_index:next_stage_index] = φ_list

    return (
        final_δ_list,
        final_u_list,
        final_v_list,
        final_r_list,
        final_p_list,
        final_x_list,
        final_y_list,
        final_ψ_list,
        final_φ_list,
    )