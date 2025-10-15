#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""mmg_3dof.

* MMG (3DOF) simulation code

.. math::

        m (\\dot{u}-vr)&=-m_x\\dot{u}+m_yvr+X_H+X_P+X_R

        m (\\dot{v}+ur)&=-m_y\\dot{v}+m_xur+Y_H+Y_R

        I_{zG}\\dot{r}&=-J_Z\\dot{r}+N_H+N_R

"""

import dataclasses
from typing import List

import numpy as np

from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.misc import derivative
from shipmmg.wind_force_and_moment_coefficient import spl_C_X, spl_C_Y, spl_C_N
# from shipmmg.wind_force_and_moment_coefficient_estimate import spl_C_XW, spl_C_YW, spl_C_NW # MPCによる推定値


@dataclasses.dataclass
class Mmg3DofBasicParams:
    L_pp: float
    B: float
    d: float
    x_G: float
    D_p: float
    m: float
    I_zG: float
    A_R: float
    η: float
    m_x: float
    m_y: float
    J_z: float
    f_α: float
    ϵ: float
    t_R: float
    x_R: float
    a_H: float
    x_H: float
    γ_R_minus: float
    γ_R_plus: float
    l_R: float
    κ: float
    t_P: float
    w_P0: float
    x_P: float


@dataclasses.dataclass
class Mmg3DofManeuveringParams:
    k_0: float
    k_1: float
    k_2: float
    R_0_dash: float
    X_vv_dash: float
    X_vr_dash: float
    X_rr_dash: float
    X_vvvv_dash: float
    Y_v_dash: float
    Y_r_dash: float
    Y_vvv_dash: float
    Y_vvr_dash: float
    Y_vrr_dash: float
    Y_rrr_dash: float
    N_v_dash: float
    N_r_dash: float
    N_vvv_dash: float
    N_vvr_dash: float
    N_vrr_dash: float
    N_rrr_dash: float


def simulate_mmg_3dof(
    basic_params: Mmg3DofBasicParams,
    maneuvering_params: Mmg3DofManeuveringParams,
    time_list: List[float],
    δ_list: List[float],
    nps_list: List[float],
    U_W_list: List[float],
    Ψ_W_list: List[float],
    u0: float = 0.0,
    v0: float = 0.0,
    r0: float = 0.0,
    x0: float = 0.0,
    y0: float = 0.0,
    ψ0: float = 0.0,
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
        x_G=basic_params.x_G,
        D_p=basic_params.D_p,
        m=basic_params.m,
        I_zG=basic_params.I_zG,
        A_R=basic_params.A_R,
        η=basic_params.η,
        m_x=basic_params.m_x,
        m_y=basic_params.m_y,
        J_z=basic_params.J_z,
        f_α=basic_params.f_α,
        ϵ=basic_params.ϵ,
        t_R=basic_params.t_R,
        x_R=basic_params.x_R,
        a_H=basic_params.a_H,
        x_H=basic_params.x_H,
        γ_R_minus=basic_params.γ_R_minus,
        γ_R_plus=basic_params.γ_R_plus,
        l_R=basic_params.l_R,
        κ=basic_params.κ,
        t_P=basic_params.t_P,
        w_P0=basic_params.w_P0,
        x_P=basic_params.x_P,
        k_0=maneuvering_params.k_0,
        k_1=maneuvering_params.k_1,
        k_2=maneuvering_params.k_2,
        R_0_dash=maneuvering_params.R_0_dash,
        X_vv_dash=maneuvering_params.X_vv_dash,
        X_vr_dash=maneuvering_params.X_vr_dash,
        X_rr_dash=maneuvering_params.X_rr_dash,
        X_vvvv_dash=maneuvering_params.X_vvvv_dash,
        Y_v_dash=maneuvering_params.Y_v_dash,
        Y_r_dash=maneuvering_params.Y_r_dash,
        Y_vvv_dash=maneuvering_params.Y_vvv_dash,
        Y_vvr_dash=maneuvering_params.Y_vvr_dash,
        Y_vrr_dash=maneuvering_params.Y_vrr_dash,
        Y_rrr_dash=maneuvering_params.Y_rrr_dash,
        N_v_dash=maneuvering_params.N_v_dash,
        N_r_dash=maneuvering_params.N_r_dash,
        N_vvv_dash=maneuvering_params.N_vvv_dash,
        N_vvr_dash=maneuvering_params.N_vvr_dash,
        N_vrr_dash=maneuvering_params.N_vrr_dash,
        N_rrr_dash=maneuvering_params.N_rrr_dash,
        time_list=time_list,
        δ_list=δ_list,
        nps_list=nps_list,
        U_W_list=U_W_list,
        Ψ_W_list=Ψ_W_list,
        u0=u0,
        v0=v0,
        r0=r0,
        x0=x0,
        y0=y0,
        ψ0=ψ0,
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
    x_G: float,
    D_p: float,
    m: float,
    I_zG: float,
    A_R: float,
    η: float,
    m_x: float,
    m_y: float,
    J_z: float,
    f_α: float,
    ϵ: float,
    t_R: float,
    x_R: float,
    a_H: float,
    x_H: float,
    γ_R_minus: float,
    γ_R_plus: float,
    l_R: float,
    κ: float,
    t_P: float,
    w_P0: float,
    x_P: float,
    k_0: float,
    k_1: float,
    k_2: float,
    R_0_dash: float,
    X_vv_dash: float,
    X_vr_dash: float,
    X_rr_dash: float,
    X_vvvv_dash: float,
    Y_v_dash: float,
    Y_r_dash: float,
    Y_vvv_dash: float,
    Y_vvr_dash: float,
    Y_vrr_dash: float,
    Y_rrr_dash: float,
    N_v_dash: float,
    N_r_dash: float,
    N_vvv_dash: float,
    N_vvr_dash: float,
    N_vrr_dash: float,
    N_rrr_dash: float,
    time_list: List[float],
    δ_list: List[float],
    nps_list: List[float],
    U_W_list: List[float],
    Ψ_W_list: List[float],
    u0: float = 0.0,
    v0: float = 0.0,
    r0: float = 0.0,
    x0: float = 0.0,
    y0: float = 0.0,
    ψ0: float = 0.0,
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

    def mmg_3dof_eom_solve_ivp(t, X):

        u, v, r, x, y, ψ, δ, nps, U_W, Ψ_W = X

        U = np.sqrt(u**2 + (v - r * x_G) ** 2)

        β = 0.0 if U == 0.0 else np.arcsin(-(v - r * x_G) / U)
        v_dash = 0.0 if U == 0.0 else v / U
        r_dash = 0.0 if U == 0.0 else r * L_pp / U

        # w_P = w_P0
        w_P = w_P0 * np.exp(-4.0 * (β - x_P * r_dash) ** 2)

        J = 0.0 if nps == 0.0 else (1 - w_P) * u / (nps * D_p)
        K_T = k_0 + k_1 * J + k_2 * J**2
        β_R = β - l_R * r_dash
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

        X_H = (
            0.5
            * ρ
            * L_pp
            * d
            * (U**2)
            * (
                -R_0_dash
                + X_vv_dash * (v_dash**2)
                + X_vr_dash * v_dash * r_dash
                + X_rr_dash * (r_dash**2)
                + X_vvvv_dash * (v_dash**4)
            )
        )
        X_R = -(1 - t_R) * F_N * np.sin(δ)
        X_P = (1 - t_P) * ρ * K_T * nps**2 * D_p**4
        Y_H = (
            0.5
            * ρ
            * L_pp
            * d
            * (U**2)
            * (
                Y_v_dash * v_dash
                + Y_r_dash * r_dash
                + Y_vvv_dash * (v_dash**3)
                + Y_vvr_dash * (v_dash**2) * r_dash
                + Y_vrr_dash * v_dash * (r_dash**2)
                + Y_rrr_dash * (r_dash**3)
            )
        )
        Y_R = -(1 + a_H) * F_N * np.cos(δ)
        N_H = (
            0.5
            * ρ
            * (L_pp**2)
            * d
            * (U**2)
            * (
                N_v_dash * v_dash
                + N_r_dash * r_dash
                + N_vvv_dash * (v_dash**3)
                + N_vvr_dash * (v_dash**2) * r_dash
                + N_vrr_dash * v_dash * (r_dash**2)
                + N_rrr_dash * (r_dash**3)
            )
        )
        N_R = -(x_R + a_H * x_H) * F_N * np.cos(δ)
        
        # Calculate wind forces ------------------
        # KCS model
        D = 0.2389 # 深さ[m]
        A_OD = 0.594 # デッキ上の構造物の側面投影面積[m^2]
        H_BR = 0.515 # 喫水からブリッジ主要構造物の最高位[m]
        H_C = 0.198 # 喫水から側面積中心までの高さ[m]
        C = -0.0122 # 船体中心から側面積中心までの前後方向座標(船首方向を正)[m]
        
        

        A_OD = A_OD # デッキ上の構造物の側面投影面積[m^2]
        A_F = H_BR * B  # 船体の正面投影面積[m^2]
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
        
        
        
        X_wind = ρ_air * A_F * spl_C_X(Ψ_A) / 2 * U_A**2
        Y_wind = ρ_air * A_L * spl_C_Y(Ψ_A) / 2 * U_A**2
        N_wind = ρ_air * A_L * L_pp * spl_C_N(Ψ_A) / 2 * U_A**2
        # ----------------------------------------
        
        
        d_u = ((X_H + X_R + X_P + X_wind) + (m + m_y) * v * r) / (m + m_x)
        d_v = ((Y_H + Y_R + Y_wind) - (m + m_x) * u * r) / (m + m_y)
        d_r = (N_H + N_R + N_wind) / (I_zG + J_z)
        # d_u = ((X_H + X_R + X_P + X_wind) + (m + m_y) * v * r + x_G * m * (r**2)) / (m + m_x)
        # d_v = (
        #     (x_G**2) * (m**2) * u * r
        #     - (N_H + N_R + N_wind) * x_G * m
        #     + ((Y_H + Y_R + Y_wind) - (m + m_x) * u * r) * (I_zG + J_z + (x_G**2) * m)
        # ) / ((I_zG + J_z + (x_G**2) * m) * (m + m_y) - (x_G**2) * (m**2))
        # d_r = (N_H + N_R + N_wind - x_G * m * (d_v + u * r)) / (I_zG + J_z + (x_G**2) * m)
        d_x = u * np.cos(ψ) - v * np.sin(ψ)
        d_y = u * np.sin(ψ) + v * np.cos(ψ)
        d_ψ = r
        d_δ = derivative(spl_δ, t)
        d_nps = derivative(spl_nps, t)
        d_U_W = derivative(spl_U_W, t)
        d_Ψ_W = derivative(spl_Ψ_W, t)
        return [d_u, d_v, d_r, d_x, d_y, d_ψ, d_δ, d_nps, d_U_W, d_Ψ_W]

    sol = solve_ivp(
        mmg_3dof_eom_solve_ivp,
        [time_list[0], time_list[-1]],
        [u0, v0, r0, x0, y0, ψ0, δ_list[0], nps_list[0], U_W_list[0], Ψ_W_list[0]],
        dense_output=True,
        method=method,
        t_eval=t_eval,
        events=events,
        vectorized=vectorized,
        **options
    )
    return sol


def get_sub_values_from_simulation_result(
    u_list: List[float],
    v_list: List[float],
    r_list: List[float],
    δ_list: List[float],
    ψ_list: List[float],
    nps_list: List[float],
    U_W_list: List[float],
    Ψ_W_list: List[float],
    basic_params: Mmg3DofBasicParams,
    maneuvering_params: Mmg3DofManeuveringParams,
    ρ: float = 1025.0,
    ρ_air: float = 1.225,
    return_all_vals: bool = False,
):
    u_A_list = list(map(lambda u, U_W, Ψ_W, ψ: u + U_W * np.cos(Ψ_W - ψ), u_list, U_W_list, Ψ_W_list, ψ_list))
    v_A_list = list(map(lambda v, U_W, Ψ_W, ψ: v + U_W * np.sin(Ψ_W - ψ), v_list, U_W_list, Ψ_W_list, ψ_list))
    U_A_list = list(map(lambda u_A, v_A: np.sqrt(u_A**2 + v_A**2), u_A_list, v_A_list))
    Ψ_A_list = list(map(lambda u_A, v_A: -np.arctan2(v_A, u_A), u_A_list, v_A_list))
    Ψ_A_list = list(map(lambda ψ_A: np.mod(ψ_A, 2 * np.pi), Ψ_A_list))
    C_X_list = list(map(lambda Ψ_A: spl_C_X(Ψ_A), Ψ_A_list))
    C_Y_list = list(map(lambda Ψ_A: spl_C_Y(Ψ_A), Ψ_A_list))
    C_N_list = list(map(lambda Ψ_A: spl_C_N(Ψ_A), Ψ_A_list))
    D = 0.2389 # 深さ[m]
    A_OD = 0.594 # デッキ上の構造物の側面投影面積[m^2]
    H_BR = 0.515 # 喫水からブリッジ主要構造物の最高位[m]
    A_F = H_BR * basic_params.B  # 船体の正面投影面積[m^2]
    A_L = (D - basic_params.d) * basic_params.L_pp + A_OD # 船体の側面投影面積[m^2]
    X_wind_list = list(map(lambda C_X, U_A: ρ_air * A_F * C_X / 2 * U_A**2, C_X_list, U_A_list))
    Y_wind_list = list(map(lambda C_Y, U_A: ρ_air * A_L * C_Y / 2 * U_A**2, C_Y_list, U_A_list))
    N_wind_list = list(map(lambda C_N, U_A: ρ_air * A_L * basic_params.L_pp * C_N / 2 * U_A**2, C_N_list, U_A_list))

    U_list = list(
        map(
            lambda u, v, r: np.sqrt(u**2 + (v - r * basic_params.x_G) ** 2),
            u_list,
            v_list,
            r_list,
        )
    )
    β_list = list(
        map(
            lambda U, v, r: 0.0
            if U == 0.0
            else np.arcsin(-(v - r * basic_params.x_G) / U),
            U_list,
            v_list,
            r_list,
        )
    )
    v_dash_list = list(map(lambda U, v: 0.0 if U == 0.0 else v / U, U_list, v_list))
    r_dash_list = list(
        map(lambda U, r: 0.0 if U == 0.0 else r * basic_params.L_pp / U, U_list, r_list)
    )
    β_P_list = list(
        map(
            lambda β, r_dash: β - basic_params.x_P * r_dash,
            β_list,
            r_dash_list,
        )
    )
    # w_P_list = [basic_params.w_P0 for i in range(len(r_dash_list))]
    w_P_list = list(
        map(lambda β_P: basic_params.w_P0 * np.exp(-4.0 * β_P**2), β_P_list)
    )
    J_list = list(
        map(
            lambda w_P, u, nps: 0.0
            if nps == 0.0
            else (1 - w_P) * u / (nps * basic_params.D_p),
            w_P_list,
            u_list,
            nps_list,
        )
    )
    K_T_list = list(
        map(
            lambda J: maneuvering_params.k_0
            + maneuvering_params.k_1 * J
            + maneuvering_params.k_2 * J**2,
            J_list,
        )
    )
    β_R_list = list(
        map(
            lambda β, r_dash: β - basic_params.l_R * r_dash,
            β_list,
            r_dash_list,
        )
    )
    γ_R_list = list(
        map(
            lambda β_R: basic_params.γ_R_minus if β_R < 0.0 else basic_params.γ_R_plus,
            β_R_list,
        )
    )
    v_R_list = list(
        map(
            lambda U, γ_R, β_R: U * γ_R * β_R,
            U_list,
            γ_R_list,
            β_R_list,
        )
    )
    u_R_list = list(
        map(
            lambda u, J, nps, K_T, w_P: np.sqrt(
                basic_params.η
                * (
                    basic_params.κ
                    * basic_params.ϵ
                    * 8.0
                    * maneuvering_params.k_0
                    * nps**2
                    * basic_params.D_p**4
                    / np.pi
                )
                ** 2
            )
            if J == 0.0
            else u
            * (1 - w_P)
            * basic_params.ϵ
            * np.sqrt(
                basic_params.η
                * (
                    1.0
                    + basic_params.κ * (np.sqrt(1.0 + 8.0 * K_T / (np.pi * J**2)) - 1)
                )
                ** 2
                + (1 - basic_params.η)
            ),
            u_list,
            J_list,
            nps_list,
            K_T_list,
            w_P_list,
        )
    )
    U_R_list = list(
        map(lambda u_R, v_R: np.sqrt(u_R**2 + v_R**2), u_R_list, v_R_list)
    )
    α_R_list = list(
        map(lambda δ, u_R, v_R: δ - np.arctan2(v_R, u_R), δ_list, u_R_list, v_R_list)
    )
    F_N_list = list(
        map(
            lambda U_R, α_R: 0.5
            * basic_params.A_R
            * ρ
            * basic_params.f_α
            * (U_R**2)
            * np.sin(α_R),
            U_R_list,
            α_R_list,
        )
    )
    X_H_list = list(
        map(
            lambda U, v_dash, r_dash: 0.5
            * ρ
            * basic_params.L_pp
            * basic_params.d
            * (U**2)
            * (
                -maneuvering_params.R_0_dash
                + maneuvering_params.X_vv_dash * (v_dash**2)
                + maneuvering_params.X_vr_dash * v_dash * r_dash
                + maneuvering_params.X_rr_dash * (r_dash**2)
                + maneuvering_params.X_vvvv_dash * (v_dash**4)
            ),
            U_list,
            v_dash_list,
            r_dash_list,
        )
    )
    X_R_list = list(
        map(lambda F_N, δ: -(1 - basic_params.t_R) * F_N * np.sin(δ), F_N_list, δ_list)
    )
    X_P_list = list(
        map(
            lambda K_T, nps: (1 - basic_params.t_P)
            * ρ
            * K_T
            * nps**2
            * basic_params.D_p**4,
            K_T_list,
            nps_list,
        )
    )
    Y_H_list = list(
        map(
            lambda U, v_dash, r_dash: 0.5
            * ρ
            * basic_params.L_pp
            * basic_params.d
            * (U**2)
            * (
                maneuvering_params.Y_v_dash * v_dash
                + maneuvering_params.Y_r_dash * r_dash
                + maneuvering_params.Y_vvv_dash * (v_dash**3)
                + maneuvering_params.Y_vvr_dash * (v_dash**2) * r_dash
                + maneuvering_params.Y_vrr_dash * v_dash * (r_dash**2)
                + maneuvering_params.Y_rrr_dash * (r_dash**3)
            ),
            U_list,
            v_dash_list,
            r_dash_list,
        )
    )
    Y_R_list = list(
        map(lambda F_N, δ: -(1 - basic_params.t_R) * F_N * np.cos(δ), F_N_list, δ_list)
    )
    N_H_list = list(
        map(
            lambda U, v_dash, r_dash: 0.5
            * ρ
            * (basic_params.L_pp**2)
            * basic_params.d
            * (U**2)
            * (
                maneuvering_params.N_v_dash * v_dash
                + maneuvering_params.N_r_dash * r_dash
                + maneuvering_params.N_vvv_dash * (v_dash**3)
                + maneuvering_params.N_vvr_dash * (v_dash**2) * r_dash
                + maneuvering_params.N_vrr_dash * v_dash * (r_dash**2)
                + maneuvering_params.N_rrr_dash * (r_dash**3)
            ),
            U_list,
            v_dash_list,
            r_dash_list,
        )
    )
    N_R_list = list(
        map(
            lambda F_N, δ: -(basic_params.x_R + basic_params.a_H * basic_params.x_H)
            * F_N
            * np.cos(δ),
            F_N_list,
            δ_list,
        )
    )
    
    if return_all_vals:
        return (
            X_H_list,
            X_R_list,
            X_P_list,
            Y_H_list,
            Y_R_list,
            N_H_list,
            N_R_list,
            U_list,
            β_list,
            v_dash_list,
            r_dash_list,
            β_P_list,
            w_P_list,
            J_list,
            K_T_list,
            β_R_list,
            γ_R_list,
            v_R_list,
            u_R_list,
            U_R_list,
            α_R_list,
            F_N_list,
            X_wind_list,
            Y_wind_list,
            N_wind_list,
        )
    else:
        return (
            X_H_list,
            X_R_list,
            X_P_list,
            Y_H_list,
            Y_R_list,
            N_H_list,
            N_R_list,
            X_wind_list,
            Y_wind_list,
            N_wind_list,
            U_A_list,
            Ψ_A_list,
            C_X_list,
            C_Y_list,
            C_N_list,
        )


def zigzag_test_mmg_3dof(
    basic_params: Mmg3DofBasicParams,
    maneuvering_params: Mmg3DofManeuveringParams,
    target_δ_rad: float,
    target_ψ_rad_deviation: float,
    time_list: List[float],
    nps_list: List[float],
    U_W_list: List[float],
    Ψ_W_list: List[float],
    δ0: float = 0.0,
    δ_rad_rate: float = 1.0 * np.pi / 180,
    u0: float = 0.0,
    v0: float = 0.0,
    r0: float = 0.0,
    x0: float = 0.0,
    y0: float = 0.0,
    ψ0: float = 0.0,
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
    final_x_list = [0.0] * len(time_list)
    final_y_list = [0.0] * len(time_list)
    final_ψ_list = [0.0] * len(time_list)

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
            x0 = x0
            y0 = y0
        else:
            δ_list[0] = final_δ_list[start_index - 1]
            u0 = final_u_list[start_index - 1]
            v0 = final_v_list[start_index - 1]
            r0 = final_r_list[start_index - 1]
            x0 = final_x_list[start_index - 1]
            y0 = final_y_list[start_index - 1]

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

        sol = simulate_mmg_3dof(
            basic_params,
            maneuvering_params,
            time_list[start_index:],
            δ_list,
            nps_list[start_index:],
            U_W_list[start_index:],
            Ψ_W_list[start_index:],
            u0=u0,
            v0=v0,
            r0=r0,
            x0=x0,
            y0=y0,
            ψ0=ψ,
            ρ=ρ,
            # TODO
        )
        sim_result = sol.sol(time_list[start_index:])
        u_list = sim_result[0]
        v_list = sim_result[1]
        r_list = sim_result[2]
        x_list = sim_result[3]
        y_list = sim_result[4]
        ψ_list = sim_result[5]
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
            final_x_list[start_index:next_stage_index] = x_list[: over_index_list[0]]
            final_y_list[start_index:next_stage_index] = y_list[: over_index_list[0]]
            final_ψ_list[start_index:next_stage_index] = ψ_list[: over_index_list[0]]
        else:
            final_δ_list[start_index:next_stage_index] = δ_list
            final_u_list[start_index:next_stage_index] = u_list
            final_v_list[start_index:next_stage_index] = v_list
            final_r_list[start_index:next_stage_index] = r_list
            final_x_list[start_index:next_stage_index] = x_list
            final_y_list[start_index:next_stage_index] = y_list
            final_ψ_list[start_index:next_stage_index] = ψ_list

    return (
        final_δ_list,
        final_u_list,
        final_v_list,
        final_r_list,
        final_x_list,
        final_y_list,
        final_ψ_list,
    )
