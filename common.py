"""共用設定：電路參數、方波、轉移函數 H、解析解（拿來當標準答案）、期刊圖風格。

兩個電路都寫成同一條一階 ODE（狀態變數 x）：
    τ·dx/dt + x = v_in(t),   x(0) = 0
    RC 串聯：x = V_C ，τ = R·C
    RL 串聯：x = V_R = R·i ，τ = L/R ，再用 KVL 得 V_L = v_in − V_R
選 V_C 和電感電流當 x，是因為它們「不能瞬間跳」，初始條件 x(0)=0 才有意義。
"""
import numpy as np
import matplotlib as mpl

# ---------------- 電路與方波參數（沿用 HW1 / HW2 的數字） ----------------
V0 = 10.0            # 方波高電位 (V)，低電位 0 V
T = 20.0             # 方波週期 (s)：半週期 10 s = 5τ_RC，電容充得滿 → 圓角方波
W0 = 2 * np.pi / T   # 基頻 ω0 (rad/s)

R_RC, C = 2.0, 1.0   # RC：R = 2 Ω, C = 1 F
R_RL, L = 2.0, 1.0   # RL：R = 2 Ω, L = 1 H
TAU_RC = R_RC * C    # τ = 2 s
TAU_RL = L / R_RL    # τ = 0.5 s

N_PERIODS = 3        # 畫 0 ~ 3T


def square_wave(t):
    """0 < t mod T < T/2 時為 V0，否則 0；剛好在跳點上取平均 V0/2（傅立葉級數收斂到的值）。"""
    phase = np.mod(t, T)
    v = np.where(phase < T / 2, V0, 0.0)
    on_jump = np.isclose(phase, 0.0) | np.isclose(phase, T / 2) | np.isclose(phase, T)
    return np.where(on_jump, V0 / 2, v)


def H(omega, tau):
    """狀態變數的轉移函數：把 x = X e^{jωt} 代入 τ x' + x = v  →  (1 + jωτ) X = V。"""
    return 1.0 / (1.0 + 1j * omega * tau)


def general_solution(t, xp_func, tau):
    """通解 = 特解 + 齊次解。
    特解 x_p：方波「永遠在跑」時的週期穩態（傅立葉算的就是它）。
    齊次解 x_h = A·e^{−t/τ}：用 x(0) = 0 決定 A = −x_p(0)。
    """
    A = -xp_func(np.array([0.0]))[0]
    xp = xp_func(t)
    xh = A * np.exp(-t / tau)
    return xp + xh, xp, xh


def exact_state(t, tau):
    """標準答案：逐段解析解。每半個週期輸入是常數 u，x 以 e^{−Δt/τ} 往 u 靠近。"""
    t = np.asarray(t, dtype=float)
    x = np.empty_like(t)
    half = T / 2
    seg = np.floor(t / half + 1e-12).astype(int)        # 第幾個半週期
    x_start = np.zeros(seg.max() + 1)                    # 每段開頭的 x
    for k in range(1, seg.max() + 1):
        u_prev = V0 if (k - 1) % 2 == 0 else 0.0
        x_start[k] = u_prev + (x_start[k - 1] - u_prev) * np.exp(-half / tau)
    u = np.where(seg % 2 == 0, V0, 0.0)
    x[:] = u + (x_start[seg] - u) * np.exp(-(t - seg * half) / tau)
    return x


# ---------------- 期刊（APS）圖風格 ----------------
COLORS = {               # Okabe–Ito 色盲友善色
    "exact": "#000000",
    "m1": "#0072B2",     # 藍：傳統傅立葉
    "m2": "#D55E00",     # 橘紅：線性代數 DFT
    "m3": "#009E73",     # 綠：QFT
    "input": "#999999",
    "rl": "#CC79A7",
}
COL1, COL2 = 3.4, 7.0    # APS 單欄 / 雙欄寬 (inch)


def set_journal_style():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.0,
        "lines.markersize": 3.5,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.minor.size": 1.5,
        "ytick.minor.size": 1.5,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "legend.frameon": False,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
    })


def panel_label(ax, text):
    ax.text(-0.13, 1.0, text, transform=ax.transAxes, fontsize=9,
            fontweight="bold", va="top", ha="left")
