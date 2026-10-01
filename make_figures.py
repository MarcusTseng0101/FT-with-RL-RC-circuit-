"""三種方法一起跑，跟解析解比對，輸出期刊規格的圖。
執行：python make_figures.py   →   fig1_response, fig2_convergence, fig3_decomposition (pdf/png)
"""
import numpy as np
import matplotlib.pyplot as plt
from common import (V0, T, W0, TAU_RC, TAU_RL, N_PERIODS, COLORS, COL1, COL2, H,
                    square_wave, general_solution, exact_state, set_journal_style, panel_label)
from m1_fourier_series import particular_series, square_wave_coeffs
from m2_linear_algebra import particular_linalg, check_eigenvectors
from m3_qft import particular_qft

N_QUBITS = 10
N = 2 ** N_QUBITS                       # 一個週期取 1024 點
M_HARM = 2001                           # 方法一用的諧波數


def periodic(xp_one_period):
    """把一個週期 N 點的特解，變成可以在任何「格點時間」取值的函數。"""
    def f(t):
        k = np.rint(np.mod(t, T) / T * N).astype(int) % N
        return xp_one_period[k]
    return f


def solve_all(tau):
    """回傳 dict：每個方法的 (t, x 通解, x_p 特解, x_h 齊次解)。"""
    t_grid1 = np.arange(N) * T / N                      # 一個週期的取樣
    v1 = square_wave(t_grid1)
    t_dense = np.linspace(0, N_PERIODS * T, 2001)
    t_grid = np.arange(N_PERIODS * N) * T / N           # 5 個週期的格點

    out = {}
    out["m1"] = (t_dense, *general_solution(t_dense, lambda t: particular_series(t, tau, M_HARM), tau))
    out["m2"] = (t_grid, *general_solution(t_grid, periodic(particular_linalg(v1, tau)), tau))
    xp3, info = particular_qft(v1, tau, return_info=True)
    out["m3"] = (t_grid, *general_solution(t_grid, periodic(xp3), tau))
    out["info"] = info
    return out


def to_VL(t, x):
    """RL：KVL  V_L = v_in − V_R。"""
    return square_wave(t) - x


def max_err(t, x, tau):
    return np.abs(x - exact_state(t, tau)).max()


def fig_response(rc, rl):
    fig, axs = plt.subplots(2, 2, figsize=(COL2, 4.4))
    (a, b), (c, d) = axs
    t_ex = np.linspace(0, N_PERIODS * T, 4001)
    t_in = np.linspace(0, N_PERIODS * T, 4001, endpoint=False)
    v_in = np.where(np.mod(t_in, T) < T / 2, V0, 0.0)
    sub2 = slice(32, None, 128)            # 標記點稀疏一點、避開跳點
    sub3 = slice(96, None, 128)

    def draw_methods(ax, data, transform, exact):
        ax.plot(t_in, v_in, color=COLORS["input"], lw=0.7, drawstyle="steps-post",
                label=r"$v_\mathrm{in}$")
        ax.plot(t_ex, exact, color=COLORS["exact"], lw=1.4, label="Analytic")
        t1, x1 = data["m1"][0], data["m1"][1]
        ax.plot(t1, transform(t1, x1), color=COLORS["m1"], lw=0.9, ls="--",
                label=f"(i) Fourier series, $M={M_HARM}$")
        t2, x2 = data["m2"][0], data["m2"][1]
        ax.plot(t2[sub2], transform(t2, x2)[sub2], ls="none", marker="o", mfc="none",
                mec=COLORS["m2"], mew=0.8, label=f"(ii) DFT matrix, $N={N}$")
        t3, x3 = data["m3"][0], data["m3"][1]
        ax.plot(t3[sub3], transform(t3, x3)[sub3], ls="none", marker="x",
                color=COLORS["m3"], mew=0.8, label=f"(iii) QFT, {N_QUBITS}+1 qubits")
        ax.set_xlim(0, N_PERIODS * T)
        ax.set_xlabel(r"$t$ (s)")

    # (a) RC 的 V_C
    draw_methods(a, rc, lambda t, x: x, exact_state(t_ex, TAU_RC))
    a.set_ylabel(r"$V_C$ (V)")
    a.set_ylim(-0.5, 12.5)
    panel_label(a, "(a)")
    a.text(0.98, 0.96, rf"RC, $\tau = RC = {TAU_RC:g}$ s", transform=a.transAxes,
           ha="right", va="top")

    # (b) 波形長相只看 τ/T：τ 越小（截止頻率越高），只有高頻被削 → 圓角方波
    tt = np.linspace(0, T, 1201)
    for tau_b, ls, col in [(0.025 * T, "-", COLORS["m3"]), (0.1 * T, "--", COLORS["m1"]),
                           (0.5 * T, "-.", COLORS["m2"])]:
        b.plot(tt / T, particular_series(tt, tau_b, M_HARM), ls=ls, color=col,
               label=rf"$\tau/T ={tau_b / T:g}$")
    b.plot(tt / T, square_wave(tt), color=COLORS["input"], lw=0.7, zorder=0)
    b.set_xlim(0, 1)
    b.set_ylim(-0.5, 12.5)
    b.set_xlabel(r"$t/T$ (one period, steady state)")
    b.set_ylabel(r"$V_C$ (V)")
    b.legend(loc="upper right", ncol=3, handlelength=2.0, columnspacing=1.0,
             borderaxespad=0.3)
    panel_label(b, "(b)")

    # (c) RL 的 V_L
    draw_methods(c, rl, to_VL, to_VL(t_ex, exact_state(t_ex, TAU_RL)))
    c.set_ylabel(r"$V_L$ (V)")
    c.set_ylim(-12, 14.5)
    panel_label(c, "(c)")
    c.text(0.98, 0.96, rf"RL, $\tau = L/R = {TAU_RL:g}$ s", transform=c.transAxes,
           ha="right", va="top")

    # (d) 頻域：方波諧波 → 濾波後
    n, cn = square_wave_coeffs(99)
    pos = (n > 0) & (np.abs(cn) > 0)
    n, cn = n[pos], cn[pos]
    w = np.logspace(np.log10(0.8), np.log10(130), 300)
    ref = 2 * V0 / (np.pi * w)                                # |c_n| 的包絡，乘上 2 成單邊振幅
    d.loglog(n, 2 * np.abs(cn), ls="none", marker="s", ms=2.5, mfc="none",
             mec=COLORS["input"], label=r"input $|v_n|$")
    d.loglog(n, 2 * np.abs(cn * H(n * W0, TAU_RC)), ls="none", marker="o", ms=2.5,
             color=COLORS["m1"], label=r"$|H_{RC}\,v_n|$ (low-pass)")
    d.loglog(n, 2 * np.abs(cn * (1 - H(n * W0, TAU_RL))), ls="none", marker="^", ms=2.5,
             color=COLORS["rl"], label=r"$|H_{L}\,v_n|$ (high-pass)")
    d.loglog(w, ref, color=COLORS["input"], lw=0.6)
    d.loglog(w, ref * np.abs(H(w * W0, TAU_RC)), color=COLORS["m1"], lw=0.6)
    d.loglog(w, ref * np.abs(1 - H(w * W0, TAU_RL)), color=COLORS["rl"], lw=0.6)
    d.set_xlabel(r"harmonic index $n$  ($\omega = n\omega_0$)")
    d.set_ylabel(r"single-sided amplitude (V)")
    d.set_xlim(0.8, 130)
    d.legend(loc="lower left", borderaxespad=0.3)
    panel_label(d, "(d)")

    h, l = a.get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.0),
               handlelength=2.0, columnspacing=1.2)
    fig.tight_layout(pad=0.3, w_pad=1.2, h_pad=0.8, rect=(0, 0, 1, 0.95))
    return fig


def fig_decomposition():
    """通解 = 特解 + 齊次解。用 τ/T = 0.5（例如 C = 5 F）讓暫態看得出來。"""
    tau = 0.5 * T
    t = np.linspace(0, N_PERIODS * T, 3001)
    x, xp, xh = general_solution(t, lambda tt: particular_series(tt, tau, M_HARM), tau)
    fig, ax = plt.subplots(figsize=(COL1, 2.4))
    ax.axhline(0, color="0.6", lw=0.5)
    ax.plot(t, xp, color=COLORS["m1"], ls="--", label=r"particular $x_p$ (steady state)")
    ax.plot(t, xh, color=COLORS["m2"], ls=":", lw=1.3,
            label=rf"homogeneous $x_h = A\,e^{{-t/\tau}}$, $A = {xh[0]:.2f}$ V")
    ax.plot(t, x, color=COLORS["exact"], lw=1.1, label=r"general $x = x_p + x_h$")
    ax.set_xlim(0, N_PERIODS * T)
    ax.set_ylim(-4.5, 11.5)
    ax.set_xlabel(r"$t$ (s)")
    ax.set_ylabel(r"$V_C$ (V)")
    ax.legend(loc="upper left", borderaxespad=0.3, handlelength=2.0)
    ax.text(0.98, 0.04, rf"RC, $\tau ={tau:g}$ s, $T = {T:g}$ s",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7)
    fig.tight_layout(pad=0.3)
    return fig, xh[0]


def fig_convergence():
    """最大誤差 vs 用了多少個傅立葉分量（RC 電路）。"""
    t_chk = np.linspace(0, N_PERIODS * T, 701)
    Ms = 2 ** np.arange(2, 13)
    e1 = []
    for M in Ms:
        x, _, _ = general_solution(t_chk, lambda t: particular_series(t, TAU_RC, M), TAU_RC)
        e1.append(max_err(t_chk, x, TAU_RC))

    Ns = 2 ** np.arange(3, 13)
    e2, e3, n3 = [], [], []
    for Nn in Ns:
        tg = np.arange(Nn) * T / Nn
        v = square_wave(tg)
        t5 = np.arange(N_PERIODS * Nn) * T / Nn
        idx = lambda xp: (lambda t: xp[np.rint(np.mod(t, T) / T * Nn).astype(int) % Nn])
        x2, _, _ = general_solution(t5, idx(particular_linalg(v, TAU_RC)), TAU_RC)
        e2.append(max_err(t5, x2, TAU_RC))
        if Nn <= 2 ** 9:
            x3, _, _ = general_solution(t5, idx(particular_qft(v, TAU_RC)), TAU_RC)
            e3.append(max_err(t5, x3, TAU_RC))
            n3.append(Nn)

    fig, ax = plt.subplots(figsize=(COL1, 2.5))
    # 方法一用 2M+1 個分量（n = −M..M）；方法二、三用 N 個
    K1 = 2 * Ms + 1
    ax.loglog(K1, e1, marker="s", mfc="none", color=COLORS["m1"], label="(i) Fourier series")
    ax.loglog(Ns, e2, marker="o", mfc="none", color=COLORS["m2"], label="(ii) DFT matrix")
    ax.loglog(n3, e3, ls="none", marker="x", ms=5, mew=0.9, color=COLORS["m3"],
              label="(iii) QFT (statevector)")
    kk = np.array([10.0, 5000.0])
    ax.loglog(kk, 12.0 / kk, color="0.5", lw=0.6, ls=":")      # 參考線：誤差 ∝ 1/K
    ax.text(1.3e3, 12.0 / 1.3e3 * 1.6, r"$\propto K^{-1}$", fontsize=7, color="0.35")
    ax.set_xlabel(r"number of Fourier components $K$")
    ax.set_ylabel(r"$\max_t |V_C - V_C^\mathrm{exact}|$ (V)")
    ax.legend(loc="lower left", borderaxespad=0.3)
    fig.tight_layout(pad=0.3)
    return fig, (Ms, e1), (Ns, e2), (n3, e3)


if __name__ == "__main__":
    set_journal_style()
    rc, rl = solve_all(TAU_RC), solve_all(TAU_RL)

    print(f"F D F^-1 非對角最大值 = {check_eigenvectors():.1e}   （約等於 0：DFT 把微分對角化）")
    print(f"\n{'':10s}{'RC: max|V_C 誤差|':>20s}{'RL: max|V_L 誤差|':>20s}")
    names = {"m1": "(i) 傅立葉", "m2": "(ii) 線代", "m3": "(iii) QFT"}
    for m in ["m1", "m2", "m3"]:
        t, x = rc[m][0], rc[m][1]
        tl, xl = rl[m][0], rl[m][1]
        print(f"{names[m]:10s}{max_err(t, x, TAU_RC):20.2e}{max_err(tl, xl, TAU_RL):20.2e}")
    print(f"\nQFT vs 線代 差異：RC {np.abs(rc['m3'][1] - rc['m2'][1]).max():.1e}, "
          f"RL {np.abs(rl['m3'][1] - rl['m2'][1]).max():.1e}  （只差浮點誤差）")
    for name, d in [("RC", rc), ("RL", rl)]:
        i = d["info"]
        print(f"QFT {name}: {i['n_qubits']} qubits, ancilla 成功機率 = {i['p_success']:.3f}")

    f1 = fig_response(rc, rl)
    f1.savefig("fig1_response.pdf"); f1.savefig("fig1_response.png")
    f2, *_ = fig_convergence()
    f2.savefig("fig2_convergence.pdf"); f2.savefig("fig2_convergence.png")
    f3, A = fig_decomposition()
    f3.savefig("fig3_decomposition.pdf"); f3.savefig("fig3_decomposition.png")
    print(f"主圖參數下齊次解係數 A = {-particular_series(np.array([0.0]), TAU_RC)[0]:.3f} V"
          f"（很小：半週期 = 5τ，暫態早就消失）；fig3 用 τ/T = 0.5，A = {A:.2f} V")
    print("\n已輸出 fig1_response, fig2_convergence, fig3_decomposition (pdf/png)")
