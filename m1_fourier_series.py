"""方法一：傳統傅立葉（級數）。

步驟（就是 L5 的「疊加三步驟」）：
  1. 拆：方波 = 一堆 e^{jnω0t} 的和，係數
         c_0 = V0/2,   c_n = V0/(jπn)（n 奇數），   c_n = 0（n 偶數、n≠0）
  2. 篩：每個頻率各自乘上 H(nω0) = 1/(1 + jnω0τ)
  3. 加：x_p(t) = Σ c_n·H(nω0)·e^{jnω0t}     ← 這就是特解（週期穩態）
"""
import numpy as np
from common import V0, W0, H


def square_wave_coeffs(M):
    """回傳 n = −M..M 和對應的 c_n。"""
    n = np.arange(-M, M + 1)
    c = np.zeros(n.size, dtype=complex)
    c[n == 0] = V0 / 2
    odd = (n % 2 != 0)
    c[odd] = V0 / (1j * np.pi * n[odd])
    return n, c


def particular_series(t, tau, M=2001):
    """用前 M 個諧波算特解 x_p(t)。"""
    n, c = square_wave_coeffs(M)
    out_coeffs = c * H(n * W0, tau)                       # 步驟 2：篩
    phase = np.exp(1j * np.outer(t, n * W0))              # 每個時間 × 每個頻率的 e^{jnω0t}
    return (phase @ out_coeffs).real                      # 步驟 3：加（虛部互相抵消）
