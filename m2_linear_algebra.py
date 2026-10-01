"""方法二：線性代數。

把一個週期取 N 個點 v = [v_0, ..., v_{N−1}]，「傅立葉轉換」就只是乘一個 N×N 矩陣：
    F[k, m] = e^{−2πi·k·m/N}           （DFT 矩陣）
    F⁻¹ = F^H / N                       （F 的共軛轉置除以 N）
特解一行搞定：
    x_p = F⁻¹ · diag(H) · F · v

為什麼可以？因為「週期訊號的微分」也是一個矩陣 D，而 F 的每一列剛好是 D 的特徵向量：
    F · D · F⁻¹ = diag(jω_k)
所以 ODE (τD + I)x = v 在傅立葉座標下變成對角矩陣，除一除就好。
"""
import numpy as np
from common import T, H


def dft_matrix(N):
    k = np.arange(N)
    return np.exp(-2j * np.pi * np.outer(k, k) / N)


def omegas(N):
    """第 k 列對應的角頻率：k = 0,1,...,N/2−1 是正頻率，後半段是負頻率。"""
    return 2 * np.pi / T * np.fft.fftfreq(N, d=1.0 / N)


def particular_linalg(v, tau):
    """輸入一個週期的取樣 v，回傳同一組時間點上的特解 x_p。"""
    N = v.size
    F = dft_matrix(N)
    F_inv = F.conj().T / N
    Hdiag = np.diag(H(omegas(N), tau))
    return (F_inv @ Hdiag @ F @ v).real


def derivative_matrix(N):
    """週期訊號的頻譜微分矩陣 D（在時域裡，是一個循環矩陣）。"""
    F = dft_matrix(N)
    w = omegas(N)
    if N % 2 == 0:
        w[N // 2] = 0.0          # Nyquist 頻率的微分取 0，D 才會是實數矩陣
    return (F.conj().T / N @ np.diag(1j * w) @ F).real


def check_eigenvectors(N=16):
    """驗證 F D F⁻¹ 是對角矩陣；回傳非對角元素的最大絕對值（應該 ~1e−15）。"""
    F = dft_matrix(N)
    M = F @ derivative_matrix(N) @ np.linalg.inv(F)
    off = M - np.diag(np.diag(M))
    return np.abs(off).max()
