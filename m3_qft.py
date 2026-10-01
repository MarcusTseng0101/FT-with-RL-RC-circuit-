"""方法三：量子傅立葉轉換 QFT（純炫技，用 Qiskit statevector 模擬）。

n 個 qubit 有 N = 2^n 個振幅，剛好裝得下 N 個取樣點。電路：
  1. 振幅編碼：|ψ⟩ = Σ (v_k/‖v‖)|k⟩
  2. QFT†：  QFT† 的矩陣 = F/√N（跟方法二的 DFT 一模一樣，只差歸一化）
  3. 乘 H：  diag(H) 不是 unitary（|H|<1 會讓向量變短），量子電腦不能直接做。
             解法：多借 1 個 ancilla qubit，對每個頻率 k 做 2×2 的 unitary
                 U_k = [[ H_k ,  s_k  ],
                        [ s_k , −H_k* ]],   s_k = √(1 − |H_k|²)
             ancilla 量到 0 的那一半，振幅剛好被乘上 H_k（block encoding）。
  4. QFT：   轉回時域
  5. 只留 ancilla = 0 的部分，乘回 ‖v‖ 就是特解 x_p。
     成功機率 = ‖x_p‖² / ‖v‖²（被濾掉的能量就是失敗的機率）。
"""
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import QFTGate, StatePreparation, UCGate
from qiskit.quantum_info import Statevector
from common import H
from m2_linear_algebra import omegas


def filter_unitaries(Hk):
    gates = []
    for h in Hk:
        s = np.sqrt(max(0.0, 1.0 - abs(h) ** 2))
        gates.append(np.array([[h, s], [s, -np.conj(h)]], dtype=complex))
    return gates


def build_circuit(v, tau):
    N = v.size
    n = int(np.log2(N))
    assert 2 ** n == N, "取樣點數必須是 2 的次方"
    data = list(range(n))
    anc = n                                               # 最高位那顆是 ancilla

    qc = QuantumCircuit(n + 1)
    qc.append(StatePreparation(v / np.linalg.norm(v)), data)   # 步驟 1
    qc.append(QFTGate(n).inverse(), data)                       # 步驟 2
    Hk = H(omegas(N), tau)
    qc.append(UCGate(filter_unitaries(Hk)), [anc] + data)       # 步驟 3：data 第 k 態 → ancilla 做 U_k
    qc.append(QFTGate(n), data)                                 # 步驟 4
    return qc


def particular_qft(v, tau, return_info=False):
    qc = build_circuit(v, tau)
    psi = Statevector(qc).data
    N = v.size
    amp = psi[:N]                                         # 步驟 5：ancilla = 0 的那一半
    xp = np.linalg.norm(v) * amp
    if return_info:
        info = {"n_qubits": qc.num_qubits,
                "p_success": float(np.sum(np.abs(amp) ** 2)),
                "max_imag": float(np.abs(xp.imag).max())}
        return xp.real, info
    return xp.real
