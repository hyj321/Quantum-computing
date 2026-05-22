import math
from pyqpanda3.core import CPUQVM, QProg, QCircuit, H, X, Z, CNOT, CZ, TOFFOLI, measure

TARGET = "1010"
SHOTS = 1000

def build_mcz(qubits):
    """底层标准多控 Z 分解"""
    q0, q1, q2, q3 = qubits
    circ = QCircuit()
    circ << TOFFOLI(q0, q1, 4) << TOFFOLI(4, q2, 5) << CZ(5, q3)
    circ << TOFFOLI(4, q2, 5) << TOFFOLI(q0, q1, 4)
    return circ


def build_oracle(qubits):
    """
    严格单参数 Oracle。针对 |1010> 翻转相位。
    注：根据量子计算 Gottesman-Knill 定理，仅用 Clifford 门无法实现此操作，
    故采用量子计算标准基础逻辑门 Toffoli（等价于 CNOT 与 T 门组合）。
    """
    circ = QCircuit()
    circ << X(qubits[0]) << X(qubits[2])
    circ << build_mcz(qubits)
    circ << X(qubits[0]) << X(qubits[2])
    return circ

# ✅ 彻底解决单参数问题
def build_diffusion(qubits):
    """严格单参数扩散算子"""
    circ = QCircuit()
    for q in qubits: circ << H(q)
    for q in qubits: circ << X(q)
    circ << build_mcz(qubits)
    for q in qubits: circ << X(q)
    for q in qubits: circ << H(q)
    return circ

def step1_verify_superposition():
    prog = QProg()
    for i in range(4): prog << H(i)
    qvm = CPUQVM()
    qvm.run(prog, 1)
    sv = qvm.result().get_state_vector()
    print("--- 初始概率幅验证 ---")
    for i in range(16):
        amp = sv[i]
        print(f"|{format(i,'04b')}> 幅度={amp.real:.4f}, 概率={abs(amp)**2:.4f}")

def main():
    step1_verify_superposition()
    
    qubits = [0, 1, 2, 3]
    k = int(math.pi / 4 * math.sqrt(16))
    
    prog = QProg()
    for q in qubits: prog << H(q)
    for _ in range(k):
        prog << build_oracle(qubits)    
        prog << build_diffusion(qubits)  
    for q in qubits: prog << measure(q, q)

    qvm = CPUQVM()
    qvm.run(prog, SHOTS)
    
    probs = qvm.result().get_prob_dict()
    counts = qvm.result().get_counts()
    print(f"\n--- Grover 搜索结果 (目标 |{TARGET}>, 迭代 {k} 次) ---")
    for i in range(16):
        s = format(i, '04b')
        p = probs.get(s, 0.0)
        c = counts.get(s, 0)
        tag = " << TARGET" if s == TARGET else ""
        print(f"|{s}>: {c}次 ({p:.1%}){tag}")

if __name__ == "__main__":
    main()
