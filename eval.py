#!/usr/bin/env python3
"""
Quantum Stress Regression - EVAL (QPanda3 VQA)
加载训练好的模型，对测试集进行预测并输出MAE/RMSE/R²及得分判定
"""

import csv
import numpy as np
from pyqpanda3.core import CPUQVM, QProg, RY, RZ, CNOT, measure

ALL_FEATURES = [
    "Gender", "Age", "Palpitations", "Sleep_Issues", "Headaches",
    "Irritability", "Concentration", "Low_Mood", "Health_Issues",
    "Loneliness", "Peer_Comp", "Prof_Issues", "Work_Env",
    "Relax_Struggle", "Home_Env", "Acad_Conf", "Subj_Conf", "Act_Conflict"
]

# ================= 工具函数 =================
def load_csv(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))

def clean_data_eval(rows, age_median):
    """使用训练集的age_median清洗测试集"""
    for r in rows:
        # 处理Age
        age = float(r['Age'])
        if age < 10 or age > 40:
            r['Age'] = str(age_median)
        
        # 处理Gender
        gender = float(r['Gender'])
        r['Gender'] = str(np.clip(gender, 0, 1))
        
        # 处理其他特征
        for f in ALL_FEATURES:
            if f in ['Gender', 'Age']:
                continue
            val = float(r[f])
            if not np.isfinite(val) or val < 0:
                val = 0.0
            r[f] = str(np.clip(val, 0, 5))

def build_x(rows):
    """构建特征矩阵X"""
    return np.array([[float(r[f]) for f in ALL_FEATURES] for r in rows], dtype=np.float64)

def normalize_apply(X, xmin, xrange):
    """应用训练集的归一化参数"""
    return (X - xmin) / xrange * np.pi

# ================= 振幅编码 =================
def amplitude_encoding(x, n_qubits):
    """将18维特征编码到5量子比特的32维振幅空间"""
    x_norm = x / (np.sum(np.abs(x)) + 1e-8)
    amp = np.zeros(2 ** n_qubits, dtype=np.complex128)
    amp[:len(x_norm)] = x_norm
    amp = amp / np.sqrt(np.sum(np.abs(amp) ** 2) + 1e-8)
    return amp

def build_amplitude_circuit(amp, theta, n_qubits, n_layers):
    """构建振幅编码+变分线路"""
    prog = QProg()
    
    # 振幅编码
    for i in range(min(n_qubits, len(amp))):
        angle = 2 * np.arcsin(np.sqrt(np.abs(amp[i])))
        prog << RY(i, angle)
    
    theta = theta.reshape(n_layers, n_qubits, 2)
    
    # 变分层
    for l in range(n_layers):
        for i in range(n_qubits):
            prog << RY(i, theta[l, i, 0])
            prog << RZ(i, theta[l, i, 1])
        for i in range(n_qubits):
            prog << CNOT(i, (i + 1) % n_qubits)
    
    return prog

def quantum_forward(qvm, X_batch, theta, n_qubits, n_layers, shots):
    """量子前向传播"""
    nq = n_qubits
    results = []
    
    for x in X_batch:
        amp = amplitude_encoding(x, n_qubits)
        prog = build_amplitude_circuit(amp, theta, n_qubits, n_layers)
        
        for i in range(nq):
            prog << measure(i, i)
        
        qvm.run(prog, shots)
        counts = qvm.result().get_counts()
        
        # 计算期望值
        z = np.zeros(nq, dtype=np.float64)
        total_shots = 0
        for bitstring, count in counts.items():
            total_shots += count
            for i in range(nq):
                if len(bitstring) > i and bitstring[-1-i] == '0':
                    z[i] += count
        
        if total_shots > 0:
            z = 2 * z / total_shots - 1
        
        # 构造特征
        features = list(z)
        for i in range(nq - 1):
            features.append(z[i] * z[i+1])
        features.append(z[0] * z[-1])
        
        results.append(features)
    
    return np.array(results, dtype=np.float64)

# ================= 指标函数 =================
def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - ss_res / (ss_tot + 1e-12)

# ================= 得分判定函数 =================
def evaluate_score(mae, rmse, r2):
    """根据MAE、RMSE、R²判定得分等级"""
    score = {}
    
    # MAE得分判定（越小越好）
    if mae < 0.5:
        score['mae'] = '优秀'
    elif mae < 1.0:
        score['mae'] = '良好'
    elif mae < 1.5:
        score['mae'] = '一般'
    else:
        score['mae'] = '较差'
    
    # RMSE得分判定（越小越好）
    if rmse < 0.7:
        score['rmse'] = '优秀'
    elif rmse < 1.2:
        score['rmse'] = '良好'
    elif rmse < 1.8:
        score['rmse'] = '一般'
    else:
        score['rmse'] = '较差'
    
    # R²得分判定（越接近1越好）
    if r2 > 0.8:
        score['r2'] = '优秀'
    elif r2 >= 0.6:
        score['r2'] = '良好'
    elif r2 >= 0.4:
        score['r2'] = '一般'
    else:
        score['r2'] = '较差'
    
    return score

# ================= 主函数 =================
def main():
    print("=" * 70)
    print("量子变分回归器 - 测试集评估")
    print("=" * 70)
    
    # 加载模型
    print("\n加载模型...")
    model = np.load("best_model/model.npz", allow_pickle=True)
    
    theta = model["theta"]
    w = model["w"]
    b = model["b"]
    xmin = model["xmin"]
    xrange = model["xrange"]
    y_mean = float(model["y_mean"])
    age_median = float(model["age_median"])
    n_qubits = int(model["n_qubits"])
    n_layers = int(model["n_layers"])
    shots = int(model["shots"])
    
    print(f"量子比特: {n_qubits} | 变分层数: {n_layers}")
    print(f"标签均值: {y_mean:.4f} | 年龄中位数: {age_median:.1f}")
    print(f"经典参数: {w.size + b.size} (≤100)")
    
    # 加载测试集
    print("\n加载测试集...")
    rows = load_csv("eval.csv")
    clean_data_eval(rows, age_median)
    
    # 构建特征（不包含Recent_Stress）
    X_test = build_x(rows)
    
    # 提取真实标签
    y_true = np.array([float(r["Recent_Stress"]) for r in rows], dtype=np.float64)
    
    # 归一化
    X_test_norm = normalize_apply(X_test, xmin, xrange)
    
    print(f"测试样本数: {len(X_test)}")
    
    # 量子推理
    print("\n执行量子推理...")
    qvm = CPUQVM()
    features = quantum_forward(qvm, X_test_norm, theta, n_qubits, n_layers, shots)
    
    # 预测
    pred_raw = (features @ w + b).flatten()
    y_pred = np.clip(pred_raw + y_mean, 0, 5)
    
    # 计算指标
    test_mae = mae(y_true, y_pred)
    test_rmse = rmse(y_true, y_pred)
    test_r2 = r2_score(y_true, y_pred)
    
    print("\n" + "=" * 70)
    print("测试集评估结果")
    print("-" * 70)
    print(f"MAE:  {test_mae:.6f}")
    print(f"RMSE: {test_rmse:.6f}")
    print(f"R²:   {test_r2:.6f}")
    print("-" * 70)
    
    # 得分判定
    score = evaluate_score(test_mae, test_rmse, test_r2)
    print("\n得分情况判定：")
    print(f"MAE得分: {score['mae']}")
    print(f"RMSE得分: {score['rmse']}")
    print(f"R²得分: {score['r2']}")
    print("-" * 70)
    print(f"样本数: {len(X_test)}")
    print("=" * 70)

if __name__ == "__main__":
    main()
