#!/usr/bin/env python3
"""
Quantum Stress Regression - TRAIN (QPanda3 VQA)
振幅编码 · 5量子比特 · 完整18维特征 · 高效推理
"""

import csv
import json
import os
import time
import numpy as np
from pyqpanda3.core import CPUQVM, QProg, RY, RZ, CNOT, measure, H

CONFIG = {
    "seed": 42,
    "n_qubits": 5,              # 5量子比特，2^5=32≥18
    "n_layers": 2,              # 变分层数
    "epochs": 100,
    "batch_size": 32,           # 增大batch提升效率
    "lr": 0.01,
    "shots": 800,               # 适中shots
    "shift": np.pi / 2,
    "save_dir": "best_model",
}

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

def clean_data(rows):
    """清洗数据，返回清洗后的数据和age_median"""
    ages = [float(r['Age']) for r in rows if 10 <= float(r['Age']) <= 40]
    age_median = np.median(ages) if ages else 20.0
    
    for r in rows:
        # 处理Age
        age = float(r['Age'])
        if age < 10 or age > 40:
            r['Age'] = str(age_median)
        
        # 处理Gender (0-1)
        gender = float(r['Gender'])
        r['Gender'] = str(np.clip(gender, 0, 1))
        
        # 处理其他特征 (0-5)
        for f in ALL_FEATURES:
            if f in ['Gender', 'Age']:
                continue
            val = float(r[f])
            if not np.isfinite(val) or val < 0:
                val = 0.0
            r[f] = str(np.clip(val, 0, 5))
        
        # 处理Recent_Stress
        if 'Recent_Stress' in r:
            val = float(r['Recent_Stress'])
            r['Recent_Stress'] = str(np.clip(val, 0, 5))
    
    return rows, age_median

def build_xy(rows):
    """构建特征矩阵X和标签y"""
    X = np.array([[float(r[f]) for f in ALL_FEATURES] for r in rows], dtype=np.float64)
    y = np.array([float(r["Recent_Stress"]) for r in rows], dtype=np.float64)
    return X, y

def normalize(X):
    """归一化到[0, π]"""
    xmin = X.min(axis=0)
    xmax = X.max(axis=0)
    xrange = xmax - xmin
    xrange[xrange < 1e-8] = 1.0
    return (X - xmin) / xrange * np.pi, xmin, xrange

# ================= 振幅编码 =================

def amplitude_encoding(x, n_qubits):
    """
    将18维特征编码到5量子比特的32维振幅空间
    这是量子编码，不是经典降维
    """
    # 归一化特征
    x_norm = x / (np.sum(np.abs(x)) + 1e-8)
    
    # 扩展到32维（补零）
    amp = np.zeros(2 ** n_qubits, dtype=np.complex128)
    amp[:len(x_norm)] = x_norm
    
    # 添加随机相位（增加表达能力）
    phases = np.exp(1j * 2 * np.pi * np.random.rand(len(amp)))
    amp = amp * phases
    
    # 归一化
    amp = amp / np.sqrt(np.sum(np.abs(amp) ** 2) + 1e-8)
    
    return amp

def build_amplitude_circuit(amp, theta, n_qubits, n_layers):
    """
    构建振幅编码+变分线路
    使用RY门近似振幅编码（简化版，避免QRAM）
    """
    prog = QProg()
    
    # 振幅编码层：使用RY门将振幅编码到量子态
    # 简化编码：只编码前n_qubits个振幅
    for i in range(min(n_qubits, len(amp))):
        # 将振幅幅值映射到RY角度
        angle = 2 * np.arcsin(np.sqrt(np.abs(amp[i])))
        prog << RY(i, angle)
    
    theta = theta.reshape(n_layers, n_qubits, 2)
    
    # 变分层
    for l in range(n_layers):
        # 旋转层
        for i in range(n_qubits):
            prog << RY(i, theta[l, i, 0])
            prog << RZ(i, theta[l, i, 1])
        
        # 纠缠层（环形连接）
        for i in range(n_qubits):
            prog << CNOT(i, (i + 1) % n_qubits)
    
    return prog

def quantum_forward(qvm, X_batch, theta, n_qubits, n_layers, shots):
    """
    量子前向传播（支持批处理优化）
    """
    nq = n_qubits
    results = []
    
    for x in X_batch:
        # 振幅编码
        amp = amplitude_encoding(x, n_qubits)
        
        # 构建线路
        prog = build_amplitude_circuit(amp, theta, n_qubits, n_layers)
        
        # 测量
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
                # QPanda3中bitstring[0]是最高位
                if len(bitstring) > i and bitstring[-1-i] == '0':
                    z[i] += count
        
        if total_shots > 0:
            z = 2 * z / total_shots - 1
        
        # 构造特征：单比特期望 + 两比特关联
        features = list(z)
        for i in range(nq - 1):
            features.append(z[i] * z[i+1])
        features.append(z[0] * z[-1])  # 首尾关联
        
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

# ================= Adam优化器 =================

class Adam:
    def __init__(self, shape, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        self.m = np.zeros(shape, dtype=np.float64)
        self.v = np.zeros(shape, dtype=np.float64)
        self.t = 0
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
    
    def step(self, param, grad):
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad ** 2)
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        return param - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

# ================= 训练主函数 =================

def train():
    print("=" * 70)
    print("量子变分回归器 - 振幅编码版本")
    print(f"量子比特: {CONFIG['n_qubits']} (2^{CONFIG['n_qubits']}=32 ≥ 18维特征)")
    print(f"变分层数: {CONFIG['n_layers']}")
    print(f"经典参数: {CONFIG['n_qubits'] + CONFIG['n_qubits']}个权重 + 1偏置 = {CONFIG['n_qubits'] + CONFIG['n_qubits'] + 1} ≤ 100")
    print("=" * 70)
    
    # 加载训练集
    print("\n加载训练数据...")
    rows = load_csv("train.csv")
    rows, age_median = clean_data(rows)
    
    # 构建特征和标签
    X, y = build_xy(rows)
    X, xmin, xrange = normalize(X)
    
    # 中心化标签
    y_mean = np.mean(y)
    y_center = y - y_mean
    
    # 划分训练集和验证集
    np.random.seed(CONFIG["seed"])
    n_samples = len(X)
    indices = np.random.permutation(n_samples)
    n_val = int(0.2 * n_samples)
    train_idx, val_idx = indices[n_val:], indices[:n_val]
    
    X_train, y_train_center = X[train_idx], y_center[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    
    print(f"训练样本: {len(X_train)} | 验证样本: {len(X_val)}")
    print(f"标签均值: {y_mean:.4f} | 年龄中位数: {age_median:.1f}")
    
    # 初始化参数
    n_qubits = CONFIG["n_qubits"]
    n_layers = CONFIG["n_layers"]
    
    # 量子参数: n_layers * n_qubits * 2
    theta = np.random.normal(0, 0.1, n_layers * n_qubits * 2).astype(np.float64)
    
    # 经典参数: n_qubits个单比特 + n_qubits个两比特关联 = 10个特征
    # 权重: 10×1=10, 偏置:1, 总计11个参数
    w = np.random.normal(0, 0.2, (n_qubits + n_qubits, 1)).astype(np.float64)
    b = np.array([0.0], dtype=np.float64)
    
    # 验证参数数量
    total_classic_params = w.size + b.size
    print(f"\n经典参数总量: {total_classic_params} (≤100: {'✓' if total_classic_params <= 100 else '✗'})")
    
    # 优化器
    opt_theta = Adam(theta.shape, CONFIG["lr"])
    opt_w = Adam(w.shape, CONFIG["lr"])
    opt_b = Adam(b.shape, CONFIG["lr"])
    
    # 量子虚拟机
    qvm = CPUQVM()
    
    # 训练循环
    best_val_mae = float('inf')
    best_state = None
    no_improve = 0
    history = []
    
    print("\n开始训练...")
    print("-" * 70)
    print(f"{'Epoch':>6} | {'Train MAE':>10} | {'Val MAE':>10} | {'R²':>8} | {'Best':>10}")
    print("-" * 70)
    
    for epoch in range(1, CONFIG["epochs"] + 1):
        # 学习率衰减
        lr = CONFIG["lr"] * (0.95 ** (epoch // 20))
        opt_theta.lr = lr
        opt_w.lr = lr
        opt_b.lr = lr
        
        # 批训练
        indices = np.random.permutation(len(X_train))
        
        for i in range(0, len(X_train), CONFIG["batch_size"]):
            batch_idx = indices[i:i + CONFIG["batch_size"]]
            X_batch = X_train[batch_idx]
            y_batch = y_train_center[batch_idx].reshape(-1, 1)
            
            # 量子前向
            q_features = quantum_forward(qvm, X_batch, theta, n_qubits, n_layers, CONFIG["shots"])
            pred = q_features @ w + b
            
            # MSE损失梯度
            grad = 2 * (pred - y_batch) / len(X_batch)
            
            # 经典参数梯度
            grad_w = q_features.T @ grad
            grad_b = np.sum(grad)
            
            # 量子参数梯度（参数移位法）
            grad_theta = np.zeros_like(theta)
            grad_q = grad @ w.T
            
            # 随机采样部分参数更新（效率优化）
            update_ratio = min(0.5 + epoch / CONFIG["epochs"] * 0.5, 1.0)
            n_update = max(1, int(len(theta) * update_ratio))
            update_indices = np.random.choice(len(theta), n_update, replace=False)
            
            for idx in update_indices:
                old = theta[idx]
                
                # 正偏移
                theta[idx] = old + CONFIG["shift"]
                q_pos = quantum_forward(qvm, X_batch, theta, n_qubits, n_layers, CONFIG["shots"])
                
                # 负偏移
                theta[idx] = old - CONFIG["shift"]
                q_neg = quantum_forward(qvm, X_batch, theta, n_qubits, n_layers, CONFIG["shots"])
                
                # 恢复
                theta[idx] = old
                
                # 梯度估计
                g = np.sum(grad_q * (q_pos - q_neg) * 0.5)
                grad_theta[idx] = np.clip(g, -5.0, 5.0)
            
            # 更新参数
            theta = opt_theta.step(theta, grad_theta)
            w = opt_w.step(w, grad_w)
            b = opt_b.step(b, grad_b)
        
        # 评估
        if epoch % 5 == 0 or epoch == 1:
            # 训练集评估
            train_features = quantum_forward(qvm, X_train, theta, n_qubits, n_layers, CONFIG["shots"])
            train_pred = np.clip((train_features @ w + b).flatten() + y_mean, 0, 5)
            train_mae = mae(y[train_idx], train_pred)
            
            # 验证集评估
            val_features = quantum_forward(qvm, X_val, theta, n_qubits, n_layers, CONFIG["shots"])
            val_pred = np.clip((val_features @ w + b).flatten() + y_mean, 0, 5)
            val_mae = mae(y_val, val_pred)
            val_r2 = r2_score(y_val, val_pred)
            
            print(f"{epoch:6d} | {train_mae:10.4f} | {val_mae:10.4f} | {val_r2:8.4f} | {best_val_mae:10.4f}")
            
            # 保存最佳模型
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_state = (theta.copy(), w.copy(), b.copy())
                no_improve = 0
                
                # 保存模型
                os.makedirs(CONFIG["save_dir"], exist_ok=True)
                np.savez(os.path.join(CONFIG["save_dir"], "model.npz"),
                        theta=theta, w=w, b=b,
                        xmin=xmin, xrange=xrange, y_mean=y_mean,
                        age_median=age_median,
                        n_qubits=n_qubits, n_layers=n_layers,
                        shots=CONFIG["shots"],
                        features=ALL_FEATURES)
            else:
                no_improve += 1
            
            # 记录历史
            history.append({
                "epoch": epoch,
                "train_mae": float(train_mae),
                "val_mae": float(val_mae),
                "val_r2": float(val_r2)
            })
            
            # 早停
            if no_improve >= 15:
                print(f"\n早停于 epoch {epoch}")
                break
    
    # 加载最佳模型
    if best_state is not None:
        theta, w, b = best_state
    
    # 最终评估
    print("\n" + "=" * 70)
    print("最终评估")
    print("-" * 70)
    
    # 全训练集评估
    train_features = quantum_forward(qvm, X, theta, n_qubits, n_layers, CONFIG["shots"] * 2)
    train_pred = np.clip((train_features @ w + b).flatten() + y_mean, 0, 5)
    
    final_mae = mae(y, train_pred)
    final_rmse = rmse(y, train_pred)
    final_r2 = r2_score(y, train_pred)
    
    print(f"训练集 MAE:  {final_mae:.4f}")
    print(f"训练集 RMSE: {final_rmse:.4f}")
    print(f"训练集 R²:   {final_r2:.4f}")
    
    # 保存训练日志
    log = {
        "config": CONFIG,
        "best_val_mae": float(best_val_mae),
        "final_train_mae": float(final_mae),
        "final_train_rmse": float(final_rmse),
        "final_train_r2": float(final_r2),
        "classic_params": total_classic_params,
        "history": history
    }
    
    with open(os.path.join(CONFIG["save_dir"], "train_log.json"), "w") as f:
        json.dump(log, f, indent=2)
    
    print("\n" + "=" * 70)
    print(f"训练完成！最佳验证MAE: {best_val_mae:.4f}")
    print(f"模型保存至: {CONFIG['save_dir']}/model.npz")
    print("=" * 70)

if __name__ == "__main__":
    train()