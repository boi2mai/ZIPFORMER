import csv
import json
import math
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn

@dataclass
class HardwareConfig:
    # 1. Kích thước mô hình
    T: int
    D_in: int
    D_ffw: int
    D_attn: int
    D_chunk: int
    
    # 2. Chiến thuật Tiling
    strategy_name: str
    ffw_tile_T: int
    ffw_tile_H: int
    attn_tile_T: int
    attn_tile_C: int
    
    # 3. Giới hạn phần cứng
    num_pe: int
    sram_limit_kb: int

    @classmethod
    def from_json(cls, filepath: str):
        """Hàm tự động đọc file JSON và nặn thành object HardwareConfig"""
        
        # 1. Lấy đường dẫn tuyệt đối của cái thư mục đang chứa file code này
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 2. Ghép nó với tên file JSON để tạo ra tọa độ chính xác 100%
        absolute_path = os.path.join(script_dir, filepath)
        
        with open(absolute_path, 'r') as f:
            data = json.load(f)
            
        return cls(
            T=data["model_shape"]["T"],
            D_in=data["model_shape"]["D_in"],
            D_ffw=data["model_shape"]["D_ffw"],
            D_attn=data["model_shape"]["D_attn"],
            D_chunk=data["model_shape"]["D_chunk"],
            
            strategy_name=data["tiling_strategy"]["name"],
            ffw_tile_T=data["tiling_strategy"]["ffw_tile_T"],
            ffw_tile_H=data["tiling_strategy"]["ffw_tile_H"],
            attn_tile_T=data["tiling_strategy"]["attn_tile_T"],
            attn_tile_C=data["tiling_strategy"]["attn_tile_C"],
            
            num_pe=data["hardware_constraints"]["num_pe"],
            sram_limit_kb=data["hardware_constraints"]["sram_limit_kb"]
        )
# ==============================================================================
# 1. HARDWARE UTILITIES (CÁC KHỐI DÙNG CHUNG)
# ==============================================================================
class Compute_block:
    """Processing Element (PE) - Float64 precision để đảm bảo sai số siêu nhỏ (~e-14)"""
    def __init__(self, num_pe=256):
        self.num_pe = num_pe
        self.input_reg = np.zeros(self.num_pe, dtype=np.float64)   
        self.weight_reg = np.zeros(self.num_pe, dtype=np.float64)  
        self.output_reg = np.zeros(1, dtype=np.float64)    
        self.current_bias = np.float64(0.0)

    def load_bias(self, bias_val): 
        self.current_bias = np.float64(bias_val)

    def load_in_local_to_Compute(self, input_vector):
        length = len(input_vector)
        self.input_reg[:length] = input_vector
        self.input_reg[length:] = 0.0

    def load_weight_to_Compute(self, weight_vector):
        length = len(weight_vector)
        self.weight_reg[:length] = weight_vector
        self.weight_reg[length:] = 0.0

    def run_mac(self):
        self.output_reg = np.sum(self.input_reg * self.weight_reg) + self.current_bias
        return self.output_reg

class HardwareProfiler:
    """Bộ đếm hiệu năng độc lập cho từng Module"""
    def __init__(self, name="Module"):
        self.name = name
        self.metrics = {'g_loads': 0, 'g_stores': 0, 'l_loads': 0, 'l_stores': 0, 'macs': 0}
    
    def track(self, metric, val): 
        self.metrics[metric] += val
    
    def print_report(self):
        g_loads = self.metrics['g_loads']
        macs = self.metrics['macs']
        intensity = macs / g_loads if g_loads > 0 else 0
        
        print("="*60)
        print(f"📊 BÁO CÁO PHÂN TÍCH HIỆU NĂNG TILING [{self.name.upper()}]")
        print("="*60)
        print(f"Tổng số phần tử Load từ GLOBAL (DRAM) : {g_loads:,.0f} words")
        print(f"Tổng số phần tử Ghi ra GLOBAL (DRAM)  : {self.metrics['g_stores']:,.0f} words")
        print(f"Tổng số phần tử Load từ LOCAL vào Core: {self.metrics['l_loads']:,.0f} words")
        print(f"Tổng số phần tử Ghi từ Core ra LOCAL  : {self.metrics['l_stores']:,.0f} words")
        print(f"Tổng số phép toán tính toán (MAC Ops) : {macs:,.0f} ops")
        print("-" * 60)
        print(f"💡 Cường độ tính toán (Arithmetic Intensity):")
        print(f"   => {intensity:.2f} phép MACs / 1 lần truy cập Global DRAM")
        print("="*60)


# ==============================================================================
# 2. FEEDFORWARD MODULE 
# ==============================================================================
class FeedforwardHardware:
    def __init__(self, config: HardwareConfig):
        # 1. Nạp cấu hình từ phễu config
        self.T = config.T
        self.D_in = config.D_in
        self.D_ffw = config.D_ffw
        self.tile_T = config.ffw_tile_T
        self.tile_H = config.ffw_tile_H
        
        self.profiler = HardwareProfiler(name=f"FFW ({config.strategy_name})")
        
        self.W1 = np.random.randn(self.D_in, self.D_ffw).astype(np.float64) / 30.0
        self.b1 = np.random.randn(self.D_ffw).astype(np.float64)
        self.W2 = np.random.randn(self.D_ffw, self.D_in).astype(np.float64) / 30.0
        self.b2 = np.random.randn(self.D_in).astype(np.float64)
        
        self.pe = Compute_block(num_pe=config.num_pe)

    def swooshL(self, x):
        return np.float64(np.logaddexp(0.0, x - 4.0) - 0.08 * x - 0.035)

    def execute(self, global_X):
        """
        ========== MODULE: FEEDFORWARD EXECUTION ==========

        Input:
          - global_X [T, D_in] (Nằm trên DRAM)

        Compute:
          - H = SwooshL(X @ W1 + b1)
          - Y = H @ W2 + b2 + X
        -> Y = SwooshL(X @ W1 + b1) @ W2 + b2 + X

        Tiling Strategy (Weight Stationary - Giữ trọng số):
          - Time Tiling: Nạp tile_T (58 hàng) Input X vào Local SRAM một lần.
          - Hidden Tiling: Nạp tile_H (192 cột) của W1, W2 vào Local SRAM để vừa khít 256 thanh ghi PE.
          - Tái sử dụng Trọng số: Trọng số nạp 1 lần từ DRAM được dùng để nhân với toàn bộ tile_T hàng X.
            -> Thủ thuật này tối ưu hóa Arithmetic Intensity lên mức > 50 MACs/Load.

        Memory Flow:
          1. Load local_X (tile_T x D_in) từ Global DRAM -> Local SRAM.
          2. Load W1_tile, W2_tile từ Global DRAM -> Local SRAM.
          3. PE tính lớp In-Proj, ghi kết quả trung gian ra local_H (SRAM).
          4. PE đọc từ local_H, tính lớp Out-Proj, cộng dồn vào local_Y (SRAM).
          5. Ghi final output từ local_Y -> Global DRAM.
        """    
        output = np.zeros((self.T, self.D_in), dtype=np.float64)
        local_X = np.zeros((self.tile_T, self.D_in), dtype=np.float64)
        local_Y = np.zeros((self.tile_T, self.D_in), dtype=np.float64)
        local_H = np.zeros((self.tile_T, self.tile_H), dtype=np.float64)
        
        for t_start in range(0, self.T, self.tile_T):
            act_T = min(self.tile_T, self.T - t_start)
            
            # 1. Load Input Tile từ DRAM vào SRAM
            local_X[:act_T, :] = global_X[t_start:t_start+act_T, :]
            self.profiler.track('g_loads', act_T * self.D_in)
            self.profiler.track('l_stores', act_T * self.D_in)
            
            self.profiler.track('g_loads', self.D_in) 
            for t in range(act_T):
                # Ghi sổ Bias b2 vào local_Y một lần duy nhất cho toàn bộ tile_T hàng (tái sử dụng khi cộng vào bước cuối)
                local_Y[t, :] = self.b2
                self.profiler.track('l_stores', self.D_in)
            
            for h_start in range(0, self.D_ffw, self.tile_H):
                # KHAI BÁO BIẾN act_H TẠI ĐÂY
                act_H = min(self.tile_H, self.D_ffw - h_start)
                
                # 2. Load Weight Tile từ DRAM vào SRAM (Sử dụng act_H)
                W1_tile = self.W1[:, h_start : h_start+act_H]
                b1_tile = self.b1[h_start : h_start+act_H]
                W2_tile = self.W2[h_start : h_start+act_H, :]
                
                # Ghi sổ Profiler theo act_H
                self.profiler.track('g_loads', self.D_in * act_H + act_H + act_H * self.D_in)
                self.profiler.track('l_stores', self.D_in * act_H + act_H + act_H * self.D_in)
                
                # --- 3. Tính toán Lớp In-Proj (X @ W1 + b1) ---
                for t in range(act_T):
                    self.pe.load_in_local_to_Compute(local_X[t, :])
                    self.profiler.track('l_loads', self.D_in)
                    
                    for h in range(act_H): # Lặp theo act_H
                        self.pe.load_weight_to_Compute(W1_tile[:, h])
                        self.pe.load_bias(b1_tile[h])
                        self.profiler.track('l_loads', self.D_in + 1)
                        
                        local_H[t, h] = self.swooshL(self.pe.run_mac())
                        self.profiler.track('macs', self.D_in)
                        self.profiler.track('l_stores', 1)
                
                # --- 4. Tính toán Lớp Out-Proj (H @ W2 + 0) ---
                for t in range(act_T):
                    # CHÚ Ý: Cắt mảng local_H theo đúng kích thước act_H để tránh lỗi
                    self.pe.load_in_local_to_Compute(local_H[t, :act_H])
                    self.profiler.track('l_loads', act_H)
                    self.pe.load_bias(0.0) 
                    
                    for d in range(self.D_in):
                        self.pe.load_weight_to_Compute(W2_tile[:, d])
                        self.profiler.track('l_loads', act_H + 1)
                        
                        local_Y[t, d] += self.pe.run_mac()
                        self.profiler.track('l_stores', 1)
                        self.profiler.track('macs', act_H) # Ghi sổ theo act_H
            
            # --- 5. Cộng Residual và Ghi ra DRAM ---
            for t in range(act_T):
                for d in range(self.D_in):
                    self.profiler.track('l_loads', 2) 
                    output[t_start+t, d] = local_Y[t, d] + local_X[t, d]
                    self.profiler.track('g_stores', 1)
                    
        return output, self.profiler
      
class FeedforwardChannelFirst:
    """
    CHIẾN THUẬT 2: CHANNEL-FIRST TILING (INPUT/WEIGHT STATIONARY)
    [ĐÃ SỬA]: Nạp Input X từ DRAM vào SRAM đúng 1 LẦN DUY NHẤT ở đầu khối!
    """
    def __init__(self, config: HardwareConfig):
        self.T = config.T
        self.D_in = config.D_in
        self.D_ffw = config.D_ffw
        self.tile_T = config.ffw_tile_T
        self.tile_H = config.ffw_tile_H
        
        self.profiler = HardwareProfiler(name=f"FFW ({config.strategy_name})")
        
        self.W1 = np.random.randn(self.D_in, self.D_ffw).astype(np.float64) / 30.0
        self.b1 = np.random.randn(self.D_ffw).astype(np.float64)
        self.W2 = np.random.randn(self.D_ffw, self.D_in).astype(np.float64) / 30.0
        self.b2 = np.random.randn(self.D_in).astype(np.float64)

    def swooshL(self, x):
        return np.float64(np.logaddexp(0.0, x - 4.0) - 0.08 * x - 0.035)

    def execute(self, global_X):
        global_Output = np.zeros((self.T, self.D_in), dtype=np.float64)
        local_Y_acc = np.zeros((self.T, self.D_in), dtype=np.float64)
        
        # =====================================================================
        # 🎯 [CHUẨN PHẦN CỨNG]: NẠP TOÀN BỘ X TỪ DRAM VÀO SRAM 1 LẦN DUY NHẤT!
        # =====================================================================
        local_X_full = global_X.copy()
        self.profiler.track('g_loads', self.T * self.D_in)
        self.profiler.track('l_stores', self.T * self.D_in)

        for h_start in range(0, self.D_ffw, self.tile_H):
            act_H = min(self.tile_H, self.D_ffw - h_start)
            
            # Load Weight Tile từ DRAM vào SRAM
            W1_tile = self.W1[:, h_start:h_start+act_H]
            b1_tile = self.b1[h_start:h_start+act_H]
            W2_tile = self.W2[h_start:h_start+act_H, :]
            
            self.profiler.track('g_loads', self.D_in * act_H + act_H + act_H * self.D_in)
            self.profiler.track('l_stores', self.D_in * act_H + act_H + act_H * self.D_in)

            for t_start in range(0, self.T, self.tile_T):
                act_T = min(self.tile_T, self.T - t_start)
                
                # PE ĐỌC X TRỰC TIẾP TỪ SRAM (l_loads), KHÔNG PHẢI TỪ DRAM (g_loads)!
                local_X = local_X_full[t_start:t_start+act_T, :]
                self.profiler.track('l_loads', act_T * self.D_in)
                
                # In-Proj
                local_H_chunk = np.dot(local_X, W1_tile) + b1_tile
                self.profiler.track('l_loads', self.D_in * act_H + act_H)
                self.profiler.track('macs', act_T * self.D_in * act_H)
                
                # Activation
                local_H_chunk = np.array([[self.swooshL(val) for val in row] for row in local_H_chunk])
                
                # Out-Proj
                partial_Y = np.dot(local_H_chunk, W2_tile)
                self.profiler.track('l_loads', act_T * act_H + act_H * self.D_in)
                self.profiler.track('macs', act_T * act_H * self.D_in)
                
                local_Y_acc[t_start:t_start+act_T, :] += partial_Y
                self.profiler.track('l_stores', act_T * self.D_in)
                
        # Residual Add & Store ra DRAM
        for t in range(self.T):
            global_Output[t, :] = local_Y_acc[t, :] + self.b2 + global_X[t, :]
            self.profiler.track('g_stores', self.D_in)
            
        return global_Output, self.profiler

class FeedforwardBlockTiling:
    """
    CHIẾN THUẬT 3: BLOCK TILING (OUTPUT-STATIONARY)
    Cắt cả ma trận thành các ô vuông. Giữ chặt ô Output trên thanh ghi (PE) để cộng dồn.
    """
    def __init__(self, config: HardwareConfig):
        self.T = config.T
        self.D_in = config.D_in
        self.D_ffw = config.D_ffw
        self.tile_T = config.ffw_tile_T
        self.tile_H = config.ffw_tile_H
        
        self.profiler = HardwareProfiler(name=f"FFW ({config.strategy_name})")
        
        self.W1 = np.random.randn(self.D_in, self.D_ffw).astype(np.float64) / 30.0
        self.b1 = np.random.randn(self.D_ffw).astype(np.float64)
        self.W2 = np.random.randn(self.D_ffw, self.D_in).astype(np.float64) / 30.0
        self.b2 = np.random.randn(self.D_in).astype(np.float64)

    def swooshL(self, x):
        return np.float64(np.logaddexp(0.0, x - 4.0) - 0.08 * x - 0.035)

    def execute(self, global_X):
        global_Output = np.zeros((self.T, self.D_in), dtype=np.float64)
        local_H_buffer = np.zeros((self.T, self.D_ffw), dtype=np.float64)
        
        # --- LỚP IN-PROJ (X @ W1 + b1) ---
        # Quét theo từng block 2D
        for t_start in range(0, self.T, self.tile_T):
            act_T = min(self.tile_T, self.T - t_start)
            # Load 1 tile Input X
            local_X = global_X[t_start:t_start+act_T, :]
            self.profiler.track('g_loads', act_T * self.D_in)
            self.profiler.track('l_stores', act_T * self.D_in) # [MỚI THÊM]
            
            for h_start in range(0, self.D_ffw, self.tile_H):
                act_H = min(self.tile_H, self.D_ffw - h_start)
                
                # Load 1 tile Weight W1
                local_W1 = self.W1[:, h_start:h_start+act_H]
                local_b1 = self.b1[h_start:h_start+act_H]
                self.profiler.track('g_loads', self.D_in * act_H + act_H)
                self.profiler.track('l_stores', self.D_in * act_H + act_H) # [MỚI THÊM]
                
                # Tính Block Output H
                local_H_block = np.dot(local_X, local_W1) + local_b1
                self.profiler.track('l_loads', act_T * self.D_in + self.D_in * act_H + act_H) # [MỚI THÊM]
                self.profiler.track('macs', act_T * self.D_in * act_H)
                
                # Activation tại chỗ trên PE
                local_H_block = np.array([[self.swooshL(val) for val in row] for row in local_H_block])
                
                # Cất Block H vào SRAM tổng
                local_H_buffer[t_start:t_start+act_T, h_start:h_start+act_H] = local_H_block
                self.profiler.track('l_stores', act_T * act_H) # [MỚI THÊM] Ghi Block H ra SRAM
                
        # --- LỚP OUT-PROJ (H @ W2) VÀ CỘNG RESIDUAL ---
        for t_start in range(0, self.T, self.tile_T):
            act_T = min(self.tile_T, self.T - t_start)
            
            for d_start in range(0, self.D_in, self.tile_H): # Tái sử dụng tile_H cho chiều D_in
                act_D = min(self.tile_H, self.D_in - d_start)
                
                # Khởi tạo Block Y (Output-Stationary) trên SRAM
                local_Y_block = np.zeros((act_T, act_D), dtype=np.float64)
                
                # Tích lũy (Reduction) dọc theo chiều D_ffw
                for h_start in range(0, self.D_ffw, self.tile_H):
                    act_H = min(self.tile_H, self.D_ffw - h_start)
                    
                    local_H_in = local_H_buffer[t_start:t_start+act_T, h_start:h_start+act_H]
                    local_W2 = self.W2[h_start:h_start+act_H, d_start:d_start+act_D]
                    self.profiler.track('g_loads', act_H * act_D) # Chỉ load Weight, H đã có trên SRAM
                    self.profiler.track('l_stores', act_H * act_D) # [MỚI THÊM]
                    
                    local_Y_block += np.dot(local_H_in, local_W2)
                    self.profiler.track('l_loads', act_T * act_H + act_H * act_D) # [MỚI THÊM]
                    self.profiler.track('macs', act_T * act_H * act_D)
                
                # Cộng Bias B2, Residual X và Ghi ra DRAM
                b2_tile = self.b2[d_start:d_start+act_D] # Lấy mẩu bias tương ứng
                local_X_res = global_X[t_start:t_start+act_T, d_start:d_start+act_D]
                
                self.profiler.track('g_loads', act_T * act_D + act_D) # Thêm track load bias
                
                global_Output[t_start:t_start+act_T, d_start:d_start+act_D] = local_Y_block + b2_tile + local_X_res
                self.profiler.track('g_stores', act_T * act_D)

        return global_Output, self.profiler

# ==============================================================================
# 3. NONLINEAR ATTENTION MODULE 
# ==============================================================================
class NonlinearAttentionHardware:
    def __init__(self, config: HardwareConfig):
        # 1. Nạp cấu hình từ phễu config
        self.T = config.T
        self.D_in = config.D_in
        self.D_attn = config.D_attn
        self.D_chunk = config.D_chunk
        self.tile_T = config.attn_tile_T
        self.tile_C = config.attn_tile_C
        
        self.profiler = HardwareProfiler(name=f"Attention ({config.strategy_name})")
        
        # SỬA Ở ĐÂY: Dùng self.D_in, self.D_attn, self.D_chunk
        self.W_in = np.random.randn(self.D_in, self.D_attn).astype(np.float64) / 30.0
        self.b_in = np.random.randn(self.D_attn).astype(np.float64)
        
        self.W_out = np.random.randn(self.D_chunk, self.D_in).astype(np.float64) / 30.0
        self.b_out = np.random.randn(self.D_in).astype(np.float64)
        
        # Nạp giới hạn phần cứng (256 PEs)
        self.pe = Compute_block(num_pe=config.num_pe)
    def execute(self, global_X, residual_X, global_attn_weights):
        output = np.zeros((self.T, self.D_in), dtype=np.float64)
        
        local_s_tanh = np.zeros((self.T, self.D_chunk), dtype=np.float64)
        local_x_gated = np.zeros((self.T, self.D_chunk), dtype=np.float64) # Đã đổi tên để chứa x sau khi nhân s_tanh
        local_y = np.zeros((self.T, self.D_chunk), dtype=np.float64)
        
        """
        --- PHASE 1: IN-PROJ (LINEAR) & CHUNK ---
        """
        tile_C = 144 
        local_X = np.zeros((self.tile_T, self.D_in), dtype=np.float64)
        
        for t_start in range(0, self.T, self.tile_T):
            act_T = min(self.tile_T, self.T - t_start)
            
            # Load Input Tile từ DRAM -> SRAM
            local_X[:act_T, :] = global_X[t_start:t_start+act_T, :]
            self.profiler.track('g_loads', act_T * self.D_in)
            self.profiler.track('l_stores', act_T * self.D_in)
            
            for c_start in range(0, self.D_attn, tile_C):
                # Load Weight Tile từ DRAM -> SRAM
                W_in_tile = self.W_in[:, c_start:c_start+tile_C]
                b_in_tile = self.b_in[c_start:c_start+tile_C]
                self.profiler.track('g_loads', self.D_in * tile_C + tile_C)
                self.profiler.track('l_stores', self.D_in * tile_C + tile_C)
                
                for t in range(act_T):
                    self.pe.load_in_local_to_Compute(local_X[t, :])
                    self.profiler.track('l_loads', self.D_in)
                    
                    for c in range(tile_C):
                        self.pe.load_weight_to_Compute(W_in_tile[:, c])
                        self.pe.load_bias(b_in_tile[c])
                        self.profiler.track('l_loads', self.D_in + 1)
                        
                        # Nhân và công Bias, sau đó phân rã trực tiếp vào S, X, Y tùy theo vị trí Cột (c_start + c)
                        val = self.pe.run_mac()
                        self.profiler.track('macs', self.D_in)
                        
                        # [CẬP NHẬT TOÁN HỌC MỚI]: Phân rã & Dung hợp tức thời (Instant Fusion)
                        global_c = c_start + c
                        if global_c < self.D_chunk:
                            # Nhánh S: Đi qua tanh và cất vào SRAM
                            local_s_tanh[t_start+t, global_c] = np.tanh(val) 
                        elif global_c < 2 * self.D_chunk:
                            # Nhánh X: Lấy S_tanh trên SRAM ra nhân NGAY LẬP TỨC với X (Element-wise)
                            d = global_c - self.D_chunk
                            local_x_gated[t_start+t, d] = local_s_tanh[t_start+t, d] * val
                            self.profiler.track('l_loads', 1) # Track 1 lần đọc s_tanh
                        else:
                            # Nhánh Y: Cất tạm vào SRAM
                            local_y[t_start+t, global_c - 2 * self.D_chunk] = val
                        self.profiler.track('l_stores', 1)

        """
        --- PHASE 2: MATMUL VỚI ATTN_WEIGHTS TỪ BÊN NGOÀI ---
        """
        local_attn_out = np.zeros((self.T, self.D_chunk), dtype=np.float64)
        for t in range(self.T):
            # 1. Nạp 1 hàng Attention Weights từ DRAM
            attn_row = global_attn_weights[t, :]
            self.pe.load_in_local_to_Compute(attn_row)
            self.profiler.track('g_loads', self.T) 
            self.pe.load_bias(0.0)
            
            for d in range(self.D_chunk):
                # 2. Lấy cột d của ma trận x_gated (đã nhân s_tanh) từ SRAM
                x_col = local_x_gated[:, d]
                self.pe.load_weight_to_Compute(x_col)
                self.profiler.track('l_loads', self.T)
                
                # 3. Thực hiện Matmul: attn_weights @ x_gated
                attn_x_val = self.pe.run_mac()
                self.profiler.track('macs', self.T)
                
                # 4. [CẬP NHẬT TOÁN HỌC MỚI]: Chỉ nhân Element-wise với nhánh Y
                local_attn_out[t, d] = attn_x_val * local_y[t, d]
                
                self.profiler.track('l_loads', 1) # Chỉ đọc y từ SRAM (ít tốn chu kỳ hơn bản cũ!)
                self.profiler.track('l_stores', 1) 

        """
        --- PHASE 3: OUT-PROJ & RESIDUAL ---
        
        Compute:
          - final_out = attn_out @ W_out + b_out + residual_X
          
        Memory Flow:
          1. Đọc local_attn_out từ SRAM.
          2. Nạp W_out từ DRAM -> SRAM -> PE.
          3. Đọc residual từ DRAM, tính toán và ghi final_out -> DRAM.
        """
        for t_start in range(0, self.T, self.tile_T):
            act_T = min(self.tile_T, self.T - t_start)
            
            for d in range(self.D_in):
                # Nạp Weight và Bias từ DRAM
                W_out_col = self.W_out[:, d]
                self.profiler.track('g_loads', self.D_chunk + 1)
                self.profiler.track('l_stores', self.D_chunk + 1)
                
                for t in range(act_T):
                    self.pe.load_in_local_to_Compute(local_attn_out[t_start+t, :])
                    self.pe.load_weight_to_Compute(W_out_col)
                    self.pe.load_bias(self.b_out[d])
                    self.profiler.track('l_loads', self.D_chunk * 2 + 1)
                    
                    final_val = self.pe.run_mac()
                    self.profiler.track('macs', self.D_chunk)
                    
                    # Cộng Residual và ghi kết quả xuất thẳng ra DRAM
                    self.profiler.track('g_loads', 1)
                    output[t_start+t, d] = final_val + residual_X[t_start+t, d]
                    self.profiler.track('g_stores', 1)

        return output, self.profiler

class NonlinearAttentionChannelFirst:
    """
    CHIẾN THUẬT 2: CHANNEL-FIRST TILING (NONLINEAR ATTENTION)
    [ĐÃ SỬA]: Nạp Input X từ DRAM vào SRAM đúng 1 LẦN DUY NHẤT!
    """
    def __init__(self, config: HardwareConfig):
        self.T, self.D_in, self.D_attn, self.D_chunk = config.T, config.D_in, config.D_attn, config.D_chunk
        self.tile_T, self.tile_C = config.attn_tile_T, config.attn_tile_C
        self.profiler = HardwareProfiler(name=f"Attention ({config.strategy_name})")
        
        self.W_in = np.random.randn(self.D_in, self.D_attn).astype(np.float64) / 30.0
        self.b_in = np.random.randn(self.D_attn).astype(np.float64)
        self.W_out = np.random.randn(self.D_chunk, self.D_in).astype(np.float64) / 30.0
        self.b_out = np.random.randn(self.D_in).astype(np.float64)

    def execute(self, global_X, residual_X, global_attn_weights):
        output = np.zeros((self.T, self.D_in), dtype=np.float64)
        local_s_tanh = np.zeros((self.T, self.D_chunk), dtype=np.float64)
        local_x_gated = np.zeros((self.T, self.D_chunk), dtype=np.float64)
        local_y = np.zeros((self.T, self.D_chunk), dtype=np.float64)

        # 🎯 NẠP X VÀO SRAM 1 LẦN DUY NHẤT
        local_X_full = global_X.copy()
        self.profiler.track('g_loads', self.T * self.D_in)
        self.profiler.track('l_stores', self.T * self.D_in)

        # --- PHASE 1: IN-PROJ ---
        for c_start in range(0, self.D_attn, self.tile_C):
            act_C = min(self.tile_C, self.D_attn - c_start)
            W_in_tile = self.W_in[:, c_start:c_start+act_C]
            b_in_tile = self.b_in[c_start:c_start+act_C]
            self.profiler.track('g_loads', self.D_in * act_C + act_C)

            for t_start in range(0, self.T, self.tile_T):
                act_T = min(self.tile_T, self.T - t_start)
                
                # Đọc từ SRAM (l_loads)
                local_X = local_X_full[t_start:t_start+act_T, :]
                self.profiler.track('l_loads', act_T * self.D_in)

                val_block = np.dot(local_X, W_in_tile) + b_in_tile
                self.profiler.track('macs', act_T * self.D_in * act_C)

                for c_idx in range(act_C):
                    global_c = c_start + c_idx
                    val_col = val_block[:, c_idx]
                    if global_c < self.D_chunk:
                        local_s_tanh[t_start:t_start+act_T, global_c] = np.tanh(val_col)
                    elif global_c < 2 * self.D_chunk:
                        d = global_c - self.D_chunk
                        local_x_gated[t_start:t_start+act_T, d] = local_s_tanh[t_start:t_start+act_T, d] * val_col
                    else:
                        local_y[t_start:t_start+act_T, global_c - 2 * self.D_chunk] = val_col

        # --- PHASE 2: ATTN MATMUL ---
        local_attn_out = np.zeros((self.T, self.D_chunk), dtype=np.float64)
        for t_start in range(0, self.T, self.tile_T):
            act_T = min(self.tile_T, self.T - t_start)
            attn_row_block = global_attn_weights[t_start:t_start+act_T, :]
            self.profiler.track('g_loads', act_T * self.T)

            for d_start in range(0, self.D_chunk, self.tile_C):
                act_D = min(self.tile_C, self.D_chunk - d_start)
                x_col_block = local_x_gated[:, d_start:d_start+act_D]
                
                attn_x_val = np.dot(attn_row_block, x_col_block)
                self.profiler.track('macs', act_T * self.T * act_D)
                
                y_col_block = local_y[t_start:t_start+act_T, d_start:d_start+act_D]
                local_attn_out[t_start:t_start+act_T, d_start:d_start+act_D] = attn_x_val * y_col_block

        # --- PHASE 3: OUT-PROJ ---
        for d_start in range(0, self.D_in, self.tile_C):
            act_D = min(self.tile_C, self.D_in - d_start)
            W_out_tile = self.W_out[:, d_start:d_start+act_D]
            b_out_tile = self.b_out[d_start:d_start+act_D]
            self.profiler.track('g_loads', self.D_chunk * act_D + act_D)

            for t_start in range(0, self.T, self.tile_T):
                act_T = min(self.tile_T, self.T - t_start)
                local_attn_in = local_attn_out[t_start:t_start+act_T, :]

                out_val = np.dot(local_attn_in, W_out_tile) + b_out_tile
                self.profiler.track('macs', act_T * self.D_chunk * act_D)

                res_X = residual_X[t_start:t_start+act_T, d_start:d_start+act_D]
                self.profiler.track('g_loads', act_T * act_D)
                output[t_start:t_start+act_T, d_start:d_start+act_D] = out_val + res_X
                self.profiler.track('g_stores', act_T * act_D)

        return output, self.profiler
    
class NonlinearAttentionBlockTiling:
    """
    CHIẾN THUẬT 3: BLOCK TILING (OUTPUT-STATIONARY)
    Cắt ô vuông 2D. Tích lũy tổng ngay trên thanh ghi PE để triệt tiêu Load/Store.
    """
    def __init__(self, config: HardwareConfig):
        self.T, self.D_in, self.D_attn, self.D_chunk = config.T, config.D_in, config.D_attn, config.D_chunk
        self.tile_T, self.tile_C = config.attn_tile_T, config.attn_tile_C
        self.profiler = HardwareProfiler(name=f"Attention ({config.strategy_name})")
        
        self.W_in = np.random.randn(self.D_in, self.D_attn).astype(np.float64) / 30.0
        self.b_in = np.random.randn(self.D_attn).astype(np.float64)
        self.W_out = np.random.randn(self.D_chunk, self.D_in).astype(np.float64) / 30.0
        self.b_out = np.random.randn(self.D_in).astype(np.float64)

    def execute(self, global_X, residual_X, global_attn_weights):
        output = np.zeros((self.T, self.D_in), dtype=np.float64)
        in_proj_full = np.zeros((self.T, self.D_attn), dtype=np.float64)
        
        # --- PHASE 1: IN-PROJ (BLOCK TILING) ---
        for t_start in range(0, self.T, self.tile_T):
            act_T = min(self.tile_T, self.T - t_start)
            for c_start in range(0, self.D_attn, self.tile_C):
                act_C = min(self.tile_C, self.D_attn - c_start)
                
                # Giữ chặt Ô vuông Output trên SRAM
                local_block = np.zeros((act_T, act_C), dtype=np.float64)
                
                # Tích lũy (Reduction) dọc theo chiều D_in
                for d_start in range(0, self.D_in, self.tile_C):
                    act_D = min(self.tile_C, self.D_in - d_start)
                    local_X = global_X[t_start:t_start+act_T, d_start:d_start+act_D]
                    W_tile = self.W_in[d_start:d_start+act_D, c_start:c_start+act_C]
                    self.profiler.track('g_loads', act_T * act_D + act_D * act_C)
                    
                    local_block += np.dot(local_X, W_tile)
                    self.profiler.track('macs', act_T * act_D * act_C)
                    
                local_block += self.b_in[c_start:c_start+act_C]
                in_proj_full[t_start:t_start+act_T, c_start:c_start+act_C] = local_block

        # Phân rã mảng siêu tốc (Vectorized)
        local_s_tanh = np.tanh(in_proj_full[:, 0:self.D_chunk])
        local_x_gated = local_s_tanh * in_proj_full[:, self.D_chunk:2*self.D_chunk]
        local_y = in_proj_full[:, 2*self.D_chunk:3*self.D_chunk]
        
        # --- PHASE 2: ATTN MATMUL (BLOCK TILING) ---
        local_attn_out = np.zeros((self.T, self.D_chunk), dtype=np.float64)
        for t_start in range(0, self.T, self.tile_T):
            act_T = min(self.tile_T, self.T - t_start)
            for d_start in range(0, self.D_chunk, self.tile_C):
                act_D = min(self.tile_C, self.D_chunk - d_start)
                
                block_acc = np.zeros((act_T, act_D), dtype=np.float64)
                for k_start in range(0, self.T, self.tile_T):
                    act_K = min(self.tile_T, self.T - k_start)
                    attn_tile = global_attn_weights[t_start:t_start+act_T, k_start:k_start+act_K]
                    x_tile = local_x_gated[k_start:k_start+act_K, d_start:d_start+act_D]
                    self.profiler.track('g_loads', act_T * act_K)
                    
                    block_acc += np.dot(attn_tile, x_tile)
                    self.profiler.track('macs', act_T * act_K * act_D)
                    
                y_tile = local_y[t_start:t_start+act_T, d_start:d_start+act_D]
                local_attn_out[t_start:t_start+act_T, d_start:d_start+act_D] = block_acc * y_tile

        # --- PHASE 3: OUT-PROJ (BLOCK TILING) ---
        for t_start in range(0, self.T, self.tile_T):
            act_T = min(self.tile_T, self.T - t_start)
            for d_start in range(0, self.D_in, self.tile_C):
                act_D = min(self.tile_C, self.D_in - d_start)
                
                block_out = np.zeros((act_T, act_D), dtype=np.float64)
                for c_start in range(0, self.D_chunk, self.tile_C):
                    act_C = min(self.tile_C, self.D_chunk - c_start)
                    attn_tile = local_attn_out[t_start:t_start+act_T, c_start:c_start+act_C]
                    W_out_tile = self.W_out[c_start:c_start+act_C, d_start:d_start+act_D]
                    self.profiler.track('g_loads', act_C * act_D)
                    
                    block_out += np.dot(attn_tile, W_out_tile)
                    self.profiler.track('macs', act_T * act_C * act_D)
                    
                block_out += self.b_out[d_start:d_start+act_D]
                res_tile = residual_X[t_start:t_start+act_T, d_start:d_start+act_D]
                self.profiler.track('g_loads', act_T * act_D)
                
                output[t_start:t_start+act_T, d_start:d_start+act_D] = block_out + res_tile
                self.profiler.track('g_stores', act_T * act_D)
                
        return output, self.profiler
# ==============================================================================
# 4. CHẠY THỬ & XÁC THỰC VỚI PYTORCH
# ==============================================================================
class PyTorchSwooshL(nn.Module):
    def forward(self, x):
        zero = torch.tensor(0.0, dtype=x.dtype, device=x.device)
        return torch.logaddexp(zero, x - 4.0) - 0.08 * x - 0.035

class ZipformerLayerPyTorch(nn.Module):
    def __init__(self, ffw_hw, attn_hw):
        super().__init__()
        self.ffw_in = nn.Linear(192, 384)
        self.ffw_act = PyTorchSwooshL()
        self.ffw_out = nn.Linear(384, 192)
        
        self.attn_in = nn.Linear(192, 432)
        self.attn_out = nn.Linear(144, 192)
        
        # BÍ QUYẾT FIX LỖI 1e-07: Ép kiểu lên float64 TRƯỚC KHI copy dữ liệu.
        self.double() 
        
        with torch.no_grad():
            self.ffw_in.weight.copy_(torch.from_numpy(ffw_hw.W1).T)
            self.ffw_in.bias.copy_(torch.from_numpy(ffw_hw.b1))
            self.ffw_out.weight.copy_(torch.from_numpy(ffw_hw.W2).T)
            self.ffw_out.bias.copy_(torch.from_numpy(ffw_hw.b2))
            
            self.attn_in.weight.copy_(torch.from_numpy(attn_hw.W_in).T)
            self.attn_in.bias.copy_(torch.from_numpy(attn_hw.b_in))
            self.attn_out.weight.copy_(torch.from_numpy(attn_hw.W_out).T)
            self.attn_out.bias.copy_(torch.from_numpy(attn_hw.b_out))

    def forward(self, x, external_attn_weights):
        # 1. Feedforward Block
        res_ffw = x
        x = self.ffw_out(self.ffw_act(self.ffw_in(x))) + res_ffw
        
        # 2. Nonlinear Attention Block 
        res_attn = x
        in_proj = self.attn_in(x) 
        s, x_k, y_v = torch.chunk(in_proj, 3, dim=-1) 
        s_tanh = torch.tanh(s)
        
        # [CẬP NHẬT TOÁN HỌC MỚI DỰA THEO SƠ ĐỒ]
        x_gated = s_tanh * x_k                                # 1. S_tanh element-wise với X
        attn_x = torch.matmul(external_attn_weights, x_gated) # 2. Reshape/Permute tàng hình -> Matmul
        attn_out = attn_x * y_v                               # 3. Permute tàng hình -> Element-wise với Y
        
        return self.attn_out(attn_out) + res_attn
# ==============================================================================
# BẢN ĐỐI CHỨNG: KỊCH BẢN KHÔNG TILING (NAIVE EXECUTION / CPU THUẦN)
# ==============================================================================
class FeedforwardNaive:
    def __init__(self, T, D_in, D_ffw):
        self.T, self.D_in, self.D_ffw = T, D_in, D_ffw
        self.profiler = HardwareProfiler(name="Feedforward (NO TILING)")
        
        # Trọng số nằm hoàn toàn trên Global DRAM
        self.W1 = np.random.randn(D_in, D_ffw).astype(np.float64) / 30.0
        self.b1 = np.random.randn(D_ffw).astype(np.float64)
        self.W2 = np.random.randn(D_ffw, D_in).astype(np.float64) / 30.0
        self.b2 = np.random.randn(D_in).astype(np.float64)

    def swooshL(self, x):
        return np.float64(np.logaddexp(0.0, x - 4.0) - 0.08 * x - 0.035)

    def execute(self, global_X):
        """
        Mô phỏng phần cứng KHÔNG có bộ nhớ đệm (SRAM) bằng các phép toán ma trận vectorized.
        Vẫn giữ ý nghĩa "naive" về mặt bộ nhớ, nhưng chạy nhanh hơn nhiều để profiler có thể hoàn tất.
        """
        self.profiler.track('macs', self.T * self.D_in * self.D_ffw)
        self.profiler.track('g_loads', self.T * self.D_ffw + self.T * self.D_in * self.D_ffw * 2)
        self.profiler.track('g_stores', self.T * self.D_ffw)

        global_H = np.dot(global_X, self.W1) + self.b1

        self.profiler.track('g_loads', self.T * self.D_ffw)
        self.profiler.track('g_stores', self.T * self.D_ffw)
        global_H = self.swooshL(global_H)

        self.profiler.track('macs', self.T * self.D_ffw * self.D_in)
        self.profiler.track('g_loads', self.T * self.D_ffw * self.D_in * 2)
        self.profiler.track('g_stores', self.T * self.D_in)
        global_Y = np.dot(global_H, self.W2) + self.b2

        self.profiler.track('g_loads', self.T * self.D_in * 2)
        self.profiler.track('g_stores', self.T * self.D_in)
        global_Output = global_Y + global_X

        return global_Output, self.profiler

class SelfAttentionPipelineHardware:
    """
    KIẾN TRÚC TỐI ƯU: SELF-ATTENTION TILED PIPELINE (PING-PONG BUFFER)
    [ĐÃ NÂNG CẤP]: Nạp toàn bộ Input X đúng 1 lần duy nhất vào SRAM để cứu băng thông DRAM!
    """
    def __init__(self, config: HardwareConfig):
        self.T = config.T
        self.N = config.D_in
        self.V = config.D_chunk
        self.num_heads = 4 
        self.vhd = self.V // self.num_heads 
        self.tile_T = config.attn_tile_T 
        
        self.profiler = HardwareProfiler(name=f"Self-Attn Pipeline")
        
        self.W_in = np.random.randn(self.N, self.V).astype(np.float64) / 30.0
        self.b_in = np.random.randn(self.V).astype(np.float64)
        self.W_out = np.random.randn(self.V, self.N).astype(np.float64) / 30.0
        self.b_out = np.random.randn(self.N).astype(np.float64)

    def execute(self, global_X, residual_X, global_attn_weights):
        if len(global_attn_weights.shape) == 2:
            attn_weights_3d = np.stack([global_attn_weights] * self.num_heads, axis=0)
        else:
            attn_weights_3d = global_attn_weights

        # --- BỘ ĐỆM SRAM NỘI BỘ ---
        local_input_full = np.zeros((self.T, self.N), dtype=np.float64) # Buffer bự ôm trọn X
        local_weight = np.zeros((self.N, self.vhd), dtype=np.float64)
        local_value_cur = np.zeros((self.T, self.vhd), dtype=np.float64)
        local_value_pre = np.zeros((self.T, self.vhd), dtype=np.float64)
        local_attn_out = np.zeros((self.T, self.vhd), dtype=np.float64)
        local_input_outproject_buffer = np.zeros((self.T, self.V), dtype=np.float64)
        local_wout = np.zeros((self.V, self.N), dtype=np.float64)
        output_buffer = np.zeros((self.T, self.N), dtype=np.float64)

        # [TỐI ƯU BĂNG THÔNG]: Load 1 phát ăn ngay toàn bộ X vào SRAM
        local_input_full[:, :] = global_X[:, :]
        self.profiler.track('g_loads', self.T * self.N)
        self.profiler.track('l_stores', self.T * self.N)

        def compute_inproj_head(head_idx):
            c0 = head_idx * self.vhd
            c1 = c0 + self.vhd
            
            local_weight[:, :] = self.W_in[:, c0:c1]
            b_in_head = self.b_in[c0:c1]
            self.profiler.track('g_loads', self.N * self.vhd + self.vhd)
            self.profiler.track('l_stores', self.N * self.vhd + self.vhd)

            for t_start in range(0, self.T, self.tile_T):
                act_T = min(self.tile_T, self.T - t_start)
                
                # PE Hút X trực tiếp từ SRAM, không ra ngoài DRAM nữa!
                local_X_tile = local_input_full[t_start:t_start+act_T, :]
                self.profiler.track('l_loads', act_T * self.N)
                self.profiler.track('l_loads', self.N * self.vhd + self.vhd)
                
                local_value_cur[t_start:t_start+act_T, :] = np.dot(local_X_tile, local_weight) + b_in_head
                self.profiler.track('macs', act_T * self.N * self.vhd)
                self.profiler.track('l_stores', act_T * self.vhd)

        def compute_attn_matmul(head_idx):
            attn_h = attn_weights_3d[head_idx] 
            self.profiler.track('g_loads', self.T * self.T) 
            
            local_attn_out[:, :] = np.dot(attn_h, local_value_pre)
            self.profiler.track('macs', self.T * self.T * self.vhd)
            self.profiler.track('l_loads', self.T * self.vhd) 
            self.profiler.track('l_stores', self.T * self.vhd) 

        # --- CHẠY PIPELINE ---
        compute_inproj_head(0)
        local_value_pre[:, :] = local_value_cur.copy()

        for h in range(1, self.num_heads):
            compute_inproj_head(h)
            compute_attn_matmul(h-1)
            
            c0 = (h-1) * self.vhd
            local_input_outproject_buffer[:, c0:c0+self.vhd] = local_attn_out.copy()
            self.profiler.track('l_stores', self.T * self.vhd)
            local_value_pre[:, :] = local_value_cur.copy()

        compute_attn_matmul(self.num_heads - 1)
        c0 = (self.num_heads - 1) * self.vhd
        local_input_outproject_buffer[:, c0:c0+self.vhd] = local_attn_out.copy()
        self.profiler.track('l_stores', self.T * self.vhd)

        # --- OUT-PROJ & RESIDUAL ---
        local_wout[:, :] = self.W_out
        self.profiler.track('g_loads', self.V * self.N)
        
        output_buffer[:, :] = residual_X
        self.profiler.track('g_loads', self.T * self.N)

        for t_start in range(0, self.T, self.tile_T):
            act_T = min(self.tile_T, self.T - t_start)
            
            local_Q_buf = local_input_outproject_buffer[t_start:t_start+act_T, :]
            self.profiler.track('l_loads', act_T * self.V)
            
            out_part = np.dot(local_Q_buf, local_wout) + self.b_out
            self.profiler.track('macs', act_T * self.V * self.N)
            
            output_buffer[t_start:t_start+act_T, :] += out_part
            self.profiler.track('g_stores', act_T * self.N)

        return output_buffer, self.profiler

# ==============================================================================
# BẢN ĐỐI CHỨNG: KỊCH BẢN KHÔNG TILING (NAIVE EXECUTION / CPU THUẦN)
# ==============================================================================
class NonlinearAttentionNaive:
    def __init__(self, config: HardwareConfig):
        self.T = config.T
        self.D_in = config.D_in
        self.D_chunk = config.D_chunk
        self.profiler = HardwareProfiler(name="Nonlin Attn (NO TILING)")
        self.W_in = np.random.randn(self.D_in, self.D_chunk).astype(np.float64) / 30.0
        self.b_in = np.random.randn(self.D_chunk).astype(np.float64)
        self.W_out = np.random.randn(self.D_chunk, self.D_in).astype(np.float64) / 30.0
        self.b_out = np.random.randn(self.D_in).astype(np.float64)

    def execute(self, global_X, residual_X, global_attn_weights):
        self.profiler.track('macs', self.T * self.D_in * self.D_chunk)
        self.profiler.track('g_loads', self.T * self.D_in * self.D_chunk * 2)
        hidden = np.dot(global_X, self.W_in) + self.b_in

        self.profiler.track('macs', self.T * self.D_chunk * self.D_in)
        self.profiler.track('g_loads', self.T * self.D_chunk * self.D_in * 2)
        output = np.dot(hidden, self.W_out) + self.b_out + residual_X

        self.profiler.track('g_stores', self.T * self.D_in)
        return output, self.profiler


class SelfAttentionNaive:
    def __init__(self, config: HardwareConfig):
        self.T = config.T
        self.N = config.D_in
        self.V = config.D_chunk
        self.profiler = HardwareProfiler(name="Self-Attn (NO TILING)")
        self.W_in = np.random.randn(self.N, self.V).astype(np.float64) / 30.0
        self.b_in = np.random.randn(self.V).astype(np.float64)
        self.W_out = np.random.randn(self.V, self.N).astype(np.float64) / 30.0
        self.b_out = np.random.randn(self.N).astype(np.float64)

    def execute(self, global_X, residual_X, global_attn_weights):
        if len(global_attn_weights.shape) == 2:
            attn_weights_3d = np.stack([global_attn_weights] * 4, axis=0)
        else:
            attn_weights_3d = global_attn_weights

        self.profiler.track('macs', self.T * self.N * self.V)
        self.profiler.track('g_loads', self.T * self.N * self.V * 2)
        values = np.dot(global_X, self.W_in) + self.b_in

        self.profiler.track('macs', self.T * self.T * self.V)
        self.profiler.track('g_loads', self.T * self.T * self.V * 2)
        attn_out = np.dot(attn_weights_3d[0], values)

        self.profiler.track('macs', self.T * self.V * self.N)
        self.profiler.track('g_loads', self.T * self.V * self.N * 2)
        out = np.dot(attn_out, self.W_out) + self.b_out + residual_X

        self.profiler.track('g_stores', self.T * self.N)
        return out, self.profiler


BYTES_PER_WORD = 2


def calculate_latency_ms(macs: int, total_traffic_words: int, num_pe: int = 256, freq_mhz: int = 200, bw_gbps: int = 10) -> float:
    cycles_compute = math.ceil(macs / num_pe)
    time_compute_sec = cycles_compute / (freq_mhz * 1e6)
    time_memory_sec = (total_traffic_words * BYTES_PER_WORD) / (bw_gbps * 1e9)
    return (time_compute_sec + time_memory_sec) * 1000.0


def get_true_strategy_name(t_tile: int, d_tile: int, T: int, D: int) -> str:
    if t_tile >= T and d_tile >= D:
        return "Full-Matrix"
    if t_tile >= T:
        return "Channel-First"
    if d_tile >= D:
        return "Time-First"
    return "Block Tiling"


def calculate_exact_hardware_sram_kb(module_type: str, strat: str, t_tile: int, d_tile: int, T: int, D_in: int, D_out: int, ping_pong: bool = True) -> float:
    P = 2 if ping_pong else 1

    if module_type == "FFW":
        buf_res = T * D_in
        buf_x = (T * D_in) if strat in ["Full-Matrix", "Channel-First"] else (P * t_tile * D_in)
        if strat in ["Full-Matrix", "Time-First"]:
            buf_w = (D_in * D_out) + (D_out * D_in)
        else:
            buf_w = P * ((D_in * d_tile) + (d_tile * D_in))
        buf_out = (T * d_tile) if strat == "Channel-First" else (t_tile * d_tile)
        total_words = buf_res + buf_x + buf_w + buf_out
    elif module_type == "NONLIN":
        buf_aux = (T * D_in) + (T * T)
        buf_x = (T * D_in) if strat in ["Full-Matrix", "Channel-First"] else (P * t_tile * D_in)
        buf_w = (D_in * D_out) if strat in ["Full-Matrix", "Time-First"] else (P * D_in * d_tile)
        buf_fusion_out = (T * d_tile) if strat == "Channel-First" else (t_tile * d_tile)
        total_words = buf_aux + buf_x + buf_w + buf_fusion_out
    else:
        vhd = d_tile // 4
        total_words = (T * D_in) + (D_in * vhd) + (T * vhd) * 2 + (T * vhd) + (T * d_tile) + (T * D_in) + (T * T)

    return (total_words * BYTES_PER_WORD) / 1024.0


def calculate_exact_dram_traffic(module_type: str, strat: str, t_tile: int, d_tile: int, T: int, D_in: int, D_out: int):
    N_T = math.ceil(T / t_tile)
    N_C = math.ceil(D_out / d_tile)

    if module_type == "FFW":
        if strat in ["Full-Matrix", "Channel-First"]:
            g_loads = (T * D_in) + (D_in * D_out) + (D_out * D_in)
        elif strat == "Time-First":
            g_loads = (T * D_in) + N_T * ((D_in * D_out) + (D_out * D_in))
        else:
            g_loads = N_C * (T * D_in) + N_T * ((D_in * D_out) + (D_out * D_in))
        g_stores = T * D_in
    elif module_type == "NONLIN":
        if strat in ["Full-Matrix", "Channel-First"]:
            g_loads = (T * D_in) + (D_in * D_out)
        elif strat == "Time-First":
            g_loads = (T * D_in) + N_T * (D_in * D_out)
        else:
            g_loads = N_C * (T * D_in) + N_T * (D_in * D_out)
        g_stores = T * D_in
    else:
        vhd = D_out // 4
        g_loads = (T * D_in) + (D_in * vhd) + (T * T)
        g_stores = T * D_in

    return g_loads, g_stores


def build_hw_modules(ffw_cfg: HardwareConfig, self_cfg: HardwareConfig, mode: str):
    ffw_classes = {
        "tiled": {
            "Full-Matrix": FeedforwardHardware,
            "Time-First": FeedforwardHardware,
            "Channel-First": FeedforwardChannelFirst,
            "Block Tiling": FeedforwardBlockTiling,
        },
        "full_matrix": {
            "Full-Matrix": FeedforwardHardware,
            "Time-First": FeedforwardHardware,
            "Channel-First": FeedforwardChannelFirst,
            "Block Tiling": FeedforwardBlockTiling,
        },
    }
    nonlin_classes = {
        "tiled": {
            "Full-Matrix": NonlinearAttentionHardware,
            "Time-First": NonlinearAttentionHardware,
            "Channel-First": NonlinearAttentionChannelFirst,
            "Block Tiling": NonlinearAttentionBlockTiling,
        },
        "full_matrix": {
            "Full-Matrix": NonlinearAttentionHardware,
            "Time-First": NonlinearAttentionHardware,
            "Channel-First": NonlinearAttentionChannelFirst,
            "Block Tiling": NonlinearAttentionBlockTiling,
        },
    }

    if mode == "naive":
        return FeedforwardNaive(ffw_cfg.T, ffw_cfg.D_in, ffw_cfg.D_ffw), NonlinearAttentionNaive(ffw_cfg), SelfAttentionNaive(self_cfg)

    if mode == "full_matrix":
        full_cfg = HardwareConfig(
            T=ffw_cfg.T,
            D_in=ffw_cfg.D_in,
            D_ffw=ffw_cfg.D_ffw,
            D_attn=ffw_cfg.D_attn,
            D_chunk=ffw_cfg.D_chunk,
            strategy_name="Full-Matrix",
            ffw_tile_T=ffw_cfg.T,
            ffw_tile_H=ffw_cfg.D_ffw,
            attn_tile_T=ffw_cfg.T,
            attn_tile_C=ffw_cfg.D_attn,
            num_pe=ffw_cfg.num_pe,
            sram_limit_kb=ffw_cfg.sram_limit_kb,
        )
        ffw = ffw_classes["full_matrix"][ffw_cfg.strategy_name](full_cfg) if ffw_cfg.strategy_name in ffw_classes["full_matrix"] else FeedforwardHardware(full_cfg)
        attn = nonlin_classes["full_matrix"][ffw_cfg.strategy_name](full_cfg) if ffw_cfg.strategy_name in nonlin_classes["full_matrix"] else NonlinearAttentionHardware(full_cfg)
        self_hw = SelfAttentionPipelineHardware(HardwareConfig(
            T=self_cfg.T,
            D_in=self_cfg.D_in,
            D_ffw=self_cfg.D_ffw,
            D_attn=self_cfg.D_attn,
            D_chunk=self_cfg.D_chunk,
            strategy_name="Pipeline",
            ffw_tile_T=self_cfg.T,
            ffw_tile_H=self_cfg.D_ffw,
            attn_tile_T=self_cfg.T,
            attn_tile_C=0,
            num_pe=self_cfg.num_pe,
            sram_limit_kb=self_cfg.sram_limit_kb,
        ))
        return ffw, attn, self_hw

    tiled_cfg = HardwareConfig(
        T=ffw_cfg.T,
        D_in=ffw_cfg.D_in,
        D_ffw=ffw_cfg.D_ffw,
        D_attn=ffw_cfg.D_attn,
        D_chunk=ffw_cfg.D_chunk,
        strategy_name=ffw_cfg.strategy_name,
        ffw_tile_T=ffw_cfg.ffw_tile_T,
        ffw_tile_H=ffw_cfg.ffw_tile_H,
        attn_tile_T=ffw_cfg.attn_tile_T,
        attn_tile_C=ffw_cfg.attn_tile_C,
        num_pe=ffw_cfg.num_pe,
        sram_limit_kb=ffw_cfg.sram_limit_kb,
    )
    ffw = ffw_classes["tiled"][ffw_cfg.strategy_name](tiled_cfg) if ffw_cfg.strategy_name in ffw_classes["tiled"] else FeedforwardHardware(tiled_cfg)
    attn = nonlin_classes["tiled"][ffw_cfg.strategy_name](tiled_cfg) if ffw_cfg.strategy_name in nonlin_classes["tiled"] else NonlinearAttentionHardware(tiled_cfg)
    self_hw = SelfAttentionPipelineHardware(HardwareConfig(
        T=self_cfg.T,
        D_in=self_cfg.D_in,
        D_ffw=self_cfg.D_ffw,
        D_attn=self_cfg.D_attn,
        D_chunk=self_cfg.D_chunk,
        strategy_name="Pipeline",
        ffw_tile_T=self_cfg.attn_tile_T,
        ffw_tile_H=0,
        attn_tile_T=self_cfg.attn_tile_T,
        attn_tile_C=0,
        num_pe=self_cfg.num_pe,
        sram_limit_kb=self_cfg.sram_limit_kb,
    ))
    return ffw, attn, self_hw


def run_stack_profile_comparison(output_excel: str = "stack_profile_comparison.xlsx", output_json: str = "stack_profile_comparison.json", json_path: str | None = None):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if json_path is None:
        json_path = os.path.join(script_dir, "golden_configs.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Không tìm thấy {json_path}. Hãy chạy DSE trước.")

    with open(json_path, "r") as f:
        golden_configs = json.load(f)

    rows: List[Dict[str, Any]] = []
    for cfg in golden_configs:
        sid = cfg["stack_id"]
        T, D_in, D_ffw, D_attn, D_cn, D_cs = (
            cfg["shape"]["T"],
            cfg["shape"]["D_in"],
            cfg["shape"]["D_ffw"],
            cfg["shape"]["D_attn"],
            cfg["shape"]["D_chunk_n"],
            cfg["shape"]["D_chunk_s"],
        )

        X = np.random.randn(T, D_in).astype(np.float64) / 10.0
        attn_w_2d = np.random.rand(T, T).astype(np.float64)
        attn_w_2d = attn_w_2d / attn_w_2d.sum(axis=-1, keepdims=True)
        attn_w_3d = np.random.rand(4, T, T).astype(np.float64)
        attn_w_3d = attn_w_3d / attn_w_3d.sum(axis=-1, keepdims=True)

        hw_cfg = HardwareConfig(
            T=T,
            D_in=D_in,
            D_ffw=D_ffw,
            D_attn=D_attn,
            D_chunk=D_cn,
            strategy_name=cfg["ffw"]["strat"],
            ffw_tile_T=cfg["ffw"]["t"],
            ffw_tile_H=cfg["ffw"]["h"],
            attn_tile_T=cfg["nonlin"]["t"],
            attn_tile_C=cfg["nonlin"]["c"],
            num_pe=256,
            sram_limit_kb=512,
        )
        self_cfg = HardwareConfig(
            T=T,
            D_in=D_in,
            D_ffw=D_ffw,
            D_attn=D_attn,
            D_chunk=D_cs,
            strategy_name=cfg["self"]["strat"],
            ffw_tile_T=cfg["self"]["t"],
            ffw_tile_H=0,
            attn_tile_T=cfg["self"]["t"],
            attn_tile_C=0,
            num_pe=256,
            sram_limit_kb=512,
        )

        for mode in ["tiled", "full_matrix", "naive"]:
            if mode == "tiled":
                ffw_strat = cfg["ffw"]["strat"]
                ffw_t = cfg["ffw"]["t"]
                ffw_h = cfg["ffw"]["h"]
                nonlin_strat = cfg["nonlin"]["strat"]
                nonlin_t = cfg["nonlin"]["t"]
                nonlin_c = cfg["nonlin"]["c"]
                self_strat = cfg["self"]["strat"]
                self_t = cfg["self"]["t"]
            elif mode == "full_matrix":
                ffw_strat = "Full-Matrix"
                ffw_t = T
                ffw_h = D_ffw
                nonlin_strat = "Full-Matrix"
                nonlin_t = T
                nonlin_c = D_attn
                self_strat = "Pipeline"
                self_t = T
            else:
                ffw_strat = "Naive"
                ffw_t = 1
                ffw_h = 1
                nonlin_strat = "Naive"
                nonlin_t = 1
                nonlin_c = 1
                self_strat = "Pipeline"
                self_t = 1

            total_macs = T * D_in * D_ffw * 2 + T * D_in * D_attn * 2 + T * D_in * D_cs * 4
            loads_f, stores_f = calculate_exact_dram_traffic("FFW", ffw_strat, ffw_t, ffw_h, T, D_in, D_ffw)
            loads_n, stores_n = calculate_exact_dram_traffic("NONLIN", nonlin_strat, nonlin_t, nonlin_c, T, D_in, D_attn)
            loads_s, stores_s = calculate_exact_dram_traffic("SELF", self_strat, self_t, D_cs, T, D_in, D_cs)
            total_loads = loads_f + loads_n + loads_s
            total_stores = stores_f + stores_n + stores_s
            total_traffic_words = total_loads + total_stores
            latency_ms = calculate_latency_ms(total_macs, total_traffic_words)
            intensity_mac_byte = total_macs / (total_traffic_words * BYTES_PER_WORD) if total_traffic_words > 0 else 0.0
            perf_gmacs_s = float((total_macs / 1e9) / (latency_ms / 1000.0)) if latency_ms > 0 else 0.0

            sram_f = calculate_exact_hardware_sram_kb("FFW", ffw_strat, ffw_t, ffw_h, T, D_in, D_ffw, ping_pong=True)
            sram_n = calculate_exact_hardware_sram_kb("NONLIN", nonlin_strat, nonlin_t, nonlin_c, T, D_in, D_attn, ping_pong=True)
            sram_s = calculate_exact_hardware_sram_kb("SELF", self_strat, self_t, D_cs, T, D_in, D_cs, ping_pong=False)
            peak_sram_kb = max(sram_f, sram_n, sram_s)

            rows.append({
                "Stack_ID": sid,
                "Mode": mode,
                "Strategy": cfg["ffw"]["strat"],
                "T": T,
                "D_in": D_in,
                "D_ffw": D_ffw,
                "D_attn": D_attn,
                "D_chunk_n": D_cn,
                "D_chunk_s": D_cs,
                "FFW_Tile_T": ffw_t,
                "FFW_Tile_H": ffw_h,
                "Nonlin_Tile_T": nonlin_t,
                "Nonlin_Tile_C": nonlin_c,
                "Self_Tile_T": self_t,
                "Peak_SRAM_KB": float(peak_sram_kb),
                "Total_MACs": int(total_macs),
                "Global_Loads": int(total_loads),
                "Global_Stores": int(total_stores),
                "Total_Traffic_Words": int(total_traffic_words),
                "Intensity_MAC_Byte": float(intensity_mac_byte),
                "Perf_GMACs_s": perf_gmacs_s,
            })
            print(f"Stack {sid} [{mode}] -> intensity={intensity_mac_byte:.3f}, perf={perf_gmacs_s:.3f} GMAC/s, sram={peak_sram_kb:.2f} KB")

    with open(os.path.join(script_dir, output_json), "w") as f:
        json.dump(rows, f, indent=2)

    try:
        import pandas as pd
    except ImportError:
        raise RuntimeError("Cần cài pandas để xuất file Excel")

    df = pd.DataFrame(rows)
    keep_cols = [
        "Stack_ID", "Mode", "Strategy", "T", "D_in", "D_ffw", "D_attn",
        "D_chunk_n", "D_chunk_s", "FFW_Tile_T", "FFW_Tile_H", "Nonlin_Tile_T",
        "Nonlin_Tile_C", "Self_Tile_T", "Peak_SRAM_KB", "Total_MACs",
        "Global_Loads", "Global_Stores", "Total_Traffic_Words", "Intensity_MAC_Byte",
        "Perf_GMACs_s"
    ]
    df = df[[c for c in keep_cols if c in df.columns]]
    df.to_excel(os.path.join(script_dir, output_excel), index=False)

    print(f"✅ Đã lưu kết quả profiler so sánh vào {output_excel} và {output_json}")
    return rows


# ==============================================================================
# PYTORCH GOLDEN MODELS (ĐỂ XÁC THỰC TOÁN HỌC TỚI 1e-14)
# ==============================================================================
class PyTorchSwooshL(nn.Module):
    def forward(self, x):
        zero = torch.tensor(0.0, dtype=x.dtype, device=x.device)
        return torch.logaddexp(zero, x - 4.0) - 0.08 * x - 0.035

class ZipformerLayerPyTorch(nn.Module):
    def __init__(self, ffw_hw, attn_hw):
        super().__init__()
        self.ffw_in = nn.Linear(192, 384)
        self.ffw_act = PyTorchSwooshL()
        self.ffw_out = nn.Linear(384, 192)
        self.attn_in = nn.Linear(192, 432)
        self.attn_out = nn.Linear(144, 192)
        self.double() 
        with torch.no_grad():
            self.ffw_in.weight.copy_(torch.from_numpy(ffw_hw.W1).T)
            self.ffw_in.bias.copy_(torch.from_numpy(ffw_hw.b1))
            self.ffw_out.weight.copy_(torch.from_numpy(ffw_hw.W2).T)
            self.ffw_out.bias.copy_(torch.from_numpy(ffw_hw.b2))
            self.attn_in.weight.copy_(torch.from_numpy(attn_hw.W_in).T)
            self.attn_in.bias.copy_(torch.from_numpy(attn_hw.b_in))
            self.attn_out.weight.copy_(torch.from_numpy(attn_hw.W_out).T)
            self.attn_out.bias.copy_(torch.from_numpy(attn_hw.b_out))

    def forward(self, x, external_attn_weights):
        res_ffw = x
        x = self.ffw_out(self.ffw_act(self.ffw_in(x))) + res_ffw
        res_attn = x
        in_proj = self.attn_in(x) 
        s, x_k, y_v = torch.chunk(in_proj, 3, dim=-1) 
        attn_out = torch.matmul(external_attn_weights, torch.tanh(s) * x_k) * y_v                               
        return self.attn_out(attn_out) + res_attn

class SelfAttentionPyTorch(nn.Module):
    def __init__(self, hw_module):
        super().__init__()
        self.num_heads = hw_module.num_heads
        self.vhd = hw_module.vhd
        self.in_proj = nn.Linear(hw_module.N, hw_module.V)
        self.out_proj = nn.Linear(hw_module.V, hw_module.N)
        self.double() 
        with torch.no_grad():
            self.in_proj.weight.copy_(torch.from_numpy(hw_module.W_in).T)
            self.in_proj.bias.copy_(torch.from_numpy(hw_module.b_in))
            self.out_proj.weight.copy_(torch.from_numpy(hw_module.W_out).T)
            self.out_proj.bias.copy_(torch.from_numpy(hw_module.b_out))

    def forward(self, x, residual, attn_weights_3d):
        T = x.shape[0]
        v = self.in_proj(x) 
        v = v.view(T, self.num_heads, self.vhd).permute(1, 0, 2)
        attn_out = torch.matmul(attn_weights_3d, v)
        attn_out = attn_out.permute(1, 0, 2).reshape(T, self.num_heads * self.vhd)
        return self.out_proj(attn_out) + residual

# ==============================================================================
# HÀM MAIN: SẠCH ĐẸP, CHỈ ĐỂ UNIT TEST & XÁC THỰC
# ==============================================================================
if __name__ == "__main__":
    np.random.seed(99)
    torch.manual_seed(99)
    
    # 1. Khởi tạo cấu hình và dữ liệu giả
    cfg = HardwareConfig(T=116, D_in=192, D_ffw=384, D_attn=432, D_chunk=144, strategy_name="Standard Tiling", ffw_tile_T=58, ffw_tile_H=192, attn_tile_T=58, attn_tile_C=144, num_pe=256, sram_limit_kb=512)
    cfg_self = HardwareConfig(T=116, D_in=192, D_ffw=384, D_attn=192, D_chunk=48, strategy_name="Pipeline", ffw_tile_T=58, ffw_tile_H=192, attn_tile_T=58, attn_tile_C=0, num_pe=256, sram_limit_kb=512)

    global_X = np.random.randn(cfg.T, cfg.D_in).astype(np.float64) / 30.0
    attn_w_2d = np.random.rand(cfg.T, cfg.T).astype(np.float64)
    attn_w_3d = np.random.rand(4, cfg.T, cfg.T).astype(np.float64)

    print("\n" + "★"*80)
    print("🚀 KIỂM TRA ĐỘC LẬP TỪNG KHỐI PHẦN CỨNG (UNIT TESTS)")
    print("★"*80)
    
    # --- TEST 1: FFW & NONLIN ATTENTION ---
    print("\n---> [TEST 1] KHỐI FEEDFORWARD & NONLINEAR ATTN")
    hw_f = FeedforwardHardware(cfg)
    hw_n = NonlinearAttentionHardware(cfg)
    
    out_f, prof_f = hw_f.execute(global_X)
    out_n, prof_n = hw_n.execute(out_f, out_f, attn_w_2d)
    
    prof_f.print_report()
    prof_n.print_report()

    pt_model = ZipformerLayerPyTorch(hw_f, hw_n)
    pt_model.eval()
    with torch.no_grad():
        golden_out = pt_model(torch.from_numpy(global_X), torch.from_numpy(attn_w_2d)).numpy()
    print(f"✅ Xác thực PyTorch (FFW+Nonlin): Sai số Max = {np.max(np.abs(golden_out - out_n)):.2e}")

  # ==============================================================================
    # [TEST 2]: SELF ATTENTION PIPELINE vs DAN POVEY'S ORIGINAL CODE
    # ==============================================================================
    print("\n---> [TEST 2] KHỐI SELF-ATTENTION PIPELINE (Tối ưu SRAM Load)")
    
    # 1. Chạy Hardware Simulator
    hw_s = SelfAttentionPipelineHardware(cfg_self)
    out_s, prof_s = hw_s.execute(out_n, out_n, attn_w_3d)

    # In báo cáo Load/Store/MACs
    prof_s.print_report()

    # 2. Khởi tạo Golden Model ĐÚNG CHUẨN CỦA DAN POVEY (Icefall/Zipformer)
    class DanPoveySelfAttention(nn.Module):
        def __init__(self, embed_dim, num_heads, value_head_dim):
            super().__init__()
            self.in_proj = nn.Linear(embed_dim, num_heads * value_head_dim, bias=True)
            self.out_proj = nn.Linear(num_heads * value_head_dim, embed_dim, bias=True)
            self.double() # Ép Float64
            
        def forward(self, x, residual, attn_weights):
            # x shape gốc của Dan Povey: (seq_len, batch_size, embed_dim)
            (seq_len, batch_size, embed_dim) = x.shape
            num_heads = attn_weights.shape[0]

            x_proj = self.in_proj(x)  # (seq_len, batch_size, num_heads * value_head_dim)
            x_proj = x_proj.reshape(seq_len, batch_size, num_heads, -1).permute(2, 1, 0, 3)
            
            # Matmul: [4, 116, 116] @ [4, 1, 116, 12] -> [4, 1, 116, 12]
            v = torch.matmul(attn_weights, x_proj)
            
            v = v.permute(2, 1, 0, 3).reshape(seq_len, batch_size, -1) # Gom head
            return self.out_proj(v) + residual

    # 3. Bơm trọng số từ Hardware sang Golden Model
    pt_dan_povey = DanPoveySelfAttention(embed_dim=192, num_heads=4, value_head_dim=12)
    pt_dan_povey.eval()
    with torch.no_grad():
        pt_dan_povey.in_proj.weight.copy_(torch.from_numpy(hw_s.W_in).T)
        pt_dan_povey.in_proj.bias.copy_(torch.from_numpy(hw_s.b_in))
        pt_dan_povey.out_proj.weight.copy_(torch.from_numpy(hw_s.W_out).T)
        pt_dan_povey.out_proj.bias.copy_(torch.from_numpy(hw_s.b_out))

        # 4. Chuyển X thành shape (seq_len, batch_size=1, embed_dim) theo ý Dan Povey
        x_dan_povey = torch.from_numpy(out_n).unsqueeze(1).double()
        res_dan_povey = x_dan_povey.clone()
        
        # [SỬA LỖI Ở ĐÂY]: Thêm .unsqueeze(1) để biến [4, 116, 116] thành [4, 1, 116, 116]
        attn_w_dan_povey = torch.from_numpy(attn_w_3d).unsqueeze(1).double()
        
        # Chạy suy luận (Inference)
        golden_s = pt_dan_povey(x_dan_povey, res_dan_povey, attn_w_dan_povey)
        
        # Ép ngược shape về [116, 192] để so sánh với Hardware Simulator
        golden_s_2d = golden_s.squeeze(1).numpy()

    # 5. So sánh chéo
    max_err_self = np.max(np.abs(golden_s_2d - out_s))
    print(f"✅ Xác thực Dan Povey (Self-Attn): Sai số Max = {max_err_self:.2e}")
    print("★"*80 + "\n")