import numpy as np
import torch
import torch.nn as nn
import time

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
    """Bộ đếm hiệu năng độc lập cho từng Module (Định dạng chuẩn theo yêu cầu)"""
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
        print(f"Tổng số phần tử (float64) Load từ GLOBAL (DRAM) : {g_loads:,.0f} elements")
        print(f"Tổng số phần tử (float64) Ghi ra GLOBAL (DRAM)  : {self.metrics['g_stores']:,.0f} elements")
        print(f"Tổng số phần tử (float64) Load từ LOCAL vào Core: {self.metrics['l_loads']:,.0f} elements")
        print(f"Tổng số phần tử (float64) Ghi từ Core ra LOCAL  : {self.metrics['l_stores']:,.0f} elements")
        print(f"Tổng số phép toán tính toán thực tế (MAC Ops)   : {macs:,.0f} ops")
        print("-" * 60)
        print(f"💡 Cường độ tính toán (Arithmetic Intensity):")
        print(f"   => {intensity:.2f} phép MACs / 1 lần truy cập Global DRAM")
        print(f"   (Chỉ số này càng cao nghĩa là Tiling càng hiệu quả, giảm nghẽn cổ chai RAM)")
        print("="*60)


# ==============================================================================
# 2. FEEDFORWARD MODULE (ĐỘC LẬP & THAM SỐ HÓA)
# ==============================================================================
class FeedforwardHardware:
    def __init__(self, T, D_in, D_ffw, tile_T=58, tile_H=192):
        self.T, self.D_in, self.D_ffw = T, D_in, D_ffw
        self.tile_T, self.tile_H = tile_T, tile_H
        
        self.pe = Compute_block(num_pe=256)
        self.profiler = HardwareProfiler(name="Feedforward Module")
        
        # DRAM Weights
        self.W1 = np.random.randn(D_in, D_ffw).astype(np.float64) / 30.0
        self.b1 = np.random.randn(D_ffw).astype(np.float64)
        self.W2 = np.random.randn(D_ffw, D_in).astype(np.float64) / 30.0
        self.b2 = np.random.randn(D_in).astype(np.float64)

    def swooshL(self, x):
        return np.float64(np.logaddexp(0.0, x - 4.0) - 0.08 * x - 0.035)

    def execute(self, global_X):
        """
        ========== MODULE: FEEDFORWARD EXECUTION ==========

        Input:
          - global_X [T, D_in]

        Compute:
          - H = SwooshL(X @ W1 + b1)
          - Y = H @ W2 + b2 + X

        Tiling Strategy:
          - Tiling theo chiều Thời gian (tile_T = 58) để nạp Input X vào Local SRAM một lần.
          - Tiling theo chiều Hidden (tile_H = 192) để chia nhỏ Weight W1, W2 nạp vừa khít 256 PE.
          - Tái sử dụng Trọng số (Weight Reuse): Trọng số nạp 1 lần được nhân với toàn bộ tile_T hàng.
            -> Kéo Arithmetic Intensity lên mức siêu cao.

        Memory Flow:
          1. Load local_X (tile_T x D_in) từ Global DRAM -> Local SRAM.
          2. Load W1_tile, W2_tile từ Global DRAM -> Local SRAM.
          3. PE tính toán lớp In-Proj, ghi kết quả trung gian ra local_H (SRAM).
          4. PE đọc từ local_H, tính toán lớp Out-Proj, cộng dồn vào local_Y (SRAM).
          5. Ghi final output từ local_Y -> Global DRAM.
        """
        output = np.zeros((self.T, self.D_in), dtype=np.float64)
        local_X = np.zeros((self.tile_T, self.D_in), dtype=np.float64)
        local_Y = np.zeros((self.tile_T, self.D_in), dtype=np.float64)
        local_H = np.zeros((self.tile_T, self.tile_H), dtype=np.float64)
        
        for t_start in range(0, self.T, self.tile_T):
            act_T = min(self.tile_T, self.T - t_start)
            
            # 1. Load Tile Input (DRAM -> SRAM)
            local_X[:act_T, :] = global_X[t_start:t_start+act_T, :]
            self.profiler.track('g_loads', act_T * self.D_in)
            self.profiler.track('l_stores', act_T * self.D_in)
            local_Y.fill(0.0)
            
            for h_start in range(0, self.D_ffw, self.tile_H):
                # 2. Load Weight Tile (DRAM -> SRAM)
                W1_tile = self.W1[:, h_start : h_start+self.tile_H]
                b1_tile = self.b1[h_start : h_start+self.tile_H]
                W2_tile = self.W2[h_start : h_start+self.tile_H, :]
                self.profiler.track('g_loads', self.D_in * self.tile_H + self.tile_H + self.tile_H * self.D_in)
                self.profiler.track('l_stores', self.D_in * self.tile_H + self.tile_H + self.tile_H * self.D_in)
                
                # --- 3. Lớp 1 (In-Proj) ---
                for t in range(act_T):
                    self.pe.load_in_local_to_Compute(local_X[t, :])
                    self.profiler.track('l_loads', self.D_in)
                    
                    for h in range(self.tile_H):
                        self.pe.load_weight_to_Compute(W1_tile[:, h])
                        self.pe.load_bias(b1_tile[h])
                        self.profiler.track('l_loads', self.D_in + 1) # Load weight col + bias
                        
                        mac_res = self.pe.run_mac()
                        self.profiler.track('macs', self.D_in)
                        
                        local_H[t, h] = self.swooshL(mac_res)
                        self.profiler.track('l_stores', 1)
                
                # --- 4. Lớp 2 (Out-Proj) ---
                for t in range(act_T):
                    self.pe.load_in_local_to_Compute(local_H[t, :])
                    self.profiler.track('l_loads', self.tile_H)
                    self.pe.load_bias(0.0)
                    
                    for d in range(self.D_in):
                        self.pe.load_weight_to_Compute(W2_tile[:, d])
                        self.profiler.track('l_loads', self.tile_H)
                        
                        # Accumulate
                        self.profiler.track('l_loads', 1) # Read Y_acc
                        local_Y[t, d] += self.pe.run_mac()
                        self.profiler.track('l_stores', 1) # Write Y_acc
                        self.profiler.track('macs', self.tile_H)
            
            # --- 5. Cộng Bias lớp 2, Residual và Xuất ra DRAM ---
            for t in range(act_T):
                for d in range(self.D_in):
                    self.profiler.track('g_loads', 1) # Load b2
                    self.profiler.track('l_loads', 2) # Load local_Y và local_X để cộng residual
                    
                    output[t_start+t, d] = local_Y[t, d] + self.b2[d] + local_X[t, d]
                    
                    self.profiler.track('g_stores', 1)
                    
        return output, self.profiler


# ==============================================================================
# 3. NONLINEAR ATTENTION MODULE (ĐỘC LẬP & THAM SỐ HÓA)
# ==============================================================================
class NonlinearAttentionHardware:
    def __init__(self, T, D_in, D_attn, D_chunk, tile_T=58):
        self.T, self.D_in = T, D_in
        self.D_attn, self.D_chunk = D_attn, D_chunk
        self.tile_T = tile_T
        
        self.pe = Compute_block(num_pe=256)
        self.profiler = HardwareProfiler(name="Nonlinear Attention Module")
        
        # DRAM Weights
        self.W_in = np.random.randn(D_in, D_attn).astype(np.float64) / 30.0
        self.b_in = np.random.randn(D_attn).astype(np.float64)
        self.W_out = np.random.randn(D_chunk, D_in).astype(np.float64) / 30.0
        self.b_out = np.random.randn(D_in).astype(np.float64)

    def execute(self, global_X, residual_X):
        output = np.zeros((self.T, self.D_in), dtype=np.float64)
        
        # Buffers Local (SRAM)
        local_s = np.zeros((self.T, self.D_chunk), dtype=np.float64)
        local_x = np.zeros((self.T, self.D_chunk), dtype=np.float64)
        local_y = np.zeros((self.T, self.D_chunk), dtype=np.float64)
        
        """
        --- PHASE 1: IN-PROJ (LINEAR) & CHUNK ---
        
        Compute:
          - In_proj = X @ W_in + b_in  -> [T, 432]
          - Chunk 3 phần -> s [144], x [144], y [144]
          - s_tanh = tanh(s)
          
        Tiling Strategy:
          - Chia ma trận W_in thành các tile cột (tile_C = 144) để phân rã thẳng vào S, X, Y.
          - Nạp tile_T hàng Input X vào SRAM để tái sử dụng nhân với Weight.
          
        Memory Flow:
          1. Load local_X (tile_T rows) từ DRAM -> SRAM.
          2. Nhân local_X với W_in_tile, lưu kết quả chia tách thẳng vào local_s, local_x, local_y (SRAM).
        """
        tile_C = 144 
        local_X = np.zeros((self.tile_T, self.D_in), dtype=np.float64)
        
        for t_start in range(0, self.T, self.tile_T):
            act_T = min(self.tile_T, self.T - t_start)
            local_X[:act_T, :] = global_X[t_start:t_start+act_T, :]
            
            self.profiler.track('g_loads', act_T * self.D_in)
            self.profiler.track('l_stores', act_T * self.D_in)
            
            for c_start in range(0, self.D_attn, tile_C):
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
                        
                        val = self.pe.run_mac()
                        self.profiler.track('macs', self.D_in)
                        
                        # Phân rã trực tiếp vào S, X, Y
                        global_c = c_start + c
                        if global_c < self.D_chunk:
                            local_s[t_start+t, global_c] = np.tanh(val) 
                        elif global_c < 2 * self.D_chunk:
                            local_x[t_start+t, global_c - self.D_chunk] = val
                        else:
                            local_y[t_start+t, global_c - 2 * self.D_chunk] = val
                        self.profiler.track('l_stores', 1)

        """
        --- PHASE 2: CROSS-TIME ATTENTION ---
        
        Compute:
          - attn_scores[t, j] = s_tanh[t] @ x[j].T   -> [116]
          - attn_out[t] = attn_scores[t] @ y         -> [144]
          
        Tiling Strategy:
          - Kích thước T=116 và D_chunk=144 đều nhỏ hơn 256 PE.
          - Xử lý trọn vẹn từng hàng không cần chia nhỏ (No Tiling required here).
          
        Memory Flow:
          1. Load s_tanh[t] và x[j] từ SRAM -> PE, tính điểm Attention scores.
          2. Đẩy điểm Attention vừa tính vào PE, nhân với các cột y từ SRAM.
          3. Ghi kết quả vào local_attn_out (SRAM).
        """
        local_attn_out = np.zeros((self.T, self.D_chunk), dtype=np.float64)
        for t in range(self.T):
            s_row = local_s[t, :]
            self.pe.load_in_local_to_Compute(s_row)
            self.profiler.track('l_loads', self.D_chunk)
            
            attn_scores = np.zeros(self.T, dtype=np.float64)
            self.pe.load_bias(0.0)
            
            for j in range(self.T):
                self.pe.load_weight_to_Compute(local_x[j, :])
                self.profiler.track('l_loads', self.D_chunk)
                attn_scores[j] = self.pe.run_mac()
                self.profiler.track('macs', self.D_chunk)
                
            self.pe.load_in_local_to_Compute(attn_scores)
            self.profiler.track('l_loads', self.T) 
            
            for d in range(self.D_chunk):
                self.pe.load_weight_to_Compute(local_y[:, d])
                self.profiler.track('l_loads', self.T)
                
                local_attn_out[t, d] = self.pe.run_mac()
                self.profiler.track('l_stores', 1)
                self.profiler.track('macs', self.T)

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
                W_out_col = self.W_out[:, d]
                self.profiler.track('g_loads', self.D_chunk + 1) # Weight + Bias
                self.profiler.track('l_stores', self.D_chunk + 1)
                
                for t in range(act_T):
                    self.pe.load_in_local_to_Compute(local_attn_out[t_start+t, :])
                    self.pe.load_weight_to_Compute(W_out_col)
                    self.pe.load_bias(self.b_out[d])
                    self.profiler.track('l_loads', self.D_chunk * 2 + 1)
                    
                    final_val = self.pe.run_mac()
                    self.profiler.track('macs', self.D_chunk)
                    
                    self.profiler.track('g_loads', 1) # Load residual
                    output[t_start+t, d] = final_val + residual_X[t_start+t, d]
                    self.profiler.track('g_stores', 1)

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
        
        with torch.no_grad():
            self.ffw_in.weight.copy_(torch.from_numpy(ffw_hw.W1).T)
            self.ffw_in.bias.copy_(torch.from_numpy(ffw_hw.b1))
            self.ffw_out.weight.copy_(torch.from_numpy(ffw_hw.W2).T)
            self.ffw_out.bias.copy_(torch.from_numpy(ffw_hw.b2))
            
            self.attn_in.weight.copy_(torch.from_numpy(attn_hw.W_in).T)
            self.attn_in.bias.copy_(torch.from_numpy(attn_hw.b_in))
            self.attn_out.weight.copy_(torch.from_numpy(attn_hw.W_out).T)
            self.attn_out.bias.copy_(torch.from_numpy(attn_hw.b_out))

    def forward(self, x):
        res_ffw = x
        x = self.ffw_out(self.ffw_act(self.ffw_in(x))) + res_ffw
        res_attn = x
        in_proj = self.attn_in(x) 
        s, x_k, y_v = torch.chunk(in_proj, 3, dim=-1) 
        s_tanh = torch.tanh(s)
        attn_scores = torch.matmul(s_tanh, x_k.transpose(-2, -1))
        attn_out = torch.matmul(attn_scores, y_v)
        return self.attn_out(attn_out) + res_attn

if __name__ == "__main__":
    np.random.seed(99)
    T, D_in = 116, 192
    global_X = np.random.randn(T, D_in).astype(np.float64) / 30.0

    print("="*80)
    print("🚀 KHỞI ĐỘNG HỆ THỐNG MODULE HARDWARE")
    print("="*80)
    
    # 1. Khởi tạo các Modules với tham số tùy chỉnh
    ffw_module = FeedforwardHardware(T=T, D_in=D_in, D_ffw=384, tile_T=58, tile_H=192)
    attn_module = NonlinearAttentionHardware(T=T, D_in=D_in, D_attn=432, D_chunk=144, tile_T=58)
    
    start_time = time.time()
    
    # 2. Pipeline Execution: Dataflow đi qua các modules
    ffw_output, ffw_metrics = ffw_module.execute(global_X)
    
    # Trong thực tế, ffw_output sẽ được lưu ở một SRAM Buffer (Activation Buffer) lớn
    # Sau đó module Attention sẽ đọc từ đó lên để làm Input và Residual.
    final_output, attn_metrics = attn_module.execute(global_X=ffw_output, residual_X=ffw_output)
    
    print(f"Tổng thời gian mô phỏng phần cứng: {time.time() - start_time:.4f}s")
    
    # 3. In Báo cáo Độc lập cho từng khối (Đã cập nhật Tiếng Việt hoàn hảo)
    ffw_metrics.print_report()
    attn_metrics.print_report()

    # 4. Kiểm chứng tổng thể với PyTorch
    print("\n" + "="*80)
    print("VERIFICATION via PyTorch Golden Model")
    print("="*80)
    
    pt_model = ZipformerLayerPyTorch(ffw_module, attn_module).double()
    pt_model.eval()
    
    with torch.no_grad():
        x_tensor = torch.from_numpy(global_X).double()
        golden_output = pt_model(x_tensor).numpy()

    max_err = np.max(np.abs(golden_output - final_output))
    is_correct = np.allclose(golden_output, final_output, atol=1e-10)
    
    print(f"  Input X shape         : {x_tensor.shape}")
    print(f"  Golden output shape   : {golden_output.shape}")
    print(f"  Simulated output shape: {final_output.shape}")
    print(f"  Max error             : {max_err:.2e}")
    print(f"  Status: {'✅ PASSED' if is_correct else '❌ FAILED'}")
    print("="*80)