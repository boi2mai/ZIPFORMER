import numpy as np
import torch
import torch.nn as nn
import time

class Compute_block:
    """Processing Element với registers (Giữ nguyên kiến trúc của bạn)"""
    def __init__(self, num_pe=256):
        self.num_pe = num_pe
        self.input_reg = np.zeros(self.num_pe, dtype=np.float32)   
        self.weight_reg = np.zeros(self.num_pe, dtype=np.float32)  
        self.output_reg = np.zeros(1, dtype=np.float32)    
        self.current_bias = np.float32(0.0)

    def load_bias(self, bias_val):
        self.current_bias = np.float32(bias_val)

    def load_in_local_to_Compute(self, input_vector):
        length = len(input_vector)
        self.input_reg[:length] = input_vector
        self.input_reg[length:] = 0.0 # Padding 0 cho đủ num_pe

    def load_weight_to_Compute(self, weight_vector):
        length = len(weight_vector)
        self.weight_reg[:length] = weight_vector
        self.weight_reg[length:] = 0.0

    def run_mac(self):
        """Thực hiện Multiply-Accumulate"""
        self.output_reg = np.sum(self.input_reg * self.weight_reg) + self.current_bias
        return self.output_reg


class FFWCore:
    def __init__(self, T=116, D_in=192, D_hidden=384, tile_T=58, tile_H=192):
        self.T = T
        self.D_in = D_in
        self.D_hidden = D_hidden
        
        # Tiling config
        self.tile_T = tile_T  # Số lượng rows xử lý 1 lần
        self.tile_H = tile_H  # Chia đôi D_hidden (384 -> 192) để vừa PE=256

        # ========== BỘ ĐẾM HIỆU NĂNG (HARDWARE PROFILER) ==========
        self.metrics = {
            'global_loads': 0,   # Số phần tử load từ Global DRAM vào SRAM
            'global_stores': 0,  # Số phần tử ghi từ SRAM ra Global DRAM
            'local_loads': 0,    # Số phần tử load từ Local SRAM vào Core (Registers)
            'local_stores': 0,   # Số phần tử ghi từ Core/Logic vào Local SRAM
            'mac_ops': 0         # Số phép tính Multiply-Accumulate (MAC)
        }

        # ========== GLOBAL MEMORY (DRAM) ==========
        self.global_X  = np.random.randn(T, D_in).astype(np.float32)
        
        # In_proj weights (Linear 1)
        self.global_W1 = np.random.randn(D_in, D_hidden).astype(np.float32) / 10.0
        self.global_b1 = np.random.randn(D_hidden).astype(np.float32)
        
        # Out_proj weights (Linear 2)
        self.global_W2 = np.random.randn(D_hidden, D_in).astype(np.float32) / 10.0
        self.global_b2 = np.random.randn(D_in).astype(np.float32)
        
        self.global_Output = np.zeros((T, D_in), dtype=np.float32)

        # ========== LOCAL MEMORY (SRAM) ==========
        self.local_X     = np.zeros((self.tile_T, self.D_in), dtype=np.float32)
        self.local_H     = np.zeros(self.tile_H, dtype=np.float32) # Chỉ cần 1 buffer nhỏ [192] cho trung gian
        self.local_Y_acc = np.zeros((self.tile_T, self.D_in), dtype=np.float32) # Accumulator cho kết quả Linear 2
        
        self.pe = Compute_block(num_pe=256)

    def track_global_to_local(self, num_elements):
        self.metrics['global_loads'] += num_elements
        self.metrics['local_stores'] += num_elements

    def track_local_to_global(self, num_elements):
        self.metrics['local_loads'] += num_elements
        self.metrics['global_stores'] += num_elements

    def track_local_to_core(self, num_elements):
        self.metrics['local_loads'] += num_elements

    def track_core_to_local(self, num_elements):
        self.metrics['local_stores'] += num_elements

    def track_mac(self, vector_len):
        """1 MAC = 1 phép nhân + 1 phép cộng. Tính trên vector độ dài N thì tốn N MACs"""
        self.metrics['mac_ops'] += vector_len

    def activation_swooshL(self, x):
        """
        Hàm SwooshL (Activation) - Cập nhật chính xác theo mã nguồn gốc
        swoosh_l(x) = log(1 + exp(x-4)) - 0.08*x - 0.035
        (Dùng np.logaddexp để đảm bảo numerical stability giống torch)
        """
        return np.float32(np.logaddexp(0.0, x - 4.0) - 0.08 * x - 0.035)

    def execute_ffw_pipeline(self):
        """
        Thực thi FFW với Operator Fusion: (Linear1 -> SwooshL -> Linear2_Accumulate)
        """
        print(f"Bắt đầu xử lý FFW Tiling (T={self.T}, D={self.D_in}, Hidden={self.D_hidden})...")
        print("Dữ liệu đang được sử dụng ở định dạng: Float32 (4 bytes/phần tử)")
        
        # Vòng lặp 1: Duyệt qua các cụm thời gian (Time tiles)
        for t_start in range(0, self.T, self.tile_T):
            actual_T = min(self.tile_T, self.T - t_start)
            
            # Load X từ Global -> Local
            self.local_X[:actual_T, :] = self.global_X[t_start:t_start+actual_T, :]
            self.track_global_to_local(actual_T * self.D_in)

            # Khởi tạo Accumulator bằng 0
            self.local_Y_acc.fill(0)
            
            # Vòng lặp 2: Duyệt qua các khối của Hidden Dimension (384 chia làm 2 khối 192)
            for h_start in range(0, self.D_hidden, self.tile_H):
                # Load weights cho khối này (Mô phỏng việc đưa trọng số vào SRAM)
                W1_tile = self.global_W1[:, h_start : h_start+self.tile_H] # [192, 192]
                b1_tile = self.global_b1[h_start : h_start+self.tile_H]    # [192]
                W2_tile = self.global_W2[h_start : h_start+self.tile_H, :] # [192, 192]

                self.track_global_to_local(self.D_in * self.tile_H)      # W1
                self.track_global_to_local(self.tile_H)                  # b1
                self.track_global_to_local(self.tile_H * self.D_in)      # W2

                # Xử lý từng hàng trong Tile
                for t in range(actual_T):
                    row_X = self.local_X[t, :]
                    
                    # === STAGE 1: IN_PROJ (LINEAR 1) + FUSION ACTIVATION ===
                    self.pe.load_in_local_to_Compute(row_X)
                    self.track_local_to_core(self.D_in) # Load X vào PE
                    
                    for h in range(self.tile_H):
                        # Tính X @ W1[:, h] + b1[h]
                        self.pe.load_weight_to_Compute(W1_tile[:, h])
                        self.track_local_to_core(self.D_in) # Load W1_col vào PE
                        
                        self.pe.load_bias(b1_tile[h])
                        self.track_local_to_core(1) # Load Bias 1 vào PE

                        mac_result = self.pe.run_mac()
                        self.track_mac(self.D_in) # Thực hiện D_in phép MAC
                        
                        # Tích hợp Activation NGAY LẬP TỨC (FUSION)
                        self.local_H[h] = self.activation_swooshL(mac_result)
                        self.track_core_to_local(1) # Lưu H trung gian vào SRAM
                        
                    # Lúc này local_H đang chứa kết quả sau Activation (kích thước 192)
                    
                    # === STAGE 2: OUT_PROJ (LINEAR 2 ACCUMULATION) ===
                    self.pe.load_in_local_to_Compute(self.local_H)
                    self.track_local_to_core(self.tile_H) # Load vector trung gian H vào PE
                    
                    self.pe.load_bias(0) # Linear 2 sẽ cộng b2 ở cuối cùng
                    
                    for d_out in range(self.D_in):
                        # Tính H @ W2[:, d_out]
                        self.pe.load_weight_to_Compute(W2_tile[:, d_out])
                        self.track_local_to_core(self.tile_H) # Load W2_col vào PE
                        
                        acc_val = self.pe.run_mac()
                        self.track_mac(self.tile_H) # Thực hiện tile_H phép MAC
                        
                        # Cộng dồn (Accumulate) vào kết quả
                        self.track_local_to_core(1) # Đọc từ Local_Y_acc để chuẩn bị cộng
                        self.local_Y_acc[t, d_out] += acc_val
                        self.track_core_to_local(1) # Ghi đè lại Local_Y_acc sau khi cộng

            # Sau khi cộng dồn xong tất cả các khối Hidden, xử lý Bias và Residual
            for t in range(actual_T):
                for d_out in range(self.D_in):
                    b2_val = self.global_b2[d_out]
                    self.track_global_to_local(1) # Load b2 từ Global
                    
                    # Cần load Y_acc và X từ Local
                    y_val = self.local_Y_acc[t, d_out]
                    x_val = self.local_X[t, d_out]
                    self.track_local_to_core(2)

                    # Output = Y_acc + b2 + Residual_X
                    final_val = np.float32(y_val + b2_val + x_val)
                    
                    # Store kết quả ra Global Memory
                    self.global_Output[t_start + t, d_out] = final_val
                    self.track_local_to_global(1) # Lưu Output ra Global

        print("Hoàn thành tính toán Khối FFW!\n")
        self.print_hardware_metrics()

    def print_hardware_metrics(self):
        g_loads = self.metrics['global_loads']
        l_loads = self.metrics['local_loads']
        macs = self.metrics['mac_ops']
        
        # Số phép tính (MACs) trên 1 phần tử load từ Global (Arithmetic Intensity)
        compute_per_global_load = macs / g_loads if g_loads > 0 else 0
        
        print("="*60)
        print("📊 BÁO CÁO PHÂN TÍCH HIỆU NĂNG TILING (HARDWARE METRICS)")
        print("="*60)
        print(f"Tổng số phần tử (float32) Load từ GLOBAL (DRAM) : {g_loads:,.0f} elements")
        print(f"Tổng số phần tử (float32) Ghi ra GLOBAL (DRAM)  : {self.metrics['global_stores']:,.0f} elements")
        print(f"Tổng số phần tử (float32) Load từ LOCAL vào Core: {l_loads:,.0f} elements")
        print(f"Tổng số phần tử (float32) Ghi từ Core ra LOCAL  : {self.metrics['local_stores']:,.0f} elements")
        print(f"Tổng số phép toán tính toán thực tế (MAC Ops)   : {macs:,.0f} ops")
        print("-" * 60)
        print(f"💡 Cường độ tính toán (Arithmetic Intensity):")
        print(f"   => {compute_per_global_load:.2f} phép MACs / 1 lần truy cập Global DRAM")
        print(f"   (Chỉ số này càng cao nghĩa là Tiling càng hiệu quả, giảm nghẽn cổ chai RAM)")
        print("="*60)


# ==========================================
# PHẦN KIỂM TRA ĐỘ CHÍNH XÁC VỚI PYTORCH
# ==========================================
class PyTorchSwooshL(nn.Module):
    def forward(self, x):
        # Cập nhật theo đúng class SwooshLFunction / SwooshLOnnx
        zero = torch.tensor(0.0, dtype=x.dtype, device=x.device)
        return torch.logaddexp(zero, x - 4.0) - 0.08 * x - 0.035

class PyTorchFFW(nn.Module):
    def __init__(self, core_instance):
        super().__init__()
        self.in_proj = nn.Linear(192, 384)
        self.act = PyTorchSwooshL()
        self.out_proj = nn.Linear(384, 192)
        
        with torch.no_grad():
            self.in_proj.weight.copy_(torch.from_numpy(core_instance.global_W1).T)
            self.in_proj.bias.copy_(torch.from_numpy(core_instance.global_b1))
            self.out_proj.weight.copy_(torch.from_numpy(core_instance.global_W2).T)
            self.out_proj.bias.copy_(torch.from_numpy(core_instance.global_b2))

    def forward(self, x):
        residual = x
        x = self.in_proj(x)
        x = self.act(x)
        x = self.out_proj(x)
        return x + residual

if __name__ == "__main__":
    np.random.seed(42) # Set seed để dễ kiểm chứng độ chính xác
    # 1. Khởi tạo và chạy Tiling Core
    core = FFWCore()
    start = time.time()
    core.execute_ffw_pipeline()
    print(f"Thời gian Tiling mô phỏng: {time.time() - start:.4f}s")

    print("\nĐang kiểm tra chéo với PyTorch chuẩn...")
    pt_model = PyTorchFFW(core)
    pt_model.eval()
    
    x_tensor = torch.from_numpy(core.global_X)
    with torch.no_grad():
        golden_output = pt_model(x_tensor).numpy()

    max_err = np.max(np.abs(golden_output - core.global_Output))
    
    if np.allclose(golden_output, core.global_Output, atol=1e-4):
        print(f"✅ TRẠNG THÁI: THÀNH CÔNG (Sai số {max_err:.2e})")
    else:
        print(f"❌ TRẠNG THÁI: THẤT BẠI (Sai số {max_err:.2e})")