import sys
from unittest.mock import MagicMock
import warnings # Bỏ cảnh báo FutureWarning để giữ cho output sạch sẽ khi chạy verify_hardware.py
warnings.filterwarnings("ignore", category=FutureWarning)
# --- BÙA CHÚ ĐÁNH LỪA PYTHON (MOCKING) ---
sys.modules['k2'] = MagicMock()
sys.modules['k2.ragged'] = MagicMock()

import torch
import numpy as np

# 1. Import các module gốc từ thư viện của Dan Povey
from zipformer import FeedforwardModule, NonlinAttention 
# 2. Import các module Hardware Simulator đã được chúng ta xây dựng
from zipformer_modular_accelerator import (
    HardwareConfig, 
    FeedforwardHardware, 
    FeedforwardChannelFirst,
    FeedforwardBlockTiling, 
    NonlinearAttentionHardware
)

def verify_zipformer_hardware(config_file="config_block_tiling.json"):
    print("="*80)
    print(f"🚀 BẮT ĐẦU XÁC THỰC HARDWARE SIMULATOR 🚀")
    print("="*80)

    config = HardwareConfig.from_json(config_file)
    print(f"[*] Đang sử dụng file config : {config_file}")
    print(f"[*] Chiến thuật Tiling       : {config.strategy_name}")

    # ------------------------------------------------------------------
    # BƯỚC 1: KHỞI TẠO MÔ HÌNH PYTORCH GỐC
    # ------------------------------------------------------------------
    print("\n[1] Đang khởi tạo mô hình PyTorch gốc...")
    official_ffw = FeedforwardModule(embed_dim=config.D_in, feedforward_dim=config.D_ffw, dropout=0.0)
    official_attn = NonlinAttention(channels=config.D_in, hidden_channels=config.D_chunk)
    
    official_ffw.eval(); official_attn.eval()
    official_ffw.double(); official_attn.double()

    with torch.no_grad():
        official_ffw.hidden_balancer = torch.nn.Identity()
        official_ffw.out_whiten = torch.nn.Identity()
        official_attn.balancer = torch.nn.Identity()
        official_attn.whiten1 = torch.nn.Identity()
        official_attn.whiten2 = torch.nn.Identity()

    # ------------------------------------------------------------------
    # BƯỚC 1.5: VÁ LỖI MAGICMOCK CỦA KHỐI FFW (THE PATCH)
    # ------------------------------------------------------------------
    # Ghi đè hàm forward của lớp ActivationDropoutAndLinear để bỏ qua k2 
    # và chỉ dùng các hàm PyTorch cơ bản (SwooshL -> Linear).
    def patched_forward(x):
        # 1. SwooshL Activation
        x = torch.logaddexp(torch.zeros_like(x), x - 4.0) - 0.08 * x - 0.035
        # 2. Linear Transform
        w = official_ffw.out_proj.weight
        b = official_ffw.out_proj.bias
        return torch.nn.functional.linear(x, w, b)
        
    # Thay ruột hàm gốc bằng hàm của chúng ta
    official_ffw.out_proj.forward = patched_forward

    # ------------------------------------------------------------------
    # BƯỚC 2: AUTO-SWITCH HARDWARE MODULE DỰA VÀO JSON
    # ------------------------------------------------------------------
    print("[2] Khởi tạo Hardware Simulator và BƠM TRỌNG SỐ...")
    
    # Tự động chọn Class FFW dựa vào tên chiến thuật trong JSON
    if "Channel-First" in config.strategy_name:
        hw_ffw = FeedforwardChannelFirst(config)
    elif "Block" in config.strategy_name:
        hw_ffw = FeedforwardBlockTiling(config) # <--- Thêm dòng này
    else:
        hw_ffw = FeedforwardHardware(config)
        
    hw_attn = NonlinearAttentionHardware(config)

    with torch.no_grad():
        hw_ffw.W1 = official_ffw.in_proj.weight.detach().numpy().T
        hw_ffw.b1 = official_ffw.in_proj.bias.detach().numpy()
        hw_ffw.W2 = official_ffw.out_proj.weight.detach().numpy().T
        hw_ffw.b2 = official_ffw.out_proj.bias.detach().numpy()

        hw_attn.W_in = official_attn.in_proj.weight.detach().numpy().T
        hw_attn.b_in = official_attn.in_proj.bias.detach().numpy()
        hw_attn.W_out = official_attn.out_proj.weight.detach().numpy().T
        hw_attn.b_out = official_attn.out_proj.bias.detach().numpy()

    # ------------------------------------------------------------------
    # BƯỚC 3: TẠO DỮ LIỆU ĐẦU VÀO VÀ SO GĂNG CẢ 2 KHỐI
    # ------------------------------------------------------------------
    print("[3] Chạy Suy luận và So găng (Inference)...")
    np_X = np.random.randn(config.T, config.D_in).astype(np.float64) / 10.0
    pt_X = torch.from_numpy(np_X).unsqueeze(1) 

    np_attn_weights = np.random.rand(1, 1, config.T, config.T).astype(np.float64) 
    np_attn_weights = (np_attn_weights / np_attn_weights.sum(axis=-1, keepdims=True))
    pt_attn_weights = torch.from_numpy(np_attn_weights)

    with torch.no_grad():
        # PyTorch chạy FFW (chưa có Residual)
        pt_ffw_out = official_ffw(pt_X).squeeze(1).numpy()
        
        # THÊM DÒNG NÀY: Bù phép cộng Residual cho khớp với Hardware
        pt_ffw_out += np_X 
        
        # PyTorch chạy Attention (đã khớp sẵn)
        pt_attn_out = official_attn(pt_X, pt_attn_weights).squeeze(1).numpy()
    hw_ffw_out, _ = hw_ffw.execute(np_X)
    hw_attn_out, _ = hw_attn.execute(global_X=np_X, residual_X=np.zeros_like(np_X), global_attn_weights=np_attn_weights[0, 0])

    # ------------------------------------------------------------------
    # BƯỚC 4: BÁO CÁO PHÁN QUYẾT TỔNG THỂ
    # ------------------------------------------------------------------
    err_ffw = np.max(np.abs(hw_ffw_out - pt_ffw_out))
    err_attn = np.max(np.abs(hw_attn_out - pt_attn_out))
    
    print("\n" + "="*80)
    print(f"📊 BÁO CÁO XÁC THỰC (VERIFICATION REPORT)")
    print("="*80)
    print(f" [Khối FFW]       Max Err: {err_ffw:.2e} -> {'✅ PASSED' if np.allclose(hw_ffw_out, pt_ffw_out, atol=1e-10) else '❌ FAILED'}")
    print(f" [Khối Attention] Max Err: {err_attn:.2e} -> {'✅ PASSED' if np.allclose(hw_attn_out, pt_attn_out, atol=1e-10) else '❌ FAILED'}")
    print("="*80)

if __name__ == "__main__":
    torch.manual_seed(99)
    np.random.seed(99)
    
    verify_zipformer_hardware("config_block_tiling.json")