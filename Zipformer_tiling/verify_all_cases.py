import sys
import os
import json
from unittest.mock import MagicMock

# --- BÙA CHÚ ĐÁNH LỪA PYTHON (MOCKING) ---
sys.modules['k2'] = MagicMock()
sys.modules['k2.ragged'] = MagicMock()

import torch
import torch.nn as nn
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from zipformer import FeedforwardModule, NonlinAttention, SelfAttention
from zipformer_modular_accelerator import (
    HardwareConfig, 
    FeedforwardHardware, FeedforwardChannelFirst, FeedforwardBlockTiling,
    NonlinearAttentionHardware, NonlinearAttentionChannelFirst, NonlinearAttentionBlockTiling,
    SelfAttentionPipelineHardware
)

def verify_golden_stacks():
    print("="*135)
    print("🚀 BẮT ĐẦU VERIFY DỰA TRÊN GOLDEN CONFIGS TỪ DSE (6 STACKS) 🚀")
    print("="*135)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(script_dir, 'golden_configs.json')
    
    if not os.path.exists(json_path):
        print(f"❌ Không tìm thấy file {json_path}. Vui lòng chạy file sweep_tiling.py trước!")
        return

    with open(json_path, 'r') as f:
        golden_configs = json.load(f)

    ffw_classes = {
        "Full-Matrix": FeedforwardHardware,
        "Time-First": FeedforwardHardware,
        "Channel-First": FeedforwardChannelFirst,
        "Block Tiling": FeedforwardBlockTiling,
    }
    attn_classes = {
        "Full-Matrix": NonlinearAttentionHardware,
        "Time-First": NonlinearAttentionHardware,
        "Channel-First": NonlinearAttentionChannelFirst,
        "Block Tiling": NonlinearAttentionBlockTiling,
    }

    # [SỬA LỖI LOG]: Thêm 3 cột Tile Size hiển thị rõ ràng thông số băm của từng khối
    header = f"{'Stk':<3} | {'Shape(T,D)':<12} | {'FFW Tile':<12} | {'Nonlin Tile':<12} | {'Self Tile':<10} | {'FFW Check':<10} | {'Nonlin Check':<12} | {'Self Check':<10}"
    print(header)
    print("-" * 135)

    for cfg in golden_configs:
        sid = cfg["stack_id"]
        T, D_in, D_ffw, D_attn, D_cn, D_cs = cfg["shape"]["T"], cfg["shape"]["D_in"], cfg["shape"]["D_ffw"], cfg["shape"]["D_attn"], cfg["shape"]["D_chunk_n"], cfg["shape"]["D_chunk_s"]

        np.random.seed(99)
        torch.manual_seed(99)
        np_X = np.random.randn(T, D_in).astype(np.float64) / 10.0
        pt_X = torch.from_numpy(np_X).unsqueeze(1) 
        
        np_attn_w_3d = np.random.rand(4, T, T).astype(np.float64)
        np_attn_w_3d = np_attn_w_3d / np_attn_w_3d.sum(axis=-1, keepdims=True)
        np_attn_w_2d = np_attn_w_3d[0]
        
        pt_attn_w_2d = torch.from_numpy(np_attn_w_2d).unsqueeze(0).unsqueeze(0)
        pt_attn_w_3d = torch.from_numpy(np_attn_w_3d).unsqueeze(1)

        official_ffw = FeedforwardModule(embed_dim=D_in, feedforward_dim=D_ffw, dropout=0.0)
        official_attn = NonlinAttention(channels=D_in, hidden_channels=D_cn)
        official_self = SelfAttention(embed_dim=D_in, num_heads=4, value_head_dim=D_cs//4)
        
        official_ffw.eval(); official_attn.eval(); official_self.eval()
        official_ffw.double(); official_attn.double(); official_self.double()

        with torch.no_grad():
            official_ffw.hidden_balancer = nn.Identity()
            official_ffw.out_whiten = nn.Identity()
            official_attn.balancer = nn.Identity()
            official_attn.whiten1 = nn.Identity()
            official_attn.whiten2 = nn.Identity()
            official_self.whiten = nn.Identity()

            def patched_forward(x):
                x = torch.logaddexp(torch.zeros_like(x), x - 4.0) - 0.08 * x - 0.035
                return nn.functional.linear(x, official_ffw.out_proj.weight, official_ffw.out_proj.bias)
            official_ffw.out_proj.forward = patched_forward

            pt_ffw_out = official_ffw(pt_X).squeeze(1) + pt_X.squeeze(1)
            pt_attn_out = official_attn(pt_X, pt_attn_w_2d).squeeze(1)
            pt_self_out = (official_self(pt_X, pt_attn_w_3d) + pt_X).squeeze(1)

        # Lấy thông số Tile Size từ JSON
        f_t, f_h = cfg["ffw"]["t"], cfg["ffw"]["h"]
        n_t, n_c = cfg["nonlin"]["t"], cfg["nonlin"]["c"]
        s_t = cfg["self"]["t"]

        hw_c_f = HardwareConfig(T, D_in, D_ffw, D_attn, D_cn, cfg["ffw"]["strat"], f_t, f_h, 0, 0, 2048, 512)
        hw_c_n = HardwareConfig(T, D_in, D_ffw, D_attn, D_cn, cfg["nonlin"]["strat"], 0, 0, n_t, n_c, 2048, 512)
        hw_c_s = HardwareConfig(T, D_in, D_ffw, D_attn, D_cs, cfg["self"]["strat"], 0, 0, s_t, 0, 2048, 512)

        hw_f = ffw_classes[cfg["ffw"]["strat"]](hw_c_f)
        hw_n = attn_classes[cfg["nonlin"]["strat"]](hw_c_n)
        hw_s = SelfAttentionPipelineHardware(hw_c_s)

        hw_f.W1, hw_f.b1 = official_ffw.in_proj.weight.detach().numpy().T, official_ffw.in_proj.bias.detach().numpy()
        hw_f.W2, hw_f.b2 = official_ffw.out_proj.weight.detach().numpy().T, official_ffw.out_proj.bias.detach().numpy()

        hw_n.W_in, hw_n.b_in = official_attn.in_proj.weight.detach().numpy().T, official_attn.in_proj.bias.detach().numpy()
        hw_n.W_out, hw_n.b_out = official_attn.out_proj.weight.detach().numpy().T, official_attn.out_proj.bias.detach().numpy()
        
        hw_s.W_in, hw_s.b_in = official_self.in_proj.weight.detach().numpy().T, official_self.in_proj.bias.detach().numpy()
        hw_s.W_out, hw_s.b_out = official_self.out_proj.weight.detach().numpy().T, official_self.out_proj.bias.detach().numpy()

        hw_ffw_out, _ = hw_f.execute(np_X)
        hw_attn_out, _ = hw_n.execute(np_X, np.zeros_like(np_X), np_attn_w_2d)
        hw_self_out, _ = hw_s.execute(np_X, np_X, np_attn_w_3d)

        def compare_with_reference(hw_out, pt_out):
            try:
                torch.testing.assert_close(
                    torch.from_numpy(hw_out),
                    pt_out.detach().cpu(),
                    atol=1e-10,
                    rtol=0.0,
                )
                return True
            except AssertionError:
                return False

        ffw_pass = "✅ PASSED" if compare_with_reference(hw_ffw_out, pt_ffw_out) else "❌ FAILED"
        attn_pass = "✅ PASSED" if compare_with_reference(hw_attn_out, pt_attn_out) else "❌ FAILED"
        self_pass = "✅ PASSED" if compare_with_reference(hw_self_out, pt_self_out) else "❌ FAILED"
        
        # [SỬA LỖI LOG]: In ra rõ mồn một các Tile
        shape_str = f"({T},{D_in})"
        tile_f_str = f"({f_t},{f_h})"
        tile_n_str = f"({n_t},{n_c})"
        tile_s_str = f"({s_t}, -)"

        print(f"Stk {sid:<1} | {shape_str:<12} | {tile_f_str:<12} | {tile_n_str:<12} | {tile_s_str:<10} | {ffw_pass:<10} | {attn_pass:<12} | {self_pass:<10}")

    print("=" * 135)

if __name__ == "__main__":
    verify_golden_stacks()












# import sys
# from unittest.mock import MagicMock

# # --- BÙA CHÚ ĐÁNH LỪA PYTHON (MOCKING) ---
# sys.modules['k2'] = MagicMock()
# sys.modules['k2.ragged'] = MagicMock()

# import torch
# import numpy as np
# import warnings
# warnings.filterwarnings("ignore", category=FutureWarning)

# from zipformer import FeedforwardModule, NonlinAttention 
# from zipformer_modular_accelerator import (
#     HardwareConfig, 
#     FeedforwardHardware, FeedforwardChannelFirst, FeedforwardBlockTiling,
#     NonlinearAttentionHardware, NonlinearAttentionChannelFirst, NonlinearAttentionBlockTiling
# )

# def verify_all_sweep_cases():
#     print("="*100)
#     print("🚀 BẮT ĐẦU VERIFY TỰ ĐỘNG TOÀN BỘ TEST CASES 🚀")
#     print("="*100)

#     T, D_in, D_ffw, D_attn, D_chunk = 116, 192, 384, 432, 144
#     np.random.seed(99)
#     torch.manual_seed(99)

#     # 1. Danh sách y hệt file sweep
#     test_cases = [
#         ("Time-First", 116, 192, "Time-First", 116, 144), 
#         ("Time-First", 58, 192,  "Time-First", 58, 72),
#         ("Channel-First", 58, 192, "Time-First", 58, 72),
#         ("Block Tiling", 29, 96, "Time-First", 58, 72),
#         ("Block Tiling", 29, 96, "Block Tiling", 29, 72),
#         ("Channel-First", 58, 192, "Channel-First", 58, 72),
#         ("Time-First", 58, 192, "Block Tiling", 29, 72),
#         ("Block Tiling", 14, 48, "Block Tiling", 14, 36)
#     ]

#     ffw_classes = {
#         "Time-First": FeedforwardHardware,
#         "Channel-First": FeedforwardChannelFirst,
#         "Block Tiling": FeedforwardBlockTiling
#     }
#     attn_classes = {
#         "Time-First": NonlinearAttentionHardware,
#         "Channel-First": NonlinearAttentionChannelFirst,
#         "Block Tiling": NonlinearAttentionBlockTiling
#     }

#     # 2. Khởi tạo PyTorch Golden Model (Dùng chung cho mọi case để tiết kiệm thời gian)
#     official_ffw = FeedforwardModule(embed_dim=D_in, feedforward_dim=D_ffw, dropout=0.0)
#     official_attn = NonlinAttention(channels=D_in, hidden_channels=D_chunk)
#     official_ffw.eval(); official_attn.eval()
#     official_ffw.double(); official_attn.double()

#     with torch.no_grad():
#         official_ffw.hidden_balancer = torch.nn.Identity()
#         official_ffw.out_whiten = torch.nn.Identity()
#         official_attn.balancer = torch.nn.Identity()
#         official_attn.whiten1 = torch.nn.Identity()
#         official_attn.whiten2 = torch.nn.Identity()

#         # Vá lỗi K2 cho FFW
#         def patched_forward(x):
#             x = torch.logaddexp(torch.zeros_like(x), x - 4.0) - 0.08 * x - 0.035
#             return torch.nn.functional.linear(x, official_ffw.out_proj.weight, official_ffw.out_proj.bias)
#         official_ffw.out_proj.forward = patched_forward

#     # 3. Tạo Input Data
#     np_X = np.random.randn(T, D_in).astype(np.float64) / 10.0
#     pt_X = torch.from_numpy(np_X).unsqueeze(1) 
#     np_attn_weights = np.random.rand(1, 1, T, T).astype(np.float64) 
#     np_attn_weights = (np_attn_weights / np_attn_weights.sum(axis=-1, keepdims=True))
#     pt_attn_weights = torch.from_numpy(np_attn_weights)

#     with torch.no_grad():
#         pt_ffw_out = official_ffw(pt_X).squeeze(1).numpy() + np_X
#         pt_attn_out = official_attn(pt_X, pt_attn_weights).squeeze(1).numpy()

#     print(f"{'FFW Strat':<14} | {'Attn Strat':<14} | {'FFW Tile':<10} | {'Attn Tile':<10} | {'FFW Check':<10} | {'Attn Check':<10}")
#     print("-" * 85)

#     # 4. Vòng lặp test từng case
#     for ffw_strat, f_t, f_h, attn_strat, a_t, a_c in test_cases:
#         config = HardwareConfig(
#             T=T, D_in=D_in, D_ffw=D_ffw, D_attn=D_attn, D_chunk=D_chunk,
#             strategy_name=f"Mix", ffw_tile_T=f_t, ffw_tile_H=f_h,
#             attn_tile_T=a_t, attn_tile_C=a_c, num_pe=256, sram_limit_kb=512
#         )

#         hw_ffw = ffw_classes[ffw_strat](config)
#         hw_attn = attn_classes[attn_strat](config)

#         # Bơm trọng số
#         hw_ffw.W1 = official_ffw.in_proj.weight.detach().numpy().T
#         hw_ffw.b1 = official_ffw.in_proj.bias.detach().numpy()
#         hw_ffw.W2 = official_ffw.out_proj.weight.detach().numpy().T
#         hw_ffw.b2 = official_ffw.out_proj.bias.detach().numpy()

#         hw_attn.W_in = official_attn.in_proj.weight.detach().numpy().T
#         hw_attn.b_in = official_attn.in_proj.bias.detach().numpy()
#         hw_attn.W_out = official_attn.out_proj.weight.detach().numpy().T
#         hw_attn.b_out = official_attn.out_proj.bias.detach().numpy()

#         # Chạy Hardware
#         hw_ffw_out, _ = hw_ffw.execute(np_X)
#         hw_attn_out, _ = hw_attn.execute(np_X, np.zeros_like(np_X), np_attn_weights[0, 0])

#         # So sánh
#         ffw_pass = "✅ PASSED" if np.allclose(hw_ffw_out, pt_ffw_out, atol=1e-10) else "❌ FAILED"
#         attn_pass = "✅ PASSED" if np.allclose(hw_attn_out, pt_attn_out, atol=1e-10) else "❌ FAILED"
        
#         print(f"{ffw_strat:<14} | {attn_strat:<14} | {f'({f_t},{f_h})':<10} | {f'({a_t},{a_c})':<10} | {ffw_pass:<10} | {attn_pass:<10}")

# if __name__ == "__main__":
#     verify_all_sweep_cases()