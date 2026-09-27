"""
SelfAttention Tiled Pipeline Simulator
=======================================
class SelfAttention(nn.Module):


    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        value_head_dim: int,
    ) -> None:
        super().__init__()
        self.in_proj = nn.Linear(embed_dim, num_heads * value_head_dim, bias=True)

        self.out_proj = nn.Linear(
            num_heads * value_head_dim, embed_dim, bias=True
        )

    def forward(
        self,
        x: Tensor,
        attn_weights: Tensor,
    ) -> Tensor:

        (seq_len, batch_size, embed_dim) = x.shape
        num_heads = attn_weights.shape[0]
        assert attn_weights.shape == (num_heads, batch_size, seq_len, seq_len)

        x = self.in_proj(x)  # (seq_len, batch_size, num_heads * value_head_dim)
        x = x.reshape(seq_len, batch_size, num_heads, -1).permute(2, 1, 0, 3)
        # now x: (num_heads, batch_size, seq_len, value_head_dim)
        value_head_dim = x.shape[-1]

        # todo: see whether there is benefit in overriding matmul
        x = torch.matmul(attn_weights, x)
        # v: (num_heads, batch_size, seq_len, value_head_dim)

        x = (
            x.permute(2, 1, 0, 3)
            .contiguous()
            .view(seq_len, batch_size, num_heads * value_head_dim)
        )
        # returned value is of shape (seq_len, batch_size, embed_dim), like the input.
        x = self.out_proj(x)

        return x
   # Ở đây chỉ là class, khi tính toán sẽ có residual net



Kiến trúc theo đúng yêu cầu của bạn:

  DRAM:
    global_input[116,192]
    global_weight_in[192,48]
    global_attn_weight[4,116,116]
    global_weight_out[48,192]

  Local SRAM (pipeline + tiling [58,192]):
    local_input          [58, 192]   ← load tile input cho InProj
    local_input_outproject_buffer [116,48] ← Q0|Q1|Q2|Q3 (điền dần ngay khi có)
    local_Q_buf          [58,  48]   ← tile của local_input_outproject_buffer cho OutProj
    local_wout           [48, 192]   ← full W_out (tile_N=192)
    output_buffer        [116,192]   ← lưu nguyên Input từ đầu, sau OutProj sẽ cộng element-wise

Pipeline (per stage):
  Stage 0 : InProj(A) full T → local_val_cur[:,0:12]
            → Bắt đầu điền Q0 sau khi AttnMul(Q0) xong

  Stage 1 : InProj(B) || AttnMul(Q0) → local_input_outproject_buffer[:, 0:12]
  Stage 2 : InProj(C) || AttnMul(Q1) → local_input_outproject_buffer[:,12:24]
  Stage 3 : InProj(D) || AttnMul(Q2) → local_input_outproject_buffer[:,24:36]
  Stage 4 :           || AttnMul(Q3) → local_input_outproject_buffer[:,36:48]

  Sau khi có đủ Q0..Q3 trong local_input_outproject_buffer:
    PHA 2: OutProj tiled theo [58,192]
      load Q_tile[58,48]
      Q_tile @ W_out[48,192] + bias_out
      → cộng element-wise ngay vào output_buffer[58,192] (đã chứa Input từ đầu)
      → ghi thẳng global_output

Residual: 
  output_buffer bắt đầu = global_input
  Sau khi tính xong OutProj Linear cho từng tile → cộng ngay (không cần buffer output riêng)
"""


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time


# ──────────────────────────────────────────────────────────────────
# Compute Block (PE)
# ──────────────────────────────────────────────────────────────────
class Compute_block:
    """Processing Element: input_reg × weight_reg → MAC → output_reg"""

    def __init__(self, num_pe: int = 192):
        self.num_pe     = num_pe
        self.input_reg  = np.zeros(num_pe)
        self.weight_reg = np.zeros(num_pe)
        self.output_reg = np.zeros(1)
        self.bias_regs  = None          # set nếu cần bias

    # ── bias helpers ─────────────────────────────────────────
    def load_bias(self, bias_vector):
        self.bias_regs = np.array(bias_vector, dtype=np.float64)

    def set_current_bias(self, col_idx: int):
        self.output_reg = float(self.bias_regs[col_idx]) if self.bias_regs is not None else 0.0

    # ── data movement ─────────────────────────────────────────
    def load_input(self, vec):
        self.input_reg[:] = vec

    def load_weight(self, vec):
        self.weight_reg[:] = vec

    # ── MAC ───────────────────────────────────────────────────
    def mac(self):
        """Multiply-Accumulate: output += sum(input * weight)"""
        self.output_reg += float(np.sum(self.input_reg * self.weight_reg))

    def mac_fresh(self):
        """Fresh MAC (không cộng dồn): output = sum(input * weight)"""
        self.output_reg = float(np.sum(self.input_reg * self.weight_reg))

    def get_output(self):
        return self.output_reg


# ──────────────────────────────────────────────────────────────────
# Core  –  SelfAttention Tiled Pipeline
# ──────────────────────────────────────────────────────────────────
class CoreSelfAttention:
    """
    Mô phỏng phần cứng thực thi SelfAttention theo pipeline tiling.

    Các tham số:
      T         = 116   (sequence length)
      N         = 192   (embed_dim)
      V         = 48    (num_heads * value_head_dim = 4 * 12)
      num_heads = 4
      tile_T    = 58    (chia T thành 2 halves)
    """

    def __init__(self, T=116, N=192, V=48, num_heads=4, tile_T=58, seed=42):
        rng = np.random.default_rng(seed)

        self.T         = T
        self.N         = N
        self.V         = V
        self.num_heads = num_heads
        self.tile_T    = tile_T
        self.vhd       = V // num_heads          # value_head_dim = 12

        # ── Global Memory (DRAM) ──────────────────────────────────
        self.global_input        = rng.random((T, N))             # [116, 192]
        self.global_weight_in    = rng.random((N, V))             # [192,  48]
        self.global_bias_in      = rng.random(V)                  # [48]
        self.global_attn_weight  = self._random_attn(num_heads, T, rng)  # [4, 116, 116]
        self.global_weight_out   = rng.random((V, N))             # [ 48, 192]
        self.global_bias_out     = rng.random(N)                  # [192]
        
        # ── Local SRAM ────────────────────────────────────────────
        # InProj tile buffers
        self.local_input   = np.zeros((tile_T, N))      # [58, 192]
        self.local_weight  = np.zeros((N, self.vhd))    # [192,  12]  – 1 head tại 1 thời điểm
        self.local_bias_in = np.zeros(())

        # Value head buffers (cur = đang tính, pre = đã xong)
        self.local_value_cur = np.zeros((T, self.vhd))  # [116, 12]
        self.local_value_pre = np.zeros((T, self.vhd))  # [116, 12]

        # AttnMatmul output tile
        self.local_attn_out  = np.zeros((T, self.vhd))  # [116, 12]  – Q_h kết quả

        # Input for outproject Linear (ghép Q0..Q3 dần dần)
        self.local_input_outproject_buffer = np.zeros((T, V))  # [116, 48]  – ghép Q0|Q1|Q2|Q3 dần dần
        self.local_Q_buf = np.zeros((tile_T, V))               # [58, 48]   – Q tile cho OutProj
        self.local_wout = np.zeros((V, N))                     # [48, 192]  – full W_out
        self.output_buffer = np.zeros((T, N))                  # [116, 192] – chứa residual + linear out

        # Assembled value concat + OutProj
        self.global_output       = np.zeros((T, N))     # [116, 192]  = kết quả cuối

        # ── Compute Blocks ────────────────────────────────────────
        self.block_inproj  = Compute_block(num_pe=N)         # 192 PE
        self.block_attnmat = Compute_block(num_pe=T)          # 116 PE (dot với 1 cột K)
        self.block_outproj = Compute_block(num_pe=V)         # 48 PE (dot với 1 cột W_out + bias)
    # ─────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────
    @staticmethod
    def _random_attn(num_heads, T, rng):
        """Tạo attn_weight ngẫu nhiên đã softmax, shape [num_heads, T, T]"""
        raw = rng.random((num_heads, T, T)).astype(np.float64)
        # softmax theo axis=-1
        raw -= raw.max(axis=-1, keepdims=True)
        exp  = np.exp(raw)
        return exp / exp.sum(axis=-1, keepdims=True)

    def _load_input_tile(self, t_start):
        """Global DRAM → Local SRAM: input tile [tile_T, N]"""
        t_end = min(t_start + self.tile_T, self.T)
        rows  = t_end - t_start
        self.local_input[:rows, :] = self.global_input[t_start:t_end, :]
        return rows

    def _load_weight_head(self, head_idx):
        """Global DRAM → Local SRAM: weight cho 1 head [N, vhd]"""
        c0 = head_idx * self.vhd
        c1 = c0 + self.vhd
        self.local_weight[:, :] = self.global_weight_in[:, c0:c1]

    # ─────────────────────────────────────────────────────────────
    # Stage 1 – InProj Linear (tiled over T, per head)
    # ─────────────────────────────────────────────────────────────
    def _compute_inproj_head(self, head_idx):
        """
        Tính A/B/C/D = Input @ W_in[:, head*vhd : (head+1)*vhd] + bias
        Kết quả → local_value_cur [116, vhd]

        Tiling T: vòng lặp ngoài theo tile_T (T rows/tile)
        PE (block_inproj, 192 PEs): mỗi lần tính 1 phần tử output
          input_reg  [192] ← row của input
          weight_reg [192] ← cột v của local_weight
          output      = dot + bias
        """
        self._load_weight_head(head_idx)
        bias_offset = head_idx * self.vhd

        for t_start in range(0, self.T, self.tile_T):
            rows = self._load_input_tile(t_start)
            for t_idx in range(rows):
                row     = self.local_input[t_idx, :]          # [192]
                row_abs = t_start + t_idx
                self.block_inproj.load_input(row)

                for v in range(self.vhd):                     # v = 0..11
                    self.block_inproj.load_weight(self.local_weight[:, v])
                    bias = float(self.global_bias_in[bias_offset + v])
                    self.block_inproj.output_reg = 0.0
                    self.block_inproj.mac()
                    self.block_inproj.output_reg += bias
                    self.local_value_cur[row_abs, v] = self.block_inproj.get_output()

    # ─────────────────────────────────────────────────────────────
    # Stage 2 – AttnMatmul: Q_h = AttnWeight[h] @ Value_h
    # ─────────────────────────────────────────────────────────────
    def _compute_attn_matmul(self, head_idx):
        """
        Q_h[i, v] = sum_j  AttnWeight[h, i, j] * Value_h[j, v]
                  = AttnWeight[h, i, :] · Value_h[:, v]

        Thực thi:
          - Outer: i  (0..115)  – query row
          - Inner: v  (0..11)   – value dimension
          block_attnmat (116 PE):
            input_reg  [116] ← AttnWeight[head, i, :]
            weight_reg [116] ← Value_h[:, v]   (từ local_value_pre)
            output      = dot
        """
        attn_h = self.global_attn_weight[head_idx]   # [116, 116]
        for i in range(self.T):
            attn_row = attn_h[i, :]                  # [116]
            self.block_attnmat.load_input(attn_row)
            for v in range(self.vhd):
                val_col = self.local_value_pre[:, v]  # [116]
                self.block_attnmat.load_weight(val_col)
                self.block_attnmat.output_reg = 0.0
                self.block_attnmat.mac()
                self.local_attn_out[i, v] = self.block_attnmat.get_output()
            

    # ─────────────────────────────────────────────────────────────
    # Stage 3 – OutProj Linear
    # ─────────────────────────────────────────────────────────────
    def _compute_outproj(self):
        """
        OutputLinear = ValueConcat @ W_out + bias_out
        OutputFinal  = Input + OutputLinear  (residual add tại output_buffer)

        ValueConcat [116, 48], W_out [48, 192], Input [116, 192]

        Tiling T: 58 rows/tile
        PE (block_inproj tái sử dụng, 192 PEs → dùng 48 PE đầu):
          input_reg  [48] ← row của ValueConcat
          weight_reg [48] ← cột n của W_out
          output      = dot + bias_out[n]

        Dòng dữ liệu theo kiến trúc:
          output_buffer khởi tạo từ global_input
          mỗi tile Q[58,48] tính linear out rồi cộng trực tiếp vào output_buffer
          ghi thẳng global_output
        """
        # Sử dụng block_outproj (48 PE)
        block = self.block_outproj

        # preload full W_out vào SRAM local theo mô tả kiến trúc
        self.local_wout[:, :] = self.global_weight_out

        # output_buffer bắt đầu bằng residual input
        self.output_buffer[:, :] = self.global_input

        for t_start in range(0, self.T, self.tile_T):
            t_end = min(t_start + self.tile_T, self.T)
            rows = t_end - t_start

            # load Q tile [58,48] từ local_input_outproject_buffer
            self.local_Q_buf[:rows, :] = self.local_input_outproject_buffer[t_start:t_end, :]

            for t_idx in range(rows):
                row_abs = t_start + t_idx
                row_v = self.local_Q_buf[t_idx, :]          # [48]
                block.load_input(row_v)
                for n in range(self.N):                    # n = 0..191
                    block.load_weight(self.local_wout[:, n])
                    block.output_reg = 0.0
                    block.mac()
                    block.output_reg += float(self.global_bias_out[n])

                    # residual cộng tại chỗ vào output_buffer
                    self.output_buffer[row_abs, n] += block.get_output()
                    self.global_output[row_abs, n] = self.output_buffer[row_abs, n]

    # ─────────────────────────────────────────────────────────────
    # Pipeline Execution
    # ─────────────────────────────────────────────────────────────
    def execute(self):
        """
        Pipeline tổng thể:

          Head 0:  InProj(A)
          ─────────────────────────────────────────────
          Head 1:  InProj(B)   ||   AttnMatmul(Q0 ← A) ←
          Head 2:  InProj(C)   ||   AttnMatmul(Q1 B)
          Head 3:  InProj(D)   ||   AttnMatmul(Q2 ← C)
                               ||   AttnMatmul(Q3 ← D)
          ─────────────────────────────────────────────
          OutProj: [Q0|Q1|Q2|Q3] @ W_out + bias_out + residual(Input)

        Quy trình dữ liệu:
          local_value_cur  : head đang tính
          local_value_pre  : head vừa xong → dùng cho AttnMatmul song song
          local_attn_out   : Q_h vừa tính xong → ghép vào local_input_outproject_buffer
        """
        print("\n" + "="*70)
        print("SELF-ATTENTION TILED PIPELINE EXECUTION")
        print("="*70)
        print(f"  T={self.T}, N={self.N}, V={self.V}, heads={self.num_heads}, vhd={self.vhd}")
        print(f"  tile_T={self.tile_T}")
        print(f"  InProj  : Input[{self.T},{self.N}] @ W_in[{self.N},{self.V}] → [{self.T},{self.V}]")
        print(f"  Attn    : AttnW[4,{self.T},{self.T}] @ Value[{self.T},{self.vhd}] → [{self.T},{self.vhd}] x4")
        print(
            f"  OutProj : ValueConcat[{self.T},{self.V}] @ W_out[{self.V},{self.N}] + residual(Input) "
            f"→ [{self.T},{self.N}]"
        )

        # ── Head 0: Chỉ InProj, chưa có pre-value để AttnMatmul ──
        print("\n[Head 0]  InProj(A) ...")
        self._compute_inproj_head(head_idx=0)
        self.local_value_pre[:, :] = self.local_value_cur.copy()   # A → pre
        print(f"  A (head-0 value) computed: shape {self.local_value_cur.shape}, "
              f"sample A[0,:3] = {self.local_value_cur[0,:3].round(4)}")

        # ── Heads 1-3: Pipeline InProj || AttnMatmul ──────────────
        for h in range(1, self.num_heads):
            print(f"\n[Head {h}]  InProj({chr(65+h)}) || AttnMatmul(Q{h-1} ← {chr(65+h-1)}) ...")

            # --- Parallel simulation ---
            # Trong phần cứng thực: 2 việc này chạy đồng thời trên 2 unit khác nhau.
            # Ở đây mô phỏng tuần tự để kiểm tra tính đúng đắn.

            # PIPELINE SLOT A: InProj cho head h → local_value_cur
            self._compute_inproj_head(head_idx=h)
            print(f"  InProj done  → {chr(65+h)}[0,:3] = {self.local_value_cur[0,:3].round(4)}")

            # PIPELINE SLOT B: AttnMatmul dùng local_value_pre (head h-1)
            self._compute_attn_matmul(head_idx=h-1)
            print(f"  AttnMul done → Q{h-1}[0,:3] = {self.local_attn_out[0,:3].round(4)}")

            # Ghi Q{h-1} vào local_input_outproject_buffer theo đúng layout Q0|Q1|Q2|Q3
            c0 = (h-1) * self.vhd
            self.local_input_outproject_buffer[:, c0:c0+self.vhd] = self.local_attn_out.copy()

            # Swap buffers: cur → pre cho vòng tiếp
            self.local_value_pre[:, :] = self.local_value_cur.copy()

        # ── Sau vòng lặp: tính Q3 (head 3) từ D ─────────────────
        print(f"\n[Post-loop]  AttnMatmul(Q3 ← D) ...")
        self._compute_attn_matmul(head_idx=self.num_heads - 1)
        print(f"  Q3[0,:3] = {self.local_attn_out[0,:3].round(4)}")
        c0 = (self.num_heads - 1) * self.vhd
        self.local_input_outproject_buffer[:, c0:c0+self.vhd] = self.local_attn_out.copy()

        # Giữ bản assembled toàn cục để debug/verify thuận tiện

        print(f"\n  Local input for outproject Linear buffer : shape {self.local_input_outproject_buffer.shape}")
        print(f"  Input_Outproject_buffer = {self.local_input_outproject_buffer[0,:].round(4)}")

        # ── OutProj ───────────────────────────────────────────────
        print("\n[OutProj]  ValueConcat @ W_out + bias_out + residual ...")
        self._compute_outproj()
        print(f"  Output[0,:4] = {self.global_output[0,:4].round(4)}")
        print("\n✓ Execution complete.")


# ──────────────────────────────────────────────────────────────────
# Golden Reference (PyTorch)
# ──────────────────────────────────────────────────────────────────
def golden_reference(core: CoreSelfAttention) -> np.ndarray:
    """
    Tính kết quả chuẩn bằng PyTorch thuần:
      1. value   = Input @ W_in.T + bias_in     → [116, 48]  (dùng conv1d style cho đơn giản)
      2. Chia thành 4 heads [116, 12] mỗi head
                3. Q_h    = attn_weight[h] @ value_h      → [116, 12]
                4. Ghép   [Q0|Q1|Q2|Q3]                   → [116, 48]
                5. output_linear = ValueConcat @ W_out + bias_out → [116, 192]
                6. output_final  = Input + output_linear          → [116, 192] (residual)
    """
    x   = torch.from_numpy(core.global_input).double()               # [116, 192]
    Win = torch.from_numpy(core.global_weight_in).double()           # [192, 48]
    bin_= torch.from_numpy(core.global_bias_in).double()             # [48]
    AW  = torch.from_numpy(core.global_attn_weight).double()         # [4, 116, 116]
    Wout= torch.from_numpy(core.global_weight_out).double()          # [48, 192]
    bout= torch.from_numpy(core.global_bias_out).double()            # [192]

    vhd = core.vhd

    # InProj
    value = x @ Win + bin_                                            # [116, 48]

    # AttnMatmul per head
    heads = []
    for h in range(core.num_heads):
        v_h   = value[:, h*vhd:(h+1)*vhd]                           # [116, 12]
        q_h   = AW[h] @ v_h                                          # [116, 12]
        heads.append(q_h)
    value_concat = torch.cat(heads, dim=1)                            # [116, 48]

    # OutProj
    out = x + (value_concat @ Wout + bout)                            # [116, 192]
    return out.numpy()


# ──────────────────────────────────────────────────────────────────
# Verification
# ──────────────────────────────────────────────────────────────────
def verify(core: CoreSelfAttention):
    print("\n" + "="*70)
    print("VERIFICATION")
    print("="*70)

    golden = golden_reference(core)
    simulated = core.global_output

    max_err  = np.max(np.abs(golden - simulated))
    passed   = np.allclose(golden, simulated, atol=1e-9)

    print(f"  Golden    output[0,:4] : {golden[0,:4].round(6)}")
    print(f"  Simulated output[0,:4] : {simulated[0,:4].round(6)}")
    print(f"  Max absolute error     : {max_err:.2e}")
    print(f"  Status: {'✅  PASSED' if passed else '❌  FAILED'}")
    return passed


# ──────────────────────────────────────────────────────────────────
# Pipeline Stage Diagram (text)
# ──────────────────────────────────────────────────────────────────
def print_pipeline_diagram():
    diagram = """
Pipeline Timeline
─────────────────────────────────────────────────────────────────
 Cycle/Stage │  Compute Unit 1 (InProj)  │  Compute Unit 2 (Attn)
─────────────┼───────────────────────────┼───────────────────────
   Stage 0   │  InProj(A) = Input@W[:,0:12]│       idle
             │  → A[116,12] saved         │
─────────────┼───────────────────────────┼───────────────────────
   Stage 1   │  InProj(B) = Input@W[:,12:24]│ Q0 = AttnW[0] @ A
             │  → B[116,12] saved          │  → Q0[116,12] saved
─────────────┼───────────────────────────┼───────────────────────
   Stage 2   │  InProj(C) = Input@W[:,24:36]│ Q1 = AttnW[1] @ B
             │  → C[116,12] saved          │  → Q1[116,12] saved
─────────────┼───────────────────────────┼───────────────────────
   Stage 3   │  InProj(D) = Input@W[:,36:48]│ Q2 = AttnW[2] @ C
             │  → D[116,12] saved          │  → Q2[116,12] saved
─────────────┼───────────────────────────┼───────────────────────
   Stage 4   │       idle                  │ Q3 = AttnW[3] @ D
             │                             │  → Q3[116,12] saved
─────────────┼───────────────────────────┼───────────────────────
    Stage 5   │  OutProj: [Q0|Q1|Q2|Q3][116,48] @ W_out[48,192] + residual
                 │  → Output[116,192]
─────────────────────────────────────────────────────────────────

Memory Data Flow:
  DRAM ──► local_input[58,192]   (tile_T = 58 rows per load)
       ──► local_weight[192,12]  (1 head at a time)
       ──► global_attn_weight    (used directly per head)
       ──► global_weight_out

  local_value_cur[116,12]  ←  InProj result (current head)
  local_value_pre[116,12]  ←  InProj result (previous head, ready for Attn)
    local_attn_out[116,12]   ←  AttnMatmul result
    local_input_outproject_buffer[116,48] ← assembled Q0..Q3
    output_buffer[116,192]   ←  Input + OutProjLinear
    global_output[116,192]   ←  final result
"""
    print(diagram)


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print_pipeline_diagram()

    core = CoreSelfAttention(
        T=116, N=192, V=48, num_heads=4, tile_T=29, seed=0
    )

    t0 = time.time()
    core.execute()
    elapsed = time.time() - t0

    passed = verify(core)

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"  Execution time : {elapsed:.6f}s")
    print(f"  Result         : {'✅  ALL PASSED' if passed else '❌  FAILED'}")
    print("="*70)