import numpy as np
import torch
import torch.nn.functional as F
import time


class Compute_block:
    """Processing Element với registers"""
    def __init__(self, num_pe=256, M=None, is_linear=True):
        self.num_pe = num_pe
        self.input_reg = np.zeros(self.num_pe)   
        self.weight_reg = np.zeros(self.num_pe)  
        self.output_reg = np.zeros(1)    
        self.all_bias_regs = np.zeros(M) if M is not None else np.zeros(0)
        self.current_bias = 0.0

    def load_entire_bias_to_core(self, full_bias_vector):
        self.all_bias_regs = np.array(full_bias_vector).copy()

    def select_bias_for_column(self, m_global_idx):
        self.current_bias = self.all_bias_regs[m_global_idx]
        self.output_reg = np.zeros(1)

    def load_in_local_to_Compute(self, input_vector):
        """Load từ Local SRAM → PE input_reg"""
        self.input_reg = np.array(input_vector)

    def load_weight_to_Compute(self, weight_vector):
        """Load từ Local SRAM → PE weight_reg"""
        self.weight_reg = np.array(weight_vector)

    def get_output_to_local(self):
        """Get output từ PE → Local SRAM"""
        return self.output_reg


class Core:
    def __init__(self, T=116, N=192, M=272, tile_T=58, tile_M=16):
        """
        Tri-stage Accelerator: Linear + Attention + Pos-emb Attention
        
        STAGE 1 - PIPELINE - LINEAR LAYER // ATTENTION LAYER (attn_scores)
          Input[116, 192] @ Weight[192, 272] + Bias[272] → Output[116, 272]

          Q = Output[:, :128]      [116, 128]
          K = Output[:, 128:256]   [116, 128]
          attn_score = Q @ K.T  → [4, 116, 116]  (4 heads x 32 dims/head)

        STAGE 2 - PIPELINE - POS-EMB ATTENTION (pos_scores) // attn_weight
          P = Output[:, 256:272]   [116, 16] -> 4 x [116, 4] 
          pos_emb                  [4, 4, 231]    (4 heads x 4 dims/head x 231 positions)
          pos_score = P_head @ pos_emb_head → [4, 116, 116]  (lấy 116 cột đầu)
        
        Memory Hierarchy:
        1. Global Memory (DRAM): global_input, global_weight, global_bias
        2. Local Memory (SRAM): local_input, local_weight, local_output
        3. PE Registers: input_reg, weight_reg
        4. Compute: run_compute_mac()
        """
        self.T, self.N, self.M = T, N, M 
        self.tile_T = tile_T
        self.tile_N = N
        self.tile_M = tile_M // 2  # 2 blocks parallel

        # ========== STAGE 1: LINEAR LAYER ==========
        # Global Memory
        self.global_input = np.random.rand(T, N)       # [116, 192]
        self.global_weight = np.random.rand(N, M)      # [192, 272]
        self.global_bias = np.random.rand(M)           # [272]
        # self.global_output = np.zeros((T, M))          # [116, 272]
        
        # Local Memory (SRAM)
        self.local_input    = np.zeros((self.tile_T, self.tile_N))        # [58, 192]
        self.local_weightA  = np.zeros((self.tile_N, self.tile_M))        # [192, 8]
        self.local_weightB  = np.zeros((self.tile_N, self.tile_M))        # [192, 8]
        self.local_outputA_Q_cur  = np.zeros((self.T, 32))        #Qx[116, 32]
        self.local_outputA_Q_pre  = np.zeros((self.T, 32))        #Qx[116, 32]
        self.local_outputB_K_cur  = np.zeros((self.T, 32))        #Kx[116, 32]
        self.local_outputB_K_pre  = np.zeros((self.T, 32))        #Kx[116, 32]

        # ========== STAGE 2: ATTENTION LAYER for attn_scores ==========
        self.global_attn_score = np.zeros((4, T, T))  # [4, 116, 116]
        self.local_attn_score = np.zeros((58, 116)) 
        
        # ========== STAGE 3: ATTENTION LAYER for pos_scores ==========
        self.global_P = None  # P[116, 16] -> P0[:,0:4], P1[:,4:8], P2[:,8:12], P3[:,12:16]
        self.global_pos_emb = np.zeros((4, 4, 231))  # pos_emb[4, 4, 231] -> pos_emb0[:,:], pos_emb1[:,:], pos_emb2[:,:], pos_emb3[:,:]
        self.global_pos_score = np.zeros((4, T, T))  # global_pos_score [4, 116, 116]
        
        self.local_P = np.zeros((58, 4))
        
        self.local_pos_emb = np.zeros((4, 173))  # chỉ cần 173 cột cho mỗi half
        self.local_pos_score = np.zeros((58, 116))  # pos_score tile [rows, :58] (58, 58)

    # ========== STAGE 1: LINEAR LAYER ==========
    def initialize_all_bias(self):
        """Load bias vào PE registers"""
        self.block_A.load_entire_bias_to_core(self.global_bias)
        self.block_B.load_entire_bias_to_core(self.global_bias)

    def load_input_global_to_local(self, t_start):
        """Load Input từ Global DRAM → Local SRAM"""
        t_end = min(t_start + self.tile_T, self.T)
        self.local_input[:t_end-t_start, :] = self.global_input[t_start:t_end, :].copy()
        return t_end - t_start

    def load_weight_global_to_local(self, m_offset, head, phase=1):
        """
        Load Weight từ Global DRAM → Local SRAM
        
        Phase 1: 2 blocks parallel
          - Block A: Weight[:, m_offset : m_offset + tile_M]
          - Block B: Weight[:, m_offset + 128 : m_offset + 128 + tile_M]
        
        Phase 2: Xử lý phần cuối
          - Block A: Weight[:, 256:264]
          - Block B: Weight[:, 264:272]
        """
        if phase == 1:
            self.local_weightA = self.global_weight[:, m_offset + head*32 : m_offset + head*32 + self.tile_M].copy()
            self.local_weightB = self.global_weight[:, m_offset + 128 + head*32 : m_offset + 128 + head*32 + self.tile_M].copy()
        elif phase == 2:
            self.local_weightA = self.global_weight[:, 256:264].copy()
            self.local_weightB = self.global_weight[:, 264:272].copy()
    # ========== STAGE 2: ATTENTION LAYER for attn_scores ==========
    def read_attn_score_local_to_global(self, head_idx, t_start, actual_range_t):
        """Write local_attn_output tile từ Local SRAM → Global DRAM (1 lần/tile)
        
        Args:
          - head_idx      : chỉ số head (0-3)
          - t_start       : hàng bắt đầu của tile trong global_attn_score
          - actual_range_t: số hàng thực tế trong tile
        """
        self.global_attn_score[head_idx, t_start:t_start+actual_range_t, :] = self.local_attn_score[:actual_range_t, :]

    # ========== STAGE 3: POS-EMB ATTENTION ==========
    def load_P_global_to_local(self, head_idx, t_start, p_start, p_end):
        """Load P_head tile từ Global DRAM → Local SRAM (58, 4) 

        Args:
          - head_idx : chỉ số head (0-3)
          - t_start: hàng bắt đầu của head
          - p_start: cột bắt đầu của head trong global_P
          - p_end  : cột kết thúc của head trong global_P
          - block  : 'A' → [rows,0:58], 'B' → [rows,58:116]
        
        Returns:
          - actual_rows: 58
        """
        self.local_P[:, :] = self.global_P[t_start:t_start+58, p_start:p_start+4].copy()
        return 58
    def load_attn_score_global_to_local(self, head_idx, t_start):
        """Load attn_score tile từ Global DRAM → Local SRAM (58, 116) để tính toán attn_weight ngay khi tính xong pos_emb"""
        self.local_attn_score[:, :] = self.global_attn_score[head_idx, t_start:t_start+58, :].copy()

    def load_pos_emb_global_to_local(self, head_idx, t_start):
        """Load 173 cột pos_emb cần thiết cho half từ Global → Local SRAM [4, 173]
        
        Half 0 (t_start=0) : load global cols 58..230 → local[0:173]
        Half 1 (t_start=58): load global cols  0..172 → local[0:173]

        Local index trong compute_pos_score_row_batch:
          Block A: 115 - p_row_idx   (giống nhau cho cả 2 halves)
          Block B: 57 - p_row_idx
        """
        if t_start == 0:
            self.local_pos_emb[:, :] = self.global_pos_emb[head_idx, :, 58:231].copy()  # 173 cột
        else:  # t_start == 58
            self.local_pos_emb[:, :] = self.global_pos_emb[head_idx, :,  0:173].copy()  # 173 cột

    def read_pos_score_local_to_global(self, head_idx, t_start):
        """Write pos_score tile từ Local SRAM → Global DRAM

        Args:
          - head_idx : chỉ số head (0-3)
          - t_start  : hàng bắt đầu của tile (0 hoặc 58)
                       → ghi local_pos_score[0:58, :] vào global_pos_score[head_idx, t_start:t_start+58, :]
        """
        self.global_pos_score[head_idx, t_start:t_start+58, :] = self.local_pos_score

    def compute_pos_score_row_batch(self, p_row_idx, col_offset):
        """Tính pos_score cho 1 hàng P nhân với 116 cột pos_emb,
        sử dụng cả 2 blocks song song.

        pos_score[p_row, col] = P_row[4] · pos_emb_col[4]  (dot product)

        Args:
          - p_row_idx : chỉ số hàng P trong local_P (0..57)
          - col_offset: cột bắt đầu trong local_pos_emb (0..230)
          - num_cols  : tổng số cột cần tính (mặc định = 116 = 58 + 58)

        Returns:
          - scores: [num_cols,]  (scores_A concat scores_B)

        Hardware parallelism — 232 PEs / 4 dims = 58 dot products/cycle:
        ─────────────────────────────────────────────────────────────────
          input_reg [232]   = P_row x 58 [← dùng chung cho cả 2 block]
                            = [pi0,pi1,pi2,pi3] x 58

          weight_reg_A[232] = flatten(pos_emb_cols[col_offset   : col_offset+58 ].T)
                            = [ pos_emb_r0_c0 - p_row_idx, pos_emb_r1_c0 - p_row_idx, pos_emb_r2_c0 - p_row_idx, pos_emb_r3_c0 - p_row_idx,
                                pos_emb_r0_c1 - p_row_idx, pos_emb_r1_c1 - p_row_idx, pos_emb_r2_c1 - p_row_idx, pos_emb_r3_c1 - p_row_idx,
                                pos_emb_r0_c2 - p_row_idx, pos_emb_r1_c2 - p_row_idx, pos_emb_r2_c2 - p_row_idx, pos_emb_r3_c2 - p_row_idx,, ...,
                                pos_emb_r0_c57 - p_row_idx, pos_emb_r1_c57 - p_row_idx, pos_emb_r2_c57 - p_row_idx, pos_emb_r3_c57 - p_row_idx]

          weight_reg_B[232] = flatten(pos_emb_cols[col_offset   : col_offset+58 ].T)
                            = [ pos_emb_r0_c58 - p_row_idx, pos_emb_r1_c58 - p_row_idx, pos_emb_r2_c58 - p_row_idx, pos_emb_r3_c58 - p_row_idx,
                                pos_emb_r0_c59 - p_row_idx, pos_emb_r1_c59 - p_row_idx, pos_emb_r2_c59 - p_row_idx, pos_emb_r3_c59 - p_row_idx,
                                pos_emb_r0_c60 - p_row_idx, pos_emb_r1_c60 - p_row_idx, pos_emb_r2_c60 - p_row_idx, pos_emb_r3_c60 - p_row_idx,, ...,
                                pos_emb_r0_c116 - p_row_idx, pos_emb_r1_c116 - p_row_idx, pos_emb_r2_c116 - p_row_idx, pos_emb_r3_c116 - p_row_idx]6]

          products[232]  = input_reg * weight_reg  (element-wise)
          → reshape [PARALLEL, p_row_size] → sum(axis=1) → [PARALLEL] scores
        """
        p_row    = self.local_P[p_row_idx, :]       # [4,]
        p_row_size     = p_row.shape[0]                    # 4
        PARALLEL = self.block_A.num_pe // p_row_size       # 232 // 4 = 58
        scores   = np.zeros(2 * PARALLEL)   # [116,]

        # --- input_reg: giống nhau cho cả 2 block ---
        p_padded = np.zeros(self.block_A.num_pe)
        p_padded[:PARALLEL * p_row_size] = np.tile(p_row, PARALLEL)  # [58*4,]

        # ===== BLOCK_A: local cols [57-p_row_idx : 57-p_row_idx + PARALLEL] =====
        # local_pos_emb [4, 173]: half0 load global[58:231], half1 load global[0:173]
        # Mọi p_row_idx: Block A bắt đầu tại local col = 57 - p_row_idx
        local_A = 57 - p_row_idx
        cols_A  = self.local_pos_emb[:, local_A : local_A + PARALLEL]  # [4, 58]
        e_A      = np.zeros(self.block_A.num_pe)
        e_A[:PARALLEL * p_row_size] = cols_A.T.flatten()          # [58*4,]

        self.block_A.load_in_local_to_Compute(p_padded)
        self.block_A.load_weight_to_Compute(e_A)

        prod_A = self.block_A.input_reg[:PARALLEL * p_row_size] * self.block_A.weight_reg[:PARALLEL * p_row_size]  # [58*4]
        scores[:PARALLEL] = prod_A.reshape(PARALLEL, p_row_size).sum(axis=1)                                       # [58]

        # ===== BLOCK_B: local cols [115-p_row_idx : 115-p_row_idx + PARALLEL] (parallel) =====
        local_B = 115 - p_row_idx
        cols_B  = self.local_pos_emb[:, local_B : local_B + PARALLEL]  # [4, 58]
        e_B      = np.zeros(self.block_B.num_pe)
        e_B[:PARALLEL * p_row_size] = cols_B.T.flatten()          # [58*4,]

        self.block_B.load_in_local_to_Compute(p_padded)
        self.block_B.load_weight_to_Compute(e_B)

        prod_B = self.block_B.input_reg[:PARALLEL * p_row_size] * self.block_B.weight_reg[:PARALLEL * p_row_size]  # [58*4]
        scores[PARALLEL:] = prod_B.reshape(PARALLEL, p_row_size).sum(axis=1)                                       # [58]

        return scores  # [116,]

    def execute_process(self, N, M):
        """
        ========== STAGE 1: LINEAR LAYER EXECUTION // ATTENTION LAYER for attn_scores ==========
        
        Pipeline Strategy (Linear & Matmul 1):
        - Đầu tiên, tính toán Q_0 và K_0 từ phép Linear.
        - Tiếp theo, tính pipeline song song:
          + Trong lúc tính Q_1 và K_1, thực hiện tính toán Matmul 1 cho attn_score_0.
          + Trong lúc tính Q_2 và K_2, thực hiện tính toán Matmul 1 cho attn_score_1.
          + Trong lúc tính Q_3 và K_3, thực hiện tính toán Matmul 1 cho attn_score_2.
        - Cuối cùng, tính nốt attn_score_3.
        
        Tiling Strategy:
        - Outer loop: T (rows) - tile_T = 58 rows
        - Middle loop: M (cols) - tile_M = 8 cols x 2 blocks

        Input: Linear output [116, 272]
        Extract:
          - Q = Output[:, :128] <=> shape [116, 128] → split into 4 heads [116, 32] each
          - K = Output[:, 128:256] <=> shape [116, 128] → split into 4 heads [116, 32] each

        Compute: For each head: Attention[head] = Q_head @ K_head.T  → [4, 116, 116]
        
        Memory Flow:
        1. Tính toán Q_0, K_0.
        2. Tính Q_head_i, K_head_i đồng thời load Q_head_{i-1}, K_head_{i-1} để tính toán.
        3. Compute MAC tính attention score cho head_{i-1}.
        4. Store kết quả attn_score_{i-1} xuống Local SRAM rồi tới Global SRAM.
        """
        # Compute Blocks (PE)
        self.block_A = Compute_block(num_pe=256, M=M, is_linear=True)
        self.block_B = Compute_block(num_pe=256, M=M, is_linear=True)
        self.initialize_all_bias()
        
        # ===== OUTER LOOP: Input rows =====
        for head in range(4):
            if head == 0:
                for t_start in range(0, self.T, self.tile_T):
                    actual_range_t = self.load_input_global_to_local(t_start)
                    for m_offset in range(0, 32, self.tile_M):
                        self.load_weight_global_to_local(m_offset, head, phase=1)
                        for t_idx in range(actual_range_t):
                            row = self.local_input[t_idx, :]
                            pad_zeros = np.zeros(64)
                            input_reg = np.concatenate([row, pad_zeros])

                            self.block_A.load_in_local_to_Compute(input_reg)
                            self.block_B.load_in_local_to_Compute(input_reg)

                            for i in range(self.tile_M):
                                self.block_A.select_bias_for_column(m_offset + head*32 + i)
                                self.block_A.load_weight_to_Compute(np.concatenate([self.local_weightA[:, i], pad_zeros]))
                                self.block_A.output_reg = np.sum(self.block_A.input_reg[:192] * self.block_A.weight_reg[:192]) + self.block_A.current_bias
                                self.local_outputA_Q_cur[t_idx + t_start, m_offset + i] = self.block_A.get_output_to_local()

                                self.block_B.select_bias_for_column(m_offset + head*32 + 128 + i)
                                self.block_B.load_weight_to_Compute(np.concatenate([self.local_weightB[:, i], pad_zeros]))
                                self.block_B.output_reg = np.sum(self.block_B.input_reg[:192] * self.block_B.weight_reg[:192]) + self.block_B.current_bias
                                self.local_outputB_K_cur[t_idx + t_start, m_offset + i] = self.block_B.get_output_to_local()
                
                self.local_outputA_Q_pre[:, :] = self.local_outputA_Q_cur.copy()
                self.local_outputB_K_pre[:, :] = self.local_outputB_K_cur.copy()

            else:
                for t_start in range(0, self.T, self.tile_T):
                    actual_range_t = self.load_input_global_to_local(t_start)
                    self.local_attn_score.fill(0)
                    for m_offset in range(0, 32, self.tile_M):
                        self.load_weight_global_to_local(m_offset, head, phase=1)
                        for t_idx in range(actual_range_t):
                            row = self.local_input[t_idx, :]
                            row_idx = t_idx + t_start
                            # 2 vòng lặp (t_start đi qua các Tiling blocks) và (t_idx đi qua các hàng trong một Tiling block)
                            # Kết hợp lại giúp row_idx duyệt qua toàn bộ biến đếm từ 0 cho tới 115.
                            # Điều này đảm bảo trích xuất chính xác và không thiếu bất cứ hàng nào trong số 116 hàng của ma trận Q.
                            q_pre_row = self.local_outputA_Q_pre[row_idx, :]

                            for i in range(self.tile_M):
                                iter_idx = m_offset + i
                                k_A1, k_A2 = iter_idx * 4, iter_idx * 4 + 1
                                k_B1, k_B2 = iter_idx * 4 + 2, iter_idx * 4 + 3

                                # Hệ thống lặp 32 lần x 4 rows = 128 rows, nhưng ma trận thực tế chỉ có self.T = 116 rows.
                                # Do đó nếu k_ix >= 116, sẽ padding bằng np.zeros(32) để tránh lỗi IndexError 
                                # và các phép nhân thừa này sẽ ra 0, không ảnh hưởng kết quả cộng dồn.
                                K1 = self.local_outputB_K_pre[k_A1, :] if k_A1 < self.T else np.zeros(32)
                                K2 = self.local_outputB_K_pre[k_A2, :] if k_A2 < self.T else np.zeros(32)
                                K3 = self.local_outputB_K_pre[k_B1, :] if k_B1 < self.T else np.zeros(32)
                                K4 = self.local_outputB_K_pre[k_B2, :] if k_B2 < self.T else np.zeros(32)

                                input_reg_AB = np.concatenate([row, q_pre_row, q_pre_row])
                                
                                self.block_A.select_bias_for_column(m_offset + head*32 + i)
                                self.block_A.load_in_local_to_Compute(input_reg_AB)
                                self.block_A.load_weight_to_Compute(np.concatenate([self.local_weightA[:, i], K1, K2]))

                                prod_A = self.block_A.input_reg * self.block_A.weight_reg

                                self.block_A.output_reg = np.sum(prod_A[:192]) + self.block_A.current_bias
                                
                                self.local_outputA_Q_cur[row_idx, m_offset + i] = self.block_A.get_output_to_local()

                                if k_A1 < self.T: self.local_attn_score[t_idx, k_A1] += np.sum(prod_A[192:224])
                                if k_A2 < self.T: self.local_attn_score[t_idx, k_A2] += np.sum(prod_A[224:256])

                                self.block_B.select_bias_for_column(m_offset + head*32 + 128 + i)
                                self.block_B.load_in_local_to_Compute(input_reg_AB)
                                self.block_B.load_weight_to_Compute(np.concatenate([self.local_weightB[:, i], K3, K4]))

                                prod_B = self.block_B.input_reg * self.block_B.weight_reg
                                
                                self.block_B.output_reg = np.sum(prod_B[:192]) + self.block_B.current_bias
                                
                                self.local_outputB_K_cur[row_idx, m_offset + i] = self.block_B.get_output_to_local()
                                
                                if k_B1 < self.T: self.local_attn_score[t_idx, k_B1] += np.sum(prod_B[192:224])
                                if k_B2 < self.T: self.local_attn_score[t_idx, k_B2] += np.sum(prod_B[224:256])

                    self.read_attn_score_local_to_global(head-1, t_start, actual_range_t)

                self.local_outputA_Q_pre[:, :] = self.local_outputA_Q_cur.copy()
                self.local_outputB_K_pre[:, :] = self.local_outputB_K_cur.copy()

        # Tính nốt attn_score_pre cho head = 3 và ===== PHASE 2: Xử lý phần còn lại (M = 256:272) =====
        for t_start in range(0, self.T, self.tile_T):
            actual_range_t = self.load_input_global_to_local(t_start)
            self.load_weight_global_to_local(256, None, phase=2)
            self.local_outputA_Q_cur.fill(0)
            self.local_outputB_K_cur.fill(0)
            self.local_attn_score.fill(0)
            
            for iter_offset in range(0, 32, self.tile_M):
                for t_idx in range(actual_range_t):
                    row = self.local_input[t_idx, :]
                    row_idx = t_idx + t_start
                    q_pre_row = self.local_outputA_Q_pre[row_idx, :]

                    for i in range(self.tile_M):
                        iter_idx = iter_offset + i
                        
                        base_idx = iter_idx * 10
                        k_A1, k_A2, k_A3, k_A4, k_A5 = base_idx, base_idx+1, base_idx+2, base_idx+3, base_idx+4
                        k_B1, k_B2, k_B3, k_B4, k_B5 = base_idx+5, base_idx+6, base_idx+7, base_idx+8, base_idx+9

                        # Trích xuất 5 vector K cho Block A
                        K_A1 = self.local_outputB_K_pre[k_A1, :] if k_A1 < self.T else np.zeros(32)
                        K_A2 = self.local_outputB_K_pre[k_A2, :] if k_A2 < self.T else np.zeros(32)
                        K_A3 = self.local_outputB_K_pre[k_A3, :] if k_A3 < self.T else np.zeros(32)
                        K_A4 = self.local_outputB_K_pre[k_A4, :] if k_A4 < self.T else np.zeros(32)
                        K_A5 = self.local_outputB_K_pre[k_A5, :] if k_A5 < self.T else np.zeros(32)
                        
                        # Trích xuất 5 vector K cho Block B
                        K_B1 = self.local_outputB_K_pre[k_B1, :] if k_B1 < self.T else np.zeros(32)
                        K_B2 = self.local_outputB_K_pre[k_B2, :] if k_B2 < self.T else np.zeros(32)
                        K_B3 = self.local_outputB_K_pre[k_B3, :] if k_B3 < self.T else np.zeros(32)
                        K_B4 = self.local_outputB_K_pre[k_B4, :] if k_B4 < self.T else np.zeros(32)
                        K_B5 = self.local_outputB_K_pre[k_B5, :] if k_B5 < self.T else np.zeros(32)

                        input_attn3 = np.concatenate([q_pre_row] * 5)
                        
                        # ================= Block A =================
                        if iter_offset == 0:
                            # Tính 96 chiều đầu tiên của Linear (0->95)
                            input_phase2 = row[:96]
                            input_reg_AB = np.concatenate([input_phase2, input_attn3])
                            self.block_A.load_in_local_to_Compute(input_reg_AB)

                            weight_phase2_A = self.local_weightA[:96, i]
                            weight_attn3_A = np.concatenate([K_A1, K_A2, K_A3, K_A4, K_A5])
                            self.block_A.load_weight_to_Compute(np.concatenate([weight_phase2_A, weight_attn3_A]))

                            prod_A = self.block_A.input_reg * self.block_A.weight_reg
                            
                            # Chưa cộng bias, chỉ lưu phần nửa đầu của phép nhân
                            self.block_A.output_reg = np.sum(prod_A[:96])
                            self.local_outputA_Q_cur[t_idx, i] = self.block_A.get_output_to_local()

                        elif iter_offset == 8:
                            # Tính 96 chiều nốt lại của Linear (96->191)
                            input_phase2 = row[96:192]
                            input_reg_AB = np.concatenate([input_phase2, input_attn3])
                            self.block_A.load_in_local_to_Compute(input_reg_AB)

                            self.block_A.select_bias_for_column(256 + i)
                            weight_phase2_A = self.local_weightA[96:192, i]
                            weight_attn3_A = np.concatenate([K_A1, K_A2, K_A3, K_A4, K_A5])
                            self.block_A.load_weight_to_Compute(np.concatenate([weight_phase2_A, weight_attn3_A]))

                            prod_A = self.block_A.input_reg * self.block_A.weight_reg
                            
                            # Lấy kết quả cũ cộng dồn với nửa sau, và cộng thêm Bias hoàn tất phép tính
                            self.block_A.output_reg = self.local_outputA_Q_cur[t_idx, i] + np.sum(prod_A[:96]) + self.block_A.current_bias
                            self.local_outputA_Q_cur[t_idx, i] = self.block_A.get_output_to_local()

                        else:
                            # Linear Phase 2 đã xong hoàn toàn, bỏ trống 96 PE đầu
                            input_phase2 = np.zeros(96)
                            input_reg_AB = np.concatenate([input_phase2, input_attn3])
                            self.block_A.load_in_local_to_Compute(input_reg_AB)

                            weight_phase2_A = np.zeros(96)
                            weight_attn3_A = np.concatenate([K_A1, K_A2, K_A3, K_A4, K_A5])
                            self.block_A.load_weight_to_Compute(np.concatenate([weight_phase2_A, weight_attn3_A]))
                            
                            prod_A = self.block_A.input_reg * self.block_A.weight_reg

                        # Tính 5 điểm Attention cho Block A
                        if k_A1 < self.T: self.local_attn_score[t_idx, k_A1] += np.sum(prod_A[96:128])
                        if k_A2 < self.T: self.local_attn_score[t_idx, k_A2] += np.sum(prod_A[128:160])
                        if k_A3 < self.T: self.local_attn_score[t_idx, k_A3] += np.sum(prod_A[160:192])
                        if k_A4 < self.T: self.local_attn_score[t_idx, k_A4] += np.sum(prod_A[192:224])
                        if k_A5 < self.T: self.local_attn_score[t_idx, k_A5] += np.sum(prod_A[224:256])

                        # ================= Block B =================
                        if iter_offset == 0:
                            input_phase2 = row[:96]
                            input_reg_AB = np.concatenate([input_phase2, input_attn3])
                            self.block_B.load_in_local_to_Compute(input_reg_AB)

                            weight_phase2_B = self.local_weightB[:96, i]
                            weight_attn3_B = np.concatenate([K_B1, K_B2, K_B3, K_B4, K_B5])
                            self.block_B.load_weight_to_Compute(np.concatenate([weight_phase2_B, weight_attn3_B]))

                            prod_B = self.block_B.input_reg * self.block_B.weight_reg
                            
                            self.block_B.output_reg = np.sum(prod_B[:96])
                            self.local_outputB_K_cur[t_idx, i] = self.block_B.get_output_to_local()

                        elif iter_offset == 8:
                            input_phase2 = row[96:192]
                            input_reg_AB = np.concatenate([input_phase2, input_attn3])
                            self.block_B.load_in_local_to_Compute(input_reg_AB)

                            self.block_B.select_bias_for_column(264 + i)
                            weight_phase2_B = self.local_weightB[96:192, i]
                            weight_attn3_B = np.concatenate([K_B1, K_B2, K_B3, K_B4, K_B5])
                            self.block_B.load_weight_to_Compute(np.concatenate([weight_phase2_B, weight_attn3_B]))

                            prod_B = self.block_B.input_reg * self.block_B.weight_reg
                            
                            self.block_B.output_reg = self.local_outputB_K_cur[t_idx, i] + np.sum(prod_B[:96]) + self.block_B.current_bias
                            self.local_outputB_K_cur[t_idx, i] = self.block_B.get_output_to_local()

                        else:
                            input_phase2 = np.zeros(96)
                            input_reg_AB = np.concatenate([input_phase2, input_attn3])
                            self.block_B.load_in_local_to_Compute(input_reg_AB)

                            weight_phase2_B = np.zeros(96)
                            weight_attn3_B = np.concatenate([K_B1, K_B2, K_B3, K_B4, K_B5])
                            self.block_B.load_weight_to_Compute(np.concatenate([weight_phase2_B, weight_attn3_B]))
                            
                            prod_B = self.block_B.input_reg * self.block_B.weight_reg
                            
                        # Tính 5 điểm Attention cho Block B
                        if k_B1 < self.T: self.local_attn_score[t_idx, k_B1] += np.sum(prod_B[96:128])
                        if k_B2 < self.T: self.local_attn_score[t_idx, k_B2] += np.sum(prod_B[128:160])
                        if k_B3 < self.T: self.local_attn_score[t_idx, k_B3] += np.sum(prod_B[160:192])
                        if k_B4 < self.T: self.local_attn_score[t_idx, k_B4] += np.sum(prod_B[192:224])
                        if k_B5 < self.T: self.local_attn_score[t_idx, k_B5] += np.sum(prod_B[224:256])

            self.read_attn_score_local_to_global(3, t_start, actual_range_t)

            # Gán giá trị kết quả Phase 2 vào biến global_P -> P[116, 16] cho Stage 3
            if self.global_P is None:
                self.global_P = np.zeros((self.T, 16))
            self.global_P[t_start:t_start+actual_range_t, 0:8] = self.local_outputA_Q_cur[:actual_range_t, :8]
            self.global_P[t_start:t_start+actual_range_t, 8:16] = self.local_outputB_K_cur[:actual_range_t, :8]

        print("\nLinear layer computation // Multi-head Attention computation completed!")

        """
        ========== STAGE 2: POS-EMB ATTENTION EXECUTION ==========

        Input:
          - P = global_output[:, 256:272]    → [116, 16]  (4 heads x 4 dims/head)
          - pos_emb                           → [4 heads, 4 dims, 231 positions]

        Compute: For each head h:
          P_head       = P[:, h*4 : (h+1)*4]        → [116, 4]
          pos_emb_head = global_pos_emb[h]           → [4, 231]
          pos_score    = P_head @ pos_emb_head       → [116, 231]
          → stride 116 → global_pos_score[h] → [116, 116]

        Tiling Strategy:
          - P rows chia 2 halves: [0:58] (block_A) và [58:116] (block_B)
          - pos_emb cols chia 2 halves: cols[0:58] (block_A) và cols[58:116] (block_B)
          - Hai blocks tính song parallel cho một P_row

        Memory Flow:
          1. Extract P từ Linear output[:, 256:272]
          2. Khởi tạo pos_emb (random / load từ model)
          3. For each head (0-3):
             a. Load pos_emb_head → local_pos_emb  [4, 231]
             b. Block_A: load P[0:58, head_dims] → local_P, tính cols [0:58]
                Block_B: load P[58:116, head_dims] → local_P, tính cols [58:116] (parallel)
             c. Ghép → global_pos_score[head, :, :]
        """
        print("\n" + "="*80)
        print("STAGE 3: POS-EMB ATTENTION EXECUTION (MULTI-HEAD)")
        print("="*80)

        # Reinit blocks (no bias needed)
        self.block_A = Compute_block(num_pe=232, M=None, is_linear=False)
        self.block_B = Compute_block(num_pe=232, M=None, is_linear=False)

        # Extract P từ Linear output (output[:, 256:272])
        # P đã được load thẳng vào self.global_P trong Phase 2 loop.
        # self.global_P = self.global_output[:, 256:272].copy()    # [116, 16]

        # Khởi tạo pos_emb ngẫu nhiên (trong thực tế load từ model)
        self.global_pos_emb  = np.random.rand(4, 4, 231)          # [4 heads, 4 dims, 231 positions]
        self.global_pos_score = np.zeros((4, self.T, self.T))     # [4, 116, 116]

        print(f"P shape        : {self.global_P.shape}")
        print(f"pos_emb shape  : {self.global_pos_emb.shape}")
        print(f"pos_score shape: {self.global_pos_score.shape}")
        print(f"Number of heads: 4, Dims per head: 4, Using first {self.T} of 231 pos positions")

        # ===== MULTI-HEAD PROCESSING =====
        for head_idx in range(4):
            print(f"\n--- Processing Head {head_idx} ---")

            p_start, p_end = head_idx * 4, (head_idx + 1) * 4

            # STEP 1: Load pos_emb của head này vào local

            # STEP 2: Load P_head (58 hàng một lần) và tính pos scores
            # Với mỗi hàng P: block_A tính cols[0:58], block_B tính cols[58:116] (parallel)
            # Lặp qua 2 halves: t_start = 0 (rows 0..57) và t_start = 58 (rows 58..115)
            for t_start in (0, 58):
                self.load_pos_emb_global_to_local(head_idx, t_start)   # local_pos_emb: [4, 173]
                self.load_attn_score_global_to_local(head_idx, t_start)        # local_attn_score [58, 116]
                rows = self.load_P_global_to_local(head_idx, t_start, p_start, p_end)   # local_P [58, 4] 
                for p_row_idx in range(rows):  # 0..57
                    scores = self.compute_pos_score_row_batch(
                        p_row_idx, col_offset=t_start + p_row_idx
                    )  # [116,]
                    scores += self.local_attn_score[p_row_idx, :]
                    self.local_pos_score[p_row_idx, :] = F.softmax(torch.from_numpy(scores), dim=0).numpy()

                self.read_pos_score_local_to_global(head_idx, t_start) # ghi local_pos_score vào global_pos_score, hay chính là global attn_weight

        print("\nPos-emb Attention computation completed!")


def verify_stages(core_instance):
    """
    Xác minh đầu ra cuối cùng (attn_weights) bằng cách gọi golden model
    RelPositionMultiheadAttentionWeights từ gloden_func.py.

    Chiến lược inject:
      - in_proj.weight  ← global_weight.T  [272, 192]
      - in_proj.bias    ← global_bias       [272]
      - linear_pos.weight ← Identity[16,16]  (bypass linear_pos)
      - pos_emb input   ← reshape global_pos_emb [4,4,231]
                          thành [1, 231, 16] đúng thứ tự trục
    """
    from gloden_func import RelPositionMultiheadAttentionWeights

    print("\n" + "="*80)
    print("VERIFICATION via RelPositionMultiheadAttentionWeights (golden model)")
    print("="*80)

    # ── Khởi tạo golden model ──────────────────────────────────────────────
    # pos_dim=16 = num_heads(4) * pos_head_dim(4)  →  linear_pos sẽ là identity
    model = RelPositionMultiheadAttentionWeights(
        embed_dim      = 192,
        pos_dim        = 16,
        num_heads      = 4,
        query_head_dim = 32,
        pos_head_dim   = 4,
    )
    model.eval()
    model.double()   # dùng float64 để khớp precision với numpy simulation

    with torch.no_grad():
        # ── Inject in_proj weight & bias từ core ──────────────────────────
        # core.global_weight [192, 272]  →  in_proj.weight cần [272, 192]
        model.in_proj.weight.copy_(torch.from_numpy(core_instance.global_weight).T)
        model.in_proj.bias.copy_(torch.from_numpy(core_instance.global_bias))

        # ── Bypass linear_pos: weight = I[16,16] ──────────────────────────
        # Khi đó: linear_pos(pos_emb_raw) = pos_emb_raw  (pass-through)
        model.linear_pos.weight.copy_(torch.eye(16, dtype=torch.float64))

        # ── Chuẩn bị x: (T, 1, 192) ───────────────────────────────────────
        x = torch.from_numpy(core_instance.global_input).unsqueeze(1)  # [116, 1, 192]

        # ── Chuẩn bị pos_emb: (1, 231, 16) ───────────────────────────────
        # global_pos_emb shape: [4 heads, 4 dims, 231 positions]
        # Cần pos_emb_raw[0, pos, head*4+dim] = global_pos_emb[head, dim, pos]
        # Bước: transpose(2,0,1) → [231, 4, 4]  rồi reshape → [1, 231, 16]
        pos_emb_np = np.transpose(core_instance.global_pos_emb, (2, 0, 1))  # [231, 4, 4]
        pos_emb_np = pos_emb_np.reshape(1, 231, 16)                          # [1, 231, 16]
        pos_emb_t  = torch.from_numpy(pos_emb_np)                            # [1, 231, 16]

        # ── Forward ───────────────────────────────────────────────────────
        golden_output = model(x, pos_emb_t)   # [4, 1, 116, 116]


    # ── So sánh ───────────────────────────────────────────────────────────
    golden_np  = golden_output.squeeze(1).numpy()   # [4, 116, 116]
    simulated  = core_instance.global_pos_score      # [4, 116, 116]

    max_err    = np.max(np.abs(golden_np - simulated))
    is_correct = np.allclose(golden_np, simulated, atol=1e-10)

    print(f"\n  Input  x shape        : {x.shape}")
    print(f"  pos_emb shape         : {pos_emb_t.shape}")
    print(f"  Golden output shape   : {golden_np.shape}")
    print(f"  Simulated output shape: {simulated.shape}")
    print(f"  Max error             : {max_err:.2e}")
    print(f"  Status: {'✅ PASSED' if is_correct else '❌ FAILED'}")

    return is_correct



if __name__ == "__main__":
    print("\n" + "="*80)
    print("TRI-STAGE ACCELERATOR: LINEAR + ATTENTION + POS-EMB")
    print("="*80)
    
    # Create Core
    core = Core(T=116, N=192, M=272, tile_T=29, tile_M=16)
    
    print(f"\nConfiguration:")
    print(f"  Stage 1 (Linear)  : Input[116,192] @ Weight[192,272] + Bias[272] → [116,272]")
    print(f"  Stage 2 (Attention): Q[116,128] @ K[116,128].T → [4, 116,116]")
    print(f"  Stage 3 (Pos-emb) : P[116,16] @ pos_emb[4,4,231] → [4, 116,116]")
    print(f"\nMemory Hierarchy:")
    print(f"  • Global Memory (DRAM): Stores all inputs, weights, outputs")
    print(f"  • Local Memory (SRAM): Tiling buffers")
    print(f"  • PE Registers: input_reg, weight_reg")
    print(f"  • Compute: MAC operations (multiply-accumulate)")
    
    # ===== EXECUTION =====
    start_time = time.time()
    core.execute_process(N=232, M=272)
    time_total = time.time() - start_time
    print(f"\nTotal execution time: {time_total:.6f} seconds")
    
    # VERIFICATION
    all_passed = verify_stages(core)
    
    # SUMMARY
    print("\n" + "="*80)
    print("EXECUTION SUMMARY")
    print("="*80)
    print(f"Total execution time: {time_total:.6f}s")
    print(f"\nOverall: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    print("="*80)
