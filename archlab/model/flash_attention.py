import torch
import math

import triton
import triton.language as tl

@triton.autotune(
    configs=[
        # 1. 首选方案：Q=128, K=32, stages=4（计算密度最高，占用约 96KB，安全）
        triton.Config({'Q_TILE_SIZE': 128, 'K_TILE_SIZE': 32}, num_warps=4, num_stages=4),
        # 2. 稳健方案：Q=64, K=64, stages=3（占用约 88KB，非常安全，适合反向传播）
        triton.Config({'Q_TILE_SIZE': 64, 'K_TILE_SIZE': 64}, num_warps=4, num_stages=3),
        # 3. 保底方案：Q=64, K=32, stages=4（占用极小，约 72KB，防止极短序列出现边界效应）
        triton.Config({'Q_TILE_SIZE': 64, 'K_TILE_SIZE': 32}, num_warps=4, num_stages=4),
        # 4. 仅作为前向大块容错，但反向必须禁用此项（如果反向报错，直接注释掉反向kernel里的这一行）
        triton.Config({'Q_TILE_SIZE': 128, 'K_TILE_SIZE': 64}, num_warps=8, num_stages=2), 
    ],
    key=['N_QUERIES', 'N_KEYS', 'HEAD_DIM'], 
)
@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    query_tile_idx = tl.program_id(0)
    batch_idx = tl.program_id(1)

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_idx * stride_qb,
        shape=(N_QUERIES, D,),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_idx * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_idx * stride_kb,
        shape=(N_KEYS, D,),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_idx * stride_vb,
        shape=(N_KEYS, D,),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    # 输出 Q 指针
    O_block_ptr= tl.make_block_ptr(
        O_ptr + batch_idx * stride_ob,
        shape=(N_QUERIES, D,),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_idx * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    # 输出 logsumexp 指针
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_idx * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_idx * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    q_tile = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")

    # FP32
    o_tile = tl.zeros([Q_TILE_SIZE, D], dtype=tl.float32) # 输出块
    l_tile = tl.zeros([Q_TILE_SIZE], dtype=tl.float32) # 累加和
    m_tile = tl.full([Q_TILE_SIZE], -float('inf'), dtype=tl.float32) # 最大值

    q_start = query_tile_idx * Q_TILE_SIZE
    q_offset = tl.arange(0, Q_TILE_SIZE)[:, None] # [Q_TILE_SIZE, 1]
    k_offset = tl.arange(0, K_TILE_SIZE)[None, :] # [1, K_TILE_SIZE]

    for k_start in range(0, N_KEYS, K_TILE_SIZE):
        # 跳过当前 k_tile 不在 mask 可见范围
        if q_start + Q_TILE_SIZE > k_start:
            k_tile = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
            v_tile = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")
            s = tl.dot(q_tile, tl.trans(k_tile)) * scale # [Q_TILE_SIZE, K_TILE_SIZE]

            # 绝对位置索引，用于 causal mask
            q_abs = q_offset + q_start
            k_abs = k_offset + k_start

            # 分离非掩码与对角线
            if k_start + K_TILE_SIZE <= q_start:
                m_cur = tl.maximum(m_tile, tl.max(s, axis=1)) # [Q_TILE_SIZE]
                p = tl.exp(s - m_cur[:, None])
                m_shift = tl.exp(m_tile - m_cur)
            else:
                mask = k_abs <= q_abs
                s = tl.where(mask, s, -1e6)

                m_cur = tl.maximum(m_tile, tl.max(s, axis=1)) # [Q_TILE_SIZE]
                p = tl.exp(s - m_cur[:, None])
                m_shift = tl.exp(m_tile - m_cur)

            # 更新 row-wise
            l_tile = m_shift * l_tile + tl.sum(p, axis=1) # 修正当前累加和
            o_tile = tl.dot(p.to(V_ptr.type.element_ty), v_tile, acc=m_shift[:, None] * o_tile) # 修正当前输出
            m_tile = m_cur

        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    o_tile = o_tile / l_tile[:, None]
    l_tile = m_tile + tl.log(l_tile)

    tl.store(O_block_ptr, o_tile.to(Q_ptr.type.element_ty))
    tl.store(L_block_ptr, l_tile)

@triton.autotune(
    configs=[
        # 1. 首选方案：Q=128, K=32, stages=4（计算密度最高，占用约 96KB，安全）
        triton.Config({'Q_TILE_SIZE': 128, 'K_TILE_SIZE': 32}, num_warps=4, num_stages=4),
        # 2. 稳健方案：Q=64, K=64, stages=3（占用约 88KB，非常安全，适合反向传播）
        triton.Config({'Q_TILE_SIZE': 64, 'K_TILE_SIZE': 64}, num_warps=4, num_stages=3),
        # 3. 保底方案：Q=64, K=32, stages=4（占用极小，约 72KB，防止极短序列出现边界效应）
        triton.Config({'Q_TILE_SIZE': 64, 'K_TILE_SIZE': 32}, num_warps=4, num_stages=4),
    ],
    key=['N_QUERIES', 'N_KEYS', 'HEAD_DIM'], 
)
@triton.jit
def flash_bwd_dk_dv_kernel(
    Q_ptr, K_ptr, V_ptr,  # [bsz, dq, HEAD_DIM]
    dO_ptr,               # [bsz, dv, HEAD_DIM]
    L_ptr, D_ptr,         # [bsz, dq]
    dK_ptr, dV_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_dob, stride_doq, stride_dod,
    stride_lb, stride_lq,
    stride_db, stride_dq,
    stride_dkb, stride_dkk, stride_dkd,
    stride_dvb, stride_dvk, stride_dvd,
    N_QUERIES, N_KEYS,
    scale,
    HEAD_DIM: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    k_tile_idx = tl.program_id(0)
    batch_idx = tl.program_id(1)

    # 当前 K/V 的全局起点
    k_start = k_tile_idx * K_TILE_SIZE
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_idx * stride_kb,
        shape=(N_KEYS, HEAD_DIM),
        strides=(stride_kk, stride_kd),
        offsets=(k_start, 0),
        block_shape=(K_TILE_SIZE, HEAD_DIM),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_idx * stride_vb,
        shape=(N_KEYS, HEAD_DIM),
        strides=(stride_vk, stride_vd),
        offsets=(k_start, 0),
        block_shape=(K_TILE_SIZE, HEAD_DIM),
        order=(1, 0),
    )

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_idx * stride_qb,
        shape=(N_QUERIES, HEAD_DIM),
        strides=(stride_qq, stride_qd),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, HEAD_DIM),
        order=(1, 0),
    )

    dO_block_ptr = tl.make_block_ptr(
        dO_ptr + batch_idx * stride_dob,
        shape=(N_QUERIES, HEAD_DIM),
        strides=(stride_doq, stride_dod),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, HEAD_DIM),
        order=(1, 0),
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_idx * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(0,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )
    D_block_ptr = tl.make_block_ptr(
        D_ptr + batch_idx * stride_db,
        shape=(N_QUERIES,),
        strides=(stride_dq,),
        offsets=(0,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    # 加载 K、V 分块
    k_tile = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero") # [Bk, HEAD_DIM]
    v_tile = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")

    # 初始化 dK、dV 累加器
    dk_tile = tl.zeros([K_TILE_SIZE, HEAD_DIM], dtype=tl.float32)
    dv_tile = tl.zeros([K_TILE_SIZE, HEAD_DIM], dtype=tl.float32)

    k_abs = k_start + tl.arange(0, K_TILE_SIZE)[None, :]
    q_offset = tl.arange(0, Q_TILE_SIZE)[:, None]

    for q_start in range(0, N_QUERIES, Q_TILE_SIZE):
        if q_start + Q_TILE_SIZE > k_start:
            q_tile = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
            do_tile = tl.load(dO_block_ptr, boundary_check=(0, 1), padding_option="zero")
            l_tile = tl.load(L_block_ptr, boundary_check=(0,), padding_option="zero")
            d_tile = tl.load(D_block_ptr, boundary_check=(0,), padding_option="zero") 

            # S = Q @ K^T * scale
            s = tl.dot(q_tile, tl.trans(k_tile)) * scale # [Bq, Bk] fp32

            # 分离非掩码与对角线区域
            if k_start + K_TILE_SIZE > q_start:
                q_abs = q_start + q_offset
                mask = k_abs <= q_abs
                s = tl.where(mask, s, -1e6)
            
            p = tl.exp(s - l_tile[:, None]) # [Bq, Bk]

            dv_tile = tl.dot(tl.trans(p), do_tile.to(tl.float32), acc=dv_tile)

            dp = tl.dot(do_tile.to(V_ptr.type.element_ty), tl.trans(v_tile))

            ds = p * (dp - d_tile[:, None])

            dk_tile += tl.dot(tl.trans(ds).to(Q_ptr.type.element_ty), q_tile) * scale

        Q_block_ptr = Q_block_ptr.advance((Q_TILE_SIZE, 0))
        dO_block_ptr = dO_block_ptr.advance((Q_TILE_SIZE, 0))
        L_block_ptr = L_block_ptr.advance((Q_TILE_SIZE,))
        D_block_ptr = D_block_ptr.advance((Q_TILE_SIZE,))

    # 写回 dK, dV
    dK_block_ptr = tl.make_block_ptr(
        dK_ptr + batch_idx * stride_dkb,
        shape=(N_KEYS, HEAD_DIM),
        strides=(stride_dkk, stride_dkd),
        offsets=(k_start, 0),
        block_shape=(K_TILE_SIZE, HEAD_DIM),
        order=(1, 0),
    )

    dV_block_ptr = tl.make_block_ptr(
        dV_ptr + batch_idx * stride_dvb,
        shape=(N_KEYS, HEAD_DIM),
        strides=(stride_dvk, stride_dvd),
        offsets=(k_start, 0),
        block_shape=(K_TILE_SIZE, HEAD_DIM),
        order=(1, 0),
    )

    tl.store(dK_block_ptr, dk_tile.to(K_ptr.type.element_ty))
    tl.store(dV_block_ptr, dv_tile.to(V_ptr.type.element_ty))

@triton.autotune(
    configs=[
        # 1. 首选方案：Q=128, K=32, stages=4（计算密度最高，占用约 96KB，安全）
        triton.Config({'Q_TILE_SIZE': 128, 'K_TILE_SIZE': 32}, num_warps=4, num_stages=4),
        # 2. 稳健方案：Q=64, K=64, stages=3（占用约 88KB，非常安全，适合反向传播）
        triton.Config({'Q_TILE_SIZE': 64, 'K_TILE_SIZE': 64}, num_warps=4, num_stages=3),
        # 3. 保底方案：Q=64, K=32, stages=4（占用极小，约 72KB，防止极短序列出现边界效应）
        triton.Config({'Q_TILE_SIZE': 64, 'K_TILE_SIZE': 32}, num_warps=4, num_stages=4),
    ],
    key=['N_QUERIES', 'N_KEYS', 'HEAD_DIM'], 
)
@triton.jit
def flash_bwd_q_kernel(
    Q_ptr, K_ptr, V_ptr,
    dO_ptr,
    L_ptr, D_ptr,
    dQ_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_dob, stride_doq, stride_dod,
    stride_lb, stride_lq,
    stride_db, stride_dq,
    stride_dqb, stride_dqq, stride_dqd,
    N_QUERIES, N_KEYS,
    scale,
    HEAD_DIM: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
  ):
    q_tile_idx = tl.program_id(0)
    batch_idx = tl.program_id(1)

    q_start = q_tile_idx * Q_TILE_SIZE
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_idx * stride_qb,
        shape=(N_QUERIES, HEAD_DIM),
        strides=(stride_qq, stride_qd),
        offsets=(q_start, 0),
        block_shape=(Q_TILE_SIZE, HEAD_DIM),
        order=(1, 0),
    )

    dO_block_ptr = tl.make_block_ptr(
        dO_ptr + batch_idx * stride_dob,
        shape=(N_QUERIES, HEAD_DIM),
        strides=(stride_doq, stride_dod),
        offsets=(q_start, 0),
        block_shape=(Q_TILE_SIZE, HEAD_DIM),
        order=(1, 0),
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_idx * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(q_start,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    D_block_ptr = tl.make_block_ptr(
        D_ptr + batch_idx * stride_db,
        shape=(N_QUERIES,),
        strides=(stride_dq,),
        offsets=(q_start,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_idx * stride_kb,
        shape=(N_KEYS, HEAD_DIM),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, HEAD_DIM),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_idx * stride_vb,
        shape=(N_KEYS, HEAD_DIM),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, HEAD_DIM),
        order=(1, 0),
    )

    q_tile = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
    do_tile = tl.load(dO_block_ptr, boundary_check=(0, 1), padding_option="zero")
    l_tile = tl.load(L_block_ptr, boundary_check=(0,), padding_option="zero")
    d_tile = tl.load(D_block_ptr, boundary_check=(0,), padding_option="zero")

    dq_tile = tl.zeros([Q_TILE_SIZE, HEAD_DIM], dtype=tl.float32)

    q_abs = q_start + tl.arange(0, Q_TILE_SIZE)[:, None]
    k_offset = tl.arange(0, K_TILE_SIZE)[None, :]

    for k_start in range(0, N_KEYS, K_TILE_SIZE):
        if q_start + Q_TILE_SIZE > k_start:
            k_tile = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
            v_tile = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")

            s = tl.dot(q_tile, tl.trans(k_tile)) * scale

            if k_start + K_TILE_SIZE > q_start:
                k_abs = k_start + k_offset
                mask =  k_abs <= q_abs
                s = tl.where(mask, s, -1e6)

            p = tl.exp(s - l_tile[:, None])

            dp = tl.dot(do_tile.to(V_ptr.type.element_ty), tl.trans(v_tile))

            ds = p * (dp - d_tile[:, None])

            dq_tile += tl.dot(ds.to(K_ptr.type.element_ty), k_tile) * scale

        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    dQ_block_ptr = tl.make_block_ptr(
        dQ_ptr + batch_idx * stride_dqb,
        shape=(N_QUERIES, HEAD_DIM),
        strides=(stride_dqq, stride_dqd),
        offsets=(q_start, 0),
        block_shape=(Q_TILE_SIZE, HEAD_DIM),
        order=(1, 0),
    )

    tl.store(dQ_block_ptr, dq_tile.to(Q_ptr.type.element_ty))

class flashattention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        B, N_q, D = Q.shape
        N_k = K.shape[-2]
        scale = 1 / math.sqrt(D)

        O = torch.zeros_like(Q)
        # L 用高精度
        L = torch.zeros(B, N_q, dtype=torch.float32, device=Q.device)

        grid = lambda meta: (triton.cdiv(N_q, meta['Q_TILE_SIZE']), B)

        flash_fwd_kernel[grid](
            Q, K, V,
            O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            N_q, N_k,
            scale,
            D=D,
            is_causal=is_causal,
        )

        ctx.save_for_backward(Q, K, V, O, L)
        ctx.scale = scale
        ctx.is_causal = is_causal

        return O

    @staticmethod
    def backward(ctx, grad_O):
        Q, K, V, O, L = ctx.saved_tensors
        scale = ctx.scale
        is_causal = ctx.is_causal

        B, Nq, HEAD_DIM = Q.shape
        Nk = K.shape[1]

        # 计算 D
        D = (O * grad_O).sum(dim=-1)   # (B, Nq)

        dQ = torch.zeros_like(Q)
        dK = torch.zeros_like(K)
        dV = torch.zeros_like(V)

        # dk, dv
        grid_dkdv = lambda meta: (triton.cdiv(Nk, meta['K_TILE_SIZE']), B)

        flash_bwd_dk_dv_kernel[grid_dkdv](
            Q, K, V, grad_O, L, D,
            dK, dV,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            grad_O.stride(0), grad_O.stride(1), grad_O.stride(2),
            L.stride(0), L.stride(1),
            D.stride(0), D.stride(1),
            dK.stride(0), dK.stride(1), dK.stride(2),
            dV.stride(0), dV.stride(1), dV.stride(2),
            Nq, Nk, scale,
            HEAD_DIM=HEAD_DIM,
            is_causal=is_causal,
        )

        # dq
        grid_dq = lambda meta: (triton.cdiv(Nq, meta['Q_TILE_SIZE']), B)

        flash_bwd_q_kernel[grid_dq](
            Q, K, V, grad_O, L, D,
            dQ,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            grad_O.stride(0), grad_O.stride(1), grad_O.stride(2),
            L.stride(0), L.stride(1),
            D.stride(0), D.stride(1),
            dQ.stride(0), dQ.stride(1), dQ.stride(2),
            Nq, Nk, scale,
            HEAD_DIM=HEAD_DIM,
            is_causal=is_causal,
        )

        return dQ, dK, dV, None