#include <metal_stdlib>

using namespace metal;

struct RvllmRmsNormParams {
    uint element_count;
    uint hidden_size;
    float epsilon;
};

struct RvllmMatmulParams {
    uint rows;
    uint cols;
    uint inner;
    uint a_row_stride;
    uint b_row_stride;
    uint c_row_stride;
};

struct RvllmRopeParams {
    uint token_count;
    uint head_dim;
    uint rotary_dim;
    float base;
};

struct RvllmAttentionParams {
    uint query_count;
    uint key_count;
    uint head_dim;
    uint value_dim;
    float scale;
};

kernel void rvllm_rms_norm(device const half *input [[buffer(0)]],
                           device const half *weight [[buffer(1)]],
                           device half *output [[buffer(2)]],
                           constant RvllmRmsNormParams &params [[buffer(3)]],
                           uint gid [[thread_position_in_grid]])
{
    if (gid >= params.element_count || params.hidden_size == 0) {
        return;
    }

    const uint row_start = (gid / params.hidden_size) * params.hidden_size;
    float square_sum = 0.0f;
    for (uint col = 0; col < params.hidden_size; ++col) {
        const float value = float(input[row_start + col]);
        square_sum += value * value;
    }

    const uint weight_index = gid - row_start;
    const float mean_square = square_sum / float(params.hidden_size);
    const float scale = rsqrt(mean_square + params.epsilon);
    output[gid] = half(float(input[gid]) * scale * float(weight[weight_index]));
}

kernel void rvllm_matmul(device const half *a [[buffer(0)]],
                         device const half *b [[buffer(1)]],
                         device half *c [[buffer(2)]],
                         constant RvllmMatmulParams &params [[buffer(3)]],
                         uint2 gid [[thread_position_in_grid]])
{
    const uint col = gid.x;
    const uint row = gid.y;
    if (row >= params.rows || col >= params.cols) {
        return;
    }

    float acc = 0.0f;
    for (uint k = 0; k < params.inner; ++k) {
        const float av = float(a[row * params.a_row_stride + k]);
        const float bv = float(b[k * params.b_row_stride + col]);
        acc += av * bv;
    }

    c[row * params.c_row_stride + col] = half(acc);
}

kernel void rvllm_rope(device const half *input [[buffer(0)]],
                       device const uint *positions [[buffer(1)]],
                       device half *output [[buffer(2)]],
                       constant RvllmRopeParams &params [[buffer(3)]],
                       uint gid [[thread_position_in_grid]])
{
    const uint element_count = params.token_count * params.head_dim;
    if (gid >= element_count || params.head_dim == 0) {
        return;
    }

    const uint token = gid / params.head_dim;
    const uint dim = gid - token * params.head_dim;
    const uint rotary_dim = params.rotary_dim & ~1u;
    if (dim >= rotary_dim) {
        output[gid] = input[gid];
        return;
    }

    const uint even_dim = dim & ~1u;
    const uint base_offset = token * params.head_dim;
    const float x = float(input[base_offset + even_dim]);
    const float y = float(input[base_offset + even_dim + 1]);
    const float theta_base = params.base > 0.0f ? params.base : 10000.0f;
    const float freq = pow(theta_base, -float(even_dim) / float(rotary_dim));
    const float angle = float(positions[token]) * freq;
    const float c = cos(angle);
    const float s = sin(angle);

    if ((dim & 1u) == 0) {
        output[gid] = half(x * c - y * s);
    } else {
        output[gid] = half(x * s + y * c);
    }
}

kernel void rvllm_attention(device const half *query [[buffer(0)]],
                            device const half *key [[buffer(1)]],
                            device const half *value [[buffer(2)]],
                            device half *output [[buffer(3)]],
                            constant RvllmAttentionParams &params [[buffer(4)]],
                            uint2 gid [[thread_position_in_grid]])
{
    const uint value_col = gid.x;
    const uint query_row = gid.y;
    if (query_row >= params.query_count || value_col >= params.value_dim) {
        return;
    }

    float max_score = -3.402823466e+38f;
    for (uint key_row = 0; key_row < params.key_count; ++key_row) {
        float score = 0.0f;
        for (uint dim = 0; dim < params.head_dim; ++dim) {
            score += float(query[query_row * params.head_dim + dim])
                * float(key[key_row * params.head_dim + dim]);
        }
        max_score = max(max_score, score * params.scale);
    }

    float weighted_sum = 0.0f;
    float weight_total = 0.0f;
    for (uint key_row = 0; key_row < params.key_count; ++key_row) {
        float score = 0.0f;
        for (uint dim = 0; dim < params.head_dim; ++dim) {
            score += float(query[query_row * params.head_dim + dim])
                * float(key[key_row * params.head_dim + dim]);
        }
        const float attention_weight = exp(score * params.scale - max_score);
        weighted_sum += attention_weight * float(value[key_row * params.value_dim + value_col]);
        weight_total += attention_weight;
    }

    const float normalized = weight_total > 0.0f ? weighted_sum / weight_total : 0.0f;
    output[query_row * params.value_dim + value_col] = half(normalized);
}

struct RvllmG4EmbeddingParams {
    uint token_id;
    uint hidden;
    float scale;
};

struct RvllmG4RmsNormParams {
    uint dim;
    float epsilon;
};

struct RvllmG4HeadNormParams {
    uint num_heads;
    uint head_dim;
    float epsilon;
};

struct RvllmG4MatVecParams {
    uint rows;
    uint cols;
};

struct RvllmG4LogitsParams {
    uint vocab;
    uint hidden;
    float softcap;
};

struct RvllmG4RopeCacheParams {
    uint position;
    uint num_heads;
    uint num_kv_heads;
    uint head_dim;
    uint rotary_dim;
    float theta;
};

struct RvllmG4AttentionParams {
    uint position;
    uint num_heads;
    uint num_kv_heads;
    uint head_dim;
    uint sliding_window;
    uint is_sliding;
    float scale;
};

struct RvllmG4NormAddParams {
    uint hidden;
    float epsilon;
    uint apply_scalar;
};

struct RvllmG4GeluParams {
    uint intermediate;
};

struct RvllmG4ArgmaxParams {
    uint vocab;
};

struct RvllmG4PleCombineParams {
    uint token_id;
    uint num_layers;
    uint ple_dim;
    float hidden_scale;
    float combine_scale;
};

struct RvllmG4PleGateParams {
    uint layer_idx;
    uint ple_dim;
};

kernel void rvllm_g4_embedding(device const half *embedding [[buffer(0)]],
                               device float *residual [[buffer(1)]],
                               constant RvllmG4EmbeddingParams &params [[buffer(2)]],
                               uint gid [[thread_position_in_grid]])
{
    if (gid >= params.hidden) {
        return;
    }
    const ulong offset = ulong(params.token_id) * ulong(params.hidden) + ulong(gid);
    residual[gid] = float(embedding[offset]) * params.scale;
}

kernel void rvllm_g4_rmsnorm(device const float *input [[buffer(0)]],
                             device const half *gamma [[buffer(1)]],
                             device float *output [[buffer(2)]],
                             constant RvllmG4RmsNormParams &params [[buffer(3)]],
                             uint gid [[thread_position_in_grid]])
{
    if (gid >= params.dim || params.dim == 0) {
        return;
    }
    float sum_sq = 0.0f;
    for (uint i = 0; i < params.dim; ++i) {
        const float v = input[i];
        sum_sq += v * v;
    }
    const float inv_rms = rsqrt(sum_sq / float(params.dim) + params.epsilon);
    output[gid] = input[gid] * inv_rms * float(gamma[gid]);
}

kernel void rvllm_g4_rmsnorm_heads(device const float *input [[buffer(0)]],
                                   device const half *gamma [[buffer(1)]],
                                   device float *output [[buffer(2)]],
                                   constant RvllmG4HeadNormParams &params [[buffer(3)]],
                                   uint gid [[thread_position_in_grid]])
{
    const uint total = params.num_heads * params.head_dim;
    if (gid >= total || params.head_dim == 0) {
        return;
    }
    const uint head = gid / params.head_dim;
    const uint dim = gid - head * params.head_dim;
    const uint base = head * params.head_dim;
    float sum_sq = 0.0f;
    for (uint i = 0; i < params.head_dim; ++i) {
        const float v = input[base + i];
        sum_sq += v * v;
    }
    const float inv_rms = rsqrt(sum_sq / float(params.head_dim) + params.epsilon);
    output[gid] = input[gid] * inv_rms * float(gamma[dim]);
}

kernel void rvllm_g4_rmsnorm_heads_no_gamma(device const float *input [[buffer(0)]],
                                            device float *output [[buffer(1)]],
                                            constant RvllmG4HeadNormParams &params [[buffer(2)]],
                                            uint gid [[thread_position_in_grid]])
{
    const uint total = params.num_heads * params.head_dim;
    if (gid >= total || params.head_dim == 0) {
        return;
    }
    const uint head = gid / params.head_dim;
    const uint base = head * params.head_dim;
    float sum_sq = 0.0f;
    for (uint i = 0; i < params.head_dim; ++i) {
        const float v = input[base + i];
        sum_sq += v * v;
    }
    const float inv_rms = rsqrt(sum_sq / float(params.head_dim) + params.epsilon);
    output[gid] = input[gid] * inv_rms;
}

kernel void rvllm_g4_matvec_half(device const half *weights [[buffer(0)]],
                                 device const float *input [[buffer(1)]],
                                 device float *output [[buffer(2)]],
                                 constant RvllmG4MatVecParams &params [[buffer(3)]],
                                 uint row [[thread_position_in_grid]])
{
    if (row >= params.rows) {
        return;
    }
    float acc = 0.0f;
    const ulong base = ulong(row) * ulong(params.cols);
    for (uint col = 0; col < params.cols; ++col) {
        acc += float(weights[base + ulong(col)]) * input[col];
    }
    output[row] = acc;
}

kernel void rvllm_g4_matvec_logits(device const half *weights [[buffer(0)]],
                                   device const float *input [[buffer(1)]],
                                   device float *logits [[buffer(2)]],
                                   constant RvllmG4LogitsParams &params [[buffer(3)]],
                                   uint row [[thread_position_in_grid]])
{
    if (row >= params.vocab) {
        return;
    }
    float acc = 0.0f;
    const ulong base = ulong(row) * ulong(params.hidden);
    for (uint col = 0; col < params.hidden; ++col) {
        acc += float(weights[base + ulong(col)]) * input[col];
    }
    if (params.softcap > 0.0f) {
        acc = params.softcap * tanh(acc / params.softcap);
    }
    logits[row] = acc;
}

kernel void rvllm_g4_rope_cache(device const float *q_in [[buffer(0)]],
                                device const float *k_in [[buffer(1)]],
                                device const float *v_in [[buffer(2)]],
                                device float *q_out [[buffer(3)]],
                                device float *key_cache [[buffer(4)]],
                                device float *value_cache [[buffer(5)]],
                                constant RvllmG4RopeCacheParams &params [[buffer(6)]],
                                uint2 gid [[thread_position_in_grid]])
{
    const uint tid = gid.x;
    const uint head = gid.y;
    const uint half_head = params.head_dim / 2;
    const uint half_rotary = params.rotary_dim / 2;
    if (tid >= half_head) {
        return;
    }

    if (head < params.num_heads) {
        const uint base = head * params.head_dim;
        if (tid < half_rotary) {
            const float freq = pow(params.theta, -2.0f * float(tid) / float(params.head_dim));
            const float angle = float(params.position) * freq;
            const float c = cos(angle);
            const float s = sin(angle);
            const float lo = q_in[base + tid];
            const float hi = q_in[base + tid + half_head];
            q_out[base + tid] = lo * c - hi * s;
            q_out[base + tid + half_head] = lo * s + hi * c;
        } else {
            q_out[base + tid] = q_in[base + tid];
            q_out[base + tid + half_head] = q_in[base + tid + half_head];
        }
    }

    if (head < params.num_kv_heads) {
        const uint base = head * params.head_dim;
        const ulong cache_base = (ulong(params.position) * ulong(params.num_kv_heads) + ulong(head))
            * ulong(params.head_dim);
        if (tid < half_rotary) {
            const float freq = pow(params.theta, -2.0f * float(tid) / float(params.head_dim));
            const float angle = float(params.position) * freq;
            const float c = cos(angle);
            const float s = sin(angle);
            const float lo = k_in[base + tid];
            const float hi = k_in[base + tid + half_head];
            key_cache[cache_base + ulong(tid)] = lo * c - hi * s;
            key_cache[cache_base + ulong(tid + half_head)] = lo * s + hi * c;
        } else {
            key_cache[cache_base + ulong(tid)] = k_in[base + tid];
            key_cache[cache_base + ulong(tid + half_head)] = k_in[base + tid + half_head];
        }
        value_cache[cache_base + ulong(tid)] = v_in[base + tid];
        value_cache[cache_base + ulong(tid + half_head)] = v_in[base + tid + half_head];
    }
}

kernel void rvllm_g4_rope_query(device const float *q_in [[buffer(0)]],
                                device float *q_out [[buffer(1)]],
                                constant RvllmG4RopeCacheParams &params [[buffer(2)]],
                                uint2 gid [[thread_position_in_grid]])
{
    const uint tid = gid.x;
    const uint head = gid.y;
    const uint half_head = params.head_dim / 2;
    const uint half_rotary = params.rotary_dim / 2;
    if (tid >= half_head || head >= params.num_heads) {
        return;
    }

    const uint base = head * params.head_dim;
    if (tid < half_rotary) {
        const float freq = pow(params.theta, -2.0f * float(tid) / float(params.head_dim));
        const float angle = float(params.position) * freq;
        const float c = cos(angle);
        const float s = sin(angle);
        const float lo = q_in[base + tid];
        const float hi = q_in[base + tid + half_head];
        q_out[base + tid] = lo * c - hi * s;
        q_out[base + tid + half_head] = lo * s + hi * c;
    } else {
        q_out[base + tid] = q_in[base + tid];
        q_out[base + tid + half_head] = q_in[base + tid + half_head];
    }
}

kernel void rvllm_g4_attention(device const float *query [[buffer(0)]],
                               device const float *key_cache [[buffer(1)]],
                               device const float *value_cache [[buffer(2)]],
                               device float *output [[buffer(3)]],
                               constant RvllmG4AttentionParams &params [[buffer(4)]],
                               uint gid [[thread_position_in_grid]])
{
    const uint q_dim = params.num_heads * params.head_dim;
    if (gid >= q_dim) {
        return;
    }
    const uint q_head = gid / params.head_dim;
    const uint dim = gid - q_head * params.head_dim;
    const uint kv_head = (q_head * params.num_kv_heads) / params.num_heads;
    const uint start = (params.is_sliding != 0 && params.position + 1 > params.sliding_window)
        ? (params.position + 1 - params.sliding_window)
        : 0;

    float max_score = -3.402823466e+38f;
    for (uint pos = start; pos <= params.position; ++pos) {
        const ulong kv_base = (ulong(pos) * ulong(params.num_kv_heads) + ulong(kv_head))
            * ulong(params.head_dim);
        float score = 0.0f;
        for (uint d = 0; d < params.head_dim; ++d) {
            score += query[q_head * params.head_dim + d] * key_cache[kv_base + ulong(d)];
        }
        max_score = max(max_score, score * params.scale);
    }

    float weight_sum = 0.0f;
    float value_sum = 0.0f;
    for (uint pos = start; pos <= params.position; ++pos) {
        const ulong kv_base = (ulong(pos) * ulong(params.num_kv_heads) + ulong(kv_head))
            * ulong(params.head_dim);
        float score = 0.0f;
        for (uint d = 0; d < params.head_dim; ++d) {
            score += query[q_head * params.head_dim + d] * key_cache[kv_base + ulong(d)];
        }
        const float weight = exp(score * params.scale - max_score);
        weight_sum += weight;
        value_sum += weight * value_cache[kv_base + ulong(dim)];
    }
    output[gid] = weight_sum > 0.0f ? value_sum / weight_sum : 0.0f;
}

kernel void rvllm_g4_norm_add_residual(device const float *input [[buffer(0)]],
                                       device const half *gamma [[buffer(1)]],
                                       device float *residual [[buffer(2)]],
                                       device const half *layer_scalar [[buffer(3)]],
                                       constant RvllmG4NormAddParams &params [[buffer(4)]],
                                       uint gid [[thread_position_in_grid]])
{
    if (gid >= params.hidden) {
        return;
    }
    float sum_sq = 0.0f;
    for (uint i = 0; i < params.hidden; ++i) {
        const float v = input[i];
        sum_sq += v * v;
    }
    const float inv_rms = rsqrt(sum_sq / float(params.hidden) + params.epsilon);
    float out = residual[gid] + input[gid] * inv_rms * float(gamma[gid]);
    if (params.apply_scalar != 0) {
        out *= float(layer_scalar[0]);
    }
    residual[gid] = out;
}

kernel void rvllm_g4_gelu_mul(device const float *gate_up [[buffer(0)]],
                              device float *output [[buffer(1)]],
                              constant RvllmG4GeluParams &params [[buffer(2)]],
                              uint gid [[thread_position_in_grid]])
{
    if (gid >= params.intermediate) {
        return;
    }
    const float g = gate_up[gid];
    const float u = gate_up[params.intermediate + gid];
    const float inner = clamp(0.7978845608f * (g + 0.044715f * g * g * g), -20.0f, 20.0f);
    const float gelu = 0.5f * g * (1.0f + tanh(inner));
    output[gid] = gelu * u;
}

kernel void rvllm_g4_argmax(device const float *logits [[buffer(0)]],
                            device uint *token [[buffer(1)]],
                            constant RvllmG4ArgmaxParams &params [[buffer(2)]],
                            uint gid [[thread_position_in_grid]])
{
    if (gid != 0 || params.vocab == 0) {
        return;
    }
    float best = -3.402823466e+38f;
    uint best_id = 0;
    for (uint i = 0; i < params.vocab; ++i) {
        const float value = logits[i];
        if (isfinite(value) && value > best) {
            best = value;
            best_id = i;
        }
    }
    token[0] = best_id;
}

kernel void rvllm_g4_ple_combine(device const float *context [[buffer(0)]],
                                 device const half *token_table [[buffer(1)]],
                                 device const half *gamma [[buffer(2)]],
                                 device float *output [[buffer(3)]],
                                 constant RvllmG4PleCombineParams &params [[buffer(4)]],
                                 uint gid [[thread_position_in_grid]])
{
    const uint total = params.num_layers * params.ple_dim;
    if (gid >= total || params.ple_dim == 0) {
        return;
    }
    const uint layer = gid / params.ple_dim;
    const uint dim = gid - layer * params.ple_dim;
    const uint base = layer * params.ple_dim;

    float sum_sq = 0.0f;
    for (uint i = 0; i < params.ple_dim; ++i) {
        const float v = context[base + i] * params.hidden_scale;
        sum_sq += v * v;
    }
    const float inv_rms = rsqrt(sum_sq / float(params.ple_dim) + 1.0e-6f);
    const float projected = context[gid] * params.hidden_scale * inv_rms * float(gamma[dim]);
    const ulong token_offset = ulong(params.token_id) * ulong(total) + ulong(gid);
    const float token_identity = float(token_table[token_offset]);
    output[gid] = (projected + token_identity) * params.combine_scale;
}

kernel void rvllm_g4_ple_gate_mul(device float *gate [[buffer(0)]],
                                  device const float *per_layer_inputs [[buffer(1)]],
                                  constant RvllmG4PleGateParams &params [[buffer(2)]],
                                  uint gid [[thread_position_in_grid]])
{
    if (gid >= params.ple_dim) {
        return;
    }
    const float g = gate[gid];
    const float inner = clamp(0.7978845608f * (g + 0.044715f * g * g * g), -20.0f, 20.0f);
    const float gelu = 0.5f * g * (1.0f + tanh(inner));
    const uint offset = params.layer_idx * params.ple_dim + gid;
    gate[gid] = gelu * per_layer_inputs[offset];
}
