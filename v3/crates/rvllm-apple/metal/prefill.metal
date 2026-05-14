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
