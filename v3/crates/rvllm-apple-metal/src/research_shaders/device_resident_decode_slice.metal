// Research-only helper for the two-token device-resident decode vertical slice.
//
// One thread advances one layer's single-sequence decode metadata in-place.
// The host supplies absolute byte offsets for (position, slot_mapping,
// context_len) for every layer. The caller must prove that position+1 remains
// inside the same physical KV page; block tables therefore remain unchanged.
#include <metal_stdlib>
using namespace metal;

kernel void research_decode_advance_single(
    device uchar *arena [[buffer(0)]],
    constant uint *offsets [[buffer(1)]],
    constant uint &num_layers [[buffer(2)]],
    uint layer [[thread_position_in_grid]])
{
    if (layer >= num_layers) return;
    uint base = layer * 3u;
    device int *position = reinterpret_cast<device int *>(arena + offsets[base]);
    device int *slot = reinterpret_cast<device int *>(arena + offsets[base + 1u]);
    device int *context = reinterpret_cast<device int *>(arena + offsets[base + 2u]);
    position[0] += 1;
    slot[0] += 1;
    context[0] += 1;
}
