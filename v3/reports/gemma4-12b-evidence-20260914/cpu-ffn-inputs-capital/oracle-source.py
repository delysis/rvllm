"""Independent pinned-Transformers oracle for captured Rust ANE layer outputs."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch
import transformers
from transformers import Gemma4UnifiedForConditionalGeneration
from transformers.cache_utils import DynamicCache

parser = argparse.ArgumentParser()
parser.add_argument('--case-dir', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--ffn-output-dir', type=Path, required=True)
args = parser.parse_args()
assert not args.output.exists()
args.ffn_output_dir.mkdir(exist_ok=False)
seed_bytes = (args.case_dir/'report.json').read_bytes()
seed = json.loads(seed_bytes)
assert seed['schema'] == 'rvllm.metal_ane_prefill_handoff.v2'
assert seed['cache_capture_after_collection']
model_dir = Path(seed['model_dir'])
assert model_dir.name == '707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7'
position = seed['ane_next_position']
assert position == len(seed['prompt_token_ids'])
torch.set_num_threads(8)
started = time.perf_counter()
model = Gemma4UnifiedForConditionalGeneration.from_pretrained(
    model_dir, dtype=torch.float16, attn_implementation='eager', local_files_only=True,
).eval()
buffers = {name: str(value.dtype) for name, value in model.named_buffers()
           if 'inv_freq' in name or 'embed_scale' in name}
assert all(dtype == 'torch.float32' for name, dtype in buffers.items() if 'inv_freq' in name)
cache = DynamicCache(config=model.config)
assert len(seed['layers']) == 48
for index, layer in enumerate(seed['layers']):
    assert index == layer['layer']
    tensors = []
    for kind in ('key', 'value'):
        filename = layer[kind + '_file']
        assert Path(filename).name == filename
        data = (args.case_dir/filename).read_bytes()
        assert hashlib.sha256(data).hexdigest() == layer[kind + '_sha256']
        assert len(data) == 2 * position * layer['kv_heads'] * layer['head_dim']
        tensor = torch.frombuffer(bytearray(data), dtype=torch.float16)
        assert torch.isfinite(tensor).all()
        tensors.append(tensor.reshape(position, layer['kv_heads'], layer['head_dim'])
                       .permute(1, 0, 2).unsqueeze(0).contiguous())
    cache.update(*tensors, index)

ffn_inputs = []
ffn_handles = []
for index in range(48):
    def capture_ffn(module, inputs, index=index):
        value = inputs[0].detach().reshape(-1).contiguous()
        assert value.dtype == torch.float16 and value.numel() == 3840 and torch.isfinite(value).all()
        data = value.numpy().tobytes()
        path = args.ffn_output_dir/f'layer-{index:02}-ffn-input.fp16'
        with path.open('xb') as output_file: output_file.write(data)
        ffn_inputs.append(dict(layer=index,file=path.name,sha256=hashlib.sha256(data).hexdigest(),maximum_abs=float(value.abs().max())))
    ffn_handles.append(model.get_submodule(f'model.language_model.layers.{index}.mlp').register_forward_pre_hook(capture_ffn))

comparisons = []
handles = []
for index in range(48):
    def capture(module, inputs, output, index=index):
        expected = output[0] if isinstance(output, tuple) else output
        expected = expected.detach().reshape(-1).float()
        assert expected.numel() == 3840 and torch.isfinite(expected).all()
        path = args.case_dir/f'ane-position-{position}-layer-{index:02}.fp16'
        if not path.exists():
            comparisons.append({'layer':index, 'ane_state_present':False})
            return
        data = path.read_bytes()
        actual = torch.frombuffer(bytearray(data), dtype=torch.float16).float()
        assert actual.shape == expected.shape and torch.isfinite(actual).all()
        difference = actual - expected
        comparison = dict(layer=index, ane_state_present=True,
                          actual_sha256=hashlib.sha256(data).hexdigest(),
                          max_abs_error=float(difference.abs().max()),
                          relative_l2_error=float(difference.norm()/expected.norm()),
                          reference_rms=float(expected.square().mean().sqrt()),
                          actual_rms=float(actual.square().mean().sqrt()))
        comparisons.append(comparison)
        print(json.dumps(comparison), flush=True)
    handles.append(model.get_submodule(f'model.language_model.layers.{index}').register_forward_hook(capture))

with torch.inference_mode():
    output = model(
        input_ids=torch.tensor([[seed['first_generated_token']]], dtype=torch.long),
        attention_mask=torch.ones((1, position + 1), dtype=torch.long),
        position_ids=torch.tensor([[position]], dtype=torch.long),
        past_key_values=cache, use_cache=True, logits_to_keep=1,
    )
    logits = output.logits[0, -1].float()
    assert torch.isfinite(logits).all()
    top_values, top_ids = logits.topk(5)
for handle in handles:
    handle.remove()
report = dict(
    schema='rvllm.ane_layer_comparison.v1', model_revision=model_dir.name,
    transformers_revision='3384908511545a9c146de65f01f3d29b90b2add6',
    transformers_version=transformers.__version__, torch_version=torch.__version__,
    load_policy='Fresh FP16 CPU eager load; original FP32 rotary buffers; no model.to(dtype)',
    seed_report_sha256=hashlib.sha256(seed_bytes).hexdigest(), buffers=buffers,
    position=position, input_token=seed['first_generated_token'],
    cpu_next_token=int(logits.argmax()), cpu_top_five=list(zip(top_ids.tolist(), top_values.tolist())),
    ffn_input_source="Pinned FP16 CPU model consuming the actual Metal KV seed; these are CPU model FFN inputs, not captured ANE FFN inputs", ffn_inputs=ffn_inputs, comparisons=comparisons, all_ane_states_present=all(x['ane_state_present'] for x in comparisons),
    total_seconds=time.perf_counter()-started,
)
with args.output.open('x') as output_file:
    json.dump(report, output_file, indent=2)
print(json.dumps({k:report[k] for k in ('cpu_next_token','cpu_top_five','all_ane_states_present','total_seconds')}), flush=True)
