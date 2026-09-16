"""Fresh pinned upstream BF16 CPU references; not an inference backend."""
import json,time,pathlib,torch
from transformers import AutoTokenizer,Gemma4UnifiedForConditionalGeneration
root=pathlib.Path.cwd(); out=root/'reports/gemma4-12b-evidence-20260914/long-context-cpu-oracle';out.mkdir(exist_ok=False)
model_dir=pathlib.Path('/Users/george/.cache/huggingface/hub/models--google--gemma-4-12B-it/snapshots/707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7')
torch.set_num_threads(8)
tokenizer=AutoTokenizer.from_pretrained(model_dir,local_files_only=True)
contents=[('copy96', 'This is an unrelated sentence about a garden. '*6+'Copy exactly this sentence and nothing else: The blue fox jumps over the quiet river.'), ('recall256','The secret word is violet. '+'This paragraph contains background details about trees and flowers. '*20+'Reply with only the secret word stated at the beginning.'), ('recall768','The secret number is 42. '+'This paragraph contains background details about trees and flowers. '*62+'Reply with only the secret number stated at the beginning.')]
model=Gemma4UnifiedForConditionalGeneration.from_pretrained(model_dir,dtype=torch.bfloat16,attn_implementation='eager',local_files_only=True).eval()
assert all(v.dtype==torch.float32 for n,v in model.named_buffers() if 'inv_freq' in n)
for name,content in contents:
 messages=[{'role':'user','content':content}];prompt=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
 inputs=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_tensors='pt',return_dict=True);ids=inputs['input_ids'][0].tolist();assert 64<len(ids)<1000
 start=time.perf_counter()
 with torch.inference_mode(): result=model.generate(**inputs,max_new_tokens=24,do_sample=False)
 generated=result[0,len(ids):].tolist()
 data=dict(model_revision=model_dir.name,transformers_revision='3384908511545a9c146de65f01f3d29b90b2add6',load_policy='Fresh BF16 CPU from_pretrained; no model.to(dtype); eager attention',user_text=content,prompt=prompt,prompt_token_ids=ids,generated_tokens=generated,generated_text=tokenizer.decode(generated,skip_special_tokens=True),cpu_elapsed_seconds=time.perf_counter()-start)
 (out/(name+'.json')).write_text(json.dumps(data,indent=2));print(name,len(ids),generated,repr(data['generated_text']),flush=True)
