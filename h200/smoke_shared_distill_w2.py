"""GPU smoke for the transferred W2 model with exactly one BOS token."""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from quantlab.benchmarks.boxed import extract_last_boxed


def main():
    from vllm import LLM, SamplingParams
    p=argparse.ArgumentParser(); p.add_argument('--model-path',required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    m=LLM(model=a.model_path,quantization='gptq',dtype='float16',max_model_len=8192,max_num_seqs=4,
          kv_cache_memory_bytes=8*1024**3,gpu_memory_utilization=0.5,enforce_eager=False)
    tok=m.get_tokenizer(); cases=[('2 + 3','5'),('13 - 5','8'),('7 * 6','42'),('24 / 4','6')]
    prompts=[]
    for question,_ in cases:
        text=tok.apply_chat_template([{'role':'user','content':f'Compute {question}. Think carefully and put the final numeric answer in \\boxed{{}}.'}],tokenize=False,add_generation_prompt=True)
        if not text.rstrip().endswith('<think>'):text+='<think>\n'
        ids=tok.encode(text,add_special_tokens=False)
        if ids.count(tok.bos_token_id)!=1:raise ValueError('Expected exactly one BOS')
        prompts.append({'prompt_token_ids':ids})
    outputs=m.generate(prompts,SamplingParams(temperature=0,max_tokens=4096),use_tqdm=False)
    rows=[]
    for (question,expected),result in zip(cases,outputs):
        answer=result.outputs[0]; parsed=extract_last_boxed(answer.text)
        rows.append({'question':question,'expected':expected,'parsed':parsed,'correct':parsed is not None and parsed.strip()==expected,'text':answer.text})
    good=sum(r['correct'] for r in rows); a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps({'correct':good,'cases':rows},ensure_ascii=False,indent=2))
    print('W2_SMOKE',good,'/',len(rows),flush=True)
    if good<2:raise RuntimeError('Fewer than two correct arithmetic answers')


if __name__=='__main__':main()
