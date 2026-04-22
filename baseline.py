import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import time

# TARGET_MODEL = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"   # 预量化版本，电脑带不动8B原版
TARGET_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DRAFT_MODEL = "meta-llama/Llama-3.2-1B-Instruct"

DEVICE = "cuda"
MAX_NEW_TOKENS = 256
GAMMA = 8
TEMPERATURE = 0.7

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

print("Loading models with 4-bit quantization...")

target_model = AutoModelForCausalLM.from_pretrained(
    TARGET_MODEL,
    quantization_config=quant_config,
    device_map="auto",
    dtype=torch.bfloat16,
    trust_remote_code=True
).eval()

draft_model = AutoModelForCausalLM.from_pretrained(
    DRAFT_MODEL,
    device_map="auto",
    dtype=torch.bfloat16,
    trust_remote_code=True
).eval()

tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def speculative_generate(prompt: str):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
    generated = input_ids.clone()                    # shape: [1, seq_len]
    
    print(f"Starting generation... Target: {TARGET_MODEL} | Draft: {DRAFT_MODEL} | gamma={GAMMA}")
    start_time = time.time()
    
    with torch.no_grad():
        step = 0
        total_drafted = 0
        total_accepted = 0
        accepted_lengths = [0 for _ in range(GAMMA)]
        while generated.shape[1] < MAX_NEW_TOKENS + input_ids.shape[1]:
            step += 1
            curr_len = generated.shape[1]
            
            # 1. Draft (greedy)
            draft_input = generated
            draft_tokens_list = []
            for _ in range(GAMMA):
                outputs = draft_model(draft_input)
                next_token = outputs.logits[:, -1:].argmax(dim=-1)   # [1, 1]
                draft_tokens_list.append(next_token)
                draft_input = torch.cat([draft_input, next_token], dim=1)
            
            draft_tokens = torch.cat(draft_tokens_list, dim=1)       # [1, gamma]
            total_drafted += draft_tokens.shape[1]

            # 2. Verification
            verify_input = torch.cat([generated, draft_tokens], dim=1)
            target_outputs = target_model(verify_input)
            target_logits = target_outputs.logits[:, curr_len-1 : curr_len + GAMMA]  # [1, gamma, vocab]
            
            # 3. simple acceptance
            # print the draft text and target text for debugging
            draft_texts = [tokenizer.decode(draft_tokens[0, i].item()) for i in range(GAMMA)]
            target_texts = [tokenizer.decode(target_logits[0, i].argmax().item()) for i in range(GAMMA)]
            print(f"Draft tokens: {draft_texts}")
            print(f"Target tokens: {target_texts}")

            accepted_tokens = []
            for i in range(GAMMA):
                draft_tok = draft_tokens[0, i]
                target_tok = target_logits[0, i].argmax()
                
                if draft_tok == target_tok:
                    accepted_tokens.append(draft_tok)
                else:
                    accepted_tokens.append(target_tok)
                    break
            
            if accepted_tokens:
                new_tokens = torch.tensor(accepted_tokens, device=DEVICE).unsqueeze(0)
                generated = torch.cat([generated, new_tokens], dim=1)
                total_accepted += len(accepted_tokens)
                accepted_lengths[len(accepted_tokens)-1] += 1
            else:
                fallback_out = target_model(generated)
                next_tok = fallback_out.logits[:, -1:].argmax(dim=-1)
                generated = torch.cat([generated, next_tok], dim=1)
            
            if step > 100:
                break
    
    output_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    elapsed = time.time() - start_time
    accept_rate = total_accepted / total_drafted if total_drafted > 0 else 0
    print(f"Completed in {elapsed:.2f} seconds, generated {generated.shape[1] - input_ids.shape[1]} tokens")
    print(f"Drafted {total_drafted} tokens, accepted {total_accepted} tokens, acceptance rate: {accept_rate:.2%}")
    print(f"Accepted token lengths distribution: {accepted_lengths}")
    return output_text

if __name__ == "__main__":
    prompt = "Give me ONLY ONE benefit of using a speculatively generated decoding strategy in language models using less than 20 words. Do not write anything else."
    result = speculative_generate(prompt)
    print("\n=== Outputs ===\n")
    print(result)