import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List

INFERENCE_MICRO_BATCH_SIZE = 64

class LocalOracle:
    def __init__(self, model_id="meta-llama/Llama-Guard-3-8B"):
        print(f"Loading Oracle (BF16): {model_id}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left" 

    def check_safety_batch(self, prompts: List[str], responses: List[str]) -> List[bool]:
        """Judges a batch of (prompt, response) pairs."""
        
        all_results = []
        for i in range(0, len(prompts), INFERENCE_MICRO_BATCH_SIZE):
            batch_prompts = prompts[i : i + INFERENCE_MICRO_BATCH_SIZE]
            batch_responses = responses[i : i + INFERENCE_MICRO_BATCH_SIZE]
            
            chats = [
                [
                    {"role": "user", "content": p},
                    {"role": "assistant", "content": r}
                ]
                for p, r in zip(batch_prompts, batch_responses)
            ]
            
            templated_prompts = [
                self.tokenizer.apply_chat_template(chat, tokenize=False)
                for chat in chats
            ]

            inputs = self.tokenizer(
                templated_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=4096 
            ).to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=20, 
                    pad_token_id=self.tokenizer.pad_token_id
                )
                
            input_token_len = inputs.input_ids.shape[1]
            decoded_texts = self.tokenizer.batch_decode(outputs[:, input_token_len:], skip_special_tokens=True)
            
            results = []
            for text in decoded_texts:
                if "unsafe" in text.lower():
                    results.append(False)
                else:
                    results.append(True)
            all_results.extend(results)
        
        return all_results

_oracle = None
def ask_oracle_batch(prompts: List[str], responses: List[str]) -> List[bool]:
    global _oracle
    if _oracle is None:
        _oracle = LocalOracle()
    
    if not prompts: 
        return []
        
    return _oracle.check_safety_batch(prompts, responses)