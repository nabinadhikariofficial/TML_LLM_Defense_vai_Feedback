import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List

INFERENCE_MICRO_BATCH_SIZE = 64

class BaseTaskModel:
    def __init__(self, model_id="meta-llama/Llama-3.1-8B-Instruct"):
        print(f"Loading Base Model (BF16): {model_id}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        
        self.tokenizer.pad_token_id = 128001 
        self.tokenizer.padding_side = "left" 
        print("Base model loaded.")

    def _get_assistant_response(self, text: str) -> str:
        if "assistant<|end_header_id|>\n\n" in text:
            return text.split("assistant<|end_header_id|>\n\n")[-1].strip()
        return text

    def get_responses(self, user_prompts: List[str]) -> List[str]:
        """
        Generates responses for a BATCH of prompts, using micro-batching.
        """
        all_final_responses = []
        
        for i in range(0, len(user_prompts), INFERENCE_MICRO_BATCH_SIZE):
            batch_prompts = user_prompts[i : i + INFERENCE_MICRO_BATCH_SIZE]
            
            messages_list = [
                [{"role": "system", "content": "You are a helpful assistant."},
                 {"role": "user", "content": prompt}]
                for prompt in batch_prompts
            ]
            
            templated_prompts = [
                self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                for messages in messages_list
            ]
            
            inputs = self.tokenizer(
                templated_prompts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=2048
            ).to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.6,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            decoded_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            final_responses = [self._get_assistant_response(text) for text in decoded_texts]
            all_final_responses.extend(final_responses)
            
        return all_final_responses