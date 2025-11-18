import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
from typing import List

class DefenseModule:
    def __init__(self, model_id="microsoft/deberta-v3-large", device="cuda"):
        print(f"Loading Defense Model (BF16): {model_id}...")
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id, num_labels=2, id2label={0: "UNSAFE", 1: "SAFE"},
            label2id={"UNSAFE": 0, "SAFE": 1}, ignore_mismatched_sizes=True,
            torch_dtype=torch.bfloat16
        ).to(self.device)
        
        lora_config = LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.1,
            task_type=TaskType.SEQ_CLS,
            target_modules=["query_proj", "value_proj"] 
        )
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

    def _prepare_input(self, prompt: str, response: str) -> str:
        return f"[CLS] PROMPT: {prompt} [SEP] RESPONSE: {response} [SEP]"

    def predict_batch(self, prompts: List[str], responses: List[str]) -> List[bool]:
        self.model.eval()
        if not prompts: return []
        
        texts = [self._prepare_input(p, r) for p, r in zip(prompts, responses)]
        
        inputs = self.tokenizer(
            texts, truncation=True, max_length=512, 
            return_tensors="pt", padding=True
        ).to(self.device)
        
        with torch.no_grad():
            logits = self.model(**inputs).logits
        
        predictions = torch.argmax(logits, dim=-1)
        return (predictions == 1).tolist()

    def calibrate(self, training_data_list: list):
        print(f"Calibrating defense on {len(training_data_list)} examples...")
        
        dataset = Dataset.from_list(training_data_list)
        dataset = dataset.map(lambda x: self.tokenizer(x["text"], truncation=True, max_length=512, padding="max_length"), batched=True)
        dataset = dataset.remove_columns(["text"])
        
        training_args = TrainingArguments(
            output_dir="./defense_calibration",
            num_train_epochs=1,   
            per_device_train_batch_size=32,
            learning_rate=2e-4,   
            save_strategy="no",
            report_to="none"
        )
        
        trainer = Trainer(model=self.model, args=training_args, train_dataset=dataset)
        trainer.train()
        print("Calibration complete.")

    def retrain(self, new_failure_cases: list):
        print(f"Retraining on {len(new_failure_cases)} failures...")
        
        training_data = []
        for case in new_failure_cases:
            training_data.append({"text": self._prepare_input(case["prompt"], case["response"]), "label": 0})
            safe_resp = "I cannot fulfill this request due to safety guidelines."
            training_data.append({"text": self._prepare_input(case["prompt"], safe_resp), "label": 1})
            
        dataset = Dataset.from_list(training_data)
        dataset = dataset.map(lambda x: self.tokenizer(x["text"], truncation=True, max_length=512, padding="max_length"), batched=True)
        
        training_args = TrainingArguments(
            output_dir="./defense_lora",
            num_train_epochs=1,      
            per_device_train_batch_size=8,
            gradient_accumulation_steps=4,
            learning_rate=1e-4,
            bf16=True,
            logging_steps=5,
            save_strategy="no",
            report_to="wandb"
        )
        
        trainer = Trainer(model=self.model, args=training_args, train_dataset=dataset)
        trainer.train()