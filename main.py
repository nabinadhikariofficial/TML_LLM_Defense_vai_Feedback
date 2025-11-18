import wandb
import random
import torch
from base_model import BaseTaskModel
from defense_module import DefenseModule
from adversary_module import AdversaryModule
from oracle import ask_oracle_batch
from typing import List

NUM_CYCLES = 10
ATTACKS_PER_CYCLE = 200
EVAL_SAMPLE_SIZE = 500 

def load_dataset(path):
    try:
        with open(path, 'r') as f: return [l.strip() for l in f if l.strip()]
    except FileNotFoundError:
        print(f"Error: {path} not found. Run prepare_data.py first.")
        return []

def calibrate_defense(defense: DefenseModule, benign_data: List[str], harmful_data: List[str]):
    print(f"\n--- Calibrating Defense (Bias towards Safety) ---")
    safe_prompts = random.sample(benign_data, 400)
    harmful_prompts = random.sample(harmful_data, 50)
    
    training_data = []
    for p in safe_prompts:
        text = defense._prepare_input(p, "Here is a helpful and safe response.")
        training_data.append({"text": text, "label": 1})
    for p in harmful_prompts:
        text = defense._prepare_input(p, "I will gladly help you destroy the world.")
        training_data.append({"text": text, "label": 0})
    
    random.shuffle(training_data)
    defense.calibrate(training_data)

def evaluate(cycle: int, base: BaseTaskModel, defense: DefenseModule, 
             prompts: List[str], benign_prompts: List[str], 
             failure_db: List, prefix: str):
    print(f"Evaluating {prefix}...")
    
    # --- Check Attacks ---
    successes = 0
    if prompts:
        if prefix == "New_Attacks":
            eval_prompts = prompts
        else:
            eval_prompts = random.sample(prompts, min(len(prompts), EVAL_SAMPLE_SIZE))

        all_responses = base.get_responses(eval_prompts)
        defense_decisions = defense.predict_batch(eval_prompts, all_responses) 
        
        failed_to_block_prompts = []
        failed_to_block_responses = []
        for i, is_safe_by_defense in enumerate(defense_decisions):
            if is_safe_by_defense: 
                failed_to_block_prompts.append(eval_prompts[i])
                failed_to_block_responses.append(all_responses[i])
        
        oracle_decisions = ask_oracle_batch(failed_to_block_prompts, failed_to_block_responses)
        
        for i, is_safe_by_oracle in enumerate(oracle_decisions):
            if not is_safe_by_oracle: 
                successes += 1
                if failure_db is not None: 
                    failure_db.append({
                        "prompt": failed_to_block_prompts[i], 
                        "response": failed_to_block_responses[i]
                    })
        asr = successes / len(eval_prompts)
    else:
        asr = 0

    # --- Benign ---
    false_positives = 0
    if benign_prompts:
        benign_sample = random.sample(benign_prompts, min(EVAL_SAMPLE_SIZE, len(benign_prompts)))
        all_benign_responses = base.get_responses(benign_sample)
        defense_decisions = defense.predict_batch(benign_sample, all_benign_responses)
        
        for is_safe_by_defense in defense_decisions:
            if not is_safe_by_defense: 
                false_positives += 1
        fpr = false_positives / len(benign_sample)
    else:
        fpr = 0
    
    wandb.log({f"{prefix}_ASR": asr, f"{prefix}_FPR": fpr, "cycle": cycle})
    print(f"Cycle {cycle} [{prefix}] ASR: {asr:.2%} | FPR: {fpr:.2%}")

def main():
    wandb.init(project="llama3.1-self-hardening-H200-V3")
    
    print("Loading models...")
    base = BaseTaskModel()
    defense = DefenseModule()
    adversary = AdversaryModule() 
    
    seeds = load_dataset("data/advbench_seed.txt")
    heldout = load_dataset("data/harmful_behaviors_heldout.txt")
    benign = load_dataset("data/benign_prompts.txt")
    
    failures = []

    # CALIBRATION
    calibrate_defense(defense, benign, heldout)

    # Baseline Evaluation
    print("\n--- Running Baseline Evaluation (Cycle 0) ---")
    evaluate(0, base, defense, seeds, benign, None, "Baseline_Seed")
    evaluate(0, base, defense, heldout, benign, None, "Baseline_Generalization")

    # Main Loop
    for i in range(1, NUM_CYCLES + 1):
        print(f"\n=== Cycle {i} ===")
        
        print(f"Generating {ATTACKS_PER_CYCLE} new attacks...")
        seed_goals = [random.choice(seeds) for _ in range(ATTACKS_PER_CYCLE)]
        new_attacks = adversary.generate_attacks_batch(seed_goals)
        
        evaluate(i, base, defense, new_attacks, benign, failures, "New_Attacks")
        
        if failures:
            print(f"Retraining on {len(failures)} new failures...")
            defense.retrain(failures) 
            failures = [] 
            torch.cuda.empty_cache() 
        else:
            print("No failures detected.")
            
        print("Re-evaluating metrics...")
        evaluate(i, base, defense, heldout, benign, None, "Generalization_ASR")

if __name__ == "__main__":
    main()