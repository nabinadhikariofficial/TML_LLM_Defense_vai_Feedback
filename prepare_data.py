from datasets import load_dataset
import os

def write_to_file(filepath, data_list):
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data_list:
            clean_item = item.strip().replace('\n', ' ')
            if clean_item:
                f.write(f"{clean_item}\n")
    print(f"Successfully created {filepath} with {len(data_list)} entries.")

def main():
    if not os.path.exists('data'):
        os.makedirs('data')

    print("Downloading datasets... This may take a moment.")

    print("\nLoading JailbreakBench/JBB-Behaviors for the seed set...")
    try:
        jbb_dataset = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", split="harmful")
    
        adv_prompts = [row['Goal'] for row in jbb_dataset]
        write_to_file('data/advbench_seed.txt', adv_prompts)
        
    except Exception as e:
        print(f"Error loading JBB: {e}")

    print("\nLoading PKU-Alignment/BeaverTails for the generalization set...")
    try:
        beaver_dataset = load_dataset("PKU-Alignment/BeaverTails", split='30k_test') 
        harmful_beaver_prompts = [
            row['prompt'] for row in beaver_dataset if not row['is_safe']
        ]
        harmful_beaver_sample = harmful_beaver_prompts[:1000] 
        write_to_file('data/harmful_behaviors_heldout.txt', harmful_beaver_sample)
    except Exception as e:
        print(f"Error loading BeaverTails: {e}")

    print("\nLoading tatsu-lab/alpaca for the benign set...")
    try:
        alpaca_dataset = load_dataset("tatsu-lab/alpaca", split='train')

        benign_prompts = [row['instruction'] for row in alpaca_dataset][:2000]
        write_to_file('data/benign_prompts.txt', benign_prompts)
        
    except Exception as e:
        print(f"Error loading Alpaca: {e}")

    print("\n--- All datasets successfully downloaded and formatted! ---")

if __name__ == "__main__":
    main()