import glob
import os
import pandas as pd

def process_files(input_dir):
    input_files = sorted(glob.glob(os.path.join(input_dir, 'Configuration_ex*.csv')))
    total_relocation = 0
    total_num = len(input_files)

    for input_path in input_files:
        input_df = pd.read_csv(input_path)
        relocation = input_df['reloc'].tolist()
        total_relocation += sum(relocation)

    mean_relocation = total_relocation / total_num if total_num > 0 else 0
    return mean_relocation

def main():
    base_numbers = [27 for _ in range(5)]
    initial_values = [0, 5, 7, 10, 15]
    new_values = [27, 22, 20, 17, 12]

    mean_relocations = []

    for base_number, initial, new in zip(base_numbers, initial_values, new_values):
        input_dir = f'C:\\Users\\user\\OneDrive\\바탕 화면\\stacking_non_relocation\\Stacking_container\\removing_ideal\\Output_Data_{base_number}(stack_6_tier_5)\\Heuristic_1\\Initial_{initial}\\New_{new}'
        
        mean_relocation = process_files(input_dir)
        
        mean_relocations.append({
            'num of container': base_number,
            'Initial': initial,
            'New': new,
            'mean_relocation': mean_relocation
        })

    for result in mean_relocations:
        print(f"Num of container_{result['num of container']} Initial_{result['Initial']} New_{result['New']}: mean_relocation = {result['mean_relocation']}")


main()
