import glob
import os
import pandas as pd

def process_files(input_dir):
    input_files = sorted(glob.glob(os.path.join(input_dir, 'Configuration_ex*.csv')))

    total_num = len(input_files)
    total_relocation = 0

    for input_path in input_files:
        input_df = pd.read_csv(input_path)
        input_df = input_df.sort_values(by=['loc_x', 'loc_z'])

        stacks = {loc_x: [None] * 5 for loc_x in range(1,7)}

        # DataFrame을 리스트로 변환
        loc_x_list = input_df['loc_x'].tolist()
        loc_z_list = input_df['loc_z'].tolist()
        weight_list = input_df['weight'].tolist()

        # 각 loc_x와 loc_z에 맞게 스택에 weight 값을 추가
        for loc_x, loc_z, weight in zip(loc_x_list, loc_z_list, weight_list):
            stacks[loc_x][int(loc_z)] = weight

        relocation = {loc_x : [0] * 5 for loc_x in range(1,7)}
        for loc_x, stack in stacks.items():
            for tier in range(len(stack)):
                if stack[tier] is not None:
                    for upper_tier in range(tier + 1, len(stack)):
                        if stack[upper_tier] is not None and stack[tier] > stack[upper_tier]:
                            relocation[loc_x][tier] = 1

                            for t in range(tier + 1, len(stack)):
                                if stack[t] is not None:
                                    relocation[loc_x][t] = 1

        relocation_count = sum(1 for stack in relocation.values() for r in stack if r == 1)
        total_relocation += relocation_count

    mean_relocation = total_relocation / total_num if total_num > 0 else 0
    return mean_relocation

def main():
    base_numbers = [23 for _ in range(5)]  # 원하는 숫자로 변경
    initial_values = [0, 5, 7, 10, 15]  # Initial 숫자 리스트
    new_values = [23, 18, 16, 13, 8]  # New 숫자 리스트

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

    # 모든 mean_relocation 출력
    for result in mean_relocations:
        print(f"Num of container_{result['num of container']} Initial_{result['Initial']} New_{result['New']}: mean_relocation = {result['mean_relocation']}")

# 모든 로직 실행
main()

