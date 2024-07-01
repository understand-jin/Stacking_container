import glob
import os
import pandas as pd

def main():
    input_dir = 'C:\\Users\\user\\OneDrive\\바탕 화면\\stacking_non_relocation\\Stacking_container\\experiment\\Output_Data_25\\Heuristic_2\\Initial_15,New_10'
    input_files = sorted(glob.glob(os.path.join(input_dir, 'Relocation_ex*.csv')))

    total_num = len(input_files)
    total_relocation = 0

    for input_path in input_files:
        input_df = pd.read_csv(input_path)
        input_df = input_df.sort_values(by=['loc_x', 'loc_z'])

        print(f'input Path : {input_path}')

        stacks = {loc_x: [None] * 5 for loc_x in range(1,7)}

        # DataFrame을 리스트로 변환
        loc_x_list = input_df['loc_x'].tolist()
        loc_z_list = input_df['loc_z'].tolist()
        weight_list = input_df['weight'].tolist()

        # 각 loc_x와 loc_z에 맞게 스택에 weight 값을 추가
        for loc_x, loc_z, weight in zip(loc_x_list, loc_z_list, weight_list):
            stacks[loc_x][loc_z] = weight

        # 각 스택의 내용을 출력
        for loc_x, stack in stacks.items():
            print(f"Stack at loc_x = {loc_x}: {stack}")

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
           

        for loc_x, stack in relocation.items():
            print(f'relocation at loc_x = {loc_x} : {stack}')

            relocation = sum(1 for r in stack if r == 1)
            print(f"Total number of relocations required: {relocation}")

            total_relocation = total_relocation + relocation
            print(f'total_relocation{total_relocation}')

    mean_relocation = total_relocation / total_num
    print(f'total_relocation : {total_relocation}')
    print(f'total_num : {total_num}')
    print(f'mean_relocation : {mean_relocation}')

main()

        

