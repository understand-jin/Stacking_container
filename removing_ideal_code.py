import pandas as pd
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

#스택 최종 상태 시각화 
def save_stacks_image(stacks, output_path):
    fig, ax = plt.subplots(figsize=(10, 6))

    max_tiers = max(len(stack) for stack in stacks)

    for i, stack in enumerate(stacks):
        for j, weight in enumerate(stack):
            if weight is not None:
                ax.add_patch(plt.Rectangle((i, j), 1, 1, fill=True, edgecolor='black', facecolor = 'skyblue'))
                ax.text(i + 0.5, j + 0.5, f'{weight:.2f}', ha='center', va='center', color='white')

    ax.set_xlim(0, len(stacks))
    ax.set_ylim(-0, max_tiers)
    ax.set_xticks(np.arange(len(stacks)))
    ax.set_xticklabels([f'Stack {i + 1}' for i in range(len(stacks))])
    ax.set_yticks(np.arange(max_tiers) )
    ax.set_yticklabels([f'Tier {i + 1}' for i in (range(max_tiers))])

    plt.grid(which='both', color='grey', linestyle='-', linewidth=0.5)
    plt.title('Final Stack Configuration')
    plt.savefig(output_path)
    plt.close()



#group 값 할당한 new score
def calculate_score(weight, group):

    if isinstance(weight, (list, np.ndarray)):
        scores = []
        for w, g in zip(weight, group):
            if g == 0:
                score = w
            elif g == 1:
                score = w + 100
            elif g ==2:
                score = w + 200
            else:
                score = w + 300
            scores.append(score)
        return scores
    else:
        if group == 0:
            return weight
        elif group == 1:
            return weight + 100
        elif group == 2:
            return weight + 200
        else:
            return weight + 300


        
#output 데이터 프레임 생성
def create_dataframe_from_stacks(container_info):
    data = []
    for info in container_info.values():
        data.append({
            'idx': info['idx'],
            'loc_x': info['loc_x'],
            'loc_y': 0,
            'loc_z': info['loc_z'],
            'weight': info['weight'],
            'seq' : info['seq'],
            'reloc': info['relocations'],
            'size(ft)': info['size'],
            'group' : info['group'],
            'score' : info['score']
        })
    return pd.DataFrame(data)

#input csv 파일 로드하여 초기 상태와 새로운 컨테이너 데이터를 로드 
def load_and_transform_data(initial_state_path, container_path):
    initial_state_df = pd.read_csv(initial_state_path)
    container_df = pd.read_csv(container_path)

    initial_state_weights = initial_state_df['weight'].tolist()
    container_weights = container_df['weight'].tolist()

    new_weight = container_df['weight'].tolist()
    group1 = container_df['group'].tolist()
    score1 = calculate_score(new_weight, group1)
    new_weights = score1
    

    container_info = {}
    for _, row in initial_state_df.iterrows():
        idx = int(row['idx'])
        weight = row['weight']
        loc_x = int(row['loc_x'])
        loc_y = 0
        loc_z = int(row['loc_z'])
        group = int(row['group'])
        size = int(row['size(ft)'])
        score = calculate_score(weight, group)
        new_value = score

        container_info[idx] = {
            'idx': idx,
            'weight': weight,
            'score': score,
            'relocations': 0,
            'loc_x': loc_x,
            'loc_y': loc_y,
            'loc_z': loc_z,
            'seq' : 0,
            'group' : group,
            'size' : size
        }

    for _, row in container_df.iterrows():
        idx = int(row['idx'])
        weight = row['weight']
        seq = int(row['seq'])
        group = int(row['group'])
        size = int(row['size(ft)'])
        score = calculate_score(weight, group)
        new_value = score

        container_info[idx] = {
            'idx': idx,
            'weight': weight,
            'score': score,
            'relocations': 0,
            'loc_x': None,
            'loc_y': None,
            'loc_z': None,
            'seq' : seq,
            'group' : group,
            'size' : size
        }

    stacks = [[None] * 5 for _ in range(6)] # stacks 생성

    for _, row in initial_state_df.iterrows():
        x = int(row['loc_x']) - 1
        z = int(row['loc_z'])
        idx = int(row['idx'])
        stacks[x][z] = container_info[idx]['score']

    return stacks, new_weights, container_info

#무게 차이를 고려한 stacking
def final_relocation_single(stacks, weight, container_info):
    best_stack = -1
    best_difference = float('inf')

    for stack_num in range(len(stacks)):
        for tier_num in range(len(stacks[stack_num])):
            if stacks[stack_num][tier_num] is None:
                if tier_num == 0 or stacks[stack_num][tier_num - 1] is not None:
                    if tier_num > 0:
                        top_weight = stacks[stack_num][tier_num - 1]
                        difference = weight - top_weight
                    else:
                        difference = weight

                    if difference >= 0 and difference < best_difference and not is_peak_stack(stacks, stack_num, tier_num):
                        best_stack = stack_num
                        best_difference = difference

    if best_stack == -1:
        best_stack = 0
        best_difference = float('inf')
        for stack_num in range(len(stacks)):
            for tier_num in range(len(stacks[stack_num])):
                if stacks[stack_num][tier_num] is None:
                    if tier_num == 0 or stacks[stack_num][tier_num - 1] is not None:
                        if tier_num > 0:
                            top_weight = stacks[stack_num][tier_num - 1]
                            difference = weight - top_weight
                        else:
                            difference = weight

                        if abs(difference) < abs(best_difference) and not is_peak_stack(stacks, stack_num, tier_num):
                            best_stack = stack_num
                            best_difference = difference

    for tier_num in range(len(stacks[best_stack])):
        if stacks[best_stack][tier_num] is None:
            stacks[best_stack][tier_num] = weight
            for idx, info in container_info.items():
                if info['idx'] == idx and info['score'] == weight and info['loc_x'] is None:
                    info['loc_x'] = best_stack + 1
                    info['loc_z'] = tier_num
                    print(f"Container index: {idx}, score: {info['score']}, loc_x: {info['loc_x']}, loc_z: {info['loc_z']}")
                    break
            print(f"Placed weight {weight} in Stack {best_stack + 1}, Tier {tier_num + 1}")
            break

#컨테이너 시각화
def print_stacks(stacks):
    for i, stack in enumerate(stacks):
        print(f"Stack {i+1}: {stack}")

def print_weight_positions(weight_positions):
    for weight, positions in weight_positions.items():
        print(f"Weight {weight}: {positions}")

def print_level_positions(level_positions):
    for level, positions in level_positions.items():
        print(f"Level {level}: {positions}")

#기존 컨테이너 재정렬
def relocate_top_containers(stacks, container_info):
    relocation_count = 0

    while True:
        top_weights = []
        for i, stack in enumerate(stacks):
            for tier in range(len(stack) - 1, -1, -1):
                if stack[tier] is not None:
                    top_weights.append((stack[tier], i, tier))
                    break

        top_weights.sort(key=lambda x: (x[0], x[2])) # 무게 같으면 tier 낮은 거 먼저 탐색

        relocated = False
        for weight, stack_num, tier in top_weights:
            best_stack = stack_num
            best_difference = float('inf')

            for target_stack_num in range(len(stacks)):
                if target_stack_num != stack_num:
                    for target_tier in range(len(stacks[target_stack_num])):
                        if stacks[target_stack_num][target_tier] is None:
                            if target_tier == 0 or stacks[target_stack_num][target_tier - 1] is not None:
                                if target_tier > 0:
                                    top_weight = stacks[target_stack_num][target_tier - 1]
                                    difference = weight - top_weight
                                else:
                                    difference = weight

                                if 0 <= difference < best_difference and difference < abs((weight - (stacks[stack_num][tier - 1] if tier > 0 else 0))):
                                    best_stack = target_stack_num
                                    best_difference = difference

                            break

            if best_stack != stack_num:
                for target_tier in range(len(stacks[best_stack])):
                    if stacks[best_stack][target_tier] is None:
                        stacks[best_stack][target_tier] = weight
                        stacks[stack_num][tier] = None
                        relocation_count += 1

                        print(f"Relocated weight {weight} from Stack {stack_num + 1} to Stack {best_stack + 1}")

                        # Update container_info for the relocated container
                        for info in container_info.values():
                            if info['score'] == weight and info['loc_x'] == stack_num + 1 and info['loc_z'] == tier:
                                info['relocations'] += 1
                                info['loc_x'] = best_stack + 1
                                info['loc_z'] = target_tier
                                break

                        relocated = True
                        break

            if relocated:
                break

        if not relocated:
            break

    print(f"Total relocations: {relocation_count}")
    return relocation_count

# 피크스택 확인 함수
def is_peak_stack(stacks, stack_num, tier_num):

    stacks[stack_num][tier_num] = True 
    temp_height = sum(1 for tier in stacks[stack_num] if tier is not None)

    left_stack_height = sum(1 for tier in stacks[stack_num - 1] if tier is not None) if stack_num > 0 else 0
    right_stack_height = sum(1 for tier in stacks[stack_num + 1] if tier is not None) if stack_num < len(stacks) - 1 else 0

    # 임시로 쌓은 컨테이너 제거
    stacks[stack_num][tier_num] = None

    if temp_height - left_stack_height >= 3 and temp_height - right_stack_height >= 3:
        return True
    return False


#컨테이너 배치 과정 총 로직
def container_placement_process(initial_stacks, new_weights, container_info):
    print("Initial stacks before any new weights are added:")
    print_stacks(initial_stacks)
    print("-------------------------------------------------------------------")

    total_relocations = relocate_top_containers(initial_stacks, container_info)
    print("Stacks after relocating initial stacks:")
    print_stacks(initial_stacks)
    print("-------------------------------------------------------------------")


    for i, new_weight in enumerate(new_weights):
        print(f"\nStep {i + 1}: Placing new value {new_weight}")

        final_relocation_single(initial_stacks, new_weight, container_info)

        print_stacks(initial_stacks)

    return total_relocations



def main():
    # 사용자가 지정하는 숫자 변수
    user_defined_number = 23  
    initial_numbers = [3]  
    new_numbers = [20]  

    input_base_dir = f'C:\\Users\\user\\OneDrive\\바탕 화면\\CLT_Data-main\\Ungrouped\\Input_Data_{user_defined_number}(stack_6_tier_5)'
    output_base_dir = f'C:\\Users\\user\\OneDrive\\바탕 화면\\stacking_non_relocation\\Stacking_container\\removing_ideal\\Output_Data_{user_defined_number}(stack_6_tier_5)'
    visual_base_dir = f'C:\\Users\\user\\OneDrive\\바탕 화면\\stacking_non_relocation\\Stacking_container\\removing_ideal\\Output_Data_{user_defined_number}(stack_6_tier_5)'

    if len(initial_numbers) != len(new_numbers):
        print("Error: The lengths of initial_numbers and new_numbers lists do not match.")
        return

    for i in range(len(initial_numbers)):
        initial_num = initial_numbers[i]
        new_num = new_numbers[i]

        input_dir = os.path.join(input_base_dir, f'Initial_{initial_num}', f'New_{new_num}')
        output_dir = os.path.join(output_base_dir, 'Heuristic_1', f'Initial_{initial_num}', f'New_{new_num}')
        visual_dir = os.path.join(visual_base_dir, 'Heuristic_1', f'Initial_{initial_num}', f'New_{new_num}')

        initial_files = sorted(glob.glob(os.path.join(input_dir, 'Initial_state_ex*.csv')))
        container_files = sorted(glob.glob(os.path.join(input_dir, 'Container_ex*.csv')))

        if len(initial_files) != len(container_files):
            print(f"Error: Mismatched number of initial state and container files for configuration Initial_{initial_num} - New_{new_num}.")
            continue

        for j in range(len(initial_files)):
            initial_state_path = initial_files[j]
            container_path = container_files[j]

            # Extract the example number from the initial state file name
            example_num = os.path.basename(initial_state_path).split('_ex')[1].split('.')[0]

            output_file_name = f'Configuration_ex{example_num}.csv'
            output_file_path = os.path.join(output_dir, output_file_name)

            print(f"Processing input files: {os.path.basename(initial_state_path)} and {os.path.basename(container_path)}")

            initial_stacks, new_weights, container_info = load_and_transform_data(initial_state_path, container_path)

            print("Initial stacks before any new weights are added:")
            print_stacks(initial_stacks)
            print("-------------------------------------------------------------------")

            total_relocations = container_placement_process(initial_stacks, new_weights, container_info)
            output = create_dataframe_from_stacks(container_info)

            print(output)
            print(f"Saving output to: {output_file_path}")
            output.to_csv(output_file_path, index=False)

            image_output_path = os.path.join(visual_dir, f'Configuration_{example_num}.png')
            save_stacks_image(initial_stacks, image_output_path)

# 모든 로직 실행
main()

