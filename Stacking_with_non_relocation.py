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
            elif g == 100:
                score = w + g
            else:
                score = w + g
            scores.append(score)
        return scores
    else:
        if group == 0:
            return weight
        elif group == 100:
            return weight + group
        else:
            return weight + group

#emergency 값 할당
def final_score(scores, emergency):
    if isinstance(scores, list):
        w_max = max(scores)
        final_scores = []
        for score in scores:
            if emergency == 1:
                final_scores.append(w_max)
            else:
                final_scores.append(score)
        return final_scores
    else:
        return w_max if emergency == 1 else scores

        
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
            'emerg' : info['emergency'],
            'reloc': info['relocations'],
            'size(ft)': info['size']
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
    emergency1 = container_df['emerg'].tolist()
    new_weights = final_score(score1, emergency1)
    

    container_info = {}
    for _, row in initial_state_df.iterrows():
        idx = int(row['idx'])
        weight = row['weight']
        loc_x = int(row['loc_x'])
        loc_y = 0
        loc_z = int(row['loc_z'])
        group = int(row['group'])
        emergency = int(row['emerg'])
        size = int(row['size(ft)'])
        score = calculate_score(weight, group)
        new_value = final_score(score, emergency)

        container_info[idx] = {
            'idx': idx,
            'weight': weight,
            'new_value': new_value,
            'relocations': 0,
            'loc_x': loc_x,
            'loc_y': loc_y,
            'loc_z': loc_z,
            'seq' : 0,
            'group' : group,
            'emergency' : emergency,
            'size' : size
        }

    for _, row in container_df.iterrows():
        idx = int(row['idx'])
        weight = row['weight']
        seq = int(row['seq'])
        group = int(row['group'])
        emergency = int(row['emerg'])
        size = int(row['size(ft)'])
        score = calculate_score(weight, group)
        new_value = final_score(score, emergency)

        container_info[idx] = {
            'idx': idx,
            'weight': weight,
            'new_value': new_value,
            'relocations': 0,
            'loc_x': None,
            'loc_y': None,
            'loc_z': None,
            'seq' : seq,
            'group' : group,
            'emergency' : emergency,
            'size' : size
        }

    stacks = [[None] * 5 for _ in range(6)] # stacks 생성

    for _, row in initial_state_df.iterrows():
        x = int(row['loc_x']) - 1
        z = int(row['loc_z'])
        idx = int(row['idx'])
        stacks[x][z] = container_info[idx]['new_value']

    return stacks, new_weights, container_info

#이상적인 형상을 위한 무게레벨 설정(무게 레벌은 1~9)
def calculate_weight_levels(weights):
    levels = []
    interval = 8
    w_min = min(weights)
    w_max = max(weights)
    
    for w_c in weights:
        level = ((w_c - w_min) / (w_max - w_min) * interval) + 1
        levels.append(int(level))  
    
    return levels

def generate_positions_diagonal_pattern(num_stacks, num_tiers):
    positions = []

    for start in range(num_tiers):
        x, y = 0, start
        while y >= 0 and x < num_stacks:
            positions.append((x, y))
            x += 1
            y -= 1

    for start in range(1, num_stacks):
        x, y = start, num_tiers - 1
        while x < num_stacks and y >= 0:
            positions.append((x, y))
            x += 1
            y -= 1

    return positions


#이상적인 형상의 이상적인 좌표설정
def get_ideal_positions(new_weights, stacks):
    weight_levels = calculate_weight_levels(new_weights) #무게레벨 계산

    positions = generate_positions_diagonal_pattern(6, 5)
    occupied_stacks = {i for i, stack in enumerate(stacks) if any(tier is not None for tier in stack)}
    available_positions = [pos for pos in positions if pos[0] not in occupied_stacks]

    weight_positions = {level: [] for level in set(weight_levels)}

    pos_index = 0
    for level in sorted(set(weight_levels)):
        positions_needed = weight_levels.count(level)
        for _ in range(positions_needed):
            if pos_index < len(available_positions):
                weight_positions[level].append(available_positions[pos_index])
                pos_index += 1
            else:
                break

    weight_to_positions = {weight: weight_positions[level] for weight, level in zip(new_weights, weight_levels)}

    return weight_to_positions, weight_positions

#이상적인 좌표에 스택킹 가능하면 이상적인 좌표에 우선적으로 스택킹
def place_container(stacks, weight, positions, container_info):
    placed = False
    for position in positions:
        stack_num, tier_num = position
        if all(stacks[stack_num][i] is not None for i in range(tier_num)) and stacks[stack_num][tier_num] is None:
            stacks[stack_num][tier_num] = weight
            for idx, info in container_info.items():
                if info['idx'] == idx and info['new_value'] == weight and info['loc_x'] is None:
                    info['loc_x'] = stack_num + 1
                    info['loc_z'] = tier_num
                    print(f"Container index: {idx}, new_value: {info['new_value']}, loc_x: {info['loc_x']}, loc_z: {info['loc_z']}")
                    placed = True
                    break  # 내부 for 루프를 종료
            if placed:
                break  # 외부 for 루프를 종료
    return placed

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

                    if difference >= 0 and difference < best_difference:
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

                        if abs(difference) < abs(best_difference):
                            best_stack = stack_num
                            best_difference = difference

    for tier_num in range(len(stacks[best_stack])):
        if stacks[best_stack][tier_num] is None:
            stacks[best_stack][tier_num] = weight
            for idx, info in container_info.items():
                if info['idx'] == idx and info['new_value'] == weight and info['loc_x'] is None:
                    info['loc_x'] = best_stack + 1
                    info['loc_z'] = tier_num
                    print(f"Container index: {idx}, new_value: {info['new_value']}, loc_x: {info['loc_x']}, loc_z: {info['loc_z']}")
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
                            if info['new_value'] == weight and info['loc_x'] == stack_num + 1 and info['loc_z'] == tier:
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

#비어 있는 stack 확인
def get_empty_stacks(stacks):
    empty_stacks = []
    for stack_num, stack in enumerate(stacks):
        if all(tier is None for tier in stack):
            empty_stacks.append(stack_num)
    return empty_stacks

#컨테이너 배치 과정 총 로직
def container_placement_process(initial_stacks, new_weights, original_weights_mapping, container_info):
    print("Initial stacks before any new weights are added:")
    print_stacks(initial_stacks)
    print("-------------------------------------------------------------------")

    total_relocations = relocate_top_containers(initial_stacks, container_info)
    print("Stacks after relocating initial stacks:")
    print_stacks(initial_stacks)
    print("-------------------------------------------------------------------")

    empty_stacks = get_empty_stacks(initial_stacks)
    print(f"Empty stacks after relocation: {empty_stacks}")

    weight_to_positions, level_positions = get_ideal_positions(new_weights, initial_stacks)

    print("Ideal candidate location for new weights")
    print_level_positions(level_positions)
    print("-------------------------------------------------------------------")
    print_weight_positions(weight_to_positions)
    print("-------------------------------------------------------------------")

    for i, new_weight in enumerate(new_weights):
        # actual_weight = original_weights_mapping[new_weight]
        print(f"\nStep {i + 1}: Placing new value {new_weight}")

        ideal_positions = weight_to_positions[new_weight]
        if place_container(initial_stacks, new_weight, ideal_positions, container_info):
            print(f"Placed new value {new_weight}")
        else:
            final_relocation_single(initial_stacks, new_weight, container_info)

        print_stacks(initial_stacks)

    return total_relocations

def main():
    input_dir = 'C:\\Users\\user\\OneDrive\\바탕 화면\\stacking_non_relocation\\Stacking_container\\experiment\\Input_Data_20\\Initial_15\\New_5'
    output_dir = 'C:\\Users\\user\\OneDrive\\바탕 화면\\stacking_non_relocation\\Stacking_container\\experiment\\Output_Data_20\\Heuristic_1\\Initial_15\\New_5'
    visual_dir = 'C:\\Users\\user\\OneDrive\\바탕 화면\\stacking_non_relocation\\Stacking_container\\experiment\\Output_Data_20\\Heuristic_1\\Initial_15\\New_5'

    initial_files = sorted(glob.glob(os.path.join(input_dir, 'Initial_state_ex*.csv')))
    container_files = sorted(glob.glob(os.path.join(input_dir, 'Container_ex*.csv')))

    if len(initial_files) != len(container_files):
        print("Error: Mismatched number of initial state and container files.")
        return

    for i in range(len(initial_files)):
        initial_state_path = initial_files[i]
        container_path = container_files[i]
        
        # Extract the example number from the initial state file name
        example_num = os.path.basename(initial_state_path).split('_ex')[1].split('.')[0]

        output_file_name = f'Configuration_ex{example_num}.csv'
        output_file_path = os.path.join(output_dir, output_file_name)

        print(f"Processing input files: {os.path.basename(initial_state_path)} and {os.path.basename(container_path)}")

        initial_stacks, new_weights, container_info = load_and_transform_data(initial_state_path, container_path)

        print("Initial stacks before any new weights are added:")
        print_stacks(initial_stacks)
        print("-------------------------------------------------------------------")

        total_relocations = container_placement_process(initial_stacks, new_weights, {new_weight: weight for new_weight, weight in zip(new_weights, pd.read_csv(container_path)['weight'].tolist())}, container_info)
        output = create_dataframe_from_stacks(container_info)

        print(output)
        print(f"Saving output to: {output_file_path}")
        output.to_csv(output_file_path, index=False)

        image_output_path = os.path.join(visual_dir, f'Configuration_{example_num}.png')
        save_stacks_image(initial_stacks, image_output_path)

# 모든 로직 실행
main()
