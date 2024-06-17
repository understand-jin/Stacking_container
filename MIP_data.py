from docplex.mp.model import Model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

def calculate_score(weight, g):
    scores = []
    for weight, g in zip(weight, g):
        score = weight + g if g != 0 else weight
        scores.append(score)
    return scores

def calculate_final_score(scores, e):
    final_scores = []
    score_max = max(scores)
    for score, ee in zip(scores, e):
        final_scores.append(score_max if ee != 0 else score)
    return final_scores

def calculate_weight_levels(w):
    levels = []  
    interval = 8
    w_min = min(w)
    w_max = max(w)
    
    for w_c in w:
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

def get_ideal_positions(w_prime):
    weight_levels = calculate_weight_levels(w_prime)
    positions = generate_positions_diagonal_pattern(6, 5)
    weight_positions = {level: [] for level in set(weight_levels)}
    pos_index = 0
    for level in sorted(set(weight_levels)):
        positions_needed = weight_levels.count(level)
        for _ in range(positions_needed):
            weight_positions[level].append(positions[pos_index])
            pos_index += 1
    return weight_positions

def geometric_best(ideal_positions):
    geometric_center = {}
    for level, positions in ideal_positions.items():
        if positions:
            a = sum(pos[0] for pos in positions) / len(positions)
            b = sum(pos[1] for pos in positions) / len(positions)
            geometric_center[level] = (a, b)
    return geometric_center

def solve_model(weight, group, sequence, emergency):
    model = Model(name='IP model')

    # Parameters
    m = 6
    h = 5
    peak_limit = 2

    # Compute scores and levels
    first_score = calculate_score(weight, group)
    w_prime = calculate_final_score(first_score, emergency)
    n = len(weight)
    alpha = 0.5
    beta = 0.5
    levels = calculate_weight_levels(w_prime)
    ideal_position = get_ideal_positions(w_prime)
    geometric_center = geometric_best(ideal_position)
    M = 1000

    # Decision Variables
    x = model.binary_var_dict([(i, j, k) for i in range(n) for j in range(m) for k in range(h)], name='x')
    r = model.binary_var_dict([(j, k) for j in range(m) for k in range(h)], name='r')
    d = model.continuous_var_dict([i for i in range(n)], name='d')
    d_x = model.continuous_var_dict([i for i in range(n)], name='d_x')
    d_y = model.continuous_var_dict([i for i in range(n)], name='d_y')

    # Constraints
    for i in range(n):
        model.add_constraint(sum(x[i, j, k] for j in range(m) for k in range(h)) == 1)

    for j in range(m):
        for k in range(h):
            model.add_constraint(sum(x[i, j, k] for i in range(n)) <= 1)

    for j in range(m):
        model.add_constraint(sum(x[i, j, k] for k in range(h) for i in range(n)) <= h)

    for j in range(m):
        for k in range(h-1):
            model.add_constraint(sum(x[i, j, k] for i in range(n)) >= sum(x[i, j, k+1] for i in range(n)))

    for i in range(n):
        level = levels[i]
        center_x = geometric_center[level][0]
        center_y = geometric_center[level][1]
        model.add_constraint(d_x[i] >= sum(x[i, j, k] * j for j in range(m) for k in range(h)) - center_x)
        model.add_constraint(d_x[i] >= -(sum(x[i, j, k] * j for j in range(m) for k in range(h)) - center_x))
        model.add_constraint(d_y[i] >= sum(x[i, j, k] * k for j in range(m) for k in range(h)) - center_y)
        model.add_constraint(d_y[i] >= -(sum(x[i, j, k] * k for j in range(m) for k in range(h)) - center_y))
        model.add_constraint(d[i] == d_x[i] + d_y[i])

    for j in range(m-1):
        model.add_constraint(sum(x[i, j, k] for k in range(h) for i in range(n)) - sum(x[i, j+1, k] for k in range(h) for i in range(n)) <= peak_limit)
        model.add_constraint(sum(x[i, j, k] for k in range(h) for i in range(n)) - sum(x[i, j+1, k] for k in range(h) for i in range(n)) >= -peak_limit)

    for j in range(m):
        for k in range(h-1):
            for _k in range(k+1, h):
                model.add_constraint((sum(w_prime[i]*x[i, j, k] for i in range(n)) - sum(w_prime[i]*x[i, j, _k] for i in range(n))) / M <= M * (1-sum(x[i, j, _k] for i in range(n))) + r[j, k])
                model.add_constraint(r[j, k] <= M * (1-sum(x[i, j, _k] for i in range(n))) + r[j, _k])

    for j in range(m):
        for k in range(h):
            model.add_constraint(sum(x[i, j, k] for i in range(n)) >= r[j, k])

    for j in range(m):
        for k in range(h-1):
            for _k in range(k+1, h):
                model.add_constraint(sum(sequence[i] * x[i, j, k] for i in range(n)) <= M * (1 - sum(x[i, j, _k] for i in range(n))) + sum(sequence[i] * x[i, j, _k] for i in range(n)))

    for j in range(m):
        for k in range(h-1):
            for _k in range(k+1, h):
                model.add_constraint(sum(emergency[i] * x[i, j, k] for i in range(n)) <= M * (1 - sum(x[i, j, _k] for i in range(n))) + sum(emergency[i] * x[i, j, _k] for i in range(n)))

    # Objective function
    model.minimize((alpha * sum(r[j, k] for j in range(m) for k in range(h))) + (beta * sum(d[i] for i in range(n))))

    model.print_information()

    # Solve the model
    solution = model.solve()
    model.print_solution()

    if solution:
        for i in range(n):
            for j in range(m):
                for k in range(h):
                    if x[i, j, k].solution_value >= 0.99:
                        print(x[i, j, k], ' = ', x[i, j, k].solution_value, ', weight : ', weight[i], ', w_prime :', w_prime[i], 'distance : ', d[i].solution_value)
                        print(r[j, k], '=', r[j, k].solution_value)
    else:
        print('No solution found')

    return model, solution, x, r, d, geometric_center

def visualize_solution(model, solution, x, weight, group, sequence, levels, geometric_center):
    n = len(weight)
    m = 6
    h = 5
    stacks = []  
    tiers = []   
    weights = [] 
    priority = []
    level_list = []

    for i in range(n):
        for j in range(m):
            for k in range(h):
                if x[i, j, k].solution_value >= 0.99:
                    stacks.append(j)
                    tiers.append(k)
                    weights.append(weight[i])
                    priority.append(group[i])
                    level_list.append(levels[i])

    plt.figure(figsize=(10, 6))
    plt.scatter(stacks, tiers, c='white', s=100)  

    for label, x, y, p, v in zip(weights, stacks, tiers, priority, level_list):
        plt.annotate(f'W {label} ({p, v})', (x, y), textcoords="offset points", xytext=(0, 30), ha='center')

    plt.grid(True)
    plt.xticks([i + 0.5 for i in range(m)], [f'Stack {i+1}' for i in range(m)])
    plt.yticks(range(h), [f'Tier {i+1}' for i in range(h)]) 
    plt.xlim(-0.5, m - 0.5)
    plt.ylim(-0, h)
    plt.xlabel('Stacks')
    plt.ylabel('Tiers')
    plt.title('Container Stacking Solution Visualization')
    plt.show()

def load_data(initial_state_path, container_path):
    initial_df = pd.read_csv(initial_state_path)
    container_df = pd.read_csv(container_path)
    
    # 필요한 열만 선택
    initial_weight = initial_df['weight'].tolist()
    group = initial_df['group'].tolist()
    emergency = initial_df['emerg'].tolist()
    seq = [0] * len(initial_weight)
    loc_x = initial_df['loc_x'].tolist()
    loc_z = initial_df['loc_z'].tolist()
    index = initial_df['idx'].tolist()

    # 초기 상태의 컨테이너 정보
    initial_data = {
        'weight': initial_weight,
        'group': group,
        'seq' : seq,
        'emerg': emergency,
        'loc_x' : loc_x,
        'loc_z' : loc_z,
        'index' : index
    }
    
    # 새로운 컨테이너 정보
    new_weight = container_df['weight'].tolist()
    new_group = container_df['group'].tolist()
    new_sequence = container_df['seq'].tolist()
    new_emergency = container_df['emerg'].tolist()
    new_index = container_df['idx'].tolist()
    
    new_data = {
        'weight': new_weight,
        'group': new_group,
        'seq': new_sequence,
        'emerg': new_emergency,
        'index' : new_index
    }
    
    return initial_data, new_data

# 파일 경로 설정
input_dir = 'C:\\Users\\user\\OneDrive\\바탕 화면\\MIP_data\\input\\'
output_dir = 'C:\\Users\\user\\OneDrive\\바탕 화면\\MIP_data\\output\\'

initial_files = sorted(glob.glob(os.path.join(input_dir, 'Initial_state_ex*.csv')))
container_files = sorted(glob.glob(os.path.join(input_dir, 'Container_ex*.csv')))

for i in range(len(initial_files)):
    initial_state_path = initial_files[i]
    container_path = container_files[i]



def main():
    # 데이터 로드
    initial_data, new_data = load_data(initial_state_path, container_path)
    
    # 초기 데이터와 새로운 데이터 병합
    weight = initial_data['weight'] + new_data['weight']
    group = initial_data['group'] + new_data['group']
    sequence = initial_data['seq'] + new_data['seq'] 
    emergency = initial_data['emerg'] + new_data['emerg']

    # 모델 해결
    model, solution, x, r, d, geometric_center = solve_model(weight, group, sequence, emergency)

    # 솔루션 시각화
    visualize_solution(model, solution, x, weight, group, sequence, calculate_weight_levels(weight), geometric_center)

if __name__ == "__main__":
    main()

