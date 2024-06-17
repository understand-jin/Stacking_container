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

def solve_model(initial_data, new_data, result_file_path, solution_file_path, plot_file_path):
    model = Model(name='IP model')

    # Parameters
    m = 6
    h = 5
    peak_limit = 2

    initial_n = len(initial_data['weight'])
    new_n = len(new_data['weight'])
    total_n = initial_n + new_n
    alpha = 0.5
    beta = 0.5

    # Weight of containers
    weight = initial_data['weight'] + new_data['weight']
    group = initial_data['group'] + new_data['group']
    sequence = initial_data['seq'] + new_data['seq']
    emergency = initial_data['emerg'] + new_data['emerg']

    # Compute scores and levels
    first_score = calculate_score(weight, group)
    w_prime = calculate_final_score(first_score, emergency)
    levels = calculate_weight_levels(w_prime)
    ideal_position = get_ideal_positions(w_prime)
    geometric_center = geometric_best(ideal_position)
    M = 1000

    # Decision Variables
    x = model.binary_var_dict([(i, j, k) for i in range(total_n) for j in range(m) for k in range(h)], name='x')
    r = model.binary_var_dict([(j, k) for j in range(m) for k in range(h)], name='r')
    d = model.continuous_var_dict([i for i in range(total_n)], name='d')
    d_x = model.continuous_var_dict([i for i in range(total_n)], name='d_x')
    d_y = model.continuous_var_dict([i for i in range(total_n)], name='d_y')

    # Constraints for initial containers
    for i in range(initial_n):
        j = initial_data['loc_x'][i] - 1
        k = initial_data['loc_z'][i]
        model.add_constraint(x[i, j, k] == 1)

    # Constraint1 : Container i muse be assigned to ecactly one stack and on tier
    for i in range(total_n):
       model.add_constraint(sum(x[i,j,k] for j in range(m) for k in range(h)) == 1)

    # Constraint 2 : one slot can only have one container
    for j in range(m):
        for k in range(h):
            model.add_constraint(sum(x[i, j, k] for i in range(total_n)) <= 1)

    # constraint 3 : the height of stack j must be less than or equal to h
    for j in range(m):
        model.add_constraint(sum(x[i, j, k] for k in range(h) for i in range(total_n)) <= h)

    # constraint 4 : you can't stack a container on slot k if there is no container on slot k-1
    for j in range(m):
        for k in range(h-1):
            model.add_constraint(sum(x[i, j, k] for i in range(total_n)) >= sum(x[i, j, k+1] for i in range(total_n)))

    # constraint 5: define d_i
    for i in range(total_n):
        level = levels[i]
        center_x = geometric_center[level][0]
        center_y = geometric_center[level][1]
        model.add_constraint(d_x[i] >= sum(x[i, j, k] * j for j in range(m) for k in range(h)) - center_x)
        model.add_constraint(d_x[i] >= -(sum(x[i, j, k] * j for j in range(m) for k in range(h)) - center_x))
        model.add_constraint(d_y[i] >= sum(x[i, j, k] * k for j in range(m) for k in range(h)) - center_y)
        model.add_constraint(d_y[i] >= -(sum(x[i, j, k] * k for j in range(m) for k in range(h)) - center_y))
        model.add_constraint(d[i] == d_x[i] + d_y[i])

    # Constraint 6 : prevent peak stacks
    for j in range(m-1):
        model.add_constraint(sum(x[i, j, k] for k in range(h) for i in range(total_n)) - sum(x[i, j+1, k] for k in range(h) for i in range(total_n)) <= peak_limit)
        model.add_constraint(sum(x[i, j, k] for k in range(h) for i in range(total_n)) - sum(x[i, j+1, k] for k in range(h) for i in range(total_n)) >= -peak_limit)

    # Constraint 7 : define r_jk
    for j in range(m):
        for k in range(h-1):
            for _k in range(k+1, h):
                model.add_constraint((sum(w_prime[i] * x[i, j, k] for i in range(total_n)) - sum(w_prime[i] * x[i, j, _k] for i in range(total_n))) / M <= M * (1-sum(x[i, j, _k] for i in range(total_n))) + r[j, k])
                model.add_constraint(r[j, k] <= M * (1-sum(x[i, j, _k] for i in range(total_n))) + r[j, _k])

    for j in range(m):
        for k in range(h):
            model.add_constraint(sum(x[i, j, k] for i in range(total_n)) >= r[j, k])

    for j in range(m):
        for k in range(h-1):
            for _k in range(k+1, h):
                model.add_constraint(sum(sequence[i] * x[i, j, k] for i in range(total_n)) <= M * (1 - sum(x[i, j, _k] for i in range(total_n))) + sum(sequence[i] * x[i, j, _k] for i in range(total_n)))

    for j in range(m):
        for k in range(h-1):
            for _k in range(k+1, h):
                model.add_constraint(sum(emergency[i] * x[i, j, k] for i in range(total_n)) <= M * (1 - sum(x[i, j, _k] for i in range(total_n))) + sum(emergency[i] * x[i, j, _k] for i in range(total_n)))

    # Objective function
    model.minimize((alpha * sum(r[j, k] for j in range(m) for k in range(h))) + (beta * sum(d[i] for i in range(total_n))))

    model.print_information()

    # Solve the model
    solution = model.solve()
    model.print_solution()

    results = []
    if solution:
        with open(solution_file_path, 'w') as f:
            for i in range(total_n):
                for j in range(m):
                    for k in range(h):
                         if x[i, j, k].solution_value >= 0.99:
                            results.append([
                                i + 1,
                                j + 1,
                                0,
                                k,
                                weight[i],
                                group[i],
                                w_prime[i],
                                sequence[i],
                                emergency[i],
                                r[j, k].solution_value,
                                20,
                            ])
                            f.write(f'{x[i, j, k]} = {x[i, j, k].solution_value}, weight: {weight[i]}, w_prime: {w_prime[i]}, distance: {d[i].solution_value}\n')
                            f.write(f'{r[j, k]} = {r[j, k].solution_value}\n')
    else:
        print('No solution found')
        with open(solution_file_path, 'w') as f:
            f.write('No solution found')

    # Convert results to DataFrame
    result_df = pd.DataFrame(results, columns=[
        'idx', 'loc_x', 'loc_y', 'loc_z', 'weight', 'group', 'score', 'seq', 'emerg', 'reloc', 'size(ft)'
    ])

    # Save DataFrame to CSV
    result_df.to_csv(result_file_path, index=False)
    print(f'Results saved to {result_file_path}')

    # Save visualization
    visualize_solution(model, solution, x, weight, group, sequence, calculate_weight_levels(weight), geometric_center, plot_file_path)

    return model, solution, x, r, d, geometric_center


def visualize_solution(model, solution, x, weight, group, sequence, levels, geometric_center, plot_file_path):
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
    
    # Save plot to file
    plt.savefig(plot_file_path)
    plt.close()

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
        'seq': seq,
        'emerg': emergency,
        'loc_x': loc_x,
        'loc_z': loc_z,
        'index': index
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
        'index': new_index
    }
    
    return initial_data, new_data

def main():
    input_dir = 'C:\\Users\\user\\OneDrive\\바탕 화면\\MIP_data\\input\\'
    output_dir = 'C:\\Users\\user\\OneDrive\\바탕 화면\\MIP_data\\output\\'

    initial_files = sorted(glob.glob(os.path.join(input_dir, 'Initial_state_ex*.csv')))
    container_files = sorted(glob.glob(os.path.join(input_dir, 'Container_ex*.csv')))

    for i in range(len(initial_files)):
        initial_state_path = initial_files[i]
        container_path = container_files[i]
        
        # Extract example number from file name
        example_num = os.path.basename(initial_state_path).split('_ex')[1].split('.')[0]

        # Load data
        initial_data, new_data = load_data(initial_state_path, container_path)

        # Result file paths
        result_file_path = os.path.join(output_dir, f'Configuration_ex{example_num}.csv')
        solution_file_path = os.path.join(output_dir, f'Solution_ex{example_num}.txt')
        plot_file_path = os.path.join(output_dir, f'Visualization_ex{example_num}.png')

        # Solve model
        model, solution, x, r, d, geometric_center = solve_model(initial_data, new_data, result_file_path, solution_file_path, plot_file_path)

        # Visualization
        weight = initial_data['weight'] + new_data['weight']
        group = initial_data['group'] + new_data['group']
        sequence = initial_data['seq'] + new_data['seq']
        visualize_solution(model, solution, x, weight, group, sequence, calculate_weight_levels(weight), geometric_center, plot_file_path)


main()