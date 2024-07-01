from docplex.mp.model import Model
import numpy as np
import pandas as pd
import os
import glob

model = Model(name = 'IP model')

#Redefine weight considering priority and weight
#if some container have priority value, we just add this priority to max weight of container 
def calculate_score(weight, p):
    scores = []
    w_max = max(weight)
    for weight, p in zip(weight, p):
        if p != 0:
            score = w_max + p
        else:
            score = weight
        scores.append((score))
    return scores

#Calculate level of weight
def calculate_weight_levels(w):
    levels = []  
    interval = 8  # we're gonna use 9level
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

    m = 6
    h = 5
    peak_limit = 2

    weight = initial_data['weight'] + new_data['weight']
    sequence = initial_data['seq'] + new_data['seq']

    initial_n = len(initial_data['weight'])
    n = len(weight)

    alpha = 1
    beta = 1

    ideal_position = get_ideal_positions(weight)
    geometric_center = geometric_best(ideal_position)
    M = 10000
        # Decision Variables
x = model.binary_var_dict([(i,j,k) for i in range(n) for j in range(m) for k in range(h)], name = 'x')
r = model.binary_var_dict([(j,k) for j in range(m) for k in range(h)], name = 'r')

d = model.continuous_var_dict([i for i in range(n)], name = 'd')
d_x = model.continuous_var_dict([i for i in range(n)], name = 'd_x')
d_y = model.continuous_var_dict([i for i in range(n)], name = 'd_y')

# Constraint1 : Container i muse be assigned to ecactly one stack and on tier
for i in range(n):
    model.add_constraint(sum(x[i,j,k] for j in range(m) for k in range(h)) == 1)

# Constraint 2 : one slot can only have one container
for j in range(m):
    for k in range (h):
        model.add_constraint(sum(x[i,j,k] for i in range(n)) <= 1)

# constraint 3 : the hight of stack j must be less than or equal to h
for j in range(m):
    model.add_constraint(sum(x[i,j,k]for k in range(h) for i in range(n)) <= h)

# constraint 4 : you can't stack a container on slot k if there is no container on slot k-1
for j in range(m):
    for k in range(h-1):
        model.add_constraint(sum(x[i,j,k] for i in range(n)) >= sum(x[i,j,k+1] for i in range(n)))

# constraint 5 : define d_i
for i in range(n):
    model.add_constraint(d_x[i] >= sum(x[i,j,k] * j for j in range(m) for k in range(h)) - geometric_center[i][0])
    model.add_constraint(d_x[i] >= -(sum(x[i,j,k] * j for j in range(m) for k in range(h)) - geometric_center[i][0]))
    model.add_constraint(d_y[i] >= sum(x[i,j,k] * k for j in range(m) for k in range(h)) - geometric_center[i][1])
    model.add_constraint(d_y[i] >= -(sum(x[i,j,k] * k for j in range(m) for k in range(h)) - geometric_center[i][1]))
    model.add_constraint(d[i] == d_x[i] + d_y[i])

# Constraint 6 : prevent peak stacks
for j in range(m-1):
    model.add_constraint(sum(x[i,j,k]for k in range(h) for i in range(n)) - sum(x[i,j+1,k] for k in range(h) for i in range(n)) <= l)
    model.add_constraint(sum(x[i,j,k]for k in range(h) for i in range(n)) - sum(x[i,j+1,k] for k in range(h) for i in range(n)) >= - l)

# Constraint 7 : define r_jk
for j in range(m):
    for k in range(h-1):
        for _k in range(k+1, h):
            model.add_constraint((sum(w[i]*x[i,j,k] for i in range(n))-sum(w[i]*x[i,j,_k] for i in range(n)))/M <= M * (1-sum(x[i,j,_k] for i in range(n)))+ r[j,k])
            model.add_constraint(r[j,k] <= M * (1-sum(x[i,j,_k]for i in range(n))) + r[j,_k])
            
# for j in range(m):
#     for k in range(h):
#         model.add_constraint(sum(x[i,j,k]for i in range(n)) >= r[j,k])

for j in range(m):
        for k in range(h-1):
            for _k in range(k+1, h):
                model.add_constraint(sum(sequence[i] * x[i, j, k] for i in range(n)) <= M * (1 - sum(x[i, j, _k] for i in range(n))) + sum(sequence[i] * x[i, j, _k] for i in range(n)))

#Objective function
model.minimize((alpha * sum(r[j,k] for j in range(m) for k in range(h))) + (beta * sum(d[i]for i in range(n))))

model.print_information()
# Solve the model
solution = model.solve()

model.print_solution()

if solution:
    
    for i in range(n):
        for j in range(m):
            for k in range(h):
                if x[i,j,k].solution_value >= 0.99:
                    print(x[i,j,k], ' = ', x[i,j,k].solution_value, ', weight : ',w[i], 'distance : ', d[i].solution_value)
                    print(r[j,k], '=', r[j,k].solution_value)

    
    
else:
    print('No solution found')



#visualization
import matplotlib.pyplot as plt

stacks = []  
tiers = []   
weights = [] 
priority = []
levels = []

for i in range(n):
    for j in range(m):
        for k in range(h):
            
            if x[i, j, k].solution_value >= 0.99:
                stacks.append(j)
                tiers.append(k)
                weights.append(w[i])
                priority.append(p[i])
                levels.append(v[i])

plt.figure(figsize=(10, 6))
plt.scatter(stacks, tiers, c='white', s=100)  

for label, x, y, p, v in zip(weights, stacks, tiers, priority, levels):
    #annotate ; 각 점 근처에 컨테이너 무게 표시하는 텍스트 라벨 
    plt.annotate(f'W {label} ({p, v})', (x, y), textcoords="offset points", xytext=(0,30), ha='center')

plt.grid(True)
plt.xticks([i +0.5 for i in range(m)], [f'Stack {i+1}' for i in range(m)])
plt.yticks(range(h), [f'Tier {i+1}' for i in range(h)]) 
plt.xlim(-0.5, m-0.5 )
plt.ylim(-0, h )
plt.xlabel('Stacks')
plt.ylabel('Tiers')
plt.title('Container Stacking Solution Visualization')
plt.show()

def load_data(initial_state_path, container_path):
    initial_df = pd.read_csv(initial_state_path)
    container_df = pd.read_csv(container_path)

    initial_weight = initial_df['weight'].tolist()
    initial_seq = initial_df['seq'].tolist()
    loc_x = initial_df['loc_x'].tolist()
    loc_z = initial_df['loc_z'].tolist()
    
    initial_data = {
        'weight' : initial_weight,
        'seq' : initial_seq,
        'loc_x' : loc_x,
        'loc_z' : loc_z
    }

    new_weight = container_df['weight'].tolist()
    new_seq = container_df['seq'].tolist()

    new_data = {
        'weight' : new_weight,
        'seq' : new_seq
    }

    return initial_data, new_data




def main():
    input_dir = 'C:\\Users\\user\\OneDrive\\바탕 화면\\MIP_data\\input'
    output_dir = 'C:\\Users\\user\\OneDrive\\바탕 화면\\MIP_data\\output\\trying'

    
    initial_files = sorted(glob.glob(os.path.join(input_dir, 'Initial_state_ex*.csv')))
    container_files = sorted(glob.glob(os.path.join(input_dir, 'Container_ex*.csv')))

    for i in range(len(initial_files)):
        initial_state_path = initial_files[i]
        container_path = container_files[i]
        
        print(initial_state_path)
        print(container_path, '\n')
        # Extract example number from file name
        example_num = os.path.basename(initial_state_path).split('_ex')[1].split('.')[0]

        initial_data, new_data = load_data(initial_state_path, container_path)

        result_file_path = os.path.join(output_dir, f'Configuration_ex{example_num}.csv')
        solution_file_path = os.path.join(output_dir, f'Solution_ex{example_num}.txt')
        plot_file_path = os.path.join(output_dir, f'Visualization_ex{example_num}.png')

        model, solution = solve_model(initial_data, new_data, result_file_path, solution_file_path, plot_file_path)