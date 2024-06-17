from docplex.mp.model import Model
import pandas as pd
import numpy as np
import os
import glob


model = Model(name = 'IP model')

#Redefine weight considering priority and weight
#if some container have priority value, we just add this priority to max weight of container 
def calculate_score(weight, g):
    scores = []
    for weight, g in zip(weight, g):
        if g != 0:
            score = weight + g
        else:
            score = weight
        scores.append(score)
    return scores

def calculate_final_score(scores, e):
    final_scores = []
    score_max = max(scores)
    for score, ee in zip(scores, e ):
        if ee != 0:
            score = score_max 
        else:
            score = score
        final_scores.append(score)
    return final_scores

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

    weight_positions = {level : [] for level in set(weight_levels)}

    pos_index = 0
    for level in sorted(set(weight_levels)):
        positions_needed = weight_levels.count(level)
        for _ in range(positions_needed):
            weight_positions[level].append(positions[pos_index])
            pos_index += 1
    
    return weight_positions

def geometric_best(ideal_positions):
    geometric_center ={}
    for level, positions in ideal_positions.items():
        if positions:
            a = sum(pos[0] for pos in positions) / len(positions)
            b = sum(pos[1] for pos in positions) / len(positions)
            geometric_center[level] = (a, b)
    return geometric_center

        

# Parameters

# Number of stacks
m = 6

# Capacity of tiers
h = 5

# limit of peak stacks
l = 2

peak_limit = 2

# Weight of containers
weight = [100,200,300,400,500,600,700,800,900,100,100]

#Priority of containers(0~10)
# 0 means we don't care sequence of this container.
# 10 means this container have to pick up firstly as the highest priority
g = [0, 0, 0, 0, 0, 0, 0, 0, 0 ,0 ,0 ,0]

s = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

e = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

#Container's new weight which consider priority
first_score = calculate_score(weight, g)
w_prime = calculate_final_score(first_score, e)

# Number of Containers
n = len(weight)

# weight of objective function
alpha = 0.5
beta = 0.5

#Weight of level
levels = calculate_weight_levels(w_prime)

ideal_position = get_ideal_positions(w_prime)
#Geometric centers
geometric_center = geometric_best(ideal_position)

#BIg M
M = 1000

# Decision Variables
x = model.binary_var_dict([(i,j,k) for i in range(n) for j in range(m) for k in range(h)], name = 'x')
r = model.binary_var_dict([(j,k) for j in range(m) for k in range(h)], name = 'r')

d = model.continuous_var_dict([i for i in range(n)], name = 'd')
d_x = model.continuous_var_dict([i for i in range(n)], name = 'd_x')
d_y = model.continuous_var_dict([i for i in range(n)], name = 'd_y')

# # Constraint 0 :  Define w_prime
# w_max = max(weight)
# w_prime = []
# for i in range(n):
#     w_prime.append(w_max + p[i])

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

# constraint 5: define d_i
for i in range(n):
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
    model.add_constraint(sum(x[i,j,k]for k in range(h) for i in range(n)) - sum(x[i,j+1,k] for k in range(h) for i in range(n)) <= peak_limit)
    model.add_constraint(sum(x[i,j,k]for k in range(h) for i in range(n)) - sum(x[i,j+1,k] for k in range(h) for i in range(n)) >= - peak_limit)

# Constraint 7 : define r_jk
for j in range(m):
    for k in range(h-1):
        for _k in range(k+1, h):
            model.add_constraint((sum(w_prime[i]*x[i,j,k] for i in range(n))-sum(w_prime[i]*x[i,j,_k] for i in range(n)))/M <= M * (1-sum(x[i,j,_k] for i in range(n)))+ r[j,k])
            model.add_constraint(r[j,k] <= M * (1-sum(x[i,j,_k]for i in range(n))) + r[j,_k])
            
for j in range(m):
    for k in range(h):
        model.add_constraint(sum(x[i,j,k]for i in range(n)) >= r[j,k])

# Constraint 8 : Sequence
for j in range(m):
    for k in range(h-1):
        for _k in range(k+1, h):
            model.add_constraint(sum(s[i] * x[i,j,k]for i in range(n)) <= M * (1 - sum(x[i,j,_k]for i in range (n))) + sum(s[i]*x[i,j,_k]for i in range (n)))

# Constraint 9 : Emergency
for j in range(m):
    for k in range(h-1):
        for _k in range(k+1, h):
            model.add_constraint(sum(e[i] * x[i,j,k] for i in range(n)) <= M * (1 - sum(x[i,j,_k]for i in range(n))) + sum(e[i] + x[i,j,_k]for i in range(n)))

# Initial Container location setting



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
                    print(x[i,j,k], ' = ', x[i,j,k].solution_value, ', weight : ',weight[i], ', w_prime :', w_prime[i],  'distance : ', d[i].solution_value)
                    print(r[j,k], '=', r[j,k].solution_value)

    
    
else:
    print('No solution found')

print(geometric_center)

#visualization
import matplotlib.pyplot as plt

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
                priority.append(g[i])
                level_list.append(levels[i])

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