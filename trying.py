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

positions = generate_positions_diagonal_pattern(10, 6)
print(positions)
