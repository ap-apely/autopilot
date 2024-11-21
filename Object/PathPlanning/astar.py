import heapq
import math
import numpy as np

# Define the heuristic function using diagonal distance
def diagonal_distance(current, goal):
    dx = abs(current[0] - goal[0])
    dy = abs(current[1] - goal[1])
    return max(dx, dy) + (math.sqrt(2) - 1) * min(dx, dy)

# Define the A* algorithm function
def a_star(start, goal, grid):
    # Initialize the open and closed sets
    open_set = [(0, start)]
    closed_set = set()

    # Initialize the cost and parent dictionaries
    cost = {start: 0}
    parent = {start: None}

    # Loop until the open set is empty
    while open_set:
        # Pop the node with the lowest cost from the open set
        current_cost, current = heapq.heappop(open_set)

        # If the current node is the goal, we have found a path
        if current == goal:
            path = []
            while current:
                path.append(current)
                current = parent[current]
            return path[::-1]

        # Add the current node to the closed set
        closed_set.add(current)

        # Loop through the neighbors of the current node
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
            neighbor = (current[0] + dx, current[1] + dy)

            # Skip if the neighbor is not in the grid or is an obstacle or is in the closed set
            if not (0 <= neighbor[0] < grid.shape[0] and 0 <= neighbor[1] < grid.shape[1]) or grid[neighbor[0]][neighbor[1]] == 1 or neighbor in closed_set:
                continue

            # Calculate the tentative cost to reach the neighbor
            tentative_cost = cost[current] + math.sqrt(dx ** 2 + dy ** 2)

            # If the neighbor is not in the open set or the tentative cost is lower than the existing cost,
            # update the cost and parent dictionaries and add the neighbor to the open set
            if neighbor not in cost or tentative_cost < cost[neighbor]:
                cost[neighbor] = tentative_cost
                priority = tentative_cost + diagonal_distance(neighbor, goal)
                heapq.heappush(open_set, (priority, neighbor))
                parent[neighbor] = current

    # If we reach here, there is no path to the goal
    return None

def update_grid(grid, coord, length):
    row, col = coord
    # Create a mask for circular region
    y, x = np.ogrid[-length:length+1, -length:length+1]
    mask = x*x + y*y <= length*length
    
    # Calculate bounds for the region to update
    row_start = max(0, row - length)
    row_end = min(grid.shape[0], row + length + 1)
    col_start = max(0, col - length)
    col_end = min(grid.shape[1], col + length + 1)
    
    # Calculate the corresponding section of the mask
    mask_row_start = max(0, length - row)
    mask_row_end = mask.shape[0] - max(0, row + length + 1 - grid.shape[0])
    mask_col_start = max(0, length - col)
    mask_col_end = mask.shape[1] - max(0, col + length + 1 - grid.shape[1])
    
    # Update the grid using the mask
    grid[row_start:row_end, col_start:col_end] |= mask[mask_row_start:mask_row_end, mask_col_start:mask_col_end]
    
    return grid

def path_planning(obs):
    grid_size = 500
    grid = np.zeros((grid_size, grid_size), dtype=np.uint8)
    
    # Update grid for all obstacles at once
    for i in obs:
        grid = update_grid(grid, (i[0], i[1]), 50)
    
    start = (grid_size - 1, grid_size // 2)
    goal = (0, grid_size // 2)
    
    path = a_star(start, goal, grid)
    # for i in path:
    #     grid[i[0]][i[1]]=7
    # for i in grid:
    #     print(i)
    # if path:
    #     print("Path found:", path)
    # else:
    #     print("error")
    return path
