import numpy as np

def parse_input():
    grid_size, tolerance = input().split()
    grid_size = int(grid_size)
    tolerance = float(tolerance)
    
    # Читаем граничные условия для четырех сторон
    left_boundary = list(map(float, input().split()))
    right_boundary = list(map(float, input().split()))
    bottom_boundary = list(map(float, input().split()))
    top_boundary = list(map(float, input().split()))
    
    return grid_size, tolerance, left_boundary, right_boundary, bottom_boundary, top_boundary

def solve_laplace(N, precision, left, right, bottom, top):
    step = 1.0 / N
    potential = np.zeros((N+1, N+1))

    for col in range(N+1):
        potential[0][col] = left[col]         
        potential[N][col] = right[col]        
    
    for row in range(N+1):
        potential[row][0] = bottom[row]    
        potential[row][N] = top[row]    

    previous_state = potential.copy()
    relaxation = 2.0 / (1.0 + np.sin(np.pi / N))  
    
    while True:
        max_error = 0.0
        for row in range(1, N):
            for col in range(1, N):
                updated = 0.25 * (
                    previous_state[row+1][col] + 
                    potential[row-1][col] + 
                    previous_state[row][col+1] + 
                    potential[row][col-1]
                )
                delta = abs(potential[row][col] - updated)
                max_error = max(max_error, delta)
                potential[row][col] = (1 - relaxation)*potential[row][col] + relaxation*updated
        
        if max_error < precision:
            break
        previous_state = potential.copy()

    solution = []
    for col in range(1, N):
        for row in range(1, N):
            solution.append(str(potential[row][col]))
    
    return solution

M, eps, g1, g2, g3, g4 = parse_input()
result = solve_laplace(M, eps, g1, g2, g3, g4)
print(' '.join(result))