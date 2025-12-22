import numpy as np

# Grid: 0=empty, -1=obstacle, 1=dirty
grid_template = np.array([[1, 0, 1],
                          [0,-1, 0],
                          [1, 0, 1]])
rows, cols = grid_template.shape

actions = [(-1,0),(1,0),(0,-1),(0,1)]  # Up, Down, Left, Right
Q = np.zeros((rows, cols, len(actions)))
alpha, gamma, epsilon, episodes = 0.1, 0.9, 0.2, 500

def choose_action(s):
    r,c = s
    if np.random.rand() < epsilon:
        return np.random.randint(0,4)
    return np.argmax(Q[r,c])

for _ in range(episodes):
    grid = grid_template.copy()   # reset dirty cells each episode
    state = (0,0)
    a = choose_action(state)
    while True:
        r,c = state
        dr, dc = actions[a]
        nr, nc = max(0,min(r+dr,rows-1)), max(0,min(c+dc,cols-1))
        reward = 1 if grid[nr,nc]==1 else -0.1 if grid[nr,nc]==-1 else 0
        if grid[nr,nc]==1:
            grid[nr,nc] = 0  # cleaned
        na = choose_action((nr,nc))
        Q[r,c,a] += alpha * (reward + gamma*Q[nr,nc,na] - Q[r,c,a])
        state, a = (nr,nc), na
        if np.all(grid != 1):  # all cleaned
            break

# Print learned policy
dir_map = ['↑','↓','←','→']
print("Learned Policy:")
for r in range(rows):
    row_policy = []
    for c in range(cols):
        if grid_template[r,c]==-1:
            row_policy.append('X')
        else:
            row_policy.append(dir_map[np.argmax(Q[r,c])])
    print(row_policy)
