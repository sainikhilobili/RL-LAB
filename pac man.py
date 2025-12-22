import numpy as np

# Grid: 0=empty, 1=food, -1=ghost
grid = np.array([[0, 1, 0],
                 [0,-1, 0],
                 [1, 0, 0]])
rows, cols = grid.shape

actions = [(-1,0),(1,0),(0,-1),(0,1)]  # Up, Down, Left, Right
Q = np.zeros((rows, cols, len(actions)))
alpha, gamma, epsilon, episodes = 0.1, 0.9, 0.2, 500

def choose_action(s):
    r,c = s
    if np.random.rand() < epsilon:
        return np.random.randint(0,4)
    return np.argmax(Q[r,c])

for _ in range(episodes):
    state = (2,0)  # start position
    while True:
        r,c = state
        a = choose_action(state)
        dr, dc = actions[a]
        nr, nc = max(0,min(r+dr,rows-1)), max(0,min(c+dc,cols-1))
        reward = grid[nr,nc]
        Q[r,c,a] += alpha * (reward + gamma * np.max(Q[nr,nc]) - Q[r,c,a])
        state = (nr,nc)
        if grid[nr,nc]!=0:
            break

# Print learned policy
dir_map = ['↑','↓','←','→']
print("Learned Policy:")
for r in range(rows):
    row_policy = []
    for c in range(cols):
        if grid[r,c]==-1:
            row_policy.append('G')   # Ghost
        elif grid[r,c]==1:
            row_policy.append('F')   # Food
        else:
            row_policy.append(dir_map[np.argmax(Q[r,c])])
    print(row_policy)
