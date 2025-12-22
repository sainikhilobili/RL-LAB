import numpy as np

# Maze: 0=empty, -1=trap, 1=goal
maze = np.array([[0, 0, 0, 1],
                 [0, -1, 0, -1],
                 [0, 0, 0, 0]])
rows, cols = maze.shape

actions = [(-1,0),(1,0),(0,-1),(0,1)]  # Up, Down, Left, Right
V = np.zeros_like(maze, dtype=float)
alpha, gamma, episodes = 0.1, 0.9, 500

for _ in range(episodes):
    state = (2,0)  # start position
    while True:
        r,c = state
        a = actions[np.random.randint(0,4)]
        nr,nc = max(0,min(r+a[0],rows-1)), max(0,min(c+a[1],cols-1))
        reward = maze[nr,nc] if maze[nr,nc]!=0 else 0
        V[r,c] += alpha * (reward + gamma*V[nr,nc] - V[r,c])
        state = (nr,nc)
        if maze[nr,nc] != 0:
            break

# Print final value function
print("Value Function (TD(0)):")
for row in V:
    print(["{0:.2f}".format(v) for v in row])
