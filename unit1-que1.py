import random

# 0=empty, 1=dirt(+1), -1=obstacle(blocked)
grid = [
    [0, 1, 0, 0, 1],
    [0,-1, 1, 0, 0],
    [1, 0, 0,-1, 1],
    [0, 1, 0, 0, 0],
    [1, 0,-1, 1, 0]
]

actions = [(0,1),(1,0),(0,-1),(-1,0)]  # R,D,L,U
x = y = 0
reward = 0
steps = 0

def dirt_left():
    return any(1 in row for row in grid)

while dirt_left():
    dx, dy = random.choice(actions)
    nx, ny = x+dx, y+dy

    # ✔ Move only if inside grid AND not obstacle
    if 0 <= nx < 5 and 0 <= ny < 5 and grid[nx][ny] != -1:
        x, y = nx, ny
        if grid[x][y] == 1:
            reward += 1
            grid[x][y] = 0  # clean dirt
    steps += 1

print("All dirt cleaned!")
print("Total Steps:", steps)
print("Total Reward:", reward)
print("Final Position:", (x,y))
print("Final Grid:")
for r in grid: print(r)
