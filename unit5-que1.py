import numpy as np
import matplotlib.pyplot as plt

grid = np.zeros((10,10))
obstacles = [(3,3),(3,4),(4,3),(6,6),(6,7)]
for o in obstacles:
    grid[o] = -1

start, goal = (0,0), (9,9)
pos = start
path = [pos]

# High-level route (hierarchical decision)
route = [(0,9),(9,9)]

for sg in route:
    while pos != sg:
        x,y = pos
        gx,gy = sg

        if y < gy: nxt = (x,y+1)
        elif x < gx: nxt = (x+1,y)
        else: break

        if grid[nxt] != -1:
            pos = nxt
            path.append(pos)

print("Reached Goal:", pos == goal)
print("Total Steps:", len(path))

plt.imshow(grid, cmap='gray_r')
px,py = zip(*path)
plt.plot(py,px,'b-o')
plt.scatter(goal[1],goal[0],c='green',s=120)
plt.title("Hierarchical Vehicle Navigation")
plt.show()
