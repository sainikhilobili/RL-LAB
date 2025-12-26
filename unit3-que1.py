import numpy as np
np.random.seed(0)

grid = np.zeros((5,5))
dirt = [(1,2),(2,3),(4,1)]
obstacles = [(1,1),(3,3)]

for d in dirt: grid[d]=1
for o in obstacles: grid[o]=-1

actions = [(0,1),(1,0),(0,-1),(-1,0)]  # R D L U
theta = np.zeros(4)  # policy parameters
alpha, gamma = 0.1, 0.9

def softmax(x): return np.exp(x)/np.sum(np.exp(x))

for episode in range(200):
    pos=(0,0); traj=[]
    while grid.sum()>0:
        probs = softmax(theta)
        a = np.random.choice(4,p=probs)
        nx = (max(0,min(4,pos[0]+actions[a][0])),
              max(0,min(4,pos[1]+actions[a][1])))
        r = grid[nx]
        if r==1: grid[nx]=0
        traj.append((a,r))
        pos=nx

    G=0
    for a,r in reversed(traj):
        G = r + gamma*G
        theta[a] += alpha*G

print("Learned Policy Probabilities:", softmax(theta))
