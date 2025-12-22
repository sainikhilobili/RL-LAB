import numpy as np

G=[[0,0,0,0,5],
   [0,-1,0,0,0],
   [0,0,0,-1,0],
   [0,0,0,0,5],
   [0,0,0,0,0]]

A=[(-1,0),(1,0),(0,-1),(0,1)]
V=np.zeros((5,5))
P=np.zeros((5,5,2),int)
g=0.9

def step(i,j,a):
    x,y=max(0,min(4,i+a[0])),max(0,min(4,j+a[1]))
    return x,y,G[x][y]-1

while True:
    for _ in range(20):
        for i in range(5):
            for j in range(5):
                x,y=P[i][j]
                V[i][j]=step(i,j,(x,y))[2]+g*V[step(i,j,(x,y))[0]][step(i,j,(x,y))[1]]

    stable=True
    for i in range(5):
        for j in range(5):
            best=max(A,key=lambda a:step(i,j,a)[2]+g*V[step(i,j,a)[0]][step(i,j,a)[1]])
            if tuple(P[i][j])!=best:
                P[i][j]=best; stable=False
    if stable: break

print("Optimal Policy:\n",P)
