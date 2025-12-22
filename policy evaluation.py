import numpy as np

R=[[0,0,2,0,0],
   [0,-2,0,0,0],
   [0,0,0,-2,0],
   [0,2,0,0,5],
   [0,0,0,0,0]]

A=[(-1,0),(1,0),(0,-1),(0,1)]
V=np.zeros((5,5))
g=0.9

for _ in range(50):
    for i in range(5):
        for j in range(5):
            V[i][j]=sum(
                (R[max(0,min(4,i+a[0]))][max(0,min(4,j+a[1]))]
                +g*V[max(0,min(4,i+a[0]))][max(0,min(4,j+a[1]))])
                for a in A)/4

print(V.round(2))
