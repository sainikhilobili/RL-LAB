import numpy as np

pickup=(4,4)
A=[(-1,0),(1,0),(0,-1),(0,1)]
V=np.zeros((5,5))
g=0.9

for _ in range(50):
    for i in range(5):
        for j in range(5):
            if (i,j)==pickup:
                V[i][j]=10
            else:
                V[i][j]=max(
                    -1 + g*V[max(0,min(4,i+a[0]))][max(0,min(4,j+a[1]))]
                    for a in A
                )

print("Value Function:\n",V.round(2))
