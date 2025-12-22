import numpy as np

V=np.zeros(10); g=0.9
demand=2

for _ in range(50):
    for s in range(10):
        V[s]=min(
            (abs(s+a-demand)+g*V[min(9,max(0,s+a-demand))])
            for a in range(3)
        )

policy=[min(range(3),key=lambda a:(abs(s+a-demand)+g*V[min(9,max(0,s+a-demand))])) for s in range(10)]
print("Optimal Policy (order qty):",policy)
