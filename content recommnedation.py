import random

p=[0.2,0.4,0.6]     # true engagement rates
eps=0.1
runs,steps=20,1000

def simulate():
    Q,N=[0]*3,[0]*3
    reward=0
    for _ in range(steps):
        a=random.randrange(3) if random.random()<eps else Q.index(max(Q))
        r=1 if random.random()<p[a] else 0
        N[a]+=1; Q[a]+= (r-Q[a])/N[a]; reward+=r
    return reward/steps

avg=sum(simulate() for _ in range(runs))/runs
print("Average Engagement Rate:",round(avg,3))
