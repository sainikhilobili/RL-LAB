import random, math

p=[0.1,0.3,0.5]; T=1000
def pull(i): return 1 if random.random()<p[i] else 0

def eps_greedy(e=0.1):
    Q,N=[0]*3,[0]*3; r=0
    for _ in range(T):
        i=random.randrange(3) if random.random()<e else Q.index(max(Q))
        x=pull(i); N[i]+=1; Q[i]+= (x-Q[i])/N[i]; r+=x
    return r/T

def ucb():
    Q,N=[0]*3,[0]*3; r=0
    for t in range(1,T+1):
        i=t-1 if t<=3 else max(range(3),key=lambda i:Q[i]+math.sqrt(2*math.log(t)/N[i]))
        x=pull(i); N[i]+=1; Q[i]+= (x-Q[i])/N[i]; r+=x
    return r/T

print("ε-Greedy CTR:",eps_greedy())
print("UCB CTR:",ucb())
