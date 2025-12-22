import random, math

p=[0.05,0.10,0.20]   # true CTR of ads
T=1000

def pull(i): return 1 if random.random()<p[i] else 0

def eps_greedy(e=0.1):
    Q,N=[0]*3,[0]*3; c=0
    for _ in range(T):
        i=random.randrange(3) if random.random()<e else max(range(3),key=lambda x:Q[x])
        r=pull(i); N[i]+=1; Q[i]+= (r-Q[i])/N[i]; c+=r
    return c/T

def ucb():
    Q,N=[0]*3,[0]*3; c=0
    for t in range(1,T+1):
        i=t-1 if t<=3 else max(range(3),key=lambda x:Q[x]+math.sqrt(2*math.log(t)/N[x]))
        r=pull(i); N[i]+=1; Q[i]+= (r-Q[i])/N[i]; c+=r
    return c/T

def thompson():
    S,F=[1]*3,[1]*3; c=0
    for _ in range(T):
        i=max(range(3),key=lambda x:random.betavariate(S[x],F[x]))
        r=pull(i); c+=r; (S if r else F)[i]+=1
    return c/T

print("Epsilon-Greedy CTR:",eps_greedy())
print("UCB CTR:",ucb())
print("Thompson CTR:",thompson())
