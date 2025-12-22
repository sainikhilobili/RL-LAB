import random, math

prices=[10,20,30]
p=[0.6,0.4,0.2]   # true buy probabilities
T=500

def pull(i): return prices[i] if random.random()<p[i] else 0

def eps_greedy(e=0.1):
    Q,N=[0]*3,[0]*3
    r=0
    for t in range(T):
        i=random.randrange(3) if random.random()<e else max(range(3),key=lambda x:Q[x])
        rew=pull(i); N[i]+=1; Q[i]+= (rew-Q[i])/N[i]; r+=rew
    return r

def ucb():
    Q,N=[0]*3,[0]*3
    r=0
    for t in range(1,T+1):
        i=t-1 if t<=3 else max(range(3),key=lambda x:Q[x]+math.sqrt(2*math.log(t)/N[x]))
        rew=pull(i); N[i]+=1; Q[i]+= (rew-Q[i])/N[i]; r+=rew
    return r

def thompson():
    S,F=[1]*3,[1]*3
    r=0
    for _ in range(T):
        i=max(range(3),key=lambda x:random.betavariate(S[x],F[x]))
        rew=pull(i); r+=rew
        (S if rew else F)[i]+=1
    return r

print("Epsilon-Greedy:",eps_greedy())
print("UCB:",ucb())
print("Thompson:",thompson())
