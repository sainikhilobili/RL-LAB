import random, math

T=1000
Qe=[0,0,0]
Qs=[0,0,0]

def play(a): return 1 if a==2 else 0   # winning action

# Epsilon-Greedy
wins_e=0
eps=0.1
for _ in range(T):
    a=random.randrange(3) if random.random()<eps else Qe.index(max(Qe))
    r=play(a); wins_e+=r
    Qe[a]+=0.1*(r-Qe[a])

# Softmax
wins_s=0
temp=0.5
for _ in range(T):
    p=[math.exp(q/temp) for q in Qs]
    a=random.choices(range(3),p)[0]
    r=play(a); wins_s+=r
    Qs[a]+=0.1*(r-Qs[a])

print("Epsilon-Greedy Win Rate:",wins_e/T)
print("Softmax Win Rate:",wins_s/T)
