import random, math

p = [0.2, 0.4, 0.6]  # true probabilities of success
T = 1000

def pull(i):
    return 1 if random.random() < p[i] else 0

# Epsilon-Greedy
def eps_greedy(e=0.1):
    Q, N = [0]*3, [0]*3
    r = 0
    for _ in range(T):
        if random.random() < e:
            i = random.randrange(3)
        else:
            i = Q.index(max(Q))
        x = pull(i)
        N[i] += 1
        Q[i] += (x - Q[i]) / N[i]
        r += x
    return r / T

# Upper Confidence Bound (UCB)
def ucb():
    Q, N = [0]*3, [0]*3
    r = 0
    for t in range(1, T+1):
        if t <= 3:
            i = t-1
        else:
            i = max(range(3), key=lambda x: Q[x] + math.sqrt(2*math.log(t)/N[x]))
        x = pull(i)
        N[i] += 1
        Q[i] += (x - Q[i]) / N[i]
        r += x
    return r / T

# Thompson Sampling
def thompson():
    S, F = [1]*3, [1]*3
    r = 0
    for _ in range(T):
        i = max(range(3), key=lambda x: random.betavariate(S[x], F[x]))
        x = pull(i)
        r += x
        if x == 1:
            S[i] += 1
        else:
            F[i] += 1
    return r / T

# Run simulations
print("ε-Greedy CTR:", eps_greedy())
print("UCB CTR:", ucb())
print("Thompson CTR:", thompson())
