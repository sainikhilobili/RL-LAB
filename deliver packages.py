import random

# Grid size and goal
N = 4
goal = (3, 3)
actions = [(0,1),(0,-1),(1,0),(-1,0)]  # R L D U

Q = {}
returns = {}
policy = {}

def step(s, a):
    x, y = s
    dx, dy = a
    ns = (max(0,min(N-1,x+dx)), max(0,min(N-1,y+dy)))
    reward = 20 if ns == goal else -1
    return ns, reward

# Monte Carlo Control
for _ in range(500):
    s = (0,0)
    episode = []

    while s != goal:
        a = policy.get(s, random.choice(actions))
        ns, r = step(s,a)
        episode.append((s,a,r))
        s = ns

    G = 0
    for s,a,r in reversed(episode):
        G += r
        returns.setdefault((s,a), []).append(G)
        Q[(s,a)] = sum(returns[(s,a)]) / len(returns[(s,a)])
        policy[s] = max(actions, key=lambda x: Q.get((s,x),0))

print("Learned Policy:")
for s in policy:
    print(s, "→", policy[s])
