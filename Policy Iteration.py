states = range(6)      # queue size 0–5
gamma = 0.9
V = [0]*6
policy = [0]*6        # 0=Red, 1=Green

for _ in range(5):
    for s in states:
        # choose best action directly
        red   = -s + gamma * V[min(s+1,5)]
        green = -s + gamma * V[max(s-1,0)]
        if green > red:
            policy[s] = 1
            V[s] = green
        else:
            policy[s] = 0
            V[s] = red

print("Queue → Action")
for s in states:
    print(s, "→", "GREEN" if policy[s] else "RED")
