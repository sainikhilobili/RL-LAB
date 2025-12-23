import numpy as np
MAX_QUEUE = 5
PASS_LIMIT = 2      
ARRIVAL = 1         
GAMMA = 0.9
states = [(i, j) for i in range(MAX_QUEUE + 1)
                  for j in range(MAX_QUEUE + 1)]
actions = [0, 1]  
def transition(state, action):
    q_ns, q_ew = state
    if action == 0:  # NS green
        q_ns = max(0, q_ns - PASS_LIMIT)
    else:            # EW green
        q_ew = max(0, q_ew - PASS_LIMIT)
    # New arrivals
    q_ns = min(MAX_QUEUE, q_ns + ARRIVAL)
    q_ew = min(MAX_QUEUE, q_ew + ARRIVAL)
    return (q_ns, q_ew)
def reward(state):
    q_ns, q_ew = state
    return -(q_ns + q_ew)
def policy_iteration():
    V = {s: 0 for s in states}
    policy = {s: np.random.choice(actions) for s in states}
    stable = False
    while not stable:
        for _ in range(50):
            for s in states:
                a = policy[s]
                s_next = transition(s, a)
                V[s] = reward(s) + GAMMA * V[s_next]
        stable = True
        for s in states:
            old_action = policy[s]
            action_values = {}
            for a in actions:
                s_next = transition(s, a)
                action_values[a] = reward(s) + GAMMA * V[s_next]
            best_action = max(action_values, key=action_values.get)
            policy[s] = best_action
            if old_action != best_action:
                stable = False
    return policy, V
optimal_policy, optimal_value = policy_iteration()
print("Sample Optimal Policy:")
for s in [(0,5), (3,1), (4,4)]:
    direction = "NS Green" if optimal_policy[s] == 0 else "EW Green"
    print(f"State {s} -> {direction}")
