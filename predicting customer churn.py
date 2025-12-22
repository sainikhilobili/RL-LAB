import random

def episode(policy):
    state="at-risk"
    reward=0
    for _ in range(10):
        action=policy(state)
        churn_prob=0.1 if action=="retain" else 0.4
        if random.random()<churn_prob:
            break
        reward+=1
    return reward

def mc_value(policy, N=1000):
    return sum(episode(policy) for _ in range(N))/N

retain_policy=lambda s:"retain"
ignore_policy=lambda s:"ignore"

print("Retain Policy Value:",mc_value(retain_policy))
print("Ignore Policy Value:",mc_value(ignore_policy))
