import random

def episode(policy):
    free_agents=3
    reward=0
    for _ in range(10):
        if free_agents>0 and policy()=="ASSIGN":
            free_agents-=1
            reward+=1
        free_agents+=random.choice([0,1])  # agent becomes free
    return reward

def mc_value(policy, N=1000):
    return sum(episode(policy) for _ in range(N))/N

greedy=lambda:"ASSIGN"
random_policy=lambda:random.choice(["ASSIGN","WAIT"])

print("Greedy Policy Value:",mc_value(greedy))
print("Random Policy Value:",mc_value(random_policy))
