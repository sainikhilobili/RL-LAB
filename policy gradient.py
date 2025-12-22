import random

theta=0.5      # policy parameter
lr=0.01        # learning rate

for _ in range(1000):
    action=theta + random.gauss(0,0.1)     # policy
    reward=action * random.uniform(-1,2)   # market return
    theta += lr * reward                   # gradient ascent

print("Optimized Policy Parameter:",round(theta,3))
