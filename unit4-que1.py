import numpy as np

# Environment
dt = 0.1
target = np.array([1.0, 1.0, 1.0])
state = np.zeros(3)

# Policy: a = W * (target - state)
W = np.eye(3) * 0.5        # policy parameters
lr = 0.1                  # learning rate
steps = 20

for episode in range(10):
    state = np.zeros(3)
    total_loss = 0

    for _ in range(steps):
        error = target - state
        action = W @ error                 # control
        next_state = state + action * dt   # dynamics

        loss = np.sum((target - next_state)**2)
        total_loss += loss

        # ---- Analytic Gradient ----
        grad_W = -2 * np.outer((target - next_state), error) * dt
        W -= lr * grad_W

        state = next_state

    print(f"Episode {episode+1}, Loss: {total_loss:.4f}")

print("Final position:", state)
print("Target:", target)
