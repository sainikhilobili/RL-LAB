import os, numpy as np, random
os.environ['TF_ENABLE_ONEDNN_OPTS']='0'
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

# Simple highway environment
n_states, n_actions = 3, 3   # [speed, lane, distance] | actions: slow, keep, fast
gamma, epsilon = 0.9, 0.1

def build_model():
    m = Sequential([
        Dense(16, input_shape=(n_states,), activation='relu'),
        Dense(16, activation='relu'),
        Dense(n_actions, activation='linear')
    ])
    m.compile('adam','mse')
    return m

model = build_model()
state = np.array([50, 1, 10]).reshape(1,-1)  # initial state

for step in range(200):
    a = random.randint(0,2) if random.random()<epsilon else np.argmax(model.predict(state,0))
    speed, lane, dist = state[0]
    reward = speed*0.1 - abs(dist-10)        # safe & efficient driving reward
    speed += [-5,0,5][a]                     # action effect
    next_state = np.array([speed, lane, dist-1]).reshape(1,-1)
    target = model.predict(state,0)
    target[0][a] = reward + gamma*np.max(model.predict(next_state,0))
    model.fit(state, target, verbose=0)
    state = next_state

print("✅ DQN Training Completed for Autonomous Vehicle")
