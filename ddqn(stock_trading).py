import os, numpy as np, random
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense

# Simulated stock prices
prices = np.random.randint(50,150,30)
n_states, n_actions = 3, 3      # price, holding, buy_price
gamma, eps = 0.9, 0.1

# Build neural network
def build_model():
    m = Sequential([Input(shape=(n_states,)),
                    Dense(16, activation='relu'),
                    Dense(16, activation='relu'),
                    Dense(n_actions, activation='linear')])
    m.compile('adam','mse'); return m

Q, target = build_model(), build_model()
target.set_weights(Q.get_weights())

state = np.array([prices[0],0,0]).reshape(1,-1)
profit = 0
policy = []

for t in range(len(prices)-1):
    # Epsilon-greedy
    a = random.randint(0,2) if random.random()<eps else np.argmax(Q.predict(state,0))
    price, hold, buy = state[0]; r=0

    if a==1 and hold==0: hold, buy = 1, price      # BUY
    elif a==2 and hold==1: hold, r, profit = 0, price-buy, profit+(price-buy)  # SELL

    next_state = np.array([prices[t+1], hold, buy]).reshape(1,-1)
    a_next = np.argmax(Q.predict(next_state,0))
    y = Q.predict(state,0)
    y[0][a] = r + gamma*target.predict(next_state,0)[0][a_next]

    Q.fit(state, y, verbose=0)
    state = next_state
    policy.append(['H','B','S'][a])

    if t%5==0: target.set_weights(Q.get_weights())

print("Prices:", prices.tolist())
print("Trading Policy:", policy)
print("Total Profit:", profit)
