import os, numpy as np
os.environ['TF_ENABLE_ONEDNN_OPTS']='0'
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

prices = np.random.randint(50,150,20)
n_states, n_actions, gamma = 2, 3, 0.9

def build_model():
    m = Sequential([Dense(16,input_shape=(n_states,),activation='relu'),
                    Dense(16,activation='relu'),Dense(n_actions,activation='linear')])
    m.compile('adam','mse'); return m

model = build_model(); target_model = build_model()
state = np.array([prices[0],0]).reshape(1,-1)
policy = []

for t in range(len(prices)-1):
    a = np.argmax(model.predict(state,verbose=0))
    holding = state[0,1]; price = prices[t+1]; reward=0
    if a==1 and holding==0: holding=1
    elif a==2 and holding==1: reward=price-state[0,0]; holding=0
    next_state = np.array([price,holding]).reshape(1,-1)
    a_next = np.argmax(model.predict(next_state,verbose=0))
    target = model.predict(state,verbose=0)
    target_val = target_model.predict(next_state,verbose=0)
    target[0][a] = reward + gamma*target_val[0][a_next]
    model.fit(state,target,verbose=0); state = next_state
    policy.append(['H','B','S'][a])

print("Learned Trading Policy:", policy)
