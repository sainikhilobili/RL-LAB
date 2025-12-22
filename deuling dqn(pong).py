import os, numpy as np, random
os.environ['TF_ENABLE_ONEDNN_OPTS']='0'
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, Lambda
from tensorflow.keras.optimizers import Adam

# Simple environment: state=[ball_x, ball_y, paddle_y], actions=0:left,1:stay,2:right
n_states, n_actions = 3, 3
gamma, eps = 0.9, 0.1

# Dueling DQN model
def build_dueling():
    inp = Input(shape=(n_states,))
    x = Dense(32, activation='relu')(inp)
    x = Dense(32, activation='relu')(x)

    # Value stream
    V = Dense(1)(x)
    # Advantage stream
    A = Dense(n_actions)(x)
    # Combine into Q-values
    Q = Lambda(lambda a: a[0] + (a[1]-np.mean(a[1],axis=1,keepdims=True)))([V,A])
    model = Model(inputs=inp, outputs=Q)
    model.compile(Adam(0.001),'mse')
    return model

Q, target = build_dueling(), build_dueling()
target.set_weights(Q.get_weights())

# Initial state
state = np.array([0.5,0.5,0.5]).reshape(1,-1)
policy = []

for step in range(50):  # short training loop
    a = random.randint(0,2) if random.random()<eps else np.argmax(Q.predict(state,0))
    ball_x, ball_y, paddle_y = state[0]

    # Simple dynamics
    ball_x += (a-1)*0.05
    reward = 1 if abs(ball_x-paddle_y)<0.1 else -1
    next_state = np.array([ball_x, ball_y, paddle_y]).reshape(1,-1)

    # Double DQN target
    a_next = np.argmax(Q.predict(next_state,0))
    y = Q.predict(state,0)
    y[0][a] = reward + gamma*target.predict(next_state,0)[0][a_next]

    Q.fit(state,y,verbose=0)
    state = next_state
    policy.append(['Left','Stay','Right'][a])

    if step%10==0: target.set_weights(Q.get_weights())

print("Learned Policy:", policy)
