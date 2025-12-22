import numpy as np, matplotlib.pyplot as plt

V=np.zeros((5,5)); g=0.9
policy1=lambda i,j:(1,0)     # always DOWN
policy2=lambda i,j:(0,1)     # always RIGHT

def eval_policy(policy):
    V=np.zeros((5,5))
    for _ in range(50):
        for i in range(5):
            for j in range(5):
                ni,nj=min(4,i+policy(i,j)[0]),min(4,j+policy(i,j)[1])
                V[i][j]=-1+g*V[ni][nj]
    return V

plt.subplot(1,2,1); plt.title("Policy DOWN"); plt.imshow(eval_policy(policy1))
plt.subplot(1,2,2); plt.title("Policy RIGHT"); plt.imshow(eval_policy(policy2))
plt.show()
