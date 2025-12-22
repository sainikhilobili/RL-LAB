import random

states=[("A","GREEN"),("B","RED"),("B","GREEN"),("C","GREEN")]
dest="C"

def policy(state):
    pos,light=state
    return "STOP" if light=="RED" else "MOVE"

reward=0
state=("A","GREEN")

for _ in range(10):
    action=policy(state)
    pos,light=state
    
    if action=="MOVE" and light=="GREEN":
        reward+=1
        pos=chr(ord(pos)+1)
    elif action=="MOVE" and light=="RED":
        reward-=5
    else:
        reward+=0
    
    if pos==dest:
        reward+=10
        break
    
    state=(pos, random.choice(["RED","GREEN"]))

print("Total Reward:",reward)
