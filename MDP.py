import random

G=[[0,1,0,0,-1],[0,-1,1,0,0],[0,0,0,-1,1],[1,0,-1,0,0],[0,0,0,1,0]]
A=[(-1,0),(1,0),(0,-1),(0,1)]

def sim(p):
    x=y=r=0
    for _ in range(20):
        dx,dy=p(x,y)
        x,y=max(0,min(4,x+dx)),max(0,min(4,y+dy))
        r+=G[x][y]; G[x][y]=0
    return r

print("Random:",sim(lambda x,y:random.choice(A)))
print("Greedy:",sim(lambda x,y:max(A,key=lambda a:G[max(0,min(4,x+a[0]))][max(0,min(4,y+a[1]))])))
