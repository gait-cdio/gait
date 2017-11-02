
from tracker import Track
t = Track(1,1,3,-1, 1)

t.predict()
print(t.x)
t.update(2.02,2.1)
print(t.x)


