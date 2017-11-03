import math

from tracker import Track, match
t = Track(1,1,3,-1, 1)

t.predict()
t.update(2.02,2.1)

a = [(2,3), (2,2), (4,5)]
b = [(1,2), (4,5)]

def dist_fun(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

match_list = match(a, b, dist_fun, 10)
print(match_list)
