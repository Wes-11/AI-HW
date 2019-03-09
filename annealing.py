import math
import copy
from random import *
vertex_list=list()
vertex=list()
color = dict()
pred = dict()
discover = dict()
adj = dict()
finish = dict()
time = 0
vertex = list()
def euclidian_distance(x1, y1, x2, y2):
    e = round(math.sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2)))
    return e
def parse_data(filename):
    data=list()
    with open(filename, 'r') as f:
        q = f.readline()
        t = q.split()
        n = int(t[1][2:20])
        for x in range(0, 9):

            a = f.readline()
        for y in range(0, n):
            b = f.readline()
            u = b.split()
            data.append((int(u[0]), float(u[1]), float(u[2])))
            vertex_list.append(y)
    return(data)
    #print(data)

def build_graph(d):
    n = len(d)
    Matrix = [[0 for i in range(n)] for j in range(n)]
    for x in range(0, n):
        for y in range(0, n):
            x1, y1 = d[x][1], d[x][2]
            x2, y2 = d[y][1], d[y][2]
            w = euclidian_distance(x1,y1,x2,y2)
            Matrix[x][y] = w
    return Matrix

d = parse_data()

a = build_graph(parse_data())
g = int(len(vertex_list)/2)
for i in range(0, g):
    vertex.append(vertex_list[i])
DFS = [(0, 9), (9, 13), (13, 20), (20, 1), (1, 3), (3, 2), (2, 4), (4, 6),
       (6, 7), (7, 12), (12, 14), (14, 19), (19, 22), (22, 24), (24, 25),
       (25, 21), (21, 23), (23, 27), (27, 8), (8, 11), (11, 15), (15, 16),
       (16, 17), (17, 18), (18, 5), (5, 10), (10, 26), (26, 30), (30, 35),(35, 28),
       (28, 29), (29, 31), (31, 34), (34, 36), (36, 37), (37, 32), (32, 33),(32,33),(33,0)]

something = [(0, 9), (2, 4), (29, 31), (19, 19), (34, 34), (17, 17), (18, 18), (32, 33), (27, 27), (12, 14), (8, 7), (32, 33),
  (22, 22), (13, 0), (33, 32), (8, 7), (37, 37), (1, 1), (11, 11), (12, 14), (6, 5), (35, 35), (31, 29), (5, 6),
  (26, 26), (30, 30), (20, 20),
  (9, 13), (15, 15), (28, 28), (3, 3), (36, 36), (23, 21), (25, 24), (21, 23), (4, 2), (25, 24), (10, 10), (16, 16)]
print(len(DFS))
def get_edge_weight(G, E):
    root, destination = E[0], E[1] 
    weight = G[root][destination]
    return weight
def get_path_weight(G, L):
    weight = 0
    for i in L:
        root, destination = i[0], i[1]
        weight += G[root][destination]
    return weight
def randomize(U,V,W):
    A,B = U[0],U[1]
    C,D = V[0],V[1]
    E,F = W[0],W[1]
    #[(A,B),(C,D),(E,F)],
    newEdges = [[(A,B),(C,D),(E,F)],[(A,C),(B,F),(E,D)], [(A,D),(B,E),(C,F)], [(A,F),(B,D),(C,E)],[(A,E),(B,C),(D,F)]]
    return newEdges
def get_edges(P, E):
    edges = list()
    at  = P.index(E)
    while len(edges) < 3:
        S = randint(1, 6)
        check  = at + S
        if check < len(P):
            if P[check] not in edges:
                edges.append(P[check])
        elif check > len(P):
            check = at - S
            if P[check] not in edges:
                edges.append(P[check])
    return edges
def get_lightest_group(L):
    lightest = math.inf
    for i in L:
        weight = 0
        counter=0
        for j in i:
            counter+= 1
            C = get_edge_weight(a, j)
            weight += C
        if weight < lightest:
            lightest = weight
            LG = i
    return LG
def three_opt(G):
    g = set()
    H = copy.deepcopy(G)
    #print(G)
    for i in range(len(G)):
        E = H[i]
        edges = get_edges(H, E)
        H.remove(edges[0]), H.remove(edges[1]), H.remove(edges[2])#removing selected edges    
        ER = randomize(edges[0],edges[1],edges[2]) #returns all combinations of edges
        usedEdges = get_lightest_group(ER) #gets the lightest combination of three edges
        H.append(usedEdges[0]), H.append(usedEdges[1]), H.append(usedEdges[2])#adding lightest group into graph
    return H
def get_adj(V, G):
    temp = list()
    n = len(G)
    for i in range(n):
        if G[i][0] == V:           
            temp.append(G[i][1])
    adj[V] = temp
    return None
def DFS_util(G):
    n = len(G)
    for i in vertex:
        color[i] = "white"
        pred[i] = "nil"   
    for i in vertex:
        if color[i] == "white":
            DFS_Visit(G,i)
    return None
def DFS_Visit(G,i):
    global time
    time = time+1
    discover[i] = time
    color[i] = "gray"
    a = get_adj(i, G)  
    for v in adj[i]:
        if color[v] == "white":
            pred[v] = i
            DFS_Visit(G, v)
    color[i] = "black"
    time = time+1
    finish[i] = time
    return None
def build_path(L):
    newList = list()
    for i in range(len(L)-1):
        newList.append((L[i], L[i+1]))
    newList.append((L[-1],L[0]))
    return newList
estimate = get_path_weight(a, DFS)
best = get_path_weight(a, DFS)
#print(get_path_weight(a, R))
bestPath = DFS
print(best)
counter = 0
def acceptance(O, N, T):
    e = 2.71828
    beta = O - N
    K = beta/T
    a = e*T
    return a
def anneal(X):
    A = get_path_weight(a, X)
    Temp = 1.0
    Temp_min = .0000001
    alpha = .9
    while Temp > Temp_min:
        i=1
        while i <= 100:
            Y = three_opt(X)
            B = get_path_weight(a, Y)
            prob = acceptance(A, B, Temp)
            if prob > random():
                X = Y
                A = B
            i += 1
        print(Temp, alpha)
        Temp = Temp*alpha
    return X, get_path_weight(a, X)

def run_three_opt(P):
    global best
    discover.clear()
    Y = three_opt(P)
    DFS_util(Y)
    b = list(discover.keys())
    R = build_path(b)
    W = get_path_weight(a, R)
    if best > W:
        bestPath = R
        best = W
        print("new best path ", R, W)
        counter = 0
stop = 10000
three_opt(bestPath)
while best > estimate/2:
    counter += 1
    print(counter)
    run_three_opt(bestPath)

    if counter == stop:
        break
print("finish")

