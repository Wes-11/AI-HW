import math
import copy
from random import *
from collections import defaultdict
import sys
vertex_list = list()
color = defaultdict(list)
adj = defaultdict(list)
pred, discover, finish, parent, rank = dict(), dict(), dict(), dict(), dict()
time = 0
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
            #9 was chosen to skip the comments that were present in the provided file
            a = f.readline()
        for y in range(0, n):
            b = f.readline()
            u = b.split()
            data.append((int(u[0]), float(u[1]), float(u[2])))
            vertex_list.append(y)
    return(data)
def build_graph(d):
    #builds a matrix made up of the weights of edges
    n = len(d)
    Matrix = [[0 for i in range(n)] for j in range(n)]
    for x in range(0, n):
        for y in range(0, n):
            x1, y1 = d[x][1], d[x][2]
            x2, y2 = d[y][1], d[y][2]
            w = euclidian_distance(x1,y1,x2,y2)
            Matrix[x][y] = w
    return Matrix
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
#edge weight and path weight are seperate for convience in the other functions
def upper_triangle(G):
    #because the graph is unconnected we only need half of the matrix
    edgeSet = set()
    for i in range(0, len(G)):       
        for j in range(0, len(G)):
            if j > i:
                edgeSet.add((G[i][j], i, j))
    return edgeSet
def make_set(X):
    parent[X] = X
    rank[X] = 0
def find_set(x):                        #make_set, find_set, and union are functions of the kruskals
    if x != parent[x]:                  #algorithm implementation. 
        parent[x] = find_set(parent[x])
    return parent[x]
def union(u,v):
    r1 = find_set(u)
    r2 = find_set(v)
    if rank[r1] != rank[r2]:
        parent[r2] = r1
    else:
        parent[r1] = r2
    if rank[r1] == rank[r2]: rank[r2] += 1
def kruskal_MST(G):
     """accepts a list argument and computes the MST
        the list should be made up of 3 tuples of (weight, root, destination)"""
     g = (len(vertex_list))
     A = set()
     F = upper_triangle(G)
     n = len(F)
     Q = sorted(F, key = lambda weight : weight[0])
     for v in range(g):
        make_set(v)
     for u in range(n):
         w,u,v = Q[u]
         if find_set(u) != find_set(v):
             union(u,v)
             A.add((w,u,v))
     return A
def is_valid(E):
    #detirmines if an edge is a valid selection
    #this is a helper function for the randomize function
    #to prevent duplicates from appearing
    for i in E:
        if i[0] == i[1]:
            return False
def edge_list(L):
    #parses verticies from the MST graph
    #the original MST sontains weights as well
    edgeL =list()
    for i in range(len(L)):
        edgeL.append((L[i][1], L[i][2]))
    return edgeL
def get_adj(V, G):
    #gets all adjacent verticies to the current vertex
    temp = list()
    n = len(G)
    for i in range(n):
        if G[i][0] == V:           
            temp.append(G[i][1])
    adj[V] = temp
    return None
def DFS_util(G):
    """the main driver for the depth first search functions modeled
       off the algorithm found in cormens intro to algorithms this accepts a
       list  of 2 tuples arguement made up of (root, destination)"""
    n = len(G)
    for i in vertex_list:
        color[i] = "white"
        pred[i] = "nil"   
    for i in vertex_list:
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
    """builds a path based off the discover order,
        accepts the discover dictionary that has been turned into a list as it's
        arguement"""
    newList = list()
    for i in range(len(L)-1):
        newList.append((L[i], L[i+1]))
    newList.append((L[-1], L[0]))
    return newList
def randomize(U,V,W):
    A,B = U[0],U[1]
    C,D = V[0],V[1]
    E,F = W[0],W[1]
    #returns a list of lists made up of tuples to represent all edge combinations
    newEdges = [[(A,B),(C,D),(E,F)],[(A,C),(B,F),(E,D)], [(A,D),(B,E),(C,F)], [(A,F),(B,D),(C,E)],[(A,E),(B,C),(D,F)]]
    for i in newEdges:
        A = is_valid(i)
        if A == False:
            newEdges.remove(i)
        if len(newEdges) == 0:
            return False
        else:
            return newEdges
def get_edges(P, E):
    edges = list()
    at  = P.index(E)
    while len(edges) < 3:
        S = randint(1, 7)     #this adds a constraint for how far away an edge can be from the
        check  = at + S       #current selected edge to ignore edges that are more likely to be signifigantly heavier than
        if check < len(P):    #the selected edge
            if P[check] not in edges:
                edges.append(P[check])
        elif check > len(P):
            check = at - S
            if P[check] not in edges:
                edges.append(P[check])
    return edges   
def get_lightest_group(L):
    lightest = math.inf    #accepts a list of edge groups
    for i in L:            #returns the lightest of those groups
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
    H = copy.deepcopy(G)
    for i in range(len(G)):
        E = H[i]
        edges = get_edges(H, E)
        H.remove(edges[0]), H.remove(edges[1]), H.remove(edges[2])#removing selected edges
        ER = randomize(edges[0],edges[1],edges[2]) #returns all combinations of edges
        usedEdges = get_lightest_group(ER) #gets the lightest combination of three edges
        H.append(usedEdges[0]), H.append(usedEdges[1]), H.append(usedEdges[2])#adding lightest group into graph
    return H
def run_three_opt(P):
    """the main driver for the 3-opt implementation,
        accepts the current optimal path as an arguement"""
    Y = three_opt(P)           #run 3_opt on the current best path
    discover.clear()           #without clearing the dict this will return the same graph every time
    DFS_util(Y)                #create a path from the unorganized edges generated by 3_opt
    b = list(discover.keys())  #get the list of visted nodes
    R = build_path(b)          #build a path from that list
    W = get_path_weight(a, R)  #get that paths weight
    return (R, W)              #return the path and weight as a tuple
def hill_climb(P):
    """the hill climb function slightly modified for
        using a MST and 3-opt accepts the best initial
        estimate as an arguement"""
    counter = 0
    initial = P
    bestEstimate = get_path_weight(a, P)
    W = get_path_weight(a, P)
    stop = 10000                #give up after 10K attempts with no improvement
    while W > bestEstimate/2:   #I know my estimate is at most twice the optimal path
        counter+=1              #so this works as a lower bound   
        N = run_three_opt(P)    
        if N[1] < W:            
            counter = 0
            P = N[0]
            W = N[1]
        if counter == stop:
            break
    return (P, W)

if __name__ == "__main__" :
    fileName = str(sys.argv[1]) 
    a = build_graph(parse_data(fileName))
    G = build_graph(a) 
    MST = kruskal_MST(G)
    MSTL = list(MST)
    MST_edges = edge_list(MSTL)
    DFS_util(MST_edges)
    order = list(discover.keys())
    initial = build_path(order)
    W = get_path_weight(a, initial) 
    HC = hill_climb(initial)
    print("best discovered weight" ,HC[1])
    print("path order", list(discover.keys()))
    #final vertex (the origin) is not shown but is calculated in the weight






