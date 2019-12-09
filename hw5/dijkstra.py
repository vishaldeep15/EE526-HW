def initial_graph() :
    
    return {
            
        'A': {'B':5, 'I':6, 'O':4, 'U':7},
        'B': {'A':5, 'C':2, 'J':3},
        'C': {'B':2, 'I':6, 'D':2, 'K':3},
        'D': {'C':2, 'E':2, 'J':6, 'L':3},
        'E': {'D':2, 'F':2, 'K':6, 'M':3},
        'F': {'E':2, 'G':2, 'L':6, 'N':3},
        'G': {'F':2, 'H':5, 'M':6},
        'H': {'G':5, 'N':6, 'T':4, 'Z':7},
        'I': {'A':6, 'C':6, 'J':4, 'P':1},
        'J': {'V':3, 'I':4, 'O':7, 'Q':1, 'K':4, 'D':6},
        'K': {'J':4, 'P':7, 'R':1, 'L':4, 'E':6, 'C':3},
        'L': {'K':4, 'Q':7, 'S':1, 'M':4, 'F':6, 'D':3},
        'M': {'L':4, 'R':7, 'T':1, 'N':4, 'G':6, 'E':3},
        'N': {'M':4, 'S':7, 'H':6, 'F':3},
        'O': {'A':4, 'J':7, 'P':3, 'V':6},
        'P': {'O':3, 'U':9, 'W':6, 'Q':3, 'K':7, 'I':1},
        'Q': {'P':3, 'V':9, 'X':6, 'R':10, 'L':7, 'J':1},
        'R': {'Q':10, 'W':9, 'Y':6, 'S':3, 'M':7, 'K':1},
        'S': {'R':3, 'X':9, 'Z':6, 'T':3, 'N':7, 'L':1},
        'T': {'S':3, 'Y':9, 'H':4, 'M':1},
        'U': {'A':7, 'P':9, 'V':1},
        'V': {'U':1, 'W':7, 'Q':9, 'O':6},
        'W': {'V':7, 'X':8, 'R':9, 'P':6},
        'X': {'W':8, 'Y':1, 'S':9, 'Q':6},
        'Y': {'X':1, 'Z':1, 'T':9, 'R':6},
        'Z': {'Y':1, 'S':6, 'H':7}    
            
            
            }
print(initial_graph())
    
initial = 'A'
path = {}
adj_node = {}
queue = []
graph = initial_graph()
for node in graph:
    path[node] = float("inf")
    adj_node[node] = None
    queue.append(node)
    
path[initial] = 0
while queue:
    # find min distance which wasn't marked as current
    key_min = queue[0]
    min_val = path[key_min]
    for n in range(1, len(queue)):
        if path[queue[n]] < min_val:
            key_min = queue[n]  
            min_val = path[key_min]
    cur = key_min
    queue.remove(cur)
    print(cur)
    
    for i in graph[cur]:
        alternate = graph[cur][i] + path[cur]
        if path[i] > alternate:
            path[i] = alternate
            adj_node[i] = cur
            
            
x = 'H'
print('The path between A to H')
print(x, end = '<-')
while True:
    x = adj_node[x]
    if x is None:
        print("")
        break
    print(x, end='<-')