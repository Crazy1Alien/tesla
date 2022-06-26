def get(graph, a, b=None):
    links = graph.setdefault(a, {})
    if b is None:
        return links
    else:
        return get(links,b)
class Node:
    def __init__(self, name:str, parent:str):
        self.name = name
        self.parent = parent
        self.g = 0
        self.h = 0
        self.f = 0
    def __eq__(self, other):
        return self.name == other.name
    def __lt__(self, other):
        return self.f < other.f
    def __repr__(self):
        return ('({0},{1})'.format(self.name, self.f))
def add_to_open(open, neighbor):
    for node in open:
        if (neighbor == node and neighbor.f > node.f):
            return False
    return True
def astar_search(graph, heuristics, start, end):
    open = []
    closed = []
    start_node = Node(start, None)
    goal_node = Node(end, None)
    open.append(start_node)
    while len(open) > 0:
        open.sort()
        current_node = open.pop(0)
        closed.append(current_node)
        if current_node == goal_node:
            path = []
            while current_node != start_node:
                path.append(current_node.name + ': ' + str(current_node.g))
                current_node = current_node.parent
            path.append(start_node.name + ': ' + str(start_node.g))
            return path[::-1]
        neighbors = graph.get(current_node.name)
        for key, value in neighbors.items():
            neighbor = Node(key, current_node)
            if(neighbor in closed):
                continue
            neighbor.g = current_node.g + get(graph,current_node.name, neighbor.name)
            neighbor.h = heuristics.get(neighbor.name)
            neighbor.f = neighbor.g + neighbor.h
            if(add_to_open(open, neighbor) == True):
                open.append(neighbor)
    
    return None
g={'a':{'b':4,'c':3},'b':{'f':5,'e':12},'c':{'e':10,'d':7},'d':{'e':2},'e':{'z':5},'f':{'z':16},'z':{}}
h={'a':14,'b':12,'c':11,'d':6,'e':4,'f':11,'z':0}
path = astar_search(g, h, 'a', 'z')
print(path)