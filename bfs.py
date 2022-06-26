graph={
'S':['H','R'],
'H':['I','r'],
'R':['A','M'],
'I':['','V'],
'r':[' ','i'],
'A':['Y','E'],
'M':['^'],
'V':[],
'i':[],
'Y':[],
'E':[],
'^':[],
'':[],
' ':[]
}
def bfs(visited,graph,current_node):
    queue=[]
    queue.append(current_node)
    visited.append(current_node)
    while queue:
        p=queue.pop(0)
        print('\t',p)
        for next_node in graph[p]:
            if next_node not in visited:
                visited.append(next_node)
                queue.append(next_node)
                
bfs([],graph,'S')          