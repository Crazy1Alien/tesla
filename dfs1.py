graph1={
        '1':['2','4'],
        '2':['3','5','6','1'],
        '3':['2','5'],
        '4':['6','1'],
        '5':['2','3','7','9'],
        '6':['2','4','7','9'],
        '7':['5','6'],
        '8':['9'],
        '9':['5','6','8']}
def dfs(graph, node, visited):
    if node not in visited:
        visited.append(node)
        for k in graph[node]:
            dfs(graph, k, visited)
    return visited
visited1=dfs(graph1,'1',[])
print(visited1)