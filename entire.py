8puzzle:
I=[[1,2,3],[4,0,5],[7,8,6]]  
O=[[1,2,3],[4,5,6],[7,8,0]]  
def find_space(initial):  
 for i in range(len(initial)):  
     for j in range(len(initial)):  
         if initial[i][j] == 0:  
             return [i,j]  
print(find_space(I))  
path=list()  
def valid_p(r,c):  
    if r < 0 or r>=len(I) or c <0 or c >= len(I):   
        return False  
    else:  
     return True  
def h_value(I,O):  
    h=0  
    for i in range(len(I)):  
        for j in range(len(I)):  
            if I[i][j]!= O[i][j]:  
                h=h+1  
    return h
  
def swap(r,c):  
    pos=find_space(I)  
    tmp=I[pos[0]][pos[1]]  
    I[pos[0]][pos[1]]=I[r][c]  
    I[r][c]=tmp  
  
def left(I):  
    pos=find_space(I)  
    if valid_p(pos[0],pos[1]-1): 
        swap(pos[0],pos[1]-1) 
        
def right(I):  
    pos=find_space(I)  
    if valid_p(pos[0],pos[1]+1):  
        swap(pos[0],pos[1]+1)  
        
def up(I):  
    pos=find_space(I)  
    if valid_p(pos[0]-1,pos[1]):   
        swap(pos[0]-1,pos[1]) 
        
def down(I):  
    pos=find_space(I)  
    if valid_p(pos[0]+1,pos[1]):   
        swap(pos[0]+1,pos[1]) 
        
def PRINT(In):  
    for i in range(len(In)):   
        for j in range(len(In)):   
            print(In[i][j],end=" ")   
        print() 
        
def solve(I):  
    temp=I  
    h=h_value(I,O)  
     #while h != 0:  
    right(I)  
    h=h_value(I,O)  
    PRINT(I)  
    print(h)  
    down(I)  
    h=h_value(I,O)  
    PRINT(I)  
    print(h)  
     
print("INPUT MATRIX")  
PRINT(I)  
print("8-Puzzle ")  
solve(I)  

from collections import defaultdict  
adj=defaultdict(list)  
adj={'s':['r','l','u','d'],'r':['d','u','r'],'l':['d','u','l'],'d':['d','l','r'],'u':['l','u','r']}  
I=[[1,2,3],[4,0,5],[7,8,6]]  
O=[[1,2,3],[4,5,6],[7,8,0]] 
 
def dfs(initial_node):  
    st=list()  
    visited=list()  
    visited.append(initial_node)  
    st.append(initial_node)  
    while st:  
        temp=st.pop()  
        for i in adj[temp]:  
            if i not in visited:  
                st.append(i)  
                visited.append(i)  
                if i == 'd':  
                    down(I)  
                elif i == 'l':  
                    left(I)  
                elif i == 'u':  
                    up(I)  
                elif i == 'r': 
                    right(I)     
            PRINT(I)  
            h=h_value(I,O)   
            break  
          
        if h == 0:  
            break  

dfs('s')  


A_star_search:
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



Minmax:
# -*- coding: utf-8 -*-
"""
Created on Tue May 31 13:22:46 2022

@author: admin
"""

import math
def minimax (curDepth, nodeIndex,maxTurn, scores,targetDepth):

    if (curDepth == targetDepth):
        return scores[nodeIndex]
    
    if(maxTurn):
        return max(minimax(curDepth + 1, nodeIndex * 2,False, scores, targetDepth),minimax(curDepth + 1, nodeIndex * 2 + 1,False, scores, targetDepth))

    else:
        return min(minimax(curDepth + 1, nodeIndex * 2,True, scores, targetDepth),minimax(curDepth + 1, nodeIndex * 2 + 1,True, scores, targetDepth))

# Driver code
scores = [3, 5, 2, 9, 12, 5, 23, 23]
treeDepth = math.log(len(scores), 2)
print("The optimal value is : ", end = "")
print(minimax(0, 0, True, scores, treeDepth))


AlphaBeta:
#Python3 program to demonstrate
# working of Alpha-Beta Pruning

# Initial values of Alpha and Beta
MAX, MIN = 1000, -1000

# Returns optimal value for current player
#(Initially called for root and maximizer)
def minimax(depth, nodeIndex, maximizingPlayer,

values, alpha, beta):

# Terminating condition. i.e
# leaf node is reached
    if depth == 3:
        return values[nodeIndex]
    
    if maximizingPlayer:
    
        best = MIN
        
        # Recur for left and right children
        for i in range(0, 2):
        
            val = minimax(depth + 1, nodeIndex *
            
            2 + i,
            
            False, values, alpha,
            
            beta)
            
            best = max(best, val)
            alpha = max(alpha, best)
            
            # Alpha Beta Pruning
            if beta <= alpha:
                break
        
        return best
    
    else:
        best = MAX
        
        # Recur for left and
        # right children
        for i in range(0, 2):
        
            val = minimax(depth + 1, nodeIndex *
            
            2 + i,
            
            True, values,
            
            alpha, beta)
            
            best = min(best, val)
            beta = min(beta, best)
            
            # Alpha Beta Pruning
            if beta <= alpha:
                break
        
        return best

# Driver Code
if __name__ == "__main__":

    values = [3, 5, 6, 9, 1, 2, 0, -1]
    print("The optimal value is :", minimax(0, 0,
True, values, MIN, MAX))


best_first_search:
from queue import PriorityQueue
v = 14
graph = [[] for i in range(v)]


   
def best_first_search(source, target, v):
    visited = [0] * v
    visited[source]=True
    pq = PriorityQueue()
    pq.put((0, source))
    while pq.empty() == False:
        u = pq.get()[1]
        print(u, end=" ")
        if u == target:
            break
        for v, c in graph[u]:
            if not visited[v]:
                visited[v] = True
                pq.put((c, v))
    print()
def addedge(x, y, cost):
    graph[x].append((y, cost))
    graph[y].append((x, cost))

addedge(0, 1, 3)
addedge(0, 2, 6)
addedge(0, 3, 5)
addedge(1, 4, 9)
addedge(1, 5, 8)
addedge(2, 6, 12)
addedge(2, 7, 14)
addedge(3, 8, 7)
addedge(8, 9, 5)
addedge(8, 10, 6)
addedge(9, 11, 1)
addedge(9, 12, 10)
addedge(9, 13, 2)
source = 0
target = 9
best_first_search(source, target, v)



bfs:
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


dfs:
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



HillClimbing:
import random

def randomSolution(tsp):
    cities = list(range(len(tsp)))
    solution = []
    
    for i in range(len(tsp)):
        randomCity = cities[random.randint(0, len(cities) - 1)]
        solution.append(randomCity)
        cities.remove(randomCity)
    
    return solution

def routeLength(tsp, solution):
    routeLength = 0
    for i in range(len(solution)):
        routeLength += tsp[solution[i - 1]][solution[i]]
    return routeLength

def getNeighbours(solution):
    neighbours = []
    for i in range(len(solution)):
        for j in range(i + 1, len(solution)):
            neighbour = solution.copy()
            neighbour[i] = solution[j]
            neighbour[j] = solution[i]
            neighbours.append(neighbour)
    return neighbours

def getBestNeighbour(tsp, neighbours):
    bestRouteLength = routeLength(tsp, neighbours[0])
    bestNeighbour = neighbours[0]
    for neighbour in neighbours:
        currentRouteLength = routeLength(tsp, neighbour)
        if currentRouteLength < bestRouteLength:
            bestRouteLength = currentRouteLength
            bestNeighbour = neighbour
    return bestNeighbour, bestRouteLength

def hillClimbing(tsp):
    currentSolution = randomSolution(tsp)
    currentRouteLength = routeLength(tsp, currentSolution)
    
    neighbours = getNeighbours(currentSolution)
    bestNeighbour, bestNeighbourRouteLength = getBestNeighbour(tsp,
    neighbours)
    
    while bestNeighbourRouteLength < currentRouteLength:
        currentSolution = bestNeighbour
        currentRouteLength = bestNeighbourRouteLength
        neighbours = getNeighbours(currentSolution)
        bestNeighbour, bestNeighbourRouteLength =getBestNeighbour(tsp, neighbours)
    
    return currentSolution, currentRouteLength

def main():
    tsp = [
    [0, 400, 500, 300],
    [400, 0, 300, 500],
    [500, 300, 0, 400],
    [300, 500, 400, 0]
    ]
    
    print(hillClimbing(tsp))

if __name__ == "__main__":
    main()


Huristic
def magic_square_test(my_matrix):
    iSize = len(my_matrix[0])
    sum_list = []
    
    #Horizontal Part:
    sum_list.extend([sum (lines) for lines in
    my_matrix])
    
    #Vertical Part:
    for col in range(iSize):
        sum_list.append(sum(row[col] for row in
    my_matrix))
    
    #Diagonals Part
    result1 = 0
    for i in range(0,iSize):
        result1 +=my_matrix[i][i]
    sum_list.append(result1)
    
    result2 = 0
    for i in range(iSize-1,-1,-1):
        result2 +=my_matrix[i][i]
    sum_list.append(result2)
    
    if len(set(sum_list))>1:
        return False
    return True

m=[[7, 12, 1, 14], [2, 13, 8, 11], [16, 3, 10, 5], [9,
6, 15, 4]]
print(magic_square_test(m))

m=[[2, 7, 6], [9, 5, 1], [4, 3, 8]]
print(magic_square_test(m))

m=[[2, 7, 6], [9, 5, 1], [4, 3, 7]]
print(magic_square_test(m))



nqueen:
global N
N = 8

def printout(board):
    l=list()
    for i in range(N):
        for j in range(N):
            if board[i][j] == 1:
                print ('X', end = " ")
                l.append(j)
            else:
                print('O',end=" ")
        print()

   
        
def Safetomove(board, row, col):
    for i in range(col):
        if board[row][i] == 1:
            return False

    for i, j in zip(range(row, -1, -1),range(col, -1, -1)):
        if board[i][j] == 1:
            return False

    for i, j in zip(range(row, N, 1),range(col, -1, -1)):
        if board[i][j] == 1:
            return False

    return True

def NQ(board, col):
    if col >= N:
        return True
    for i in range(N):
        if Safetomove(board, i, col):
            board[i][col] = 1
            if NQ(board, col + 1) == True:
                return True
            board[i][col] = 0
    return False

def solveNQ():
    board = [ [0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0]]

    if NQ(board, 0) == False:
        printout("Solution does not exist")
        return False

    printout(board)
    return True

solveNQ()


TicTacToe:
player, opponent = 'x', 'o'

# This function returns true if there are moves
# remaining on the board. It returns false if
# there are no moves left to play.
def isMovesLeft(board) :

    for i in range(3) :
        for j in range(3) :
            if (board[i][j] == '_') :
                return True
    return False

def evaluate(b) :
# Checking for Rows for X or O victory.
    for row in range(3) :
        if (b[row][0] == b[row][1] and b[row][1] == b[row][2]) :
            if (b[row][0] == player) :
                return 10
            elif (b[row][0] == opponent) :
                return -10

# Checking for Columns for X or O victory.
    for col in range(3) :
        if (b[0][col] == b[1][col] and b[1][col] == b[2][col]) :
            if (b[0][col] == player) :
                return 10
            elif (b[0][col] == opponent) :
                return -10

# Checking for Diagonals for X or O victory.
    if (b[0][0] == b[1][1] and b[1][1] == b[2][2]) :
        if (b[0][0] == player) :
            return 10
        elif (b[0][0] == opponent) :
            return -10

    if (b[0][2] == b[1][1] and b[1][1] == b[2][0]) :
        if (b[0][2] == player) :
            return 10
        elif (b[0][2] == opponent) :
            return -10

    # Else if none of them have won then return 0
    return 0

# This is the minimax function. It considers all
# the possible ways the game can go and returns
# the value of the board
def minimax(board, depth, isMax) :
    score = evaluate(board)
    # If Maximizer has won the game return his/her
    # evaluated score
    if (score == 10) :
        return score
    
    # If Minimizer has won the game return his/her
    # evaluated score
    if (score == -10) :
        return score
    
    # If there are no more moves and no winner then
    # it is a tie
    if (isMovesLeft(board) == False) :
        return 0
    
    # If this maximizer's move
    if (isMax) :
        best = -1000
        
        # Traverse all cells
        for i in range(3) :
            for j in range(3) :
        
        # Check if cell is empty
                if (board[i][j]=='_') :
        
        # Make the move
                    board[i][j] = player
        
        # Call minimax recursively and choose
        # the maximum value
                    best = max( best, minimax(board,
                                              depth + 1,
                                              not isMax) )
        
        # Undo the move
                    board[i][j] = '_'
        return best

# If this minimizer's move
    else :
        best = 1000
        
        # Traverse all cells
        for i in range(3) :
            for j in range(3) :
        
        # Check if cell is empty
                if (board[i][j] == '_') :
        
        # Make the move
                    board[i][j] = opponent
        
        # Call minimax recursively and choose
        # the minimum value
                    best = min(best, minimax(board, depth + 1, not isMax))
        
        # Undo the move
                    board[i][j] = '_'
        return best

# This will return the best possible move for the player
def findBestMove(board) :
    bestVal = -1000
    bestMove = (-1, -1)
    
    # Traverse all cells, evaluate minimax function for
    
    # all empty cells. And return the cell with optimal
    # value.
    for i in range(3) :
        for j in range(3) :
        
        # Check if cell is empty
            if (board[i][j] == '_') :
            
                # Make the move
                board[i][j] = player
                
                # compute evaluation function for this
                # move.
                moveVal = minimax(board, 0, False)
                
                # Undo the move
                board[i][j] = '_'
                
                # If the value of the current move is
                # more than the best value, then update
                # best/
                if (moveVal > bestVal) :
                    bestMove = (i, j)
                    bestVal = moveVal
    
    print("The value of the best Move is :", bestVal)
    print()
    return bestMove
# Driver code
board = [
[ 'x', 'o', 'x' ],

[ 'o', 'o', '_' ],
[ '_', '_', 'x' ]
]

bestMove = findBestMove(board)
board[bestMove[0]][bestMove[1]]='x'
print("The Optimal Move is :")
print("ROW:", bestMove[0]+1, " COL:", bestMove[1]+1)
for i in board:
    print(i)
 
    

WaterJug:
# This function is used to initialize the
# dictionary elements with a default value.
from collections import defaultdict
# jug1 and jug2 contain the value
# for max capacity in respective jugs
# and aim is the amount of water to be measured.
jug1, jug2, aim = 4, 3, 2
# Initialize dictionary with
# default value as false.
visited = defaultdict(lambda: False)
# Recursive function which prints the
# intermediate steps to reach the final
# solution and return boolean value
# (True if solution is possible, otherwise False).
# amt1 and amt2 are the amount of water present
# in both jugs at a certain point of time.
def waterJugSolver(amt1, amt2):
# Checks for our goal and
# returns true if achieved.
    if (amt1 == aim and amt2 == 0) or (amt2 == aim and amt1 == 0):
    
        print(amt1, amt2)
        return True
# Checks if we have already visited the
# combination or not. If not, then it proceeds further.
    if visited[(amt1, amt2)] == False:
        print(amt1, amt2)
        # Changes the boolean value of
        # the combination as it is visited.
        visited[(amt1, amt2)] = True
        # Check for all the 6 possibilities and
        # see if a solution is found in any one of them.
        return (waterJugSolver(0, amt2) or
        waterJugSolver(amt1, 0) or
        waterJugSolver(jug1, amt2) or
        waterJugSolver(amt1, jug2) or
        waterJugSolver(amt1 + min(amt2, (jug1-amt1)),
        amt2 - min(amt2, (jug1-amt1))) or
        waterJugSolver(amt1 - min(amt1, (jug2-amt2)),
        amt2 + min(amt1, (jug2-amt2))))
    
    # Return False if the combination is
    # already visited to avoid repetition otherwise
    # recursion will enter an infinite loop.
    else:
        return False

print("Steps: ")
# Call the function and pass the
# initial amount of water present in both jugs.
waterJugSolver(0, 0)