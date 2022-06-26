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
