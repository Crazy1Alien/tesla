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