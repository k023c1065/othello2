from copy import deepcopy

class othello_class:
    def __init__(self,undo_flg=False):
        self.board = [[0 for i in range(8)] for j in range(8)]
        self.board[3][3] = 1
        self.board[4][4] = 1
        self.board[3][4] = -1
        self.board[4][3] = -1
        if undo_flg:
            self.old_board = deepcopy(self.board)
        self.undo_flg = undo_flg
        self.turn = 1
        self.passed = 0
        self.winner = 0
        self.valid_moves = []
        
    def is_valid_move(self, x, y):
        if x>=8 or x<0 or y>=8 or y<0:
            return False
        if self.board[x][y] != 0:
            return False
        
        directions = [(1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1)]
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            found_opponent = False
            while 0 <= nx < 8 and 0 <= ny < 8:
                if self.board[nx][ny] == -self.turn:
                    found_opponent = True
                elif self.board[nx][ny] == self.turn:
                    if found_opponent:
                        return True
                    else:
                        break
                else:
                    break
                nx += dx
                ny += dy
        return False
    def get_valid_moves(self):
        valid_moves = []
        for x in range(8):
            for y in range(8):
                if self.is_valid_move(x, y):
                    valid_moves.append((x, y))
        return valid_moves
    def apply_move(self, x, y):
        if self.undo_flg:
            self.old_board = deepcopy(self.board)
        if not self.is_valid_move(x, y):
            self.passed += 1
            self.turn = -self.turn
            return False
        self.passed = 0
        self.board[x][y] = self.turn
        directions = [(1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1)]
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            cells_to_flip = []
            while 0 <= nx < 8 and 0 <= ny < 8:
                if self.board[nx][ny] == -self.turn:
                    cells_to_flip.append((nx, ny))
                elif self.board[nx][ny] == self.turn:
                    for cx, cy in cells_to_flip:
                        self.board[cx][cy] = self.turn
                    break
                else:
                    break
                nx += dx
                ny += dy
        self.turn = -self.turn
        return True
    def undo_move(self):
        self.board = deepcopy(self.old_board)
        self.turn = -self.turn
    def check_winner(self):
        if self.passed >= 2 or np.abs(self.board).sum() == 64:
            self.winner = 1 if sum([self.board[i][j] == 1 for i in range(8) for j in range(8)]) > sum([self.board[i][j] == -1 for i in range(8) for j in range(8)]) else -1
        return self.winner
    def get_score(self):
        score=[
            0,0
        ]
        for i in range(8):
            for j in range(8):
                if self.board[i][j]==1:
                    score[0]+=1
                elif self.board[i][j]==-1:
                    score[1]+=1
        return score
        
    def get_symbol(self,i):
        if i == 1:
            return "●"
        if i == -1:
            return "○"
        return "□"
    def print_human_view(self):
        s="\n".join([" ".join([self.get_symbol(i) for i in row]) for row in self.board])
        #Add row and column numbers
        s = "0 1 2 3 4 5 6 7\n" + s
        s = "\n".join([str(i-1) + " " + s.split("\n")[i] if i!=0 else "  " + s.split("\n")[i] for i in range(0,9)])
        return s
import numpy as np
def format_board(board):
    new_board=[[[0]*8 for _ in range(8)],[[0]*8 for _ in range(8)]]
    for i in range(8):
        for j in range(8):
            if board[i][j]==1:
                new_board[0][i][j]=1
            elif board[i][j]==-1:
                new_board[1][i][j]=1

    return np.transpose(np.array(new_board),axes=(1,2,0))
if __name__ == "__main__":
    import random,time
    s_t=time.time()
    itr=0
    dataset=[]
    while True:
        game = othello_class()
        data=[]
        while game.check_winner() == 0:
            print(game.print_human_view())
            print("turn:",game.turn)
            valid_moves=game.get_valid_moves()
            print(valid_moves)
            
            if len(valid_moves)>0:
                if game.turn==1:
                    x, y = map(int, input().split())
                else:
                    x ,y=random.choice(valid_moves)
                # x,y=random.choice(valid_moves)
            else:
                x,y=-1,-1
            data.append((x,y))
            print("Applying move...", x,",", y)
            game.apply_move(x, y)
        #print(f"score:{game.get_score()}")
        itr+=1
        dataset.append((data,game.get_score()))
        print("\r",itr/(time.time()-s_t),end="")