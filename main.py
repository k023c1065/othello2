from game import othello_class,minimax_search2,ab_search
from modellib import *
from gui import othello_gui
from glob import glob
import os
import numpy as np
class main_controler:
    def __init__(self):
        self.gui = othello_gui()
        self.gui.add_click_event(self.click_event)
        self.game = othello_class(undo_flg=True)
        self.gui.update_board(self.game.board)
        self.HUMAN_TURN=1
        self.AI=minimax_search2(model_class())
    def click_event(self,args):
        if self.game.turn == self.HUMAN_TURN:
            x,y = args.x//90,args.y//90
            if self.try_move(x,y):
                self.gui.update_board(self.game.board)
                
        if self.game.turn != self.HUMAN_TURN:
            score,ai_move = self.AI.search(self.game,1,3)
            self.game.apply_move(*ai_move)
            self.gui.update_board(self.game.board)
        if self.game.check_winner() !=0:
            pass
                
    def try_move(self,x,y):
        if (x,y) in self.game.get_valid_moves():
            self.game.apply_move(x,y)
            return True
        if self.game.get_valid_moves() == []:
            self.game.apply_move(-1,-1)
            return True
        return False
    def start(self):
        self.gui.mainloop()
        pass

if __name__ == "__main__":
    controller = main_controler()
    controller.start()
    input("Waiting...")
    import random,time
    s_t=time.time()
    itr=0
    dataset=[]
    model = model_class()
    result = [0,0]
    ab = ab_search(model)
    minimax = minimax_search2()
    while True:
        print("Starting the game...")
        game = othello_class(undo_flg=True)
        data=[]
        while game.check_winner() == 0:
            print(game.print_human_view())
            print("turn:",game.turn,"symbol:",game.get_symbol(game.turn))
            valid_moves=game.get_valid_moves()
            print("valid moves:",valid_moves)
            
            if len(valid_moves)>0:
                if game.turn==1:
                    s_t=time.time()
                    print("AI is thinking...",end="")
                    #ab.reset()
                    # print("---------------AB--------------")
                    #ab_score,(x, y) = ab.search(game=game,my_turn=1,depth=2)
                    # print("-------------minimax-----------")
                    mini_score,(x,y) = minimax.search(game=game,my_turn=1,depth=2,model=model)
                    
                    # print(f"AB:{x},{y},{ab_score} \nMinimax:{i},{j},{mini_score}")
                    # if not(i==x and y==j):
                    #     input("Press Enter to continue...")
                    print(f"Done. Took {time.time()-s_t:.2f}sec(s)")
                else:
                    while True:
                        try:
                            x ,y=random.choice(valid_moves)
                            x ,y = map(int,input("Your turn.Enter your next move:").split())
                            if (x,y) in valid_moves:
                                break
                            print("[ERROR]Invalid move. Please enter a valid move.")   
                        except ValueError:
                            print("[ERROR]Unexpected input. Please enter two integers separated by a space.")  
                            pass
                # x,y=random.choice(valid_moves)
            else:
                x,y=-1,-1
            data.append((x,y))
            print("Applying move...", x,",", y)
            game.apply_move(x, y)
            
        #print(f"score:{game.get_score()}")
        itr+=1
        dataset.append((data,game.get_score()))
        print("\r",(time.time()-s_t)/itr,end="")
        print("Game finished.")
        
        score = game.get_score()
        print(f"Score:{score[0]}vs{score[1]}")
        print("Result:",end=" ")
        if score[0]>score[1]:
            print("Player 1 wins!")
        elif score[0]<score[1]:
            print("Player 2 wins!")
        else:
            print("Draw!")
        input("Press Enter to continue...")