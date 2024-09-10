import random
from game import othello_class,minimax_search2,ab_search,format_board
from modellib import *
from gui import othello_gui,game_mode
from glob import glob
import os
import numpy as np
class main_controler:
    def __init__(self):
        self.gui = othello_gui()
        self.gui.add_click_event(self.click_event)
        self.game = othello_class(undo_flg=True)
        self.gui.update_board(self.game.board)
        self.HUMAN_TURN=-1
        self.AI=minimax_search2(model_class(),isDebug=True)
        
    def click_event(self,args):
        if self.gui.game_mode == game_mode.TITLE:
            # 先手後手の選択
            if args.x <360:
                self.HUMAN_TURN = 1
            else:
                self.HUMAN_TURN = -1
            self.gui.reset_canvas()
            self.gui.game_mode = game_mode.GAME

            self.gui.update_board(self.game.board)
        elif self.gui.game_mode == game_mode.GAME:
            if self.game.turn == self.HUMAN_TURN:
                x,y = args.x//90,args.y//90
                # moves = self.game.get_valid_moves()
                # x,y = random.choice(moves)
                if self.try_move(x,y):
                    self.gui.update_board(board = self.game.board)
            if self.game.check_winner() !=0:
                print("Game finished.")
                print(f"Result:{self.game.get_score()}")
                self.gui.game_mode = game_mode.GAMEOVER
                self.gui.update_board(score = self.game.get_score())
                return        
            if self.game.turn != self.HUMAN_TURN:
                score,ai_move = self.AI.search(self.game,self.game.turn,2)
                print(score)
                self.game.apply_move(*ai_move)
                self.gui.update_board(board = self.game.board)
            if self.game.check_winner() !=0:
                print("Game finished.")
                print(f"Result:{self.game.get_score()}")
                self.gui.game_mode = game_mode.GAMEOVER
                self.gui.update_board(score = self.game.get_score())
                return
                
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
    