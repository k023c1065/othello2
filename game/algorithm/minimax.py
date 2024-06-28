from ..othello import othello_class,format_board
import numpy as np
from copy import copy,deepcopy
class minimax_search:
    def __init__(self,load_model_before=False,isDebug=False):
        self.debug = isDebug
        
    def search(self, game:othello_class,my_turn, depth,model):
        game = deepcopy(game)
        if game.check_winner() != 0:
            return int(game.check_winner()==game.turn),(-1,-1)
        if depth == 0:
            prediction = model.predict(format_board(game.board)[np.newaxis],verbose=0).reshape(8,8)
            best_score = -100
            best_move = -1,-1
            for move in game.get_valid_moves():
                x,y=move
                score = prediction[x][y]
                print("--"+f"{x},{y} Score:{score}")
                if score > best_score:
                    best_score = score
                    best_move = move
            return best_score,best_move
        else:
            if game.turn == my_turn:
                best_score = -100
                best_move = -1,-1
                for move in game.get_valid_moves():
                    x,y=move
                    game.apply_move(x,y)
                    print((2-depth)*"-"+f"a {x},{y}")
                    score = self.search(game,my_turn,depth-1,model)[0]
                    if score>best_score:
                        best_score = score
                        best_move = move
                    print((2-depth)*"-"+f"Score:{score}")
                    game.undo_move()
                return best_score,best_move
            else:
                best_score = 100
                best_move = -1,-1
                for move in game.get_valid_moves():
                    x,y=move
                    game.apply_move(x,y)
                    print((2-depth)*"-"+f"b {x},{y}")
                    score = self.search(game,my_turn,depth-1,model)[0]
                    if score < best_score:
                        best_score = score
                        best_move = move
                    print((2-depth)*"-"+f"Score:{score}")
                    game.undo_move()
                return best_score,best_move