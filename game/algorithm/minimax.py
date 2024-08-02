import time
from ..othello import othello_class,format_board
import numpy as np
from copy import copy,deepcopy

class minimax_search:
    def __init__(self,load_model_before=False,isDebug=False):
        self.debug = isDebug
        self.predict_data_dict = dict()
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
            
class minimax_search2:
    def __init__(self,model,isDebug=False):
        self.debug = isDebug
        self.predict_input = []
        self.predict_data_dict = dict()
        self.model = model
        self.hash_main =np.array([[2**i for i in range(8)] for j in range(8)],dtype="uint8")
    def search(self, game:othello_class,my_turn, depth):
        #Get all board needed
        self.predict_input = []
        if self.debug:print("Start search")
        s_t = time.time()
        self.get_score_search(game,my_turn,depth)
        if self.debug:print(f"get board time:{time.time()-s_t}")
        #Prediction
        if self.debug:print("Get score")
        s_t = time.time()
        self.predict_data_dict = dict()
        self.get_score(self.model)
        if self.debug:print(f"get score time:{time.time()-s_t}")
        #Main search
        if self.debug:print("Main search")
        s_t = time.time()
        r = self.main_search(game,my_turn,depth)
        print(r)
        if self.debug:print(f"Main search time:{time.time()-s_t}")
        if self.debug:
            simple_r = self.model.predict(format_board(game.board)[np.newaxis],verbose=0).reshape(8,8)
            valid_move = game.get_valid_moves()
            print("---simple_r---")
            print(simple_r)
            for m in valid_move:
                print(m,simple_r[m[0]][m[1]])
            print("---simple_r end---")
            
        return r
    def get_board_hash(self,board):
        a = tuple(board.flatten())
        #print(a)
        return hash(a)
    def get_score_search(self, game:othello_class,my_turn, depth):
        game = deepcopy(game)
        if game.check_winner() != 0:
            return int(game.check_winner()==game.turn),(-1,-1)
        if depth == 0:
            self.predict_input.append(format_board(game.board))
        else:
            valid_moves = game.get_valid_moves()
            for move in valid_moves:
                game.apply_move(*move)
                self.get_score_search(game,my_turn,depth-1)
                game.undo_move()
    def get_score(self,model):
        if len(self.predict_input) == 0:
            return
        if self.debug:print("Predicting...")
        if self.debug:print(f"Predict input:{len(self.predict_input)}")
        s_t = time.time()
        prediction = model.predict(np.array(self.predict_input),verbose=0)
        if self.debug:print(f"Predict time:{time.time()-s_t}")
        if self.debug:print("Saving")
        s_t = time.time()
        for i in range(len(self.predict_input)):
            self.predict_data_dict[self.get_board_hash(self.predict_input[i])] = prediction[i]    
        if self.debug:print(f"Save time:{time.time()-s_t}")
    def main_search(self, game:othello_class,my_turn, depth):
        game = deepcopy(game)
        if game.check_winner() != 0:
            return int(game.check_winner()==game.turn),(-1,-1)
        if depth == 0:
            #prediction = model.predict(format_board(game.board)[np.newaxis],verbose=0).reshape(8,8)
            prediction = self.predict_data_dict[self.get_board_hash(format_board(game.board))].reshape(8,8)
            best_score = -100
            best_move = -1,-1
            for move in game.get_valid_moves():
                x,y=move
                score = prediction[x][y]
                if game.turn != my_turn:
                    score = -score
                if self.debug:print("--"+f"{x},{y} Score:{score}")
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
                    if self.debug:print((2-depth)*"-"+f"a {x},{y}")
                    score = self.main_search(game,my_turn,depth-1)[0]
                    if score>best_score:
                        best_score = score
                        best_move = move
                    if self.debug:print((2-depth)*"-"+f"Score:{score}")
                    game.undo_move()
                return best_score,best_move
            else:
                best_score = 100
                best_move = -1,-1
                for move in game.get_valid_moves():
                    x,y=move
                    game.apply_move(x,y)
                    if self.debug:print((2-depth)*"-"+f"b {x},{y}")
                    score = self.main_search(game,my_turn,depth-1)[0]
                    if score < best_score:
                        best_score = score
                        best_move = move
                    if self.debug:print((2-depth)*"-"+f"Score:{score}")
                    game.undo_move()
                return best_score,best_move