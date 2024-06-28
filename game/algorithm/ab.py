import numpy as np
from copy import copy,deepcopy
import random
from ..othello import othello_class,format_board  # Import necessary modules

class ab_search:
    def __init__(self,model,isDebug=False):
        self.reset()
        self.model=model
        self.debug = isDebug
    def reset(self):
        pass
    def search(self, game:othello_class, my_turn, depth,alpha=None,beta=None):
        
        game = deepcopy(game)  # Create a deep copy of the game object
        if alpha is None:
            alpha = -100
        else:
            alpha = copy(alpha)
        if beta is None:
            beta = 100
        else:
            beta = copy(beta)
        if game.check_winner() != 0:  # Check if there is a winner
            score = int(game.check_winner() == game.turn)
            if game.turn == my_turn:
                self.alpha = max(alpha, score)
            else:
                self.beta = min(beta, score)
            return score, (-1, -1)  # Return the winner and an invalid move
        
        if depth == 0:  # Check if the maximum depth has been reached
            prediction = self.model.predict(format_board(game.board)[np.newaxis], training=False).reshape(8, 8)  # Make a prediction using the model
            best_score = -100  # Initialize the score to a very low value
            best_move = -1, -1  # Initialize the best move to an invalid move
            valid_moves = game.get_valid_moves()  # Get all valid moves
            random.shuffle(valid_moves)
            for move in valid_moves:  # Iterate over all valid moves
                x, y = move
                score = prediction[x][y]  # Get the predicted score for the move
                
                if self.debug:print("--"+f"{x},{y} Score:{score}")
                if score > best_score:  # Check if the predicted score is higher than the current score
                    best_score = prediction[x][y]  # Update the score
                    best_move = move  # Update the best move
            return best_score, best_move  # Return the score and the best move
        else:  # If the maximum depth has not been reached
            if game.turn == my_turn:  # Check if it's the AI's turn
                best_score = -100  # Initialize the score to a very low value
                best_move = -1, -1  # Initialize the best move to an invalid move
                for move in game.get_valid_moves():  # Iterate over all valid moves
                    x, y = move
                    if self.debug:print((2-depth)*"-"+f"a {x},{y}")
                    game.apply_move(x, y)  # Apply the move to the game board
                    score,_ = self.search(game, my_turn, depth-1,alpha,beta)  # Recursively call the search function with a reduced depth
                    alpha = max(alpha,score)
                    game.undo_move()  # Undo the move
                    if score > best_score:
                        best_score = score
                        best_move = move
                    if self.debug:print((2-depth)*"-"+f"Score:{score} beta:{beta}")
                    if alpha >= beta:
                        break
                return score, best_move  # Return the score and the best move
            
            else:  # If it's the opponent's turn
                best_score = 100  # Initialize the score to a very high value
                best_move = -1, -1  # Initialize the best move to an invalid move
                for move in game.get_valid_moves():  # Iterate over all valid moves
                    x, y = move
                    if self.debug:print((2-depth)*"-"+f"b {x},{y}")
                    game.apply_move(x, y)  # Apply the move to the game board
                    score,_=self.search(game, my_turn, depth-1,alpha,beta)  # Recursively call the search function with a reduced depth
                    game.undo_move()  # Undo the move
                    if score < best_score:
                        best_score = score
                        best_move = move
                    if self.debug:print((2-depth)*"-"+f"Score:{score} alpha:{alpha}")
                    beta = min(beta, score)  # Update the beta value
                    if alpha>=beta:
                        break
                    
                
                return best_score, best_move  # Return the score and the best move