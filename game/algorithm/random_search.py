from ..othello import othello_class,format_board
import numpy as np
class random_class:
    def __init__(self,load_model_before=False,isDebug=False,model=None):
        self.debug = isDebug
        if model is None:
            self.model = np.ones((8,8))
        else:
            self.model = model
    def search(self, game:othello_class,my_turn, depth):
        valid_moves = game.get_valid_moves()
        