from game import othello_class,minimax_search,ab_search
from modellib import *
from glob import glob
import os
import numpy as np
import argparse
def load_model():
    model = miniResNet(input_shape=(2,8,8),output_dim=64)
    model(np.zeros((1,2,8,8)),training=False)
    model_files = glob("model/*.h5")
    #model_filesを更新日時順でソートする
    model_files.sort(key=os.path.getmtime)
    if len(model_files)>0:
        model.load_weights(model_files[-1])
    return model
def format_board(board):
    new_board=[[[0]*8 for _ in range(8)],[[0]*8 for _ in range(8)]]
    for i in range(8):
        for j in range(8):
            if board[i][j]==1:
                new_board[0][i][j]=1
            elif board[i][j]==-1:
                new_board[1][i][j]=1
    return np.array(new_board)
def get_move(model:miniResNet,board,movable_list):
    board = format_board(board)
    prediction = model(board[np.newaxis],training=False)
    prediction = prediction.numpy().reshape(8,8)
    score_list=[]
    for move in movable_list:
        x,y=move[0],move[1]
        #print(f"Move:{x},{y} Score:{prediction[x][y]}")
        score_list.append(prediction[x][y])
    best_move = random.choices(movable_list,weights = np.exp(score_list),k=1)[0]
    return best_move[0],best_move[1]
if __name__ == "__main__":
    import random,time
    s_t=time.time()
    itr=0
    dataset=[]
    model = model_class()
    result = [0,0]
    ab = ab_search(model)
    minimax = minimax_search()
    while True:
        print("Starting the game...")
        game = othello_class(undo_flg=True)
        data=[]
        while game.check_winner() == 0:
            print(game.print_human_view())
            print("turn:",game.turn,"symbol:",game.get_symbol(game.turn))
            valid_moves=game.get_valid_moves()
            print(valid_moves)
            
            if len(valid_moves)>0:
                if game.turn==1:
                    s_t=time.time()
                    print("AI is thinking...",end="")
                    ab.reset()
                    # print("---------------AB--------------")
                    ab_score,(x, y) = ab.search(game=game,my_turn=1,depth=2)
                    # print("-------------minimax-----------")
                    # mini_score,(i,j) = minimax.search(game=game,my_turn=1,depth=2,model=model)
                    
                    # print(f"AB:{x},{y},{ab_score} \nMinimax:{i},{j},{mini_score}")
                    # if not(i==x and y==j):
                    #     input("Press Enter to continue...")
                    print(f"Done. Took {time.time()-s_t:.2f}sec(s)")
                else:
                    while True:
                        try:
                            x ,y=random.choice(valid_moves)
                            x ,y = map(int,input().split())
                            if (x,y) in valid_moves:
                                break
                        except ValueError:
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
        print("Score:",score)
        if score[0]>score[1]:
            result[0]+=1
        else:
            result[1]+=1
        print("Result:",result)
        input("Press Enter to continue...")