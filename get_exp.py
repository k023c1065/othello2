import numpy as np
from game import format_board,othello_class
import random,multiprocessing
def get_move(q_result,valid_moves):
    q = -100
    move = (-1,-1)
    weights = []
    for m in valid_moves:
        weights.append(q_result[move[0]][move[1]])
    weights = np.array(weights)
    weights = np.exp(weights)/np.exp(weights).sum()
    return random.choices(valid_moves,weights=weights,k=1)[0]

def get_exp_single(game_num,pipe,result_pipe):
        exp_memory = []
        win_score = [0,0]
        pipe_send_count = 0
        for _game_num in range(game_num):
            #print(f"process name:{multiprocessing.current_process().name} Started")
            game = othello_class()
            turn = np.random.randint(0,2)
            turn = [1,-1][turn]
            game_exp = []
            while game.check_winner() == 0:
                valid_moves = game.get_valid_moves()
                move = (-1,-1)
                if len(valid_moves)>0:
                    pipe.send(format_board(game.board))
                    pipe_send_count += 1
                    r = pipe.recv()
                    r = r.numpy().reshape(8,8)
                    move = get_move(r,valid_moves=valid_moves)
                else:
                    move = (-1,-1)
                game_exp.append((game.board,move,valid_moves))
                game.apply_move(*move)
            score = game.get_score()
            first_win_ratio = score[0]/sum(score)
            if first_win_ratio**turn>=1:
                win_score[0]+=1
            else:
                win_score[1]+=1
            turn = 1
            for exp in game_exp:
                if len(exp[2])>0:
                    r = np.zeros((8,8))
                    selected_move_q = first_win_ratio
                    if first_win_ratio**(turn) < 1:
                        selected_move_q = 1-selected_move_q
                    for move in exp[2]:
                        if move == exp[1]:
                            r[move[0]][move[1]] = selected_move_q
                        else:
                            r[move[0]][move[1]] = max(1-selected_move_q,0.00001)
                    r = r/r.sum()
                    r = r.reshape(64)
                    #print(format_board(exp[0]).shape)
                    exp_memory.append(
                        [format_board(exp[0]),r]
                    )
                turn += 1
            #print(f"process name:{multiprocessing.current_process().name} game_num:{_game_num} win_score:{win_score} pipe_send_count:{pipe_send_count}")
        for _ in range(60*game_num-pipe_send_count):
            print(f"proc name:{multiprocessing.current_process().name} pipe send empty")
            pipe.send(np.empty((2,8,8)))
            pipe.recv()
            print(f"proc name:{multiprocessing.current_process().name} pipe send empty done")
        result_pipe.send((exp_memory,win_score))
        print(f"process name:{multiprocessing.current_process().name} Finished")
        return exp_memory,win_score