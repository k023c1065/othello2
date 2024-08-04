import numpy as np
from game import format_board,othello_class
import random,multiprocessing
from tqdm import tqdm
def get_move(q_result,valid_moves):
    q = -100
    move = (-1,-1)
    weights = []
    for m in valid_moves:
        weights.append(q_result[move[0]][move[1]])
    weights = np.array(weights)
    weights = np.exp(weights)/np.exp(weights.sum())
    return random.choices(valid_moves,weights=weights,k=1)[0]
    
class exp_memory_class:
    def __init__(self,target_model,best_model):
        self.memory = []
        self.latest_memory = []
        self.model = [target_model,best_model]
        self.pipes = []
    def update_model(self,target_model,best_model):
        self.model = [target_model,best_model]
    def reset(self):
        self.memory = []
    def get_exp(self,num=1024):
        random.shuffle(self.memory)
        return self.memory[:min(len(self.memory),num)]
    def join_exp(self):
        self.memory += self.latest_memory
        self.latest_memory = []
    def create_exp(self,game_num=100,proc_num=4):
        self.pipes = [multiprocessing.Pipe() for _ in range(proc_num)]
        result_pipe = [multiprocessing.Pipe() for _ in range(proc_num)]
        tmp_exp = []
        tmp_score = [0,0]
        self.processes = [multiprocessing.Process(
            target=self.get_exp_single,
            args=(game_num,self.pipes[i][0],result_pipe[i][0]),
            daemon = True) for i in range(proc_num)]
        for process in self.processes:
            process.start()
        self.model_executer(game_num,proc_num)

        for pipe in result_pipe:
            exp,win_score = pipe[1].recv()
            tmp_exp += exp
            tmp_score[0] += win_score[0]
            tmp_score[1] += win_score[1]
        if tmp_score[0]/sum(tmp_score)>=0.55:self.memory = []
        self.memory += exp
        print(f"Final win_score:{tmp_score}")
        for process in self.processes:
            print("Join process")
            process.join()
        return tmp_score

    def model_executer(self,num,proc_num):
        num = 60*num
        input_x = [np.zeros((proc_num,2,8,8)),np.zeros((proc_num,2,8,8))]
        turn2index = {
            -1:0,
            1:1,
        }
        for _ in tqdm(range(num)):
            turn_data = []
            model_flg =[False,False]
            for i,pipe in enumerate(self.pipes):
                #print(f"pipe recv id:{i}")
                turn,data = pipe[1].recv()
                turn = turn2index[turn]
                model_flg[turn]=True
                turn_data.append(turn)
                input_x[turn][i] = data
            
            output = [None,None]
            if model_flg[0]:
                output[0] = self.model[0](input_x[0])
            if model_flg[1]:
                output[1] = self.model[1](input_x[1])
            for i,pipe in enumerate(self.pipes):
                #print(f"pipe send id:{i}")
                turn = turn_data[i]
                pipe[1].send(output[turn][i])
    @staticmethod
    def get_exp_single(game_num,pipe,result_pipe):
        exp_memory = []
        win_score = [0,0]
        pipe_send_count = 0
        this_skip_count = 0
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
                    pipe.send((game.turn,format_board(game.board)))
                    pipe_send_count += 1
                    r = pipe.recv()
                    r = r.numpy().reshape(8,8)
                    move = get_move(r,valid_moves=valid_moves)
                else:
                    this_skip_count += 1
                    move = (-1,-1)
                game_exp.append((game.board,move,valid_moves))
                game.apply_move(*move)
            score = game.get_score()
            first_win_ratio = score[0]/sum(score)
            if (first_win_ratio>0.5 and turn == 1) or (first_win_ratio<0.5 and turn == -1):
                win_score[0]+=1
            else:
                win_score[1]+=1
            turn = 1
            for exp in game_exp:
                if len(exp[2])>0:
                    r = np.ones((8,8))
                    selected_move_q = first_win_ratio
                    if turn == -1:
                        selected_move_q = 1-selected_move_q
                    r = r*np.exp(1-selected_move_q)
                    r[exp[1][0]][exp[1][1]] = np.exp(selected_move_q)
                    r = r/(
                        np.exp(selected_move_q)+np.exp(1-selected_move_q)*(len(exp[2])-1)
                    )
                    r = r.reshape(64)
                    #print(format_board(exp[0]).shape)
                    exp_memory.append(
                        [format_board(exp[0]),r]
                    )
                turn *= -1
            #print(f"process name:{multiprocessing.current_process().name} game_num:{_game_num} win_score:{win_score} pipe_send_count:{pipe_send_count}")
        for _ in range(60*game_num-pipe_send_count):
            print(f"proc name:{multiprocessing.current_process().name} pipe send empty")
            pipe.send((1,np.empty((2,8,8))))
            pipe.recv()
            print(f"proc name:{multiprocessing.current_process().name} pipe send empty done")
        result_pipe.send((exp_memory,win_score))
        print(f"process name:{multiprocessing.current_process().name} Finished")
        return exp_memory,win_score