import numpy as np
from game import format_board,othello_class
import random,multiprocessing,time
from tqdm import tqdm
import random
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
        if min(len(self.memory),num)<num:print("Warning: Memory is not enough") 
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
            daemon = True
            ) for i in range(proc_num)]
        for process in self.processes:
            process.start()
        self.model_executer(game_num,proc_num)

        
        order_win=[0,0] 
        for pipe in result_pipe:
            while not pipe[1].poll():
                pass
            exp,win_score = pipe[1].recv()
            tmp_exp += exp
            tmp_score[0] += np.sum(win_score[0])
            tmp_score[1] += np.sum(win_score[1])
            
            order_win[0] += np.sum(win_score,axis=0)[0]
            order_win[1] += np.sum(win_score,axis=0)[1]
        
        self.latest_memory = tmp_exp
        print(f"Final win_score:{tmp_score}")
        print(f"Final Order win_score:{order_win}")
        for process in self.processes:
            print("Join process")
            process.join()
            process.close()
        for pipe in self.pipes:
            pipe[0].close()
            pipe[1].close()

        
        self.pipes = []

        return tmp_score
    def model_executer(self,num,proc_num):
        num = 60*num
        input_x = [np.zeros((proc_num,8,8,2)),np.zeros((proc_num,8,8,2))]
        turn2index = {
            1:0,
            -1:1,
        }
        tqdm_iter = tqdm(range(num*proc_num))
        for _ in range(num):
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
                if not isinstance(output[0],np.ndarray):
                    output[0] = output[0].numpy()
            if model_flg[1]:
                output[1] = self.model[1](input_x[1])
                if not isinstance(output[1],np.ndarray):
                    output[1] = output[1].numpy()
            for i,pipe in enumerate(self.pipes):
                #print(f"pipe send id:{i}")
                turn = turn_data[i]
                pipe[1].send(output[turn][i])
            tqdm_iter.update(proc_num)
    @staticmethod
    def get_exp_single(game_num,pipe,result_pipe):
        print(f"Proc name:{multiprocessing.current_process().name} Starting")
        exp_memory = []
        win_score = [[0,0],[0,0]]
        pipe_send_count = 0
        this_skip_count = 0
        
        for _game_num in range(game_num):
            #print(f"process name:{multiprocessing.current_process().name} Started")
            game = othello_class()
            random.seed(time.time())
            random.seed(random.randint(0,2**32))
            turn = random.choice([1,-1])
            game_exp = []
            while game.check_winner() == 0:
                valid_moves = game.get_valid_moves()
                move = (-1,-1)
                if len(valid_moves)>0:
                    this_turn = game.turn if turn == 1 else -game.turn
                    pipe.send((this_turn,format_board(game.board)))
                    pipe_send_count += 1
                    r = pipe.recv()
                    r = r.reshape(8,8)
                    move = get_move(r,valid_moves=valid_moves)
                else:
                    this_skip_count += 1
                    move = (-1,-1)
                game_exp.append((game.board,move,valid_moves))
                game.apply_move(*move)
            score = game.get_score()
            first_win_ratio = score[0]/sum(score)
            if turn == 1:
                if score[0]>score[1]:
                    win_score[0][0]+=1
                else:
                    win_score[1][0]+=1
            elif turn == -1:
                if score[0]>score[1]:
                    win_score[1][1]+=1
                else:
                    win_score[0][1]+=1
            # if (score[0]>score[1] and turn == 1) or (score[0]<score[1] and turn == -1):
            #     win_score[0]+=1
            # elif (score[0]>score[1] and turn == -1) or (score[0]<score[1] and turn == 1):
            #     win_score[1]+=1
            #print(win_score)
            turn = 1
            for exp in game_exp:
                if len(exp[2])>0:
                    r = np.zeros((8,8))
                    selected_move_q = first_win_ratio
                    if turn == -1:
                        selected_move_q = 1-selected_move_q
                    if True or selected_move_q >0.5:
                        for move in exp[2]:
                            if move == exp[1]:
                                r[move[0]][move[1]] = selected_move_q
                            else:
                                r[move[0]][move[1]] = 1 - selected_move_q
                        #r = np.exp(r)/np.exp(r).sum()
                        if np.sum(r)>0:
                            r = r/np.sum(r)
                            exp_memory.append(
                                [format_board(exp[0]),r.reshape(64)]
                            )
                            #Data Augmentation
                            exp_memory.append(
                                [format_board(np.rot90(exp[0],1)),np.rot90(r,1).reshape(64)]
                            )
                            exp_memory.append(
                                [format_board(np.rot90(exp[0],2)),np.rot90(r,2).reshape(64)]
                            )
                            exp_memory.append(
                                [format_board(np.rot90(exp[0],3)),np.rot90(r,3).reshape(64)]
                            )
                            exp_memory.append(
                                [format_board(np.flipud(exp[0])),np.flipud(r).reshape(64)]
                            )
                            exp_memory.append(
                                [format_board(np.fliplr(exp[0])),np.fliplr(r).reshape(64)]
                            )
                turn *= -1
            #print(f"process name:{multiprocessing.current_process().name} game_num:{_game_num} win_score:{win_score} pipe_send_count:{pipe_send_count}")
        for _ in range(60*game_num-pipe_send_count):
            #print(f"proc name:{multiprocessing.current_process().name} pipe send empty")
            pipe.send((1,np.empty((8,8,2))))
            pipe.recv()
            #print(f"proc name:{multiprocessing.current_process().name} pipe send empty done")
        result_pipe.send((exp_memory,win_score))
        result_pipe.close()
        print(f"process name:{multiprocessing.current_process().name} Finished")
        print(f"process name:{multiprocessing.current_process().name} win score:{np.sum(win_score[0])}vs{np.sum(win_score[1])}")
        return exp_memory,win_score