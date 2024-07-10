from game import othello_class,format_board
import multiprocessing,tqdm,random
from multiprocessing import Pool, freeze_support, RLock
import datetime,pickle
from modellib import miniResNet,model_class
from game import minimax_search,ab_search,minimax_search2
import numpy as np
def play_game(model=None):
    game = othello_class(undo_flg=True)
    data=[]
    minimax = minimax_search2(model)
    if model is not None:
        ai_turn  = random.choice([1,-1])
    while game.check_winner() == 0:
        valid_moves = game.get_valid_moves()
        #print(np.abs(game.board).sum())
        if len(valid_moves)>0:
            if ai_turn == game.turn:
                # board = format_board(game.board)
                # prediction = model(board[np.newaxis],training=False)
                # prediction = prediction.numpy().reshape(8,8)
                # weights = []
                # for move in valid_moves:
                #     x,y=move[0],move[1]
                #     weights.append(prediction[x][y])
                # x,y=random.choices(valid_moves,weights=weights,k=1)[0]
                
                _,(x,y) = minimax.search(game,game.turn,depth=2)
            else:
                x,y=random.choice(valid_moves)
        else:
            x,y=-1,-1
        game.apply_move(x,y)
        data.append((x,y))
    return data,(game.get_score())
def create_data(num=1000,pos=1):
    dataset=[]
    model_inst = model_class()
    for i in tqdm.tqdm(range(num),position=pos):
        data=play_game(model = model_inst)
        dataset.append(data)
    return dataset
def create_dataset(num=1000,proc_num=1):
    dataset = []
    if proc_num==1:
        dataset = create_data(num)
    else:
        freeze_support()
        pool = multiprocessing.Pool(proc_num,initializer=tqdm.tqdm.set_lock, initargs=(RLock(),))
        result = pool.starmap(create_data, [(num,i+1) for i in range(proc_num)])
        pool.close()
        pool.join()
        print("\n"*proc_num)
        for r in result:
            dataset+=r
    with open(f"dataset/data_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.dat","wb") as f:
        pickle.dump(dataset,f)
if __name__ == "__main__":
    dataset_num = 1500
    create_dataset(dataset_num,1)