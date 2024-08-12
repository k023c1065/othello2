from sklearn.discriminant_analysis import softmax
from game import othello_class,format_board
from modellib import model_class,miniResNet
import numpy as np
import os
from glob import glob
import random
import multiprocessing
from RLML_trainer import trainer_class
from tqdm import tqdm
import argparse
from multiprocessing import Lock
from get_exp import exp_memory_class
def get_move(q_result,valid_moves):
    q = -100
    move = (-1,-1)
    weights = []
    for m in valid_moves:
        weights.append(q_result[move[0]][move[1]])
    weights = np.array(weights)
    weights = np.exp(weights)/np.exp(weights).sum()
    return random.choices(valid_moves,weights=weights,k=1)[0]


def random_model(x):
    return np.ones((x.shape[0],64))

def parse_arg():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--game_num",type=int,default=100,help="Number of games to play")
    parser.add_argument("--proc_num",type=int,default=multiprocessing.cpu_count(),help="Number of processes to use. When under 1 it will be set to multiprocessing.cpu_count()")
    parser.add_argument("--exp_size",type=int,default=1024*8,help="Size of experience memory")
    parser.add_argument("--patience",type=int,default=3,help="Patience for early stopping. If set to 0, it will be set to epoch+1")
    parser.add_argument("--shuffle_num",type=int,default=5,help="Number of shuffle be done on a traing session")
    parser.add_argument("--batch_size",type=int,default=64,help="Batch size for training")
    parser.add_argument("--epoch",type=int,default=50,help="Number of max epoch for training.")  
    parser.add_argument("--init_model",action="store_true",help="If set, it will start from random model")
    parser.add_argument("--init_game_num",type=int,default=100,help="Number of games to play for initial model. Will be ignored if init_model is False")
    parser = parser.parse_args()
    print(f"""
          -----------------------------------
          game_num:{parser.game_num}\n
          proc_num:{parser.proc_num if parser.proc_num>0 else multiprocessing.cpu_count()}\n
          exp_size:{parser.exp_size}\n
          patience:{parser.patience if parser.patience>0 else parser.epoch+1}\n
          shuffle_num:{parser.shuffle_num}\n
          batch_size:{parser.batch_size}\n  
          epoch:{parser.epoch}\n
          init_model:{parser.init_model}\n
          init_game_num:{parser.init_game_num}
          -----------------------------------
          """)
    return {
        "game_num":parser.game_num,
        "proc_num":parser.proc_num if parser.proc_num>0 else multiprocessing.cpu_count(),
        "exp_size":parser.exp_size,
        "patience":parser.patience if parser.patience>0 else parser.epoch+1,
        "shuffle_num":parser.shuffle_num,
        "batch_size":parser.batch_size,
        "epoch":parser.epoch,
        "init_model":parser.init_model,
        "init_game_num":parser.init_game_num,
    }


def main():
    multiprocessing.set_start_method('spawn', force=True)
    global tf
    arg = parse_arg()
    gxp = exp_memory_class(miniResNet(input_shape=(8,8,2),output_dim=64),miniResNet(input_shape=(8,8,2),output_dim=64))
    trainer = trainer_class(
        patience=arg["patience"],
        shuffle_num=arg["shuffle_num"],
        dataset_size=arg["exp_size"],
        batch_size=arg["batch_size"],
        epoch=arg["epoch"],
    )
    model_files = glob("model/*.h5")
    model_name_seed= str(random.randint(0,2**62))
    target_model = miniResNet(input_shape=(8,8,2),output_dim=64,layer_num=8)
    target_model(np.zeros((1,8,8,2)),training=False)
    
    #model_filesを更新日時順でソートする
    model_files.sort(key=os.path.getmtime)
    if len(model_files)>0 and (not arg["init_model"]):
        target_model.load_weights(model_files[-1])
        print(f"model loaded:{model_files[-1]} with updated date{os.path.getmtime(model_files[-1])}")
    best_model = miniResNet(input_shape=(8,8,2),output_dim=64,layer_num=8)
    
    gxp.update_model(target_model=target_model,best_model=best_model)
    if arg["init_model"]:
        gxp.update_model(target_model=random_model,best_model=random_model)
    best_model = random_model
    generation_no = 0
    game_num  = arg["init_game_num"] if arg["init_model"] else arg["game_num"]
    while True:
        print(f"generation_no:{generation_no}")
        score = gxp.create_exp(game_num=game_num,proc_num=arg["proc_num"])
        if arg["init_model"]:
            game_num = arg["game_num"]
            arg["init_model"] = False
        #print(gxp.latest_memory[:2])
        print(f"score:{score[0]/sum(score):3f}")
        if score[0]/sum(score)>=0.55:
            print("Updated!")
            gxp.reset()
            generation_no +=1
            best_model=target_model
        else:
            pass
        target_model.save_weights(f"model/model_{model_name_seed}_gen{generation_no}.h5")
            #target_model=best_model
        gxp.join_exp()
        target_model = trainer.train(target_model,gxp)
        print("Training complete")
        
        gxp.update_model(target_model=target_model,best_model=target_model)
        
if __name__ == "__main__":
    main()