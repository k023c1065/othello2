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


    

def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game_num",type=int,default=100)
    parser.add_argument("--proc_num",type=int,default=4)
    parser.add_argument("--exp_size",type=int,default=1024*8)
    parser.add_argument("--patience",type=int,default=3)
    parser.add_argument("--shuffle_num",type=int,default=5)
    parser.add_argument("--batch_size",type=int,default=64)
    parser = parser.parse_args()
    print(f"""
          game_num:{parser.game_num}\n
          proc_num:{parser.proc_num}\n
          exp_size:{parser.exp_size}\n
          patience:{parser.patience}\n
          shuffle_num:{parser.shuffle_num}\n
          batch_size:{parser.batch_size}\n  
          
          """)
    return {
        "game_num":parser.game_num,
        "proc_num":parser.proc_num,
        "exp_size":parser.exp_size,
        "patience":parser.patience,
        "shuffle_num":parser.shuffle_num,
        "batch_size":parser.batch_size
    }

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    global tf
    arg = parse_arg()
    gxp = exp_memory_class(miniResNet(input_shape=(2,8,8),output_dim=64),miniResNet(input_shape=(2,8,8),output_dim=64))
    trainer = trainer_class(
        patience=arg["patience"],
        shuffle_num=arg["shuffle_num"],
        dataset_size=arg["exp_size"],
        batch_size=arg["batch_size"]
    )
    model_files = glob("model/*.h5")
    model_name_seed= str(random.randint(0,2**62))
    target_model = miniResNet(input_shape=(2,8,8),output_dim=64)
    target_model(np.zeros((1,2,8,8)),training=False)
    
    #model_filesを更新日時順でソートする
    model_files.sort(key=os.path.getmtime)
    # if len(model_files)>0:
    #     target_model.load_weights(model_files[-1])
    #     print(f"model loaded:{model_files[-1]} with updated date{os.path.getmtime(model_files[-1])}")
    best_model = miniResNet(input_shape=(2,8,8),output_dim=64)
    
    gxp.update_model(target_model=target_model,best_model=best_model)
    generation_no = 0
    while True:
        print(f"generation_no:{generation_no}")
        score = gxp.create_exp(game_num=arg["game_num"],proc_num=arg["proc_num"])
        
        print(f"score:{score[0]/sum(score):3f}")
        if score[0]/sum(score)>=0.55:
            print("Updated!")
            if generation_no>0:target_model.save_weights(f"model/model_{model_name_seed}_gen{generation_no}.h5")
            gxp.reset()
            generation_no +=1
            best_model=target_model
        else:
            pass
            #target_model=best_model
        gxp.update_model(target_model=target_model,best_model=best_model)
        gxp.join_exp()
        target_model = trainer.train(target_model,gxp)
        print("Training complete")
        