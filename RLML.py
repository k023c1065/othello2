from sklearn.discriminant_analysis import softmax
from game import othello_class,format_board
from modellib import model_class,miniResNet
import numpy as np
import os
from glob import glob
import random
import tensorflow as tf
from tqdm import tqdm
def get_move(model,board,valid_moves):
    q_result = model(format_board(board)[np.newaxis]).numpy().reshape((8,8))
    q = -100
    move = (-1,-1)
    weights = []
    for m in valid_moves:
        weights.append(q_result[move[0]][move[1]])
    weights = np.array(weights)
    weights = np.exp(
        weights/weights.sum()
    )
    return random.choices(valid_moves,weights=weights,k=1)[0]
def get_exp(best_model,target_model):

    exp_memory = []
    win_score = [0,0]
    game_num =75
    for _game_num in tqdm(range(game_num)):
        game = othello_class()
        turn = np.random.randint(0,2)
        turn = [1,-1][turn]
        game_exp = []
        while game.check_winner() == 0:
            valid_moves = game.get_valid_moves()
            move = (-1,-1)
            if len(valid_moves)>0:
                if turn == game.turn:
                    move = get_move(target_model,game.board,valid_moves)
                else:
                    move = get_move(best_model,game.board,valid_moves)
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
    return exp_memory,win_score
class special_MSE:
    def __init__(self):
        pass
    def __call__(self,y, pred, n):
        # Step 2: Calculate the square of (y - pred)
        #print("y:",y.numpy())
        #print("pred:",pred.numpy())
        squared_diff = tf.square(y - pred)
        #print("squared_diff",squared_diff.numpy())
        # Step 3: Sum along axis 1
        sum_squared_diff = tf.reduce_sum(squared_diff, axis=1)
        #print("sum_squared_diff",sum_squared_diff.numpy().mean())
        # Step 4: Divide by n
        divided_by_n = sum_squared_diff / n
        #print("divided_by_n",divided_by_n.numpy().mean())
        # Step 5: Calculate the mean of the result
        result_mean = tf.reduce_mean(divided_by_n)
        #print("result_mean",result_mean)
        #input("Waiting...")
        return result_mean
        
def train(model,exp):
    EPOCH = 20
    loss_object = special_MSE()
    for e in range(EPOCH):
        if e%5==0:
            print("\nDataset shuffle")
            optimizer = tf.keras.optimizers.Adam()
            random.shuffle(exp)
            dataset = exp[:min(1024,len(exp))]

            x,y = [],[]
            for data in dataset:
                x.append(data[0])
                y.append(data[1])
            x = np.array(x,dtype="float32")
            y = np.array(y,dtype="float32")
            batch_size=16
            train_ds =  tf.data.Dataset.from_tensor_slices(
                (x, y)
                )
            train_ds = train_ds.shuffle(25000,reshuffle_each_iteration=True,seed=random.randint(0,2**32))
            train_ds = train_ds.batch(batch_size)
            train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
            
        tqdm_obj = tqdm(range(len(train_ds)))
        for x,y in train_ds:
            loss_array = []
            with tf.GradientTape() as tape:
                # training=True is only needed if there are layers with different
                # behavior during training versus inference (e.g. Dropout).
                predictions = model(x, training=True)
                poss_move = (y.numpy()/(y.numpy()+1e-30)).astype("float32")
                #clipped = tf.clip_by_value(poss_move,1e-10,1.0)

                loss = loss_object(
                    (y*poss_move)
                    ,tf.clip_by_value(predictions*poss_move,1e-20,1.0) #Avoid log(0)
                    ,tf.reduce_sum(poss_move,axis=1)
                    )
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(
                    zip(gradients, model.trainable_variables)
                    )
            loss_array.append(loss.numpy())
            tqdm_obj.update(1)
            tqdm_obj.set_description(f"loss:{np.mean(loss_array):.6f}")
        tqdm_obj.close()
    return model
    

if __name__ == "__main__":
    model_name_seed= str(random.randint(0,2**62))
    target_model = miniResNet(input_shape=(2,8,8),output_dim=64)
    target_model(np.zeros((1,2,8,8)),training=False)
    model_files = glob("model/*.h5")
    #model_filesを更新日時順でソートする
    model_files.sort(key=os.path.getmtime)
    if len(model_files)>0:
        target_model.load_weights(model_files[-1])
        print(f"model loaded:{model_files[-1]} with updated date{os.path.getmtime(model_files[-1])}")
    best_model = miniResNet(input_shape=(2,8,8),output_dim=64)
    generation_no = 0
    exp = []
    while True:
        print(f"generation_no:{generation_no}")
        tmp_exp,score = get_exp(best_model=best_model,target_model=target_model)
        
        print(f"score:{score[0]/sum(score):3f}")
        if score[0]/sum(score)>=0.55:
            print("Updated!")
            if generation_no>0:target_model.save_weights(f"model/model_{model_name_seed}_gen{generation_no}.h5")
            exp = []
            generation_no +=1
            best_model=target_model
        else:
            target_model=best_model
        exp += tmp_exp
        target_model = train(target_model,exp)
        print("Training complete")
        