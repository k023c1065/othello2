import tensorflow as tf
import numpy as np
from tqdm import tqdm
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
        
def train(model,gxp):
    EPOCH = 20
    loss_object = special_MSE()
    for e in range(EPOCH):
        if e%5==0:
            print("\nDataset shuffle")
            optimizer = tf.keras.optimizers.Adam()
            dataset = gxp.get_exp(1024*8)

            x,y = [],[]
            for data in dataset:
                x.append(data[0])
                y.append(data[1])
            x = np.array(x,dtype="float32")
            y = np.array(y,dtype="float32")
            batch_size=64
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