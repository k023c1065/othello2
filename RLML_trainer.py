
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import random

class trainer_class:
    def __init__(self,patience=3,shuffle_num=5,dataset_size=1024*8,batch_size=64):
        import tensorflow as tf
        self.tf = tf
        self.patience = patience
        self.shuffle_num = shuffle_num
        self.dataset_size = dataset_size
        self.batch_size = batch_size
    def get_loss(self,y,pred,n):
        # Step 2: Calculate the square of (y - pred)
        #print("y:",y.numpy())
        #print("pred:",pred.numpy())
        squared_diff = self.tf.square(y - pred)
        #print("squared_diff",squared_diff.numpy())
        # Step 3: Sum along axis 1
        sum_squared_diff = self.tf.reduce_sum(squared_diff, axis=1)
        #print("sum_squared_diff",sum_squared_diff.numpy().mean())
        # Step 4: Divide by n
        divided_by_n = sum_squared_diff / n
        #print("divided_by_n",divided_by_n.numpy().mean())
        # Step 5: Calculate the mean of the result
        result_mean = self.tf.reduce_mean(divided_by_n)
        #print("result_mean",result_mean)
        #input("Waiting...")
        return result_mean

        
    def train(self,model,gxp):
        EPOCH = 20
        #loss_object = special_MSE()
        test_loss_array = []
        do_shuffle = True
        fail_count = 0  
        shuffle_num = 0
        best_model = None
        while True:
            if do_shuffle:
                shuffle_num+=1
                print("\nDataset shuffle")
                optimizer = self.tf.keras.optimizers.Adam()
                dataset = gxp.get_exp(self.dataset_size)
                test_loss_array = []
                fail_count = 0  
                x,y = [],[]
                for data in dataset:
                    x.append(data[0])
                    y.append(data[1])
                x = np.array(x,dtype="float32")
                y = np.array(y,dtype="float32")
                train_x,test_x,train_y,test_y = train_test_split(x,y)
                batch_size=64
                train_ds =  self.tf.data.Dataset.from_tensor_slices(
                    (train_x, train_y)
                    )
                train_ds = train_ds.shuffle(25000,reshuffle_each_iteration=True,seed=random.randint(0,2**32))
                train_ds = train_ds.batch(self.batch_size)
                train_ds = train_ds.cache().prefetch(buffer_size=self.tf.data.AUTOTUNE)
                test_ds = self.tf.data.Dataset.from_tensor_slices(
                    (test_x,test_y)
                ).batch(256).cache().prefetch(buffer_size=self.tf.data.AUTOTUNE)
                do_shuffle = False
            tqdm_obj = tqdm(range(len(train_ds)))
            for x,y in train_ds:
                loss_array = []
                poss_move = (y.numpy()/(y.numpy()+1e-30)).astype("float32")
                with self.tf.GradientTape() as tape:
                    # training=True is only needed if there are layers with different
                    # behavior during training versus inference (e.g. Dropout).
                    predictions = model(x, training=True)
                    
                    #clipped = tf.clip_by_value(poss_move,1e-10,1.0)

                    loss = self.get_loss(
                        (y*poss_move)
                        ,self.tf.clip_by_value(predictions*poss_move,1e-20,1.0) #Avoid log(0)
                        ,self.tf.reduce_sum(poss_move,axis=1)
                        )
                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(
                        zip(gradients, model.trainable_variables)
                        )
                loss_array.append(loss.numpy())
                tqdm_obj.update(1)
                tqdm_obj.set_description(f"loss:{np.mean(loss_array):.6f}")
            
            test_loss = []
            for x,y in test_ds:
                predictions = model(x, training=False)
                poss_move = (y.numpy()/(y.numpy()+1e-30)).astype("float32")
                loss = self.get_loss(
                    (y*poss_move)
                    ,self.tf.clip_by_value(predictions*poss_move,1e-20,1.0) #Avoid log(0)
                    ,self.tf.reduce_sum(poss_move,axis=1)
                    )
                test_loss.append(loss.numpy())
            test_loss = np.mean(test_loss)
            tqdm_obj.set_description(f"loss:{np.mean(loss_array):.6f} test_loss:{test_loss:.6f}")
            tqdm_obj.close()
            if len(test_loss_array)>0 and test_loss>=test_loss_array[-1]:
                fail_count += 1
            else:
                fail_count = 0
            if len(test_loss_array)<1 or min(test_loss_array)>test_loss:
                best_model = model
            test_loss_array.append(test_loss)
            if fail_count>self.patience:
                do_shuffle = True
            if shuffle_num >= self.shuffle_num:
                assert(best_model is not None,"best_model is None")
                model = best_model
                break
            
        return model