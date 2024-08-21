
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import random
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

class trainer_class:
    def __init__(self,patience=3,shuffle_num=5,dataset_size=1024*8,batch_size=64,epoch=50):
        
        self.patience = patience
        self.shuffle_num = shuffle_num
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.EPOCH = epoch
    def get_loss(self,y,pred,poss_move):
        n = tf.reduce_sum(poss_move,axis=1)
        # Use mean squared error
        #return tf.reduce_mean(tf.square(y - pred))
        
        # Use cross entropy loss
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y,pred))
        
        # Use special cross entropy loss 
        pred_log = tf.math.log(pred)
        
        loss = y*pred_log
        loss = tf.reduce_sum(loss,axis=1)
        loss = loss/n
        result_mean = -tf.reduce_mean(loss)
        return result_mean
    
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
    
    
    def train(self,model,gxp):
        @tf.function
        def train_step(x,y):
            with tf.GradientTape() as tape:
                # training=True is only needed if there are layers with different
                # behavior during training versus inference (e.g. Dropout).
                predictions = model(x, training=True)
                loss = loss_obj(y,tf.clip_by_value(predictions,1e-20,1.0))
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(
                zip(gradients, model.trainable_variables)
            )
            return loss,predictions
        EPOCH = 20
        #loss_object = special_MSE()
        test_loss_array = []
        do_shuffle = True
        fail_count = 0  
        shuffle_num = 0
        best_model = None
        epoch = 0
        # loss_obj = tf.keras.losses.CategoricalCrossentropy()
        loss_obj = tf.keras.losses.MeanSquaredError()
        # loss_obj = tf.keras.losses.MeanAbsoluteError()
        optimizer = tf.keras.optimizers.Adam(clipvalue=1)
        model.summary()
        while True:
            epoch += 1
            if do_shuffle:
                shuffle_num+=1
                epoch = 0
                
                print("\nDataset shuffle")
                print(f"shuffle_num:{shuffle_num}")
                dataset = gxp.get_exp(self.dataset_size)
                test_loss_array = []
                fail_count = 0  
                x,y = [],[]
                for data in dataset:
                    x.append(data[0])
                    y.append(data[1])
                x = np.array(x,dtype="float32")
                y = np.array(y,dtype="float32")
                #Describe the data
                print("------Data Description------")
                print(f"x mean:{x.mean()} std:{x.std()} max:{x.max()} min:{x.min()}")
                print(f"y mean:{y.mean()} std:{y.std()} max:{y.max()} min:{y.min()}")
                # print(f"y mean:{y.mean(axis=0).reshape(8,8)}")
                # print(f"y std:{y.std(axis=0).reshape(8,8)}")
                print("----------------------------")
                train_x,test_x,train_y,test_y = train_test_split(x,y)
                batch_size=64
                train_ds =  tf.data.Dataset.from_tensor_slices(
                    (train_x, train_y)
                    )
                train_ds = train_ds.shuffle(25000,reshuffle_each_iteration=True,seed=random.randint(0,2**32))
                train_ds = train_ds.batch(self.batch_size)
                train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
                test_ds = tf.data.Dataset.from_tensor_slices(
                    (test_x,test_y)
                ).batch(256).cache().prefetch(buffer_size=tf.data.AUTOTUNE)
                do_shuffle = False
            tqdm_obj = tqdm(range(len(train_ds)))
            try:
                loss_array = []
                #train_ds = train_ds.shuffle(25000,reshuffle_each_iteration=True,seed=random.randint(0,2**32))
                for x,y in train_ds:
                    loss,predictions = train_step(x,y)
                    loss_array.append(loss.numpy())
                    tqdm_obj.update(1)
                    tqdm_obj.set_description(f"epoch: {epoch} loss:{np.mean(loss_array):.6f}")
            except KeyboardInterrupt:
                print("Interrupted")
                do_shuffle = True
            test_loss = []
            pred_array = np.array([])
            for x,y in test_ds:
                predictions = model(x, training=False)
                pred_array = np.concatenate([pred_array,predictions.numpy().flatten()])
                loss = loss_obj(y,tf.clip_by_value(predictions,1e-20,1.0))
                test_loss.append(loss.numpy())
            test_loss = np.mean(test_loss)
            pred_array = np.array(pred_array).flatten()
            # plt.hist(pred_array,bins=100)
            # plt.show()
            pred_mean = pred_array.mean()
            pred_std = pred_array.std()
            tqdm_obj.set_description(f"epoch:{epoch} loss:{np.mean(loss_array):.6f} test_loss:{np.mean(test_loss):.6f} pred_mean:{pred_mean:.6f} pred_std:{pred_std:.6f} ")
            tqdm_obj.close()
            if len(test_loss_array)>0 and test_loss>=test_loss_array[-1]:
                fail_count += 1
            else:
                fail_count = 0
            if len(test_loss_array)<1 or min(test_loss_array)>test_loss:
                pass
                #best_model = model
            test_loss_array.append(test_loss)
            if fail_count>self.patience or epoch>=self.EPOCH:
                do_shuffle = True
            if shuffle_num >= self.shuffle_num and do_shuffle:
                #assert(best_model is not None,"best_model is None")
                #model = best_model
                break
        return model