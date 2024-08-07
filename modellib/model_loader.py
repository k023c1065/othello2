import numpy as np
from .model import miniResNet
from glob import glob
import os
import functools
def load_model() -> miniResNet:
    model = miniResNet(input_shape=(8,2,2),output_dim=64)
    model(np.zeros((1,8,2,2)),training=False)
    model_files = glob("model/*.h5")
    #model_filesを更新日時順でソートする
    model_files.sort(key=os.path.getmtime)
    if len(model_files)>0:
        model.load_weights(model_files[-1])
    return model
class model_class:
    def __init__(self) -> None:
        self.model = self.load_model()
        self.pattern = np.array([2**i for i in range(0,8)],dtype="uint8")
    def load_model(self) -> miniResNet:
        model = miniResNet(input_shape=(8,8,2),output_dim=64)
        model(np.zeros((1,8,8,2)),training=False)
        model_files = glob("model/*.h5")
        #model_filesを更新日時順でソートする
        model_files.sort(key=os.path.getmtime)
        if len(model_files)>0:
            model.load_weights(model_files[-1])
            print(f"model loaded:{model_files[-1]} with updated date{os.path.getmtime(model_files[-1])}")
        return model
    
    def predict(self,x:np.ndarray,training=False,**kwargs):
        #x = hashable_board(x)
        return self._predict(x,training)
    
    #@functools.lru_cache(maxsize=512)
    def _predict(self,x,training=False):
        #x = x.data
        return self.model.predict(x,verbose=0)
    
class hashable_board:
    def __init__(self,content):
        self.data = content
        self.pattern = np.array([2**i for i in range(0,64)],dtype="uint64").reshape(8,8)
    def __hash__(self) ->int:
        
        return hash(((self.data[0][0]*self.pattern).sum(),(self.data[0][0]*self.pattern).sum()))