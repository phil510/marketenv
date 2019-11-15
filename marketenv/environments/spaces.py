import abc
import numpy as np

class Space(abc.ABC):
    def __init__(self):
        self.shape = None
        self.seed()
    
    @abc.abstractmethod
    def __contains__(self, item):
        pass
    
    def sample(self):
        raise NotImplementedError
        
    def seed(self, seed = None):
        self._rng = np.random.RandomState(seed)

class TradeSpace(Space):
    def __init__(self, n_tradable):
        super().__init__()
        
        self._n = n_tradable
        self.shape = (self._n, )
        
    def __contains__(self, item):
        try:
            item = np.asarray(item)
        except:
            return False
            
        right_shape = (item.shape == self.shape)
        all_int = (item.dtype.kind in np.typecodes['AllInteger'])
        
        return (right_shape and all_int)
    
    def sample(self):
        # TODO: Not sure about the best way to implement this
        raise NotImplementedError
        
class MarketSpace(Space):
    def __init__(self, n_tradable, n_non_tradable):
        super().__init__()
        
        self._n = n_tradable
        self._m = n_non_tradable
        self.shape = (1 + self._n * 2 + self._m, )
        
    def __contains__(self, item):
        try:
            item = np.asarray(item)
        except:
            return False
            
        right_shape = (item.shape == self.shape)
        all_int = ((item.dtype.kind in np.typecodes['AllInteger']) 
                   or (item.dtype.kind in np.typecodes['AllFloat']))
        
        return (right_shape and all_int)