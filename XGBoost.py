import xgboost as xgb
import numpy as np

class XGBoost:
    '''
    Make a wrapper of xgboost to make it easy to use. The functions are written as in sklearn
    '''
    def __init__(self, num_round=100,validate = 0,params={}, early_stopping_rounds=10):
        '''
        Initiate the object. 
        Parameters = 
            num_round = Number of iterations. Default to 100
            Pass other parameters as a dictionary: Paramters available on https://github.com/dmlc/xgboost/blob/master/doc/parameter.md
        '''
        self.num_round=num_round
        self.model={}
        self.params=params
        self.validate=validate
        self.early_stopping_rounds=early_stopping_rounds
        
    def fit(self, X, y): 
        if self.validate == 0:
            dtrain=xgb.DMatrix(X.astype(np.float64),label=y)
            self.model=xgb.train(self.params, dtrain, self.num_round, [(dtrain,'train')])
            # ,early_stopping_rounds = self.early_stopping_rounds)
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
            dtrain=xgb.DMatrix(X_train.astype(np.float64),label=y_train)
            dval  =xgb.DMatrix(X_test.astype(np.float64),label=y_test)
            watchlist = [(dtrain, 'train'),(dval, 'val')]
            self.model=xgb.train(self.params, dtrain, self.num_round, watchlist)
            
    def get_fscore(self): 
        return self.model.get_fscore()
        
    def predict(self, X):
        dtest=xgb.DMatrix(X.astype(np.float64))
        return self.model.predict(dtest)

    def get_params(self, deep=True):
        return {"num_round": self.num_round, "params": self.params,"validate":self.validate,"early_stopping_rounds":self.early_stopping_rounds}
    
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            self.setattr(parameter, value)
        return self