# Advanced Modelling Utilties

This repository contains wrapper classes for advanced modelling algorithms in python. 
Using these classes, these models can be used just like another model in sklearn. 
These classes can also be used directly in sklearn.cross_validation for cross validation. Below is the usage of these algorithms.

## Neural Network

Required Package: keras

> clf=NN(inputShape = train.shape[1], layers = [128, 64], dropout = [0.5, 0.5], loss='mae', optimizer = 'adadelta', init = 'glorot_normal', nb_epochs = 5)

> clf.fit(train, labels)

> train_pred = clf.predict(train)[:,0]

## XGBoost

Required Package: xgboost

> model=params = {'booster':booster, 'max_depth':max_depth, 'objective':objective, 'eval_metric':['logloss','rmse'],'nthread':16,'eta':0.05,'min_child_weight':100}

> model.fit(train_features,y_train)

> model.predict(test_features)