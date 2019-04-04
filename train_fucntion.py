###xgb
import xgboost as xgb
import pandas as pd

test_x=test.drop('id',1)
train_x=train.drop(['id','label'],1)

dtest = xgb.DMatrix(test_x)
# dval = xgb.DMatrix(val_x,label=val_data.label)
dtrain = xgb.DMatrix(train_x, label=train.label)
params={
    'booster':'gbtree',
    'objective': 'binary:logistic',

#   'scale_pos_weight': 1500.0/13458.0,
        'eval_metric': "auc",

    'gamma':0.1,#0.2 is ok
    'max_depth':3,
#   'lambda':550,
        'subsample':0.7,
        'colsample_bytree':0.4 ,
#         'min_child_weight':2.5,
        'eta': 0.007,
#     'learning_rate':0.01,
    'seed':1024,
    'nthread':7,

    }

watchlist  = [(dtrain,'train'),
# (dval,'val')
             ]#The early stopping is based on last set in the evallist
model = xgb.train(
    params,
                  dtrain,
                  feval=feval,
#                   maximize=False,

                          num_boost_round=1500,
#                   early_stopping_rounds=10,
#                   verbose_eval =30,
                  evals=watchlist
                 )
# model=xgb.XGBClassifier(
# max_depth=4,
#     learning_rate=0.007,
#     n_estimators=1500,
#     silent=True,
#     objective='binary:logistic',
# #     booster='gbtree',
# #     n_jobs=-1,
#     nthread=7,
# #     gamma=0,
# #     min_child_weight=1,
# #     max_delta_step=0,
#     subsample=0.7,
#     colsample_bytree=0.7,
# #     colsample_bylevel=0.7,
# #     reg_alpha=0,
# #     reg_lambda=1,
#     scale_pos_weight=1,
#     base_score=0.5,
# #     random_state=0,
#     seed=1024,
#     missing=None,
# )

# xgb.cv(params,dtrain,num_boost_round=1500,nfold=10,feval=feval,early_stopping_rounds=50,)
# model.save_model('./model/xgb.model')
# print "best best_ntree_limit",model.best_ntree_limit
