#特征提取：
import pandas as pd
import numpy as np
def get_features(data,is_train):
    a=pd.DataFrame()
    data_length=len(set(data.id.values))
    for i in range(1,data_length+1):
        test=data[data.id==i]
        if len(test)!=1:
            test.index=range(len(test))
            #前后x,y值相减
            temp=test[['x','y','t']].diff(1).dropna()
            #距离的计算
            temp['distance']=np.sqrt(temp['x']**2+temp['y']**2)
            #距离➗时间
            temp['speed']=np.log1p(temp['distance'])-np.log1p(temp['t'])

            temp['angles']=np.log1p(temp['y'])-np.log1p(temp['x'])
            speed_diff=temp['speed'].diff(1).dropna()
            angle_diff=temp['angles'].diff(1).dropna()
            test['distance_aim_deltas']=np.sqrt((test['x']-test['target_x'])**2+(test['y']-test['target_y'])**2)
            distance_aim_deltas_diff=test['distance_aim_deltas'].diff(1).dropna()


            arr=pd.DataFrame(index=[0])
            arr['id']=i
            arr['speed_diff_median'] = speed_diff.median()
            arr['speed_diff_mean'] = speed_diff.mean()
            arr['speed_diff_var'] =  speed_diff.var()
            arr['speed_diff_max'] = speed_diff.max()
            arr['angle_diff_var'] =  angle_diff.var()
            arr['time_delta_min'] =  temp['t'].min()
            arr['time_delta_max'] = temp['t'].max()
            arr['time_delta_var'] = temp['t'].var()

            arr['distance_deltas_max'] =  temp['distance'].max()
            arr['distance_deltas_var'] =  temp['distance'].var()
            arr['aim_distance_last'] = test['distance_aim_deltas'].values[-1]
            arr['aim_distance_diff_max'] = distance_aim_deltas_diff.max()
            arr['aim_distance_diff_var'] = distance_aim_deltas_diff.var()
            arr['mean_speed'] = temp['speed'].mean()
            arr['median_speed'] = temp['speed'].median()
            arr['var_speed'] = temp['speed'].var()

            arr['max_angle'] = temp['angles'].max()
            arr['var_angle'] =  temp['angles'].var()
            arr['kurt_angle'] =  temp['angles'].kurt()

            arr['y_min'] = test["y"].min()
            arr['y_max'] = test["y"].max()
            arr['y_var'] = test["y"].var()
            arr['y_mean'] = test["y"].mean()
            arr['x_min'] = test["x"].min()
            arr['x_max'] = test["x"].max()
            arr['x_var'] = test["x"].var()
            arr['x_mean'] = test["x"].mean()

            arr['x_back_num'] = min( (test['x'].diff(1).dropna() > 0).sum(), (test['x'].diff(1).dropna() < 0).sum())
            arr['y_back_num'] = min( (test['y'].diff(1).dropna() > 0).sum(), (test['y'].diff(1).dropna() < 0).sum())

            arr['xs_delta_var'] = test['x'].diff(1).dropna().var()
            arr['xs_delta_max'] = test['x'].diff(1).dropna().max()
            arr['xs_delta_min'] =test['x'].diff(1).dropna().min()
            if(is_train):
                arr['label']=test['label']

            print(arr)
            a=pd.concat([a,arr])
    return a
import read_data
# train_data = read_data.get_data("/Users/xufan/Documents/txfz_training.txt")
# a = get_features(train_data,True).fillna(0)
test_data = read_data.get_test_data("/Users/xufan/Documents/txfz_test2.txt")
b = get_features(test_data,False).fillna(0)['']
b.to_csv("/Users/xufan/Documents/test2.csv")
# a.to_csv("/Users/xufan/Documents/datas.csv")
