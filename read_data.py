#####数据读取和处理
import pandas as pd
import os


def get_data(file):
    data1=[]
    count=0
    with open(file) as f:
        for i in f.readlines():
            count+=1
            arr=i.split(" ")[1].split(';')[:-1]
            for j in arr:
                temp=[count]
                temp.extend(j.split(','))
                data1.append(temp)
    count = 0
    data2=[]
    with open(file) as f:
        for i in f.readlines():
            count+=1
            arr = i.split(" ")[2]
            list = arr.split(',')
            str = i.split(" ")[3].split("\n")[0]
            list+=str
            data2.append(list)

    data=pd.DataFrame(data1,columns=["id",'x',"y","t"])
    d2=pd.DataFrame(data2,columns=["target_x","target_y","label"])
    d2['id'] = range(1,count+1)
    d2.target_y=d2.target_y.apply(lambda x:x[:-1])
    data=pd.merge(data,d2,on="id")
    data = data.astype('double')
    return data

def get_test_data(file):
    data1 = []
    count = 0
    with open(file) as f:
        for i in f.readlines():
            count += 1
            arr = i.split(" ")[1].split(';')[:-1]
            print(arr)
            for j in arr:
                temp = [count]
                temp.extend(j.split(','))
                data1.append(temp)
    count = 0
    data2 = []
    with open(file) as f:
        for i in f.readlines():
            count += 1
            arr = i.split(" ")[2]
            list = arr.split(',')
            data2.append(list)
    print(data1)
    data = pd.DataFrame(data1, columns=["id", 'x', "y", "t"])
    d2 = pd.DataFrame(data2, columns=["target_x", "target_y"])
    d2['id'] = range(1, count + 1)
    d2.target_y = d2.target_y.apply(lambda x: x[:-1])
    data = pd.merge(data, d2, on="id")
    data = data.astype('double')
    return data