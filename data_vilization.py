import matplotlib.pyplot as plt
import read_data
# plt.xticks(list(range(len(b))),b['x'].values)
import os
path='/Users/xufan/Documents/txfz_training.txt'
# os.mkdir(path)
count = 2994
data = read_data.get_data(path)
for i in range(1,count):
    b=data[data.id==i]
    print(b)
    k=list(b['x'].values)
#     k.extend(set(b['target_x'].values))
    l=list(b['y'].values)
#     l.extend(set(b['target_y'].values))
    plt.plot(k,l,'o-')
    fig = plt.gcf()
    fig.set_size_inches(30, 15)
    fig.savefig(path+'\\'+str(i)+'.png',dpi=100)
    plt.close()
