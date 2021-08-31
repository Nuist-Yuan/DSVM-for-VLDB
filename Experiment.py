import DSVM.函数模块 as fool
import numpy as np
import matplotlib.pyplot as plt

#XX,y = fool.load_iris()
XX,y = fool.wine()
print(XX.shape)
print(y.shape)

# 一共三个区域 每个区域10台设备
Device_Region = []
Device_y_Region = []
# 邻居设备集合
#Neighbour = [[1,2,3],[0],[0],[0,4],[3,5],[4]]
Neighbour1 = [[1,2,3],[0,6],[0,7,8],[0,4],[3,5,6],[4,9],[1,4],[2,8],[2,7],[5]]
Neighbour2 = [[1],[0,2],[1,3],[2,4],[3,5],[4,6],[5,7],[6,8],[7,9],[8]]
Neighbour3 = [[2,3],[3],[0,5,3],[0,1,2,4,6],[3],[2,6,8,9],[3,5],[6,9],[5,9],[5,7,8]]
Neighbour = [Neighbour1,Neighbour2,Neighbour3]
Region = 3
NoD = 10  # Number of Device
YETA = 8 # 某个我画不出来的符号
X_Region = []
Y_Region = []
V_Region = []
oldV_Region = []
U_Region = []
f_Region = []
apha_Region = []
oldApha_Region = []
lamuda_Region = []

for lhy in range(Region):
    Device = []
    Device_y = []
    # 生成X,Y
    X = []
    Y = []
    # 生成v
    V = []
    oldV = []
    # 生成U
    U = []
    # 生成f
    f = []
    # 生成alpha
    apha = []
    oldApha = []
    # 生成lamuda
    lamuda = []
    for i in range(NoD):
        a, b = fool.data_allocator_for_wine(XX, y)
        #a, b = fool.data_allocator_for_iris(XX, y)
        Device.append(a)
        Device_y.append(b)
        X.append(fool.X_generator(a))
        Y.append(fool.Y_generator(b))
        V.append(fool.v_initial(a))
        oldV.append(fool.v_initial(a))
        U.append(fool.U_generator(a, Neighbour[lhy][i], YETA))
        apha.append(np.zeros((len(XX) + 1, 1)))
        oldApha.append(np.zeros((len(XX) + 1, 1)))
        lamuda.append(0)
    Device_Region.append(Device)
    Device_y_Region.append(Device_y)
    X_Region.append(X)
    Y_Region.append(Y)
    V_Region.append(V)
    oldV_Region.append(oldV)
    U_Region.append(U)
    f_Region.append(f)
    apha_Region.append(apha)
    oldApha_Region.append(oldApha)
    lamuda_Region.append(lamuda)


# 生成验证集
XXX = fool.X_generator(XX)
XXX_Region = [XXX,XXX,XXX]

J_Region = []
oldJ_Region = []
JUDGE_Region = []
ROU = 1.5


# 计算进一步的数值
for lhy in range(Region):
    J = []
    oldJ = []
    JUDGE = []
    for i in range(NoD):
        f_Region[lhy].append(fool.f_generator(Device_Region[lhy][i],Neighbour[lhy][i],YETA,V_Region[lhy],apha_Region[lhy][i],i))
        J.append(10000000)
        oldJ.append(10000000)
        JUDGE.append(1)
    J_Region.append(J)
    oldJ_Region.append(oldJ)
    JUDGE_Region.append(JUDGE)

RISK_Region = [[],[],[]]
RISK_LOCAL_Region = [[],[],[]]
RISK_GLOBAL_Region = [[],[],[]]
RISK_GLOBAL = []

# 开始 冲冲冲
print("开始吧")
for time in range(250):
    print(time)

    federated_agg = fool.federated_average(V_Region)
    Result = np.dot(XXX_Region[0], federated_agg)
    count = 0
    for lhylhy in range(len(y)):
        if(Result[lhylhy] * y[lhylhy] >= 0) :
            count = count + 1
    RISK_GLOBAL.append((len(y)-count) / (2 * len(XXX_Region[0])))

    for lhy in range(Region):
        # 验证一次训练结果
        error = 0
        xxxxx = NoD
        for i in range(NoD):
            if (JUDGE_Region[lhy][i] == 0):
                xxxxx = xxxxx - 1
                continue
            Result = np.dot(XXX_Region[lhy], V_Region[lhy][i])
            count = 0
            for j in range(len(y)):
                if (Result[j] * y[j] > 0):
                    count = count + 1
                else:
                    error = error + 1
            print(count)
            print(str(i + 1) + " : " + str(count / len(XX[0])))
            if(i==0):
                RISK_LOCAL_Region[lhy].append(count / len(XX[0]))
        print(error)
        print("Empirical risk：")
        RISK_GLOBAL_Region[lhy].append((xxxxx * len(XX[0]) - error) / (xxxxx * len(XX[0])))
        RISK_Region[lhy].append(error / (2 * xxxxx * len(XX[0])))
        print(error / (2 * xxxxx * len(XX[0])))
        print("---------------------------------------------")


        for i in range(NoD):
            if (JUDGE_Region[lhy][i] == 0): continue
            lamuda_Region[lhy][i] = fool.lamuda_calculate(X_Region[lhy][i],Y_Region[lhy][i],U_Region[lhy][i],f_Region[lhy][i])

            oldV_Region[lhy][i] = V_Region[lhy][i]
            oldJ_Region[lhy][i] = J_Region[lhy][i]
            oldApha_Region[lhy][i] = apha_Region[lhy][i]

            V_Region[lhy][i] = fool.v_refresh(X_Region[lhy][i],Y_Region[lhy][i],lamuda_Region[lhy][i],f_Region[lhy][i],U_Region[lhy][i])

            apha_Region[lhy][i] = fool.alpha_generator(Device_Region[lhy][i],apha_Region[lhy][i],YETA,i,Neighbour[lhy][i],V_Region[lhy])

            J_Region[lhy][i] = fool.J_calculate(YETA,i,Neighbour[lhy][i],apha_Region[lhy][i],V_Region[lhy],oldApha_Region[lhy][i],oldV_Region[lhy])

            # 异常检测
            if (J_Region[lhy][i] > ROU * oldJ_Region[lhy][i]):
                print("这是第" + str(time + 1) + "轮")
                print(J_Region[lhy][i])
                JUDGE_Region[lhy][i] = 0
                print(str(i + 1) + "号设备出现异常，已关闭")
            # 异常检测 #

            f_Region[lhy][i] = fool.f_generator(Device_Region[lhy][i],Neighbour[lhy][i],YETA,V_Region[lhy],apha_Region[lhy][i],i)

        print(len(V_Region[lhy]))
        print(len(V_Region[lhy][0]))

        # 过滤
        for num in range(NoD):
            if (JUDGE_Region[lhy][num] == 1):
                max, min, centroids = fool.get_center(np.transpose(Device_Region[lhy][num]), Device_y_Region[lhy][num])
                break
        for num in range(NoD):
            if (JUDGE_Region[lhy][num] == 0):
                print("这是设备" + str(num + 1) + "的过滤过程")
                Device_Region[lhy][num], Device_y_Region[lhy][num] = fool.filter_build(np.transpose(Device_Region[lhy][num]), Device_y_Region[lhy][num], max, min,
                                                               centroids)
                X_Region[lhy][num] = fool.X_generator(Device_Region[lhy][num])
                Y_Region[lhy][num] = fool.Y_generator(Device_y_Region[lhy][num])
                print("设备" + str(num + 1) + "过滤好了")
                JUDGE_Region[lhy][num] = 1
                J_Region[lhy][num] = 10000000
                oldJ_Region[lhy][num] = 10000000
                # 过滤 #


    # 每50轮进行一次联邦聚合
    if (time != 0 and time % 50 == 0):
        federated_agg = fool.federated_average(V_Region)
        for lhy in range(len(V_Region)):
            for yhl in range(len(V_Region[0])):
                V_Region[lhy][yhl] = federated_agg

    if (time == 100):
        for i in range(len(Y_Region[0][1])-10):
            Y_Region[0][1][i][i] = -1 * Y_Region[0][1][i][i]
        for i in range(len(Y_Region[0][3])-50):
            Y_Region[0][3][i][i] = -1 * Y_Region[0][3][i][i]
        # for i in range(len(Y_Region[0][5]) - 50):
        #     Y_Region[0][5][i][i] = -1 * Y_Region[0][5][i][i]
        # for i in range(len(Y_Region[2][5]) - 50):
        #     Y_Region[2][5][i][i] = -1 * Y_Region[2][5][i][i]
        # for i in range(len(Y_Region[0][8]) - 50):
        #     Y_Region[2][8][i][i] = -1 * Y_Region[2][8][i][i]
        # for i in range(len(Y_Region[1][1]) - 10):
        #     Y_Region[1][1][i][i] = -1 * Y_Region[1][1][i][i]
        # for i in range(len(Y_Region[1][3])-50):
        #     Y_Region[1][3][i][i] = -1 * Y_Region[1][3][i][i]
        # for i in range(len(Y_Region[1][7]) - 20):
        #     Y_Region[1][7][i][i] = -1 * Y_Region[1][7][i][i]




    if (time == 150):
        for i in range(len(Y_Region[1][1])-10):
            Y_Region[1][1][i][i] = -1 * Y_Region[1][1][i][i]
        for i in range(len(Y_Region[1][3])-50):
            Y_Region[1][3][i][i] = -1 * Y_Region[1][3][i][i]
        for i in range(len(Y_Region[1][7]) - 20):
            Y_Region[1][7][i][i] = -1 * Y_Region[1][7][i][i]
        # for i in range(len(Y_Region[0][3]) - 50):
        #     Y_Region[0][3][i][i] = -1 * Y_Region[0][3][i][i]
        # for i in range(len(Y_Region[0][7]) - 50):
        #     Y_Region[0][7][i][i] = -1 * Y_Region[0][7][i][i]
        # for i in range(len(Y_Region[2][5]) - 50):
        #     Y_Region[2][5][i][i] = -1 * Y_Region[2][5][i][i]
        # for i in range(len(Y_Region[0][8]) - 50):
        #     Y_Region[2][8][i][i] = -1 * Y_Region[2][8][i][i]




import csv

#如果不添加newline=""的话，就会每条数据中间都会有空格行
with open("record_wine_two_20_percent.csv","w", newline="") as csvfile:
    # 初始化写入对象
    writer = csv.writer(csvfile)
    #写入多行用writerows
    writer.writerow(RISK_GLOBAL[:])
# with open("record_iris_YETA=10_LOCAL.csv","w", newline="") as csvfile:
#     # 初始化写入对象
#     writer = csv.writer(csvfile)
#     #写入多行用writerows
#     writer.writerow(RISK_Region[0][:])
# with open("record_global.csv","w", newline="") as csvfile:
#     # 初始化写入对象
#     writer = csv.writer(csvfile)
#     #写入多行用writerows
#     writer.writerow(RISK_GLOBAL[:])

x_zhou = []
for i in range(250):
    x_zhou.append(i+1)
#plt.plot(x_zhou[:],RISK_Region[0][:])
plt.plot(x_zhou[:],RISK_GLOBAL[:])
plt.xlabel("Times", fontsize=12)
plt.ylabel("Empirical risk", fontsize=12)
plt.show()