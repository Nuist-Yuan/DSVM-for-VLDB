import numpy as np
import random
import csv
import struct
import matplotlib.pyplot as plt
from cvxopt import matrix
from sklearn.utils import check_random_state
from sklearn import svm, datasets
import cvxopt
from sklearn.cluster import KMeans

# 生成数据
def data_generator():
    mean1 = np.transpose((1, 1))
    mean2 = np.transpose((-1, -1))
    cov = np.array([[1, 0], [0, 2]])
    x1 = np.random.multivariate_normal(mean1, cov, (10000,), 'raise')  # nx2
    x2 = np.random.multivariate_normal(mean2, cov, (10000,), 'raise')  # nx2
    X = np.zeros((20000, 2))
    y = np.zeros(20000)
    for i in range(20000):
        if (i % 2 == 1): continue
        X[i] = x1[int(i / 2)]
        y[i] = 1.0
        X[i + 1] = x2[int(i / 2)]
        y[i + 1] = -1.0
    X = np.transpose(X)
    return X,y

#读取mnist数据
def load2():
    # X_test = np.genfromtxt('train_img.csv', dtype=float, delimiter=',')
    # y_test = np.genfromtxt('train_labels.csv', dtype=float, delimiter=',')
    X_test = np.genfromtxt('9_2img.csv', dtype=float, delimiter=',')
    y_test = np.genfromtxt('9-2label.csv', dtype=float, delimiter=',')
    return X_test,y_test

# 读取wine数据 revised
def wine():
    Content = np.genfromtxt('wine_for_test.csv', dtype=float, delimiter=',')
    X = np.zeros((len(Content),len(Content[0]-1)))
    Y = []
    for i in range(len(Content)):
        for j in range(len(Content[0])):
            if j==len(Content[0])-1 :
                Y.append(Content[i][j])
                continue
            X[i][j] = Content[i][j]
    X = np.transpose(X)
    Y = np.array(Y)
    return X,Y

def load_iris() :
    iris = datasets.load_iris()
    rng = check_random_state(42)
    perm = rng.permutation(iris.target.size)
    iris_data = iris.data[perm]
    iris_target = iris.target[perm]
    iris_data = np.transpose(iris_data)
    y = np.zeros(len(iris_target))
    for i in range(len(iris_target)):
        if(iris_target[i]==0 or iris_target[i]==1 ):
            y[i] = 1.0
        else:
            y[i] = -1.0
    return iris_data,y

# 随机给每个设备分配一定量的数据
def data_allocator(X,y):
    a = random.randint(0,12000)
    A = np.zeros((2,300))
    b = np.zeros(300)
    for i in range(300):
        A[0][i] = X[0][i+a]
        A[1][i] = X[1][i+a]
        b[i] = y[i+a]
    return A,b

# 随机给每个设备分配一定量的数据 用于mnist数据集
def data_allocator_for_mnist(X,y):
    a = random.randint(0,5060)
    A = np.zeros((784,200))
    b = np.zeros(200)
    for i in range(200):
        for j in range(784):
            A[j][i] = X[j][i+a]
        b[i] = y[i+a]
    return A,b

# 随机给每个设备分配一定量的数据 用于iris数据集
def data_allocator_for_iris(X,y):
    a = random.randint(0,50)
    A = np.zeros((4,100))
    b = np.zeros(100)
    for i in range(100):
        for j in range(4):
            A[j][i] = X[j][i+a]
        b[i] = y[i+a]
    return A,b

# 随机给每个设备分配一定量的数据 用于wine数据集
def data_allocator_for_wine(X,y):
    a = random.randint(0,1400)
    A = np.zeros((12,100))
    b = np.zeros(100)
    for i in range(100):
        for j in range(12):
            A[j][i] = X[j][i+a]
        b[i] = y[i+a]
    return A,b



# 生成X  n*3
def X_generator(X):
    XX = np.zeros((len(X[0]),len(X)+1))
    for i in range(len(X[0])):
        for j in range(len(X)):
            XX[i][j] = np.transpose(X)[i][j]
        XX[i][j + 1] = 1.0
    return XX

# 生成Y n*n
def Y_generator(Y):
    YY = np.zeros((len(Y),len(Y)))
    for i in range(len(Y)):
        YY[i][i] = Y[i]
    return YY

# 初始化v 3*1
def v_initial(X):
    v = np.zeros((len(X)+1,1))
    for i in range(len(X)+1):
        v[i][0] = -3
    return v

# 更新v
def v_refresh(X,Y,lamuda,f,U):
    # 这里的X是XX
    temp = np.dot(np.dot(np.transpose(X), Y), lamuda) - f
    return np.dot(U,temp)

# 生成U 3*3 len(X)+1 = 3
def U_generator(X,Neighbour,YETA):
    U = np.zeros((len(X)+1, len(X)+1))
    I = np.zeros((len(X)+1, len(X)+1))
    II = np.zeros((len(X)+1, len(X)+1))
    II[len(X)][len(X)] = 1
    for i in range(len(X)+1):
        I[i][i] = 1
    U = np.linalg.inv((1+2*YETA*len(Neighbour))*I - II)
    return U

# 生成 f len(X)+1 * 1
def f_generator(X,Neighbour,YETA,V,alpha,now):
    # now 表示当前节点的序号
    # V 表示存储所有节点设备的v的总集合
    vv = np.zeros((len(X)+1,1))
    for i in range(len(Neighbour)):
        vv = vv + V[int(now)] + V[int(Neighbour[i])]
    f = 2 * alpha - YETA * vv
    return f

# 更新 alpha len(X)+1 * 1
def alpha_generator(X,alpha,YETA,now,Neighbour,V):
    vv = np.zeros((len(X)+1,1))
    for i in range(len(Neighbour)):
        vv = vv + V[int(now)] - V[int(Neighbour[i])]
    alpha = alpha + (YETA/2) * vv
    return alpha

# 求解argmax的函数
def reg_LASSO(A, B, XX):
    q = matrix(np.transpose(-B))  # coefficient matrix of x^1
    G = np.zeros((2*len(XX),len(XX)))
    for i in range(len(XX)):
        G[i][i] = 1.0
        G[i+len(XX)][i] = -1.0
    G = matrix(G)
    h = np.zeros((2*len(XX),1))
    for i in range(len(XX)):
        h[i][0] = 1.0 #-2
        h[i+len(XX)][0] = 0.0
    h = matrix(h)
    cvxopt.solvers.options['show_progress'] = False
    the = cvxopt.solvers.qp(matrix(A), q, G, h)['x']  # output of this function is a dict
    return the

# 看看它能不能救我
def quadprog(H, f, L=None, k=None, Aeq=None, beq=None, lb=None, ub=None):
    """
    Input: Numpy arrays, the format follows MATLAB quadprog function: https://www.mathworks.com/help/optim/ug/quadprog.html
    Output: Numpy array of the solution
    """
    n_var = H.shape[1]

    P = cvxopt.matrix(H, tc='d')
    q = cvxopt.matrix(f, tc='d')

    if L is not None or k is not None:
        assert(k is not None and L is not None)
        if lb is not None:
            L = np.vstack([L, -np.eye(n_var)])
            k = np.vstack([k, -lb])

        if ub is not None:
            L = np.vstack([L, np.eye(n_var)])
            k = np.vstack([k, ub])

        L = cvxopt.matrix(L, tc='d')
        k = cvxopt.matrix(k, tc='d')

    if Aeq is not None or beq is not None:
        assert(Aeq is not None and beq is not None)
        Aeq = cvxopt.matrix(Aeq, tc='d')
        beq = cvxopt.matrix(beq, tc='d')

    cvxopt.solvers.options['show_progress'] = False
    sol = cvxopt.solvers.qp(P, q, L, k, Aeq, beq)

    return np.array(sol['x'])


# 求那个lamuda  len(X) * 1  好像不太对
def lamuda_calculate(XX,YY,U,f):
    # 计算A部分
    A = np.dot(YY,XX)
    A = np.dot(A,U)
    A = np.dot(A,np.transpose(XX))
    A = np.dot(A,YY)

    # 计算B部分
    YI = np.zeros((len(XX), 1))  #全是1的矩阵
    for i in range(len(XX)):
        YI[i] = 1
    B = np.transpose(YI + np.dot(np.dot(np.dot(YY, XX), U), f))
#    B = YI + np.dot(np.dot(np.dot(YY, XX), U), f)
    # G = np.zeros((2 * len(XX), len(XX)))
    # for i in range(len(XX)):
    #     G[i][i] = 1.0
    #     G[i+len(XX)][i] = -1.0
    # G = matrix(G)
    # h = np.zeros((2*len(XX),1))
    # for i in range(len(XX)):
    #     h[i][0] = 0.0 #-2
    #     h[i+len(XX)][0] = 0.0
    # h = matrix(h)
    lamuda = reg_LASSO(A,B,XX)
#    lamuda = quadprog(A,-B,G,h)
    return lamuda

# federated average
def federated_average(V):
    lens = len(V)
    lens_local = len(V[0])

    X = V[0][0]
    for lhy in range(lens):
        for yhl in range(lens_local):
            X = X + V[lhy][yhl]

    return (X - V[0][0]) / (lens * lens_local)





# 计算J
def J_calculate(YETA,now,Neighboour,alpha,V,oldAlpha,oldV):
    res1 = 0
    for i in range(len(Neighboour)):
        res1 = res1 + pow(np.linalg.norm(0.5*(V[int(now)]+V[int(Neighboour[i])]-oldV[int(now)]-oldV[int(Neighboour[i])])),2)
    res2 = (2/YETA) * pow(np.linalg.norm(alpha-oldAlpha),2)
    return YETA*res1 + res2

# 获取聚类中心 给出边界值
def get_center(X,Y):
    estimator = KMeans(n_clusters=2)  # 构造聚类器
    estimator.fit(X)  # 聚类
    centroids = estimator.cluster_centers_  # 获取聚类中心
    #label_pred = estimator.labels_  # 获取聚类标签
    dist = np.zeros((len(Y),1))
    for i in range(len(Y)):
        if(centroids[0][0] < 0):
            if(Y[i] == 1.0):
                dist[i] = np.linalg.norm(X[i] - centroids[1])
            else:
                dist[i] = np.linalg.norm(X[i] - centroids[0])
        else:
            if (Y[i] == 1.0):
                dist[i] = np.linalg.norm(X[i] - centroids[0])
            else:
                dist[i] = np.linalg.norm(X[i] - centroids[1])
    max = 0
    min = 0
    for i in range(len(dist)):
        if(dist[i]>max):
            max = dist[i]
        if(dist[i]<min):
            min = dist[i]
    return max,min,centroids

# 过滤
def filter_build(X,Y,max,min,center):
    dist = np.zeros((len(Y),1))
    filterX = []
    filterY = []
    for i in range(len(Y)):
        if(center[0][0] < 0):
            if(Y[i] == 1.0):
                dist[i] = np.linalg.norm(X[i] - center[1])
            else:
                dist[i] = np.linalg.norm(X[i] - center[0])
        else:
            if (Y[i] == 1.0):
                dist[i] = np.linalg.norm(X[i] - center[0])
            else:
                dist[i] = np.linalg.norm(X[i] - center[1])
    # 要让最大值变小 让最小值变大
    # if(max > 0):
    #     max = 0.4 * max
    # else:
    #     max = 1.5 * max
    # if(min > 0):
    #     min = 1.5 * min
    # else:
    #     min = 0.4 * min

    n = 0
    for i in range(len(dist)):
        if(dist[i] <= max and dist[i] >= min):
            filterX.append(X[n])
            filterY.append(Y[n])
            n = n + 1
    filterX = np.array(filterX)
    filterY = np.array(filterY)
    print(filterX.shape)
    print(filterY.shape)
    return np.transpose(filterX),filterY




if __name__ == '__main__':
    X,y = data_generator()
    print(X.shape)
    print(y.shape)

    print(U_generator(X,[1,2],0.1))