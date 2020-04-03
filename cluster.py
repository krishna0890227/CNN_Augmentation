import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import math
#from pandas.tools.plotting import autocorrelation_plot
from pandas.plotting import autocorrelation_plot
from pandas import Series
from random import gauss
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from matplotlib import pyplot
from sklearn.decomposition import PCA

#-------------------------------------------------------------
def CSM(vectorseries):   # CSM=convert series to matrix
    matrix=[]
    for i in range(0, len(vectorseries)//24):
       matrix.append(vectorseries[i*24: (i+1)*24])
    matrix_load=np.array(matrix)
    return matrix_load

def kmeans_apply(data, a):
    work_clusdata=data
    clus_base = []
    clus_resi = []
    resi_all=[]

    for i in range(1, 7):
        kmeans = KMeans(init='k-means++', n_clusters=i, n_init=10)
        cluster = kmeans.fit_predict(work_clusdata)
        centroids = kmeans.cluster_centers_
        for ii in range(i):
            clus_base.append(centroids[ii, :])
            clus_resi.append(work_clusdata - centroids[ii, :])
    K_expand = len(clus_resi)

    #plt.figure()
    cen1=centroids.T
    # plt.plot(work_clusdata.T[:,1], 'k-', label='Historical Load Patterns')
    # plt.plot(work_clusdata.T[:,1:], 'k-' )
    # plt.plot(cen1[:,3], 'r-s', label='Average Load Pattern',linewidth=3.5)
    # #plt.plot(cen1[:,2:], 'r-*')
    # plt.legend(loc='upper right', fontsize=12)
    # plt.ylabel('Energy Consumption (kWh)',fontsize=14)
    # #plt.axis([0,23, 0, 1.2])
    # plt.xlabel('Daily Hours', fontsize=14)
    # plt.tight_layout()
    # plt.show()
    #
    print('------------Plotting all------------------------------- ')
    print(' We have to plot  each centroid obtained from iterative k-means')
    iterative_plot(work_clusdata.T, clus_base)
    print('---------------------------------------------------------')
    print(len(clus_resi))
    #np.savetxt('Residual_alll.csv', clus_resi, delimiter=",")


    error = []
    for i in range(0, len(clus_resi)):
        resi_extract1 = clus_resi[i]
        resi_extract = resi_extract1.flatten()
        error.append(sum(resi_extract * resi_extract.T))
        resi_all.append(resi_extract)

    error1 = list(enumerate(error))
    error2 = sorted(error1, key=lambda x: x[1])
    resi_all = np.array(resi_all)
    # resi_all.flatten()
    np.savetxt('resi_collect_all.csv', resi_all, delimiter=",")
    clu_idx, clu_values = error2[K_expand // 2]
    print(clu_idx)
    min_aug = []
    value1 = []
    extract_base = []
    for i, value in enumerate(error):
        if value <= clu_values:
            min_aug.append(np.array(clus_resi[i]))
            extract_base.append(clus_base[i])
            value1.append(value)
    work_len = len(min_aug)
    print('+++++++++++++++ Hello! k_exapnd here+++++++++++++++')
    print(len(min_aug))
    print(len(extract_base))
    # plt.figure()
    # plt.plot(error, 'k-s', label=' All k_iter')
    # plt.plot(value1, 'r-*', label='Selected k_iter')
    # # plt.axis([0,5, 0,100])
    # plt.xlabel('No. of Cluster Count', fontsize=12)
    # plt.ylabel('Frobenius Norm Measure', fontsize=12)
    # plt.grid(True)
    # plt.legend(loc='upper left')
    # plt.tight_layout()
    #plt.savefig('distance.jpeg')
    plt.show()
    min_array_aug = []
    resi_only=[]
    for i in range(0, len(min_aug)):
        res_1 = np.array(min_aug[i])
        #res_1 = res_1.flatten()
        min_array_aug.append(res_1)
        res_ok=res_1.flatten()
        resi_only.append(res_ok)


    input_data = np.array(min_array_aug)
    print(input_data.shape)
    input_data = input_data.flatten()
    print(input_data.shape)
   #  print('##################################################')
   #  resi_ok=np.array(resi_only)
   #  np.savetxt('resi_collect.csv', resi_ok, delimiter=",")
   #  resi_plot = np.array(min_array_aug[3])
   #  resi_plot = resi_plot.flatten()
   #
   #  cen_re = np.array(cen1[:, 3])
   #  cen_re = cen_re.flatten()
   #  a = len(work_clusdata) + 1
   #  cen_rep = np.tile(cen_re, a)### repeatation of the same data
   #  decompose_data=np.concatenate((resi_plot, cen_rep), axis=0)
   #  np.savetxt('decomp_result.csv', decompose_data, delimiter=",")
   #  plt.figure(figsize=(6,4))
   #  plt.plot(cen_rep, 'k--', label='Y_avg')
   #  plt.legend(loc='upper right')
   # # plt.xlabel('Time Index (Hours)', fontsize=12)
   # # plt.ylabel(' Energy consumption (kWh)', fontsize=12)
   #  plt.axis([0, 720, 0, 0.75])
   #  plt.grid(True)
   #  plt.show()

    # plt.figure(figsize=(6,4))
    # plt.plot(resi_plot, 'k--', label='Residuals')
    # plt.legend(loc='upper right')
    # #plt.xlabel('Hours (one Month)', fontsize=12)
    # #plt.ylabel(' Noise Measure', fontsize=12)
    # #plt.axis([0,24,0,1.5])
    # plt.grid(True)
    # # plt.savefig('base_1.jpeg')
    # plt.tight_layout()
    # plt.show()


    # print('###########################################################################')
    # series_resi=Series(resi_plot)
    # #fig, (ax1, ax2)=plt.subplots(2,1)
    # series_gauss = [gauss(0.0, 1.0) for i in range(len(series_resi))]
    # plot_acf(series_resi)
    # plt.axis([0,200,-0.25, 0.6])
    # plt.grid(True)
    # #ax1.set_title('AC Coefficients of Residual Load Series', fontsize=12)
    # #plt.xlabel('Lags', fontsize=12)
    # #plt.ylabel('Coefficents', fontsize=12)
    # plt.show()
    #
    #
    # plot_pacf(series_resi)
    # plt.axis([0,200, -0.3, 0.3])
    # #ax2.set_title('PAC Coefficients of Residual Load Series', fontsize=12)
    # #plt.xlabel('Lags', fontsize=12)
    # #plt.ylabel('Coefficents', fontsize=12)
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()



    Ra = np.array(extract_base[work_len - 3])
    Rb = np.array(extract_base[work_len - 2])
    Rc = np.array(extract_base[work_len - 1])

    # plt.figure()
    # plt.plot(Ra, 'r--', label='Centroid 1')
    # plt.plot(Rb, 'k-', label='Centroid 2')
    # plt.plot(Rc, '.b-', label='Centroid 3')
    # plt.legend(loc='upper right', fontsize=12)
    # plt.xlabel('Daily Hours', fontsize=14)
    # plt.ylabel(' Extracted energy Consumption', fontsize=14)
    # plt.axis([0,24,0,1.5])
    # plt.grid(True)
    # #plt.savefig('base_1.jpeg')
    # plt.tight_layout()
    # plt.show()
    all_data = input_data
    print(all_data.shape)
    print(type(Rc))
    print(Rc.shape)
    test_pattern=input_data[(len(input_data)-24):]+ Rc
    print(test_pattern)
    return all_data, Rc

def convertSeriesToMatrix(vectorSeries, sequence_length):
    matrix=[]
    for i in range(len(vectorSeries)-sequence_length+1):
        matrix.append(vectorSeries[i:i+sequence_length])
    return matrix

def MApe(target, predict):
    mape=0.0
    x=np.mean(target)
    for i in range(len(target)):
        mape=mape + (abs((target[i]-predict[i])/target[i])*100)
    return mape/len(target)

def RMse(target, predict):
    rmse_sq=0.0
    for i in range(len(target)):
        rmse_sq=rmse_sq+(target[i]-predict[i])*(target[i]-predict[i])
    return math.sqrt((rmse_sq)/len(target))



def DataSeperation(data, ratio1, ratio2, baseline):
    matrix_load = np.array(data)
    print(matrix_load.shape)
    train_row1 = int(round(ratio1 * matrix_load.shape[0]))
    train_row2 = int(round(ratio2 * matrix_load.shape[0]))
    train_set1 = matrix_load[:train_row1, :]
    train_set2 = matrix_load[:train_row2, :]
    np.random.seed(230)
    # shuffle the training set (but do not shuffle the test set)
    np.random.shuffle(train_set1)
    np.random.shuffle(train_set2)
    # the training set
    X_train = train_set1[:, :-1]
    # the last column is the true value to compute the mean-squared-error loss
    y_train = train_set1[:, -1]
    # the test set
    X_test = matrix_load[train_row1:, :-1]
    y_test = matrix_load[train_row1:, -1]
    #########################################################
    X_train_3D = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test_3D = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    y_test=np.array(y_test)
    y_test = y_test + baseline
    print(y_test)
    return X_train, y_train, X_test, y_test, X_train_3D, X_test_3D

def DataSep (data, ratio1, ratio2):
    matrix_load = np.array(data)
    print(matrix_load.shape)
    shifted_value = matrix_load.mean()
    print(shifted_value)
    matrix_load -= shifted_value
    # split dataset: 90% for training and 10% for testing
    train_row1 = int(round(ratio1 * matrix_load.shape[0]))
    train_row2 = int(round(ratio2 * matrix_load.shape[0]))
    train_set1 = matrix_load[:train_row1, :]
    train_set2 = matrix_load[:train_row2, :]
    #train_set=signal.wiener(train_set1, 3)
    np.random.seed(230)
    # shuffle the training set (but do not shuffle the test set)
    #np.random.shuffle(train_set1)
   # np.random.shuffle(train_set2)
    # the training set
    X_train = train_set1[:, :-1]
    # the last column is the true value to compute the mean-squared-error loss
    y_train = train_set1[:, -1]
    # the test set
    X_test = matrix_load[train_row1:, :-1]
    y_test = matrix_load[train_row1:, -1]
    #########################################################
    # scalerX = StandardScaler().fit(X_train)
    # y_train=np.reshape(y_train.shape[0],1)
    # scalery = StandardScaler().fit(y_train)
    # X_train = scalerX.transform(X_train)
    # y_train = scalery.transform(y_train)
    # X_test = scalerX.transform(X_test)
    X_train_3D =  np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test_3D = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    #print(len(y_test))
    y_test=y_test + shifted_value
    return X_train, y_train, X_test,y_test,X_train_3D, X_test_3D, shifted_value


def iterative_plot(data, cent_data):
    # figsize=(10,8)
    # colns=3
     a=-1
    # print(len(cent_data))
    # rows= len(cent_data)//colns
    # print (rows)
    # fig1, axes = plt.subplots(rows, colns, figsize=figsize, constrained_layout=True)
    # for  i in range(0, rows):
    #      for ii in range(0, colns):
    #          a=a+1
    #          axes[i, ii].plot(data, 'k-')
    #          axes[i, ii].plot(cent_data[a], 'r-*', linewidth=2.5)
    #          axes[i, ii].set_xlabel('Time Steps (hour)')
    #          axes[i, ii].set_ylabel('Load (kWh)')
    #
    # plt.show()


def clus_plot(data):
    # samples=len(data)
    print(data.shape)
    # random_state=10
    # #pca=PCA(n_components=2)
    # #X=pca.fit_transform(data)]
    #
    # data_plot=data.T
    # data_save=data
    # k_means = KMeans(init='k-means++', n_clusters=1, n_init=10)
    # y_pred = k_means.fit_predict(data)
    # centers = k_means.cluster_centers_
    # print(centers.shape)
    # data_save=np.concatenate((data_save, centers), axis=0)
    # plt.figure(figsize=(6,4))
    # # plt.figure()
    # plt.plot(data_plot[:, 0], 'k-', label='Historical Load Profiles (all $\mathregular{Y_d}$)')
    # plt.plot(data_plot[:, 0:], 'k-')
    # plt.plot(centers[0, :], 'r-s', label='Centroid Load Profile $\mathregular{C_1}$', linewidth=3.5)
    # plt.xlabel('Daily hours', fontsize=14)
    # plt.ylabel('Energy (kWh)', fontsize=14)
    # plt.legend(loc='upper left')
    # plt.title('Number of Clusters=1')
    # plt.show()
    # # plt.show()
    # k_means = KMeans(init='k-means++', n_clusters=2, n_init=10)
    # y_pred=k_means.fit_predict(data)
    # centers=k_means.cluster_centers_
    # data_save = np.concatenate((data_save, centers), axis=0)
    # print(centers.shape)
    # plt.figure(figsize=(12,8))
    # plt.subplot(221)
    # #plt.figure()
    # plt.plot(data_plot[:, 1], 'k-', label=' Historical Load profiles (all $\mathregular{Y_d}$)')
    # plt.plot(data_plot[:, 1:], 'k-')
    # plt.plot(centers[0,:], 'r-s',  label='Centroid Load Profile $\mathregular{C_2}$', linewidth=3.5)
    # plt.plot(centers[1,:], 'b-*', label='Centroid Load Profile $\mathregular{C_3}$', linewidth=3.5)
    # plt.xlabel('Daily hours', fontsize=14)
    # plt.ylabel('Energy (kWh)', fontsize=14)
    # plt.legend(loc='upper left')
    # plt.title('Number of Clusters=2')
    # #plt.show()
    #
    # k_means = KMeans(init='k-means++', n_clusters=3, n_init=10)
    # y_pred = k_means.fit_predict(data)
    # centers = k_means.cluster_centers_
    # data_save = np.concatenate((data_save, centers), axis=0)
    # print(centers)
    # plt.subplot(222)
    # #plt.figure()
    # plt.plot(data_plot[:, 0], 'k-', label='Historical Load Profiles (all $\mathregular{Y_d}$)')
    # plt.plot(data_plot[:, 1:], 'k-')
    # plt.plot(centers[0,:], 'r-s', label='Centroid Load Profile $\mathregular{C_4}$', linewidth=3.5)
    # plt.plot(centers[1,:], 'b-*', label='Centroid Load Profile $\mathregular{C_5}$', linewidth=3.5)
    # plt.plot(centers[2, :], 'y-o', label='Centroid Load Profile $\mathregular{C_6}$', linewidth=3.5)
    # plt.xlabel('Daily hours', fontsize=14)
    # plt.ylabel('Energy (kWh)',fontsize=14)
    # plt.legend(loc='upper left')
    # plt.title('Number of Clusters=3')
    # #plt.show()
    # #
    # k_means = KMeans(init='k-means++', n_clusters=4, n_init=10)
    # y_pred = k_means.fit_predict(data)
    # centers = k_means.cluster_centers_
    # data_save = np.concatenate((data_save, centers), axis=0)
    # print(centers)
    # plt.subplot(223)
    # #plt.figure()
    # plt.plot(data_plot[:, 0], 'k-', label='Historical Load Profiles (all $\mathregular{Y_d}$)')
    # plt.plot(data_plot[:, 0:], 'k-')
    # plt.plot(centers[0, :], 'r-s', label='Centroid Load Profile $\mathregular{C_7}$', linewidth=3.5)
    # plt.plot(centers[1, :], 'b-*', label='Centroid Load Profile $\mathregular{C_8}$', linewidth=3.5)
    # plt.plot(centers[2, :], 'y-o', label='Centroid Load Profile $\mathregular{C_9}$', linewidth=3.5)
    # plt.plot(centers[3, :], 'c-^', label='Centroid Load Profile $\mathregular{C_{10}}$', linewidth=3.5)
    # plt.xlabel('Daily hours', fontsize=14)
    # plt.ylabel('Energy (kWh)', fontsize=14)
    # plt.title('Number of Clusters=4')
    # plt.legend(loc='upper left')
    # #plt.show()
    #
    #
    # k_means = KMeans(init='k-means++', n_clusters=5, n_init=10)
    # y_pred = k_means.fit_predict(data)
    # centers = k_means.cluster_centers_
    # data_save = np.concatenate((data_save, centers), axis=0)
    # np.savetxt('cluster_result.csv', data_save, delimiter=",")
    # print(centers)
    # plt.subplot(224)
    # #plt.figure()
    # plt.plot(data_plot[:, 0], 'k-', label='Historical Load Profiles (all $\mathregular{Y_d}$)')
    # plt.plot(data_plot[:, 0:], 'k-')
    # plt.plot(centers[0, :], 'r-s', label='Centroid Load Profile $\mathregular{C_{l1}}$', linewidth=3.5)
    # plt.plot(centers[1, :], 'b-*', label='Centroid Load Profile $\mathregular{C_{l2}}$', linewidth=3.5)
    # plt.plot(centers[2, :], 'g-o', label='Centroid Load Profile $\mathregular{C_{l3}}$', linewidth=3.5)
    # plt.plot(centers[3, :], 'c-^', label='Centroid Load Profile $\mathregular{C_{l4}}$', linewidth=3.5)
    # plt.plot(centers[4, :], 'm->', label='Centroid Load Profile $\mathregular{C_{l5}}$', linewidth=3.5)
    # plt.xlabel('Daily hours', fontsize=14)
    # plt.ylabel('Energy (kWh)', fontsize=14)
    # plt.title('Number of Clusters=5')
    # plt.legend(loc='upper left')
    # plt.show()
    # plt.tight_layout()
    # plt.show()

