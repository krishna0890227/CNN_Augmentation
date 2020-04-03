
import numpy as np
import pandas as pd
import Agg_loadSeries as ALS
import cluster as cl
import Neural_networks as NN
import tensorflow as tf
import matplotlib.pyplot as plt

#-----------------------------------------------------------------------------
Household_num=40 ### first=Main 3 (40), Second =Main 4 (0032), Third=main 5(0028)  Fourth=main 6(0018) main 7= 0012

conc_load_0, conc_load_3, conc_load_6=ALS.concat_Indi_load(Household_num)
Pool=6
print(len(conc_load_0))
peak_day_data=cl.CSM(conc_load_0)

print(len(conc_load_6))
print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
print(conc_load_0[720:])
print(conc_load_6[5040:])  #### for 3=2880 and for 6=5040  for april: 5184 becaus
print('up to here......')
print('=========================================================================')
ANN_Vr=[]
ANN_Vm=[]
ANN_result = []
CNN_Vr=[]
CNN_Vm=[]
CNN_result = []
CNN_Vr_aug, CNN_Vm_aug, CNN_result_aug=[], [], []
LSTM_Vr=[]
LSTM_Vm=[]
LSTM_result = []
# print('=================================================================================') 5064,
for i in range(0, 28):
    dayofforecast=i+1
    daily_data1=conc_load_0[(i*24):(744+i*24)]
    pooled_data=conc_load_0[(i*24):((744+i*24))]
    daily_pattern=cl.CSM(daily_data1)
    daily_big_pool=cl.CSM(pooled_data)
    print(daily_pattern.shape)
    cl.clus_plot(daily_pattern)
    sequence_length = 24
    all_data, baseline = cl.kmeans_apply(daily_pattern, Household_num)
    matrix_load1 = cl.convertSeriesToMatrix(all_data, sequence_length)
    matrix_load_p=cl.convertSeriesToMatrix(pooled_data, sequence_length)
    matrix_load_aug=np.array(matrix_load1)
    matrix_load_pool=np.array(matrix_load_p)
    # # # 4297=> 0.9944, 3577=>0.9933, 8617=>0.9972
    # # # 7177=> 0.99665, 5017=> 0.9952, 5737=>0.9958
    # # #3577 =>     2857=>0.9915
    print('=================================================')
    ratio_aug1=(1-24/(matrix_load_aug.shape[0]))
    ratio_aug2 = (1 - 48 / (matrix_load_aug.shape[0]))
    ratio_pool1=(1-24/(matrix_load_pool.shape[0]))
    ratio_pool2 = (1 - 48 / (matrix_load_pool.shape[0]))
    print(ratio_pool1, ratio_aug1)
    X_train, y_train, X_test, y_test, X_train_3D, X_test_3D= cl.DataSeperation(matrix_load_aug, ratio_aug1, ratio_aug2, baseline)
    X_train1, y_train1, X_test1, y_test1, X_train_3D1, X_test_3D1, shifted_value1 = cl.DataSep( matrix_load_pool, ratio_pool1, ratio_pool2)
    # # print('Single Household Forecasting of %s th Day Using MLP NN Pool  ' % dayofforecast)
    # ANN_pred, ANN_mape, ANN_rmse = NN.dense_pooled(X_train1, y_train1, X_test1, y_test1, shifted_value1)
    # ANN_Vm.append(ANN_mape)
    # ANN_Vr.append(ANN_rmse)
    # ANN_result.append(ANN_pred)
    # print('Single Household Forecasting of %s th Day Using Augmented CNN ' % dayofforecast)
    # CNN_pred_aug, CNN_rmse_aug, CNN_mape_aug = NN.CNN_network(X_train_3D, y_train, X_test_3D, y_test, baseline)
    # CNN_Vm_aug.append(CNN_mape_aug)
    # CNN_Vr_aug.append(CNN_rmse_aug)
    # CNN_result_aug.append(CNN_pred_aug)
    # print('Single Household Forecasting of %s th Day Using CNN ' % dayofforecast)
    # CNN_pred, CNN_rmse, CNN_mape = NN.CNN_net(X_train_3D1, y_train1, X_test_3D1, y_test1, shifted_value1)
    # CNN_Vm.append(CNN_mape)
    # CNN_Vr.append(CNN_rmse)
    # CNN_result.append(CNN_pred)
    print('Single Household Forecasting of %s th Day Using LSTM ' % dayofforecast)
    LSTM_pred, LSTM_rmse, LSTM_mape = NN.LSTM_network(X_train_3D, y_train, X_test_3D, y_test, baseline)
    LSTM_Vm.append(LSTM_mape)
    LSTM_Vr.append(LSTM_rmse)
    LSTM_result.append(LSTM_pred)
    #
    # plt.figure(figsize=(8,6))
    # plt.title('The Day-ahead Load Forecasting of Household ID:00%s' %Household_num, fontsize=14)
    # plt.plot(y_test, 'k-o', label="Actual Load")
    # plt.plot(ANN_pred, 'b--', label='Pooled MLP')
    # plt.plot(CNN_pred, 'b-.', label='Pooled CNN')
    # plt.plot(LSTM_pred, 'r-x', label='Pooled LSTM')
    # plt.plot(CNN_pred_aug, 'g-s', label='Augmented CNN')
    # plt.xlabel('Time index (in hours)', fontsize=14)
    # plt.ylabel('Energy Consumption (kWh)', fontsize=14)
    # plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
    #                       ("0:AM","01:AM","02:AM","03:AM","04:AM","05:AM","06:AM","07:AM","08:AM","09:AM","10:AM","11:AM",
    #                     "12:PM","01:PM","02:PM","03:PM","04:PM","05:PM","06:PM","07:PM","08:PM","09:PM","10:PM","11:PM"), rotation =45, label=13)
    # plt.axis([0,23, 0.1,0.9])
    # plt.xlabel('Time index (in hours)', fontsize=14)
    # plt.grid(True)
    # plt.legend(loc='upper left', fontsize=14)
    # plt.tight_layout()
    # plt.show()


# plt.figure(figsize=(8,6))
# plt.plot(ANN_Vm, 'k--', label='Pooled MLP')
# plt.plot(CNN_Vm, 'b-.', label='Pooled CNN ')
# plt.plot(LSTM_Vm, '.r-', label='Pooled LSTM ')
# plt.plot(CNN_Vm_aug, 'b-s', label='Augmented CNN ')
# plt.xlabel('Number of Forecasting Days',fontsize=14)
# plt.ylabel('MAPE',fontsize=14)
# plt.title('Performance Measure of Household ID:00%s ' %Household_num, fontsize=14)
# plt.legend(loc='upper right',fontsize=12)
# plt.axis([0,28,0,70])
# plt.grid('True')
# plt.tight_layout()
# plt.show()
# #
# plt.figure(figsize=(8,6))
# plt.plot(ANN_Vr, 'k--', label='Pooled MLP ')
# plt.plot(CNN_Vr, 'b-.', label='Pooled CNN')
# plt.plot(LSTM_Vr, '.r-', label='Pooled LSTM')
# plt.plot(CNN_Vr_aug, 'b-s', label='Augmented CNN')
# plt.xlabel('Number of Forecasting Days', fontsize=14)
# plt.ylabel(' RMSE', fontsize=14)
# plt.title('Performance Measure of Household ID:00%s' %Household_num, fontsize=14)
# plt.legend(loc='upper right',fontsize=12)
# plt.axis([0,28,0,0.25])
# plt.grid('True')
# plt.tight_layout()
# plt.show()