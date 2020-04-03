import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cluster as cl

def concat_load(Household_num):
    a=Household_num
    print('##################################################################################')
    print('Reading 2016-08- load series .....')
    excel08 = pd.ExcelFile('Household/H201608.xlsx')
    sheet108 = excel08.sheet_names
    data108 = pd.DataFrame()
    for sheet108 in sheet108:
        data108 = pd.concat([data108, excel08.parse(sheet108)])

    data108 = data108.fillna(0)
    data08 = data108.iloc[:, 1:]
    # print(data08.head())
    print(data08.shape)
    #print(data08.head())
    House1_08 = data08.iloc[:, :(a - 1)]
    #House1_081=data08.iloc[:,(a+100):]
    House1_08['aug'] = House1_08.sum(axis=1)
    #House1_081['aug1'] = House1_081.sum(axis=1)
    print(House1_08.tail())
    agg_Aug = House1_08.iloc[:,-1]
    #agg_Aug2 = House1_081.iloc[:, -1]


    print('##################################################################################')
    print('Reading 2016-09- load series .....')
    excel = pd.ExcelFile('Household/H201609.xlsx')
    sheets = excel.sheet_names
    data1 = pd.DataFrame()

    for sheet in sheets:
        data1 = pd.concat([data1, excel.parse(sheet)])

    data1 = data1.fillna(0)
    data09 = data1.iloc[:, 1:]
   # print(data09.tail())
    print(data09.shape)
    House1_09 = data09.iloc[:, :(a - 1)]
    # print(House1_09.head())
    House1_09['sep'] = House1_09.sum(axis=1)
    print(House1_09.tail())
    agg_sept = House1_09.iloc[:,-1]

    print('##################################################################################')
    print('Reading 2016-10- load series .....')
    excel10 = pd.ExcelFile('Household/201610.xlsx')
    sheet110 = excel10.sheet_names
    data110 = pd.DataFrame()
    for sheet110 in sheet110:
        data110 = pd.concat([data110, excel10.parse(sheet110)])

    data110 = data110.fillna(0)
    data10 = data110.iloc[:, 1:]
    # print(data08.head())
    print(data10.shape)
    print(data10.head())
    House1_10 = data10.iloc[:, :(a - 1)]
    House1_10['Oct'] = House1_10.sum(axis=1)
    print(House1_10.head())
    agg_oct = House1_10.iloc[:,-1]
    #
    print('##################################################################################')
    print('Reading 2016-11- load series .....')
    excel11 = pd.ExcelFile('Household/201611.xlsx')
    sheet111 = excel11.sheet_names
    data111 = pd.DataFrame()
    for sheet111 in sheet111:
        data111 = pd.concat([data111, excel11.parse(sheet111)])

    data111 = data111.fillna(0)
    data11 = data111.iloc[:, 1:]
    # print(data08.head())
    print(data11.shape)
    #print(data11.head())
    House1_11 = data11.iloc[:, :(a - 1)]
    House1_11['Nov'] = House1_11.sum(axis=1)
    print(House1_11.head())
    agg_Nov = House1_11.iloc[:,-1]
    #
    print('##################################################################################')
    print('Reading 2016-12- load series .....')
    excel12 = pd.ExcelFile('Household/201612.xlsx')
    sheet112 = excel12.sheet_names
    data112 = pd.DataFrame()
    for sheet112 in sheet112:
        data112 = pd.concat([data112, excel12.parse(sheet112)])

    data112 = data112.fillna(0)
    data12 = data112.iloc[:, 1:]
    # print(data08.head())
    print(data12.shape)
    #print(data12.head())
    House1_12 = data12.iloc[:, :(a - 1)]
    House1_12['Dec'] = House1_12.sum(axis=1)
    print(House1_12.head())
    agg_Dec = House1_12.iloc[:,-1]
    #
    print('##################################################################################')
    print('Reading 2017-01- load series .....')
    excel201701 = pd.ExcelFile('Household/201701.xlsx')
    sheet201701 = excel201701.sheet_names
    data201701 = pd.DataFrame()
    for sheet201701 in sheet201701:
        data201701 = pd.concat([data201701, excel201701.parse(sheet201701)])

    data201701 = data201701.fillna(0)
    data701 = data201701.iloc[:, 1:]
    # print(data08.head())
    print(data701.shape)
    print(data701.head())
    House1_701 = data701.iloc[:, :(a - 1)]
    House1_701['Jan'] = House1_701.sum(axis=1)
    print(House1_701.head())
    agg_Jan = House1_701.iloc[:,-1]
    #
    print('##################################################################################')
    print('Reading 2017-02- load series .....')
    excel201702 = pd.ExcelFile('Household/201702.xlsx')
    sheet201702 = excel201702.sheet_names
    data201702 = pd.DataFrame()
    for sheet201702 in sheet201702:
        data201702 = pd.concat([data201702, excel201702.parse(sheet201702)])

    data201702 = data201702.fillna(0)
    data702 = data201702.iloc[:, 1:]
    # print(data08.head())
    print(data702.shape)
    print(data702.head())
    House1_702 = data702.iloc[:, :(a - 1)]
    House1_702['Feb'] = House1_702.sum(axis=1)
    print(House1_702.head())
    agg_Feb = House1_702.iloc[:,-1]
    #
    print('##################################################################################')
    print('Reading 2017-03- load series .....')
    excel201703 = pd.ExcelFile('Household/201703.xlsx')
    sheet201703 = excel201703.sheet_names
    data201703 = pd.DataFrame()
    for sheet201703 in sheet201703:
        data201703 = pd.concat([data201703, excel201703.parse(sheet201703)])

    data201703 = data201703.fillna(0)
    data703 = data201703.iloc[:, 1:]
    print(data703.shape)
    print(data703.head())
    House1_703 = data703.iloc[:, :(a - 1)]
    House1_703['March'] = House1_703.sum(axis=1)
    print(House1_703.head())
    agg_Mar = House1_703.iloc[:,-1]
    #
    print('##################################################################################')
    print('Reading 2017-04- load series .....')
    excel201704 = pd.ExcelFile('Household/201704.xlsx')
    sheet201704 = excel201704.sheet_names
    data201704 = pd.DataFrame()
    for sheet201704 in sheet201704:
        data201704 = pd.concat([data201704, excel201704.parse(sheet201704)])

    data201704 = data201704.fillna(0)
    data704 = data201704.iloc[:, 1:]
    print(data704.shape)
    print(data704.head())
    House1_704 = data704.iloc[:, :(a - 1)]
    House1_704['April'] = House1_704.sum(axis=1)
    print(House1_704.head())
    agg_april = House1_704.iloc[:,-1]
    #
    print('##################################################################################')
    print('Reading 2017-05- load series .....')
    excel201705 = pd.ExcelFile('Household/201705.xlsx')
    sheet201705 = excel201705.sheet_names
    data201705 = pd.DataFrame()
    for sheet201705 in sheet201705:
        data201705 = pd.concat([data201705, excel201705.parse(sheet201705)])

    data201705 = data201705.fillna(0)
    data705 = data201705.iloc[:, 1:]
    print(data705.shape)
    print(data705.head())
    House1_705 = data705.iloc[:, :(a - 1)]
    House1_705['May'] = House1_705.sum(axis=1)
    print(House1_705.head())
    agg_may = House1_705.iloc[:,-1]
    #
    print('##################################################################################')
    print('Reading 2017-06- load series .....')
    excel201706 = pd.ExcelFile('Household/201706.xlsx')
    sheet201706 = excel201706.sheet_names
    data201706 = pd.DataFrame()
    for sheet201706 in sheet201706:
        data201706 = pd.concat([data201706, excel201706.parse(sheet201706)])

    data201706 = data201706.fillna(0)
    data706 = data201706.iloc[:, 1:]
    print(data706.shape)
    print(data706.head())
    House1_706 = data706.iloc[:, :(a - 1)]
    House1_706['June'] = House1_706.sum(axis=1)
    print(House1_706.head())
    agg_june = House1_706.iloc[:,-1]
    #
    print('##################################################################################')
    print('Reading 2017-07- load series .....')
    excel201707 = pd.ExcelFile('Household/201707.xlsx')
    sheet201707 = excel201707.sheet_names
    data201707 = pd.DataFrame()
    for sheet201707 in sheet201707:
        data201707 = pd.concat([data201707, excel201707.parse(sheet201707)])

    data201707 = data201707.fillna(data201707.mean())
    data707 = data201707.iloc[:, 1:]
    print(data707.shape)
    print(data707.head())
    House1_707 = data707.iloc[:, :(a - 1)]
    House1_707['July'] = House1_707.sum(axis=1)
    print(House1_707.head())
    agg_july = House1_707.iloc[:,-1]

    print('############################################################################')
    print(House1_08.shape, House1_09.shape,House1_10.shape, House1_11.shape,House1_12.shape, House1_701.shape,House1_702.shape)
    print(House1_703.shape, House1_704.shape, House1_705.shape, House1_706.shape, House1_707.shape)
    House1 = pd.concat([agg_Aug, agg_sept, agg_oct, agg_Nov, agg_Dec, agg_Jan, agg_Feb, agg_Mar, agg_april, agg_may, agg_june, agg_july])
    #print(House1.head())
    house11=np.array(House1)
    house11[house11>1200]=np.mean(house11)
    house11[house11<150]=np.mean(house11)
    #print(House1.tail())
    plt.plot(house11,'k:')
    plt.xlabel('Time index (hour)')
    plt.ylabel('Energy Consumption (kWhs)')
    plt.title('Aggregated Energy Consumption of %s Household'%a)
    plt.show()
    print(House1.shape)
    return house11


def concat_Indi_load(Household_num):
    a=Household_num
    print('############################################################################')
    def data_collect(data_send, a):
        House_1 = data_send.iloc[:, a - 1]
        House_2 = data_send.iloc[:, a]
        House_3 = data_send.iloc[:, a + 1]
        House_4 = data_send.iloc[:, a + 2]
        House_5 = data_send.iloc[:, a + 3]
        House_6 = data_send.iloc[:, a + 4]
        House_7 = data_send.iloc[:, a + 5]
        #----------------------------------------------------
        indi_1 = House_1.replace(0, House_1.mean())
        indi_2 = House_2.replace(0, House_2.mean())
        indi_3 = House_3.replace(0, House_3.mean())
        indi_4 = House_4.replace(0, House_4.mean())
        indi_5 = House_5.replace(0, House_5.mean())
        indi_6 = House_6.replace(0, House_6.mean())
        indi_7 = House_7.replace(0, House_7.mean())
        return indi_1, indi_2, indi_3, indi_4, indi_5, indi_6, indi_7

   #  print('##################################################################################')
   #  print('Reading 2016-08- load series .....')
   #  excel08 = pd.ExcelFile('Household/H201608.xlsx')
   #  sheet108 = excel08.sheet_names
   #  data108 = pd.DataFrame()
   #  for sheet108 in sheet108:
   #      data108 = pd.concat([data108, excel08.parse(sheet108)])
   #
   #  data108 = data108.fillna(0)
   #  data08 = data108.iloc[:, 1:]
   #  # print(data08.head())
   #  print(data08.shape)
   #  #print(data08.head())
   #  indi_Aug1, indi_Aug2, indi_Aug3, indi_Aug4,indi_Aug5, indi_Aug6, indi_Aug7=data_collect(data08, a)
   # #
   #  print('##################################################################################')
   #  print('Reading 2016- september-- load series .....')
   #  excel = pd.ExcelFile('Household/H201609.xlsx')
   #  sheets = excel.sheet_names
   #  data1 = pd.DataFrame()
   #
   #  for sheet in sheets:
   #      data1 = pd.concat([data1, excel.parse(sheet)])
   #
   #  data1 = data1.fillna(0)
   #  data09 = data1.iloc[:, 1:]
   # # print(data09.tail())
   #  indi_sept1, indi_sept2, indi_sept3, indi_sept4, indi_sept5, indi_sept6, indi_sept7 = data_collect(data09, a)

    print('##################################################################################')
    # print('Reading 2016-october- load series .....')
    # excel10 = pd.ExcelFile('Household/201610.xlsx')
    # sheet110 = excel10.sheet_names
    # data110 = pd.DataFrame()
    # for sheet110 in sheet110:
    #     data110 = pd.concat([data110, excel10.parse(sheet110)])
    #
    # data110 = data110.fillna(0)
    # data10 = data110.iloc[:, 1:]
    # # print(data08.head())
    # print(data10.shape)
    # indi_oct1, indi_oct2, indi_oct3, indi_oct4, indi_oct5, indi_oct6, indi_oct7 = data_collect(data10, a)

    # print('##################################################################################')
    # print('Reading 2016-November- load series .....')
    # excel11 = pd.ExcelFile('Household/201611.xlsx')
    # sheet111 = excel11.sheet_names
    # data111 = pd.DataFrame()
    # for sheet111 in sheet111:
    #     data111 = pd.concat([data111, excel11.parse(sheet111)])
    #
    # data111 = data111.fillna(0)
    # data11 = data111.iloc[:, 1:]
    # # print(data08.head())
    # print(data11.shape)
    # print(data11.head())
    # indi_nov1, indi_nov2, indi_nov3, indi_nov4, indi_nov5, indi_nov6, indi_nov7 = data_collect(data11, a)

    # print('##################################################################################')
    # print('Reading 2016-12- load series .....')
    # excel12 = pd.ExcelFile('Household/201612.xlsx')
    # sheet112 = excel12.sheet_names
    # data112 = pd.DataFrame()
    # for sheet112 in sheet112:
    #     data112 = pd.concat([data112, excel12.parse(sheet112)])
    #
    # data112 = data112.fillna(0)
    # data12 = data112.iloc[:, 1:]
    # # print(data08.head())
    # print(data12.shape)
    # #print(data12.head())
    # indi_dec1, indi_dec2, indi_dec3, indi_dec4, indi_dec5, indi_dec6, indi_dec7 = data_collect(data12, a)
   #  #
   #  print('##################################################################################')
   #  print('Reading 2017-01- load series .....')
   #  excel201701 = pd.ExcelFile('Household/201701.xlsx')
   #  sheet201701 = excel201701.sheet_names
   #  data201701 = pd.DataFrame()
   #  for sheet201701 in sheet201701:
   #      data201701 = pd.concat([data201701, excel201701.parse(sheet201701)])
   #
   #  data201701 = data201701.fillna(0)
   #  data701 = data201701.iloc[:, 1:]
   #  # print(data08.head())
   #  #print(data701.shape)
   #  #print(data701.head())
   #  indi_jan1, indi_jan2, indi_jan3, indi_jan4, indi_jan5, indi_jan6, indi_jan7 = data_collect(data701, a)

   #  print('##################################################################################')
   #  print('Reading 2017-02- load series .....')
   #  excel201702 = pd.ExcelFile('Household/201702.xlsx')
   #  sheet201702 = excel201702.sheet_names
   #  data201702 = pd.DataFrame()
   #  for sheet201702 in sheet201702:
   #      data201702 = pd.concat([data201702, excel201702.parse(sheet201702)])
   #
   #  data201702 = data201702.fillna(0)
   #  data702 = data201702.iloc[:, 1:]
   #  # print(data08.head())
   #  print(data702.shape)
   #  #print(data702.head())
   #  indi_feb1, indi_feb2, indi_feb3, indi_feb4, indi_feb5, indi_feb6, indi_feb7 = data_collect(data702, a)
   # #  #
   #  print('##################################################################################')
   #  print('Reading 2017-03- load series .....')
   #  excel201703 = pd.ExcelFile('Household/201703.xlsx')
   #  sheet201703 = excel201703.sheet_names
   #  data201703 = pd.DataFrame()
   #  for sheet201703 in sheet201703:
   #      data201703 = pd.concat([data201703, excel201703.parse(sheet201703)])
   #
   #  data201703 = data201703.fillna(0)
   #  data703 = data201703.iloc[:, 1:]
   #  print(data703.shape)
   #  #print(data703.head())
   #  indi_mar1, indi_mar2, indi_mar3, indi_mar4, indi_mar5, indi_mar6, indi_mar7 = data_collect(data703, a)
   #  #
   #  print('##################################################################################')
   #  print('Reading 2017-04- load series .....')
   #  excel201704 = pd.ExcelFile('Household/201704.xlsx')
   #  sheet201704 = excel201704.sheet_names
   #  data201704 = pd.DataFrame()
   #  for sheet201704 in sheet201704:
   #      data201704 = pd.concat([data201704, excel201704.parse(sheet201704)])
   #
   #  data201704 = data201704.fillna(0)
   #  data704 = data201704.iloc[:, 1:]
   #  print(data704.shape)
   # # print(data704.head())
   #  indi_apr1, indi_apr2, indi_apr3, indi_apr4, indi_apr5, indi_apr6, indi_apr7 = data_collect(data704, a)
   #  #
   #  print('##################################################################################')
   #  print('Reading 2017-05- load series .....')
   #  excel201705 = pd.ExcelFile('Household/201705.xlsx')
   #  sheet201705 = excel201705.sheet_names
   #  data201705 = pd.DataFrame()
   #  for sheet201705 in sheet201705:
   #      data201705 = pd.concat([data201705, excel201705.parse(sheet201705)])
   #
   #  data201705 = data201705.fillna(0)
   #  data705 = data201705.iloc[:, 1:]
   #  print(data705.shape)
   #  #print(data705.head())
   #  indi_may1, indi_may2, indi_may3, indi_may4, indi_may5, indi_may6, indi_may7 = data_collect(data705, a)
   # #  #
    print('##################################################################################')
    print('Reading 2017-06- load series .....')
    excel201706 = pd.ExcelFile('Household/201706.xlsx')
    sheet201706 = excel201706.sheet_names
    data201706 = pd.DataFrame()
    for sheet201706 in sheet201706:
        data201706 = pd.concat([data201706, excel201706.parse(sheet201706)])

    data201706 = data201706.fillna(0)
    data706 = data201706.iloc[:, 1:]
    print(data706.shape)
    #print(data706.head())
    indi_jun1, indi_jun2, indi_jun3, indi_jun4, indi_jun5, indi_jun6, indi_jun7 = data_collect(data706, a)
   #  #
    print('##################################################################################')
    print('Reading 2017-07- load series .....')
    excel201707 = pd.ExcelFile('Household/201707.xlsx')
    sheet201707 = excel201707.sheet_names
    data201707 = pd.DataFrame()
    for sheet201707 in sheet201707:
        data201707 = pd.concat([data201707, excel201707.parse(sheet201707)])

    data201707 = data201707.fillna(0)
    data707 = data201707.iloc[:, 1:]
    print(data707.shape)
    indi_jul1, indi_jul2, indi_jul3, indi_jul4, indi_jul5, indi_jul6, indi_jul7 = data_collect(data707, a)
    print('Peak of July measure.%f'%indi_jul1.max())
    result = np.where(indi_jul1 == np.amax(indi_jul1))
    result = result[0]
    result = int(result)
    a = result // 24
    print(a, a + 1)
    index=max_load_day(indi_jul1)
    print('Maximum day load in July measure.%f' % indi_jul1.max())

    print('############################################################################')
    # print(House1_08.shape, House1_09.shape,House1_10.shape, House1_11.shape,House1_12.shape, House1_701.shape,House1_702.shape)
    # print(House1_703.shape, House1_704.shape, House1_705.shape, House1_706.shape, House1_707.shape)
    House1 = pd.concat([ indi_jun1, indi_jul1])
                        #agg_oct1, agg_Nov1, agg_Dec1, agg_Jan1, agg_Feb1, agg_Mar1, agg_april1, agg_may1, agg_june1, agg_july1])
    #print(House1.head())
    house11=House1
    # agg_sept2=(np.mean(agg_sept1)/np.mean(agg_sept2)*agg_sept2)
    # House2=pd.concat([agg_sept2, House1])
    house112=conc_indi_series(indi_jul1, indi_jul2, indi_jul3, indi_jul4)
    house116=conc_indi_series_6(indi_jul1, indi_jul2, indi_jul3, indi_jul4, indi_jul5, indi_jul6, indi_jul7)
    #agg_oct1=np.array(indi_sept1)
    House2=np.concatenate([house112, indi_jul1])
    House6=np.concatenate([house116, indi_jul1])
    print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
    house11[house11>2.5]=2.5
    House2[House2 > 2.5] = 2.5
    House6[House6>2.5]=2.5
    #print(House1.tail())
    # #plt.plot(np.array(House1),'k--')
    # plt.plot(np.array(House2),'r-.')
    # plt.xlabel('Time index (hour)')
    # plt.ylabel('Energy Consumption (kWhs)')
    # plt.title('Individual Energy Consumption of %s Household'%a)
    # #plt.axis([100, 200, 0, 2])
    # plt.show()
    print(House2.shape)
   # b=index+1
    #print('Peak day is: ' + str(a) + ' Maximum Load Day: ' + str(b))
    return house11, House2 , House6

def conc_indi_series (data1, data2, data3, data4):
    data1=np.array(data1)
    data2=np.array(data2)
    data3=np.array(data3)
    data4=np.array(data4)
    mean1=np.mean(data1)
    mean2=np.mean(data2)
    mean3=np.mean(data3)
    mean4=np.mean(data4)
    data2=(mean1/mean2)*data2
    data3=(mean1/mean3)*data3
    data4=(mean1/mean4)*data4
    data=np.concatenate((data2, data3), axis=None)
    data=np.concatenate((data,data4), axis=None)
    data = np.concatenate((data, data1), axis=None)
    return data

def conc_indi_series_6 (data1, data2, data3, data4, data5, data6, data7):
    data1=np.array(data1)
    data2=np.array(data2)
    data3=np.array(data3)
    data4=np.array(data4)
    mean1=np.mean(data1)
    mean2=np.mean(data2)
    mean3=np.mean(data3)
    mean4=np.mean(data4)
    mean5=np.mean(data5)
    mean6=np.mean(data6)
    mean7=np.mean(data7)
    data2=(mean1/mean2)*data2
    data3=(mean1/mean3)*data3
    data4=(mean1/mean4)*data4
    data5=(mean1/mean5)*data5
    data6=(mean1/mean6)*data6
    data7=(mean1/mean7)*data7
    data=np.concatenate((data2, data3), axis=None)
    data=np.concatenate((data,data4), axis=None)
    data=np.concatenate((data, data5), axis=None)
    data=np.concatenate((data,data6), axis=None)
    data=np.concatenate((data, data7), axis=None)
    data = np.concatenate((data, data1), axis=None)
    return data


def max_load_day (data):
    max_load=cl.CSM(data)
    max_load=pd.DataFrame(max_load)
    max_load['sum_load']=max_load.sum(axis=1)
    print('YEsssssssssssssssssssssss')
    print(max_load)
    index=max_load['sum_load'].idxmax()
    print(index)
    return index
