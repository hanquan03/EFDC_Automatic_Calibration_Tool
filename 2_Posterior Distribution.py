# env3.8
"""

1) Based on the results of 500 sets EFDC models and the objective function,
   the cumulative probability was calculated;
2) Update the posterior distribution of EFDC hydrodynamic model parameters;
3) Plot the posterior distribution for each parameters.

"""


import os
import math
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
import seaborn as sns


#######################################################
# DATA PREPARATION
#======================================================

# Load parameters information
os.chdir(r"G:\AHydro_2016\code\Data\\")
param_names = np.loadtxt("hydro_paras_name.txt", dtype=str)
param_df = pd.DataFrame(param_names)
param_df.columns = ["para", "min", "max"]
param_df = param_df.sort_values(by="para")


# Load parameters sampling results
values = np.loadtxt("para_LHS500.txt", unpack=False)
values_df = pd.DataFrame(values)
column_names = param_names[:, 0]
values_df.columns = column_names
values_df['ID'] = range(len(values_df))


# Load objective function results
# obj_fun = pd.read_excel(r"G:\AHydro_2016\code\Res\Obj_fun_res.xlsx", usecols=["ID", "NSE"])
obj_fun = pd.read_excel(r"G:\AHydro_2016\code\Res\Obj_fun_res.xlsx", usecols=[0, 3])
opt_num = obj_fun[obj_fun['NSE'] > 0.5]
data = pd.merge(values_df, opt_num, on='ID', how='right')


# save to excel
output_excel_path = r"G:\AHydro_2016\code\Res\\para_cum_data.xlsx"
with pd.ExcelWriter(output_excel_path, engine='xlsxwriter') as writer:
    param_df.to_excel(writer, sheet_name='para_range', index=False)
    data.to_excel(writer, sheet_name='para_data', index=False)

print("1. Prepare data for parameters cumulative probability calculation ")



#######################################################
# UPDATE POSTERIOR DISTRIBUTION
#======================================================

os.chdir(r"G:\AHydro_2016\code\Res\\")

data_1km = pd.read_excel("para_cum_data.xlsx",sheet_name="para_data")
del data_1km["ID"]
output = os.path.join("G:\AHydro_2016\code\Res\\",'para_cum_res.xlsx')


# Obtain the likelihood function probability
data_1km["NSE_normal"] = data_1km["NSE"].apply(lambda x: x / data_1km["NSE"].sum())
data_1km = data_1km.dropna(how="any")


# The cumulative probability calculation for each parameters
for i in range(0,11):
    print(i, "Parameter Done")
    data_temp_1km = data_1km.sort_values(by=data_1km.columns[i],ascending=True)
    sum_clom_1km = data_1km.columns[i]+"_cum_pro"
    data_1km[sum_clom_1km] = data_temp_1km["NSE_normal"].cumsum()

da = data_1km.sort_index(axis=1)
del da['NSE_normal']

with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
    da.to_excel(writer, sheet_name='para_cum', index=False)

print("2. EFDC hydro_parameters cumulative probability calculation")



#######################################################
# PLOT FOR EACH PARAMETERS
#======================================================

os.chdir(r"G:\AHydro_2016\code\Res\\")

# Load data
para = pd.read_excel("para_cum_data.xlsx",sheet_name="para_range")
data = pd.read_excel("para_cum_res.xlsx",sheet_name="para_cum")
data.insert(0, "NSE", data.pop("NSE"))


#Basic figure settings
fig = plt.figure(dpi=300,figsize=(14, 10))
plt.subplot(3, 4, 1)
plt.subplots_adjust(hspace=0.5, wspace=0.5)


# Plot
sta_line = np.linspace(0,1,len(data["NSE"]))
for i in range(11):
    n = 2*i+1
    m = n +1
    data = data.sort_values(by=data.columns[n],ascending=True)
    ax1 = fig.add_subplot(3, 4, i+1)
    x_name = list(data)[n]
    x_min = float(para.iloc[i, 1])
    x_max = float(para.iloc[i, 2])
    xtick = np.linspace(x_min,x_max,4)
    xtick = np.round(xtick, 3)

    plt.figure(figsize=(30, 30))
    ax1.scatter(data.iloc[:, n], data["NSE"], alpha = 0.5, c="#C0504D") #红色
    ax1.set_xlim([x_min, x_max])
    ax1.set_xticks(xtick)
    ax1.set_xticklabels(xtick)
    ax1.set_ylim([0.5, 0.8])
    ax1.set_ylabel('Likelihood', fontsize=10, labelpad=1)
    ax1.set_xlabel('Paramter range', fontdict={'family': 'SimSun', 'size': 10}, labelpad=5)

    sta_x = np.linspace(x_min,x_max,len(data["NSE"]))
    ax2 = ax1.twinx()
    ax2.plot(data.iloc[:,n], data.iloc[:,m], "red")
    ax2.plot(sta_x, sta_line, "darkgray", linestyle='--', alpha=0.5)
    ax2.set_ylim([0,1])
    ax2.set_ylabel('cum_prob', fontdict={'family': 'SimSun', 'size': 10}, labelpad=2)
    ax2.set_title(x_name,fontdict={'weight': 'bold'})
    print(i)

plt.show()
fig.savefig(r"G:\AHydro_2016\code\Res\\hydro_para_cumul.jpg",dpi=300,bbox_inches = 'tight')
print("3. Plot Done")
