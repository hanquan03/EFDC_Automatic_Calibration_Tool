# coding=utf-8
"""
1) Based on the parameter information and LHS sampling results (500 sets), 
models are established and run;
2) The objective function is also calculated.
"""



import os
import shutil
import math
import string
import netCDF4 as nc
import numpy as np
import pandas as pd
from datetime import datetime
import sys


#Read the parameter range file
param_name = "G:/AHydro_2016/code/Data/hydro_paras_name.txt"

#######################################################
# FUNCTION: READ MODEL PARAMETER FILE
#======================================================
def read_param_file(filename):
    with open(filename, "r") as file:
        names = []
        bounds = []
        num_vars = 0

        for row in [line.split() for line in file if not line.strip().startswith('#')]:
            num_vars += 1
            names.append(row[0])
            bounds.append([float(row[1]), float(row[2])])

    return {'names': names, 'bounds': bounds, 'num_vars': num_vars}


#######################################################
# FUNCTION: GENERATE MODEL INPUT FILE
#======================================================
def genAppInputFile(inputData,appInputFile,nInputs,inputNames):
    """
    inputData: parameter values
    appInputFile: Model input files
    nInputs: Number of parameters need to be calibrated
    inputNames: Parameters list
    """
    outfile = open(appInputFile, "r")
    lineIn = outfile.readlines()
    lineOut = []
    outfile.close()
    for newLine in lineIn:
        lineLen = len(newLine)
        if nInputs > 0:
            for fInd in range(nInputs):
                strLen = len(inputNames[fInd])
                sInd = str.find(newLine, inputNames[fInd])
                if sInd >= 0 :
                    break
            if sInd >= 0:
                while(1):
                    sdata = '%10.3f' % inputData[fInd]
                    strdata = str(sdata)
                    next = sInd + strLen
                    lineTemp = newLine[0:sInd] + strdata + " " + newLine[next:lineLen+1]
                    sInd = lineTemp.find(inputNames[fInd])
                    if sInd >= 0 :
                        newLine = lineTemp
                    else:
                        break
                lineOut.append(lineTemp)
            else:
                lineOut.append(newLine)
    outfile = open(appInputFile, "w")
    outfile.writelines(lineOut)
    outfile.close()
    return


#######################################################
# FUNCTION: RUN EFDC MODELS
#======================================================
def runEFDC(run_bat):
    """
    run_bat:  Model run executables
    """
    sysComm = run_bat
    os.system(sysComm)
    return


#######################################################
# MAIN PROGRAM
# =====================================================
def predict(values):
    """
    values: LHS parameter sampling results
    """
    pf = read_param_file(param_name)
    for n in range(pf['num_vars']):
        pf['names'][n] = 'EACT_' + pf['names'][n]

    Y = np.empty([values.shape[0]])
    os.chdir('G:/AHydro_2016/500m_Model')

    for i, row in enumerate(values):
        inputData = values[i]
        if os.path.exists('./test' + str(i)):
            shutil.rmtree('./test' + str(i))
        os.mkdir('./test' + str(i))
        tmplate_path = os.listdir('./EFDC_Template')
        tmplate_path.remove("#analysis")
        tmplate_path.remove("#output")
        for line in tmplate_path:
            shutil.copy('./EFDC_Template/' + line, './test' + str(i) + '/' + line)
        os.chdir('./test' + str(i))
        for j in os.listdir('./'):
            if os.path.splitext(j)[0] == 'efdc':
                for nn in range(6):
                    genAppInputFile(inputData[nn:nn+1], j, 1,pf['names'][nn:nn+1])
            if os.path.splitext(j)[0] == 'lxly':
                genAppInputFile(inputData[6:7],j,1,pf['names'][6:7])
            if os.path.splitext(j)[0] == 'dxdy':
                genAppInputFile(inputData[7:8],j,1,pf['names'][7:8])
            if os.path.splitext(j)[0] == 'vege':
                genAppInputFile(inputData[8:9], j, 1, pf['names'][8:9])
                genAppInputFile(inputData[9:10], j, 1, pf['names'][9:10])
                genAppInputFile(inputData[10:11], j, 1, pf['names'][10:11])
            if os.path.splitext(j)[1] == '.bat':
                file = open(j, "r")
                lines = file.readlines()
                lines[3] = "TITLE G:\\AHydro_2016\\500m_Model\\test"+str(i)+ "\n"
                lines[5] = 'CD "G:\\AHydro_2016\\500m_Model\\test'+str(i)+'"\n'
                file = open(j, "w")
                file.writelines(lines)
                file.close()
        run_bat = str(os.getcwd())+"/0run_efdc.bat"
        runEFDC(run_bat)
        print ("Run EFDC model ID " + str(i))
        os.chdir("../")
    return Y

# Upload the generate samples file
values = np.loadtxt("G:/AHydro_2016/code/Data/para_LHS500.txt", unpack=False)

# Run models and save the output
res = predict(values)

print("1. Estiblise EFDC models and save simulation results")

#######################################################
# UPGRADE OBJECTIVE FUNCTION
# INCLUDING MRE, RMSE, NSE, and R2
#======================================================

origin = sys.stdout
results_df = pd.DataFrame(columns=['ID', 'MRE', 'RMSE', 'NSE', 'R2'], dtype=object)
wl_obs = np.loadtxt(r"G:\AHydro_2016\code\Data\cali_wl.txt", unpack=False)

for n in range(500):
    file_path = r"G:\\AHydro_2016\\500m_Model\test" + str(n) + "\#output\DSI_EFDC.nc"
    nc_file = nc.Dataset(file_path)
    wl_model = np.empty((1, 366))
    for i in range(366):
        star_time = 24*i
        end_time = 24*(i+1)
        wl_hour = nc_file.variables["WSEL"][star_time:end_time,0,17]
        wl_model[0][i-1] = wl_hour.mean()

    MRE = np.mean(np.abs(wl_obs - wl_model) / wl_obs)
    RMSE = np.sqrt(np.mean((wl_obs - wl_model) ** 2))
    NSE = 1 - np.sum((wl_obs - wl_model) ** 2) / np.sum((wl_obs - np.mean(wl_obs)) ** 2)
    R2 = (np.corrcoef(wl_model, wl_obs)[0,1])**2


    print (MRE, RMSE, NSE, R2)
    results_df = results_df.append({
        'ID': n,
        'MRE': round(MRE, 4),
        'RMSE': round(RMSE, 4),
        'NSE': round(NSE, 4),
        'R2': round(R2, 4)
    }, ignore_index=True)
    nc_file.close()

results_df.to_excel(r"G:\AHydro_2016\code\Res\Obj_fun_res.xlsx", index=False)

print("2. Calculate Objective function")