import fnmatch
import os
import sys
import pandas as pd
import h5py
import numpy as np
import matplotlib.pyplot as plt


def combine_simulations(MC_raw_path, MC_id):
    "combine all g4simple .hdf5 simulations within a folder into one dataframe"

    #read in each hdf5 file
    files = os.listdir(MC_raw_path)
    files = fnmatch.filter(files, "*.hdf5")
    df_list = []
    for file in files:

        # print("file: ", str(file))
        file_no = file[-7]+file[-6]
        # print("raw MC file_no: ", file_no)

        g4sfile = h5py.File(MC_raw_path+file, 'r')

        g4sntuple = g4sfile['default_ntuples']['g4sntuple']
        g4sdf = pd.DataFrame(np.array(g4sntuple), columns=['event'])

        # # build a pandas DataFrame from the hdf5 datasets we will use
        g4sdf = pd.DataFrame(np.array(g4sntuple['event']['pages']), columns=['event'])
        g4sdf = g4sdf.join(pd.DataFrame(np.array(g4sntuple['step']['pages']), columns=['step']),lsuffix = '_caller', rsuffix = '_other')
        g4sdf = g4sdf.join(pd.DataFrame(np.array(g4sntuple['Edep']['pages']), columns=['Edep']),lsuffix = '_caller', rsuffix = '_other')
        g4sdf = g4sdf.join(pd.DataFrame(np.array(g4sntuple['volID']['pages']),columns=['volID']), lsuffix = '_caller', rsuffix = '_other')
        g4sdf = g4sdf.join(pd.DataFrame(np.array(g4sntuple['iRep']['pages']),columns=['iRep']), lsuffix = '_caller', rsuffix = '_other')
        g4sdf = g4sdf.join(pd.DataFrame(np.array(g4sntuple['x']['pages']),columns=['x']), lsuffix = '_caller', rsuffix = '_other')
        g4sdf = g4sdf.join(pd.DataFrame(np.array(g4sntuple['y']['pages']),columns=['y']), lsuffix = '_caller', rsuffix = '_other')
        g4sdf = g4sdf.join(pd.DataFrame(np.array(g4sntuple['z']['pages']),columns=['z']), lsuffix = '_caller', rsuffix = '_other')

        #add new column to each df for the raw MC file no
        g4sdf["raw_MC_fileno"] = file_no

        df_list.append(g4sdf)

    #concatonate
    df_total = pd.concat(df_list, axis=0, ignore_index=True)

    return df_total





def plotMCHits(detector, source, MC_raw_path, MC_id, reopen_saved=False):

    # smear, TL_model, FCCD, DLF, frac_FCCDbore = "g", "notl", 0.0, 1.0, 0.5 
    # MC_id_new = MC_id+"_"+smear+"_"+TL_model+"_FCCD"+str(FCCD)+"mm_DLF"+str(DLF)+"_fracFCCDbore"+str(frac_FCCDbore)
    # sim_path = MC_raw_path+"AV_processed/"+MC_id_new+".hdf5"
    # df_pp =  pd.read_hdf(sim_path, key="procdf")
    # print(df_pp)

    CodePath=os.path.dirname(os.path.realpath(__file__))
    outputFolder = CodePath+"/../"

    if reopen_saved == False:
        #combine simulations and save df_total temporarily
        df_total = combine_simulations(MC_raw_path, MC_id)
        df_total.to_hdf(outputFolder+"misc/hdf5/rawMCHits_"+MC_id+'.hdf5', key='procdf', mode='w')
    else:
        #reopen MC
        df_total =  pd.read_hdf(outputFolder+"misc/hdf5/rawMCHits_"+MC_id+'.hdf5', key="procdf")

    Edeps = df_total["Edep"].to_list()
    x_coords = df_total["x"].to_list()
    y_coords = df_total["y"].to_list()
    z_coords = df_total["z"].to_list()
    # print(Edeps)
    print(x_coords)

    heatmap, xedges, yedges = np.histogram2d(x_coords, y_coords, bins=100)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    plt.clf()
    plt.imshow(heatmap.T, extent=extent, origin='lower')
    plt.show()

    # arr_Edeps = np.zeros((len(y_coords), len(x_coords)))



    # # no_cols = 2*r/grid_space + 1
    # # extent = ((-0.5)*grid_space - r, (no_cols-0.5)*grid_space - r, (-0.5)*grid_space - r, (no_cols-0.5)*grid_space - r)
    # arr
    # heatmap = ax.imshow(arr, interpolation='none', origin = "lower", extent = extent, vmin=0.95*median_AoE, vmax=1.5*median_AoE)
    #     # plt.xlabel("x / mm")
    #     # plt.ylabel("y / mm")
    #     # plt.title("z="+str(z))

    # # heatmap, xedges, yedges = np.histogram2d(x, y, bins=50)
    
    # # extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    # # plt.clf()
    # # plt.imshow(heatmap.T, extent=extent, origin='lower')
    # # plt.show()





