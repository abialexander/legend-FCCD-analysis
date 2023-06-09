import json
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mc
import numpy as np
import colorsys
import pandas as pd
from tabulate import tabulate

class Vividict(dict):
    def __missing__(self, key):
        value = self[key] = type(self)()
        return value

def lighten_color(color, amount = 0.5):
    try:
        c = mc.cnames[color]
    except:
       c = color
    c = colorsys.rgb_to_hls( * mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


def collateFCCDs(order_list, source, energy_filter="cuspEmax_ctc", cuts=True, smear="g", frac_FCCDbore=0.5):
    """
    Collate all FCCDs and errors into single dict and save to json file
    args: 
        - order_list
        - source ("Ba133", "Am241_HS1")
    """

    CodePath=os.path.dirname(os.path.realpath(__file__))

    #fixed
    DLF=1.0
    TL_model="notl"

    dict_all = Vividict()

    if source == "Ba133":
        source_data = "ba_HS4" #convert source name into correct nomenclature
    elif source == "Am241_HS1":
        source_data = "am_HS1"
    elif source == "Am241_HS6":
        source_data = "am_HS6"

    #Get detector list
    detector_list = CodePath+"/../detector_list.json"
    with open(detector_list) as json_file:
        detector_list_data = json.load(json_file)

    #Iterate through detectors in each order
    for order in order_list:
        detectors = detector_list_data["order_"+str(order)]["detectors"]
        runs = detector_list_data["order_"+str(order)]["runs"][source]
        MC_source_positions = detector_list_data["order_"+str(order)]["MC_source_positions"][source]
        for ind, detector in enumerate(detectors):

            if detector=="V07646A": #result needs checking
                print("ignoring FCCD result for ",detector, " and source ", source)
                continue


            run = runs[ind]
            MC_source_position = MC_source_positions[ind] #=["top","0r","78z"]
            MC_source_pos_hyphon = MC_source_position[0]+"-"+ MC_source_position[1]+"-"+MC_source_position[2] #="top-0r-78z"

            try:
                if cuts == False:
                    with open(CodePath+"/../results/FCCD/"+detector+"/"+source+"/FCCD_"+detector+"-"+source_data+"-"+MC_source_pos_hyphon+"_"+smear+"_"+TL_model+"_fracFCCDbore"+str(frac_FCCDbore)+"_"+energy_filter+"_run"+str(run)+"_nocuts.json") as json_file:
                        FCCD_data = json.load(json_file)
                else: #4sigma cuts default
                    with open(CodePath+"/../results/FCCD/"+detector+"/"+source+"/FCCD_"+detector+"-"+source_data+"-"+MC_source_pos_hyphon+"_"+smear+"_"+TL_model+"_fracFCCDbore"+str(frac_FCCDbore)+"_"+energy_filter+"_run"+str(run)+"_cuts.json") as json_file:
                        FCCD_data = json.load(json_file)
            except:
                print("no FCCD result with ",source," for ", detector)
                continue
            
            FCCD = FCCD_data["FCCD"]
            FCCD_err_total_up = FCCD_data["FCCD_err_total_up"]
            FCCD_err_total_low = FCCD_data["FCCD_err_total_low"]
            FCCD_err_corr_up = FCCD_data["FCCD_err_corr_up"]
            FCCD_err_corr_low = FCCD_data["FCCD_err_corr_low"]
            FCCD_err_uncorr_up = FCCD_data["FCCD_err_uncorr_up"]
            FCCD_err_uncorr_low = FCCD_data["FCCD_err_uncorr_low"]

            dict = {"FCCD": FCCD, 
            "FCCD_err_total_up": FCCD_err_total_up,
            "FCCD_err_total_low": FCCD_err_total_low,
            "FCCD_err_corr_up": FCCD_err_corr_up, 
            "FCCD_err_corr_low": FCCD_err_corr_low,  
            "FCCD_err_uncorr_up":FCCD_err_uncorr_up,
            "FCCD_err_uncorr_low": FCCD_err_uncorr_low
            }
            dict_round2dp = {"FCCD": round(FCCD,2), 
            "FCCD_err_total_up": round(FCCD_err_total_up,2),
            "FCCD_err_total_low": round(FCCD_err_total_low,2),
            "FCCD_err_corr_up": round(FCCD_err_corr_up,2), 
            "FCCD_err_corr_low": round(FCCD_err_corr_low,2),  
            "FCCD_err_uncorr_up":round(FCCD_err_uncorr_up,2),
            "FCCD_err_uncorr_low": round(FCCD_err_uncorr_low,2)
            } 
        
            dict_det = {"unrounded": dict, "rounded": dict_round2dp}
            dict_all[detector]=dict_det
    

    with open(CodePath+"/../resultsAll/FCCDvalues_"+source+".json", "w") as outfile:
        json.dump(dict_all, outfile, indent=4)
    
    return dict_all




def plotResults(order_list,source_list,results_type):
    """
    Plot all results for a given order_list and source_list
    args:
    - order_list
    - source_list
    - results_type: "FCCD" or "fAV"
    """

    CodePath=os.path.dirname(os.path.realpath(__file__))

    #Get detector list
    detector_list = CodePath+"/../detector_list.json"
    with open(detector_list) as json_file:
        detector_list_data = json.load(json_file)

    fig, ax = plt.subplots(figsize=(12,8))
    colors_orders = {2:'darkviolet', 4:'deepskyblue', 5:'orangered', 7:'green', 8:'gold', 9:'magenta'}
    markers_sources = {"Ba133": "o", "Am241_HS1":"^"} #abi plot

    detectors_all = []
    orders_all = []

    for order in order_list:

        for source in source_list:

            if results_type == "FCCD":
                json_path = CodePath+"/../resultsAll/FCCDvalues_"+source+".json"
            else:
                json_path = CodePath+"/../resultsAll/AVvalues.json"
            with open(json_path) as json_file:
                FCCDs_json = json.load(json_file)
            detectors = detector_list_data["order_"+str(order)]["detectors"]

            FCCDs_source = []
            FCCD_err_ups_source = []
            FCCD_err_lows_source = []

            detectors_source = []

            for detector in detectors:

                if detector=="V07646A": #result needs checking
                    print("ignoring FCCD result for ",detector, " and source ", source)
                    continue

                detectors_all.append(detector)
                orders_all.append(order)

                try:
                    if results_type == "FCCD":
                        FCCD_source, FCCD_err_up_source, FCCD_err_low_source = FCCDs_json[detector]["rounded"]["FCCD"], FCCDs_json[detector]["rounded"]["FCCD_err_total_up"], FCCDs_json[detector]["rounded"]["FCCD_err_total_low"] 
                    else:
                        if FCCDs_json[detector][source]["FCCD"]["Central"] == 0.0:
                            print(print("no result with ",source," for ", detector))
                            continue
                        FCCD_source, FCCD_err_up_source, FCCD_err_low_source = FCCDs_json[detector][source]["AV/Volume"]["Central"], FCCDs_json[detector][source]["AV/Volume"]["ErrPos"], FCCDs_json[detector][source]["AV/Volume"]["ErrNeg"]

                    FCCDs_source.append(FCCD_source)
                    FCCD_err_ups_source.append(FCCD_err_up_source)
                    FCCD_err_lows_source.append(FCCD_err_low_source)
                    detectors_source.append(detector)
                except:
                    print("no result with ",source," for ", detector)

            cc = colors_orders[order]
            if source == "Am241_HS1":
                cc=lighten_color(cc,1.2)

            ax.errorbar(detectors_source,FCCDs_source, yerr = [FCCD_err_lows_source, FCCD_err_ups_source], marker = markers_sources[source], color=cc, linestyle = '-')
            
    for order in colors_orders:
        color = colors_orders[order]
        ax.plot(np.NaN, np.NaN, c=color, label=f'Order #'+str(order))

    ax2 = ax.twinx()
    for source in markers_sources:
        marker = markers_sources[source]
        if source in source_list:
            ax2.plot(np.NaN, np.NaN, marker=marker,c='grey',label=source)
    ax2.get_yaxis().set_visible(False)
    ax.tick_params(axis='x', labelrotation=45)
    ax.set_xlabel('Detector', fontsize=11)
    ax.grid(linestyle='dashed', linewidth=0.5)
    plt.tight_layout()

    if results_type == "FCCD":
        ax.legend(loc='upper left', bbox_to_anchor=(0, 1), fontsize=10)
        ax2.legend(loc='upper left', bbox_to_anchor=(0, 0.77), fontsize=10)
        ax.set_ylabel('FCCD (mm)', fontsize=11)
        ax.set_ylim(0.5,1.5)
        ax.set_title("FCCDs", fontsize=12)
        plt.savefig(CodePath+"/../resultsAll/FCCDs.png", bbox_inches='tight') 
    else:
        ax.legend(loc='lower left', bbox_to_anchor=(0, 0.1), fontsize=10)
        ax2.legend(loc='lower left', bbox_to_anchor=(0, 0), fontsize=10)
        ax.set_ylabel('fAV', fontsize=11)
        ax.set_ylim(0.85,1.0)
        ax.set_title("fAV", fontsize=12)
        plt.savefig(CodePath+"/../resultsAll/fAVs.png", bbox_inches='tight') 

    plt.show()



def makeLaTeXTable(order_list, source_list):

    """
    Make LaTeX Table from all results
    """

    CodePath=os.path.dirname(os.path.realpath(__file__))

    #Get detector list
    detector_list = CodePath+"/../detector_list.json"
    with open(detector_list) as json_file:
        detector_list_data = json.load(json_file)

    #Get results
    results_path = CodePath+"/../resultsAll/AVvalues.json"
    with open(results_path) as json_file:
        FCCDs_json = json.load(json_file)
        
    detectors_all = []

    for source in source_list:

        FCCD_source_all, FCCD_source_err_up_all, FCCD_source_err_low_all = [], [], []
        FCCD_source_correrr_up_all, FCCD_source_correrr_low_all, FCCD_source_uncorrerr_up_all, FCCD_source_uncorrerr_low_all = [], [], [], []
        AV_source_all, AV_source_err_up_all, AV_source_err_low_all = [], [], []
        AV_source_correrr_up_all, AV_source_correrr_low_all, AV_source_uncorrerr_up_all, AV_source_uncorrerr_low_all = [], [], [], []
        fAV_source_all, fAV_source_err_up_all, fAV_source_err_low_all = [], [], []
        fAV_source_correrr_up_all, fAV_source_correrr_low_all, fAV_source_uncorrerr_up_all, fAV_source_uncorrerr_low_all = [], [], [], []

        for order in order_list:
        
            detectors = detector_list_data["order_"+str(order)]["detectors"]

            for detector in detectors:
                
                if source == "Ba133":
                    detectors_all.append(detector)

                try:
                    if FCCDs_json[detector][source]["FCCD"]["Central"] == 0.0:
                        print("no result for detector ", detector," and ", source)
                        FCCD, FCCD_err_up, FCCD_err_low = np.nan, np.nan, np.nan
                        FCCD_correrr_up, FCCD_correrr_low, FCCD_uncorrerr_up, FCCD_uncorrerr_low = np.nan, np.nan, np.nan, np.nan
                        AV, AV_err_up, AV_err_low = np.nan, np.nan, np.nan
                        AV_correrr_up, AV_correrr_low, AV_uncorrerr_up, AV_uncorrerr_low = np.nan, np.nan, np.nan, np.nan
                        fAV, fAV_err_up, fAV_err_low = np.nan, np.nan, np.nan
                        fAV_correrr_up, fAV_correrr_low, fAV_uncorrerr_up, fAV_uncorrerr_low = np.nan, np.nan, np.nan, np.nan
                    else:
                        FCCD, FCCD_err_up, FCCD_err_low = round(FCCDs_json[detector][source]["FCCD"]["Central"],2), round(FCCDs_json[detector][source]["FCCD"]["ErrPos"],2), round(FCCDs_json[detector][source]["FCCD"]["ErrNeg"],2)
                        FCCD_correrr_up, FCCD_correrr_low, FCCD_uncorrerr_up, FCCD_uncorrerr_low = round(FCCDs_json[detector][source]["FCCD"]["ErrCorrPos"],2), round(FCCDs_json[detector][source]["FCCD"]["ErrCorrNeg"],2), round(FCCDs_json[detector][source]["FCCD"]["ErrUncorrPos"],2), round(FCCDs_json[detector][source]["FCCD"]["ErrUncorrNeg"],2)
                        
                        AV, AV_err_up, AV_err_low = round(FCCDs_json[detector][source]["ActiveVolume"]["Central"],1), round(FCCDs_json[detector][source]["ActiveVolume"]["ErrPos"],1), round(FCCDs_json[detector][source]["ActiveVolume"]["ErrNeg"],1)
                        AV_correrr_up, AV_correrr_low, AV_uncorrerr_up, AV_uncorrerr_low = round(FCCDs_json[detector][source]["ActiveVolume"]["ErrCorrPos"],1), round(FCCDs_json[detector][source]["ActiveVolume"]["ErrCorrNeg"],1), round(FCCDs_json[detector][source]["ActiveVolume"]["ErrUncorrPos"],2), round(FCCDs_json[detector][source]["ActiveVolume"]["ErrUncorrNeg"],1)
                        
                        fAV, fAV_err_up, fAV_err_low = round(FCCDs_json[detector][source]["AV/Volume"]["Central"],3), round(FCCDs_json[detector][source]["AV/Volume"]["ErrPos"],3), round(FCCDs_json[detector][source]["AV/Volume"]["ErrNeg"],3)
                        fAV_correrr_up, fAV_correrr_low, fAV_uncorrerr_up, fAV_uncorrerr_low = round(FCCDs_json[detector][source]["AV/Volume"]["ErrCorrPos"],3), round(FCCDs_json[detector][source]["AV/Volume"]["ErrCorrNeg"],3), round(FCCDs_json[detector][source]["AV/Volume"]["ErrUncorrPos"],3), round(FCCDs_json[detector][source]["AV/Volume"]["ErrUncorrNeg"],3)
                except KeyError:
                    print("no result for detector ", detector," and ", source)
                    FCCD, FCCD_err_up, FCCD_err_low = np.nan, np.nan, np.nan
                    FCCD_correrr_up, FCCD_correrr_low, FCCD_uncorrerr_up, FCCD_uncorrerr_low = np.nan, np.nan, np.nan, np.nan
                    AV, AV_err_up, AV_err_low = np.nan, np.nan, np.nan
                    AV_correrr_up, AV_correrr_low, AV_uncorrerr_up, AV_uncorrerr_low = np.nan, np.nan, np.nan, np.nan
                    fAV, fAV_err_up, fAV_err_low = np.nan, np.nan, np.nan
                    fAV_correrr_up, fAV_correrr_low, fAV_uncorrerr_up, fAV_uncorrerr_low = np.nan, np.nan, np.nan, np.nan
                
                FCCD_source_all.append(FCCD)
                FCCD_source_err_up_all.append(FCCD_err_up)
                FCCD_source_err_low_all.append(FCCD_err_low)
                FCCD_source_correrr_up_all.append(FCCD_correrr_up)
                FCCD_source_correrr_low_all.append(FCCD_correrr_low)
                FCCD_source_uncorrerr_up_all.append(FCCD_uncorrerr_up)
                FCCD_source_uncorrerr_low_all.append(FCCD_uncorrerr_low)

                AV_source_all.append(AV)
                AV_source_err_up_all.append(AV_err_up)
                AV_source_err_low_all.append(AV_err_low)
                AV_source_correrr_up_all.append(AV_correrr_up)
                AV_source_correrr_low_all.append(AV_correrr_low)
                AV_source_uncorrerr_up_all.append(AV_uncorrerr_up)
                AV_source_uncorrerr_low_all.append(AV_uncorrerr_low)

                fAV_source_all.append(fAV)
                fAV_source_err_up_all.append(fAV_err_up)
                fAV_source_err_low_all.append(fAV_err_low)
                fAV_source_correrr_up_all.append(fAV_correrr_up)
                fAV_source_correrr_low_all.append(fAV_correrr_low)
                fAV_source_uncorrerr_up_all.append(fAV_uncorrerr_up)
                fAV_source_uncorrerr_low_all.append(fAV_uncorrerr_low)
    
        if source == "Ba133":
            FCCD_Ba_all, FCCD_Ba_err_up_all, FCCD_Ba_err_low_all = FCCD_source_all, FCCD_source_err_up_all, FCCD_source_err_low_all
            FCCD_Ba_correrr_up_all, FCCD_Ba_correrr_low_all, FCCD_Ba_uncorrerr_up_all, FCCD_Ba_uncorrerr_low_all = FCCD_source_correrr_up_all, FCCD_source_correrr_low_all, FCCD_source_uncorrerr_up_all, FCCD_source_uncorrerr_low_all
            
            AV_Ba_all, AV_Ba_err_up_all, AV_Ba_err_low_all = AV_source_all, AV_source_err_up_all, AV_source_err_low_all
            AV_Ba_correrr_up_all, AV_Ba_correrr_low_all, AV_Ba_uncorrerr_up_all, AV_Ba_uncorrerr_low_all = AV_source_correrr_up_all, AV_source_correrr_low_all, AV_source_uncorrerr_up_all, AV_source_uncorrerr_low_all

            fAV_Ba_all, fAV_Ba_err_up_all, fAV_Ba_err_low_all = fAV_source_all, fAV_source_err_up_all, fAV_source_err_low_all
            fAV_Ba_correrr_up_all, fAV_Ba_correrr_low_all, fAV_Ba_uncorrerr_up_all, fAV_Ba_uncorrerr_low_all = fAV_source_correrr_up_all, fAV_source_correrr_low_all, fAV_source_uncorrerr_up_all, fAV_source_uncorrerr_low_all
        else: 
            FCCD_Am_all, FCCD_Am_err_up_all, FCCD_Am_err_low_all = FCCD_source_all, FCCD_source_err_up_all, FCCD_source_err_low_all
            FCCD_Am_correrr_up_all, FCCD_Am_correrr_low_all, FCCD_Am_uncorrerr_up_all, FCCD_Am_uncorrerr_low_all = FCCD_source_correrr_up_all, FCCD_source_correrr_low_all, FCCD_source_uncorrerr_up_all, FCCD_source_uncorrerr_low_all
            
            AV_Am_all, AV_Am_err_up_all, AV_Am_err_low_all = AV_source_all, AV_source_err_up_all, AV_source_err_low_all
            AV_Am_correrr_up_all, AV_Am_correrr_low_all, AV_Am_uncorrerr_up_all, AV_Am_uncorrerr_low_all = AV_source_correrr_up_all, AV_source_correrr_low_all, AV_source_uncorrerr_up_all, AV_source_uncorrerr_low_all
            
            fAV_Am_all, fAV_Am_err_up_all, fAV_Am_err_low_all = fAV_source_all, fAV_source_err_up_all, fAV_source_err_low_all
            fAV_Am_correrr_up_all, fAV_Am_correrr_low_all, fAV_Am_uncorrerr_up_all, fAV_Am_uncorrerr_low_all = fAV_source_correrr_up_all, fAV_source_correrr_low_all, fAV_source_uncorrerr_up_all, fAV_source_uncorrerr_low_all

    #make strings of values plus errors
    FCCD_str_Ba_list, FCCD_str_Am_list = [], []
    AV_str_Ba_list, AV_str_Am_list = [], []
    fAV_str_Ba_list, fAV_str_Am_list = [], []

    for ind, detector in enumerate(detectors_all):
        #Ba results
        if str(FCCD_Ba_all[ind]) == 'nan':
            FCCD_str_Ba = r'-'
            AV_str_Ba = r'-'
            fAV_str_Ba = r'-'
        else:
            FCCD_str_Ba = str(FCCD_Ba_all[ind])+r'$^{+'+str(FCCD_Ba_correrr_up_all[ind])+r'+'+str(FCCD_Ba_uncorrerr_up_all[ind])+r'}_{-'+str(FCCD_Ba_correrr_low_all[ind])+r'-'+str(FCCD_Ba_uncorrerr_low_all[ind])+r'}$'
            AV_str_Ba = str(AV_Ba_all[ind])+r'$^{+'+str(AV_Ba_correrr_up_all[ind])+r'+'+str(AV_Ba_uncorrerr_up_all[ind])+r'}_{-'+str(AV_Ba_correrr_low_all[ind])+r'-'+str(AV_Ba_uncorrerr_low_all[ind])+r'}$'
            fAV_str_Ba = str(fAV_Ba_all[ind])+r'$^{+'+str(fAV_Ba_correrr_up_all[ind])+r'+'+str(fAV_Ba_uncorrerr_up_all[ind])+r'}_{-'+str(fAV_Ba_correrr_low_all[ind])+r'-'+str(fAV_Ba_uncorrerr_low_all[ind])+r'}$'
            
        FCCD_str_Ba_list.append(FCCD_str_Ba)
        AV_str_Ba_list.append(AV_str_Ba)
        fAV_str_Ba_list.append(fAV_str_Ba)
        #Am results
        if str(FCCD_Am_all[ind]) == 'nan':
            FCCD_str_Am = r'-'
            AV_str_Am = r'-'
            fAV_str_Am = r'-'
        else:
            FCCD_str_Am = str(FCCD_Am_all[ind])+r'$^{+'+str(FCCD_Am_correrr_up_all[ind])+r'+'+str(FCCD_Am_uncorrerr_up_all[ind])+r'}_{-'+str(FCCD_Am_correrr_low_all[ind])+r'-'+str(FCCD_Am_uncorrerr_low_all[ind])+r'}$'
            AV_str_Am = str(AV_Am_all[ind])+r'$^{+'+str(AV_Am_correrr_up_all[ind])+r'+'+str(AV_Am_uncorrerr_up_all[ind])+r'}_{-'+str(AV_Am_correrr_low_all[ind])+r'-'+str(AV_Am_uncorrerr_low_all[ind])+r'}$'
            fAV_str_Am = str(fAV_Am_all[ind])+r'$^{+'+str(fAV_Am_correrr_up_all[ind])+r'+'+str(fAV_Am_uncorrerr_up_all[ind])+r'}_{-'+str(fAV_Am_correrr_low_all[ind])+r'-'+str(fAV_Am_uncorrerr_low_all[ind])+r'}$'
        
        FCCD_str_Am_list.append(FCCD_str_Am)
        AV_str_Am_list.append(AV_str_Am)
        fAV_str_Am_list.append(fAV_str_Am)
        
    table =tabulate({"Detector": detectors_all,"FCCD ($^{133}$Ba) / mm": FCCD_str_Ba_list,"FCCD ($^{241}$Am) / mm": FCCD_str_Am_list,"AV ($^{133}$Ba) / mm$^3$": AV_str_Ba_list,"AV ($^{241}$Am) / mm$^3$": AV_str_Am_list,"fAV ($^{133}$Ba)": fAV_str_Ba_list,"fAV ($^{241}$Am)": fAV_str_Am_list}, headers="keys", tablefmt="latex_raw")
    with open(CodePath+"/../resultsAll/ResultsLaTeXTable.txt", "w") as f:
        print(table, file=f)


def calculateFCCDshift(order_list):
    """
    Calculate mean shift between sources across all detectors for a given order_list
    args:
    - order_list
    """
    CodePath=os.path.dirname(os.path.realpath(__file__))

    #Get detector list
    detector_list = CodePath+"/../detector_list.json"
    with open(detector_list) as json_file:
        detector_list_data = json.load(json_file)
    
    #Get results
    results_path = CodePath+"/../resultsAll/AVvalues.json"
    with open(results_path) as json_file:
        FCCDs_json = json.load(json_file)
    

    detectors_all = []

    source_list = ["Ba133", "Am241_HS1"]
    for source in source_list:

        FCCD_source_all, FCCD_source_err_up_all, FCCD_source_err_low_all = [], [], []
        FCCD_source_correrr_up_all, FCCD_source_correrr_low_all, FCCD_source_uncorrerr_up_all, FCCD_source_uncorrerr_low_all = [], [], [], []

        for order in order_list:
        
            detectors = detector_list_data["order_"+str(order)]["detectors"]

            for detector in detectors:
                
                if source == "Ba133":
                    detectors_all.append(detector)

                try:
                    if FCCDs_json[detector][source]["FCCD"]["Central"] == 0.0:
                        print("no result for detector ", detector," and ", source)
                        FCCD, FCCD_err_up, FCCD_err_low = np.nan, np.nan, np.nan
                        FCCD_correrr_up, FCCD_correrr_low, FCCD_uncorrerr_up, FCCD_uncorrerr_low = np.nan, np.nan, np.nan, np.nan
                    else:
                        FCCD, FCCD_err_up, FCCD_err_low = round(FCCDs_json[detector][source]["FCCD"]["Central"],2), round(FCCDs_json[detector][source]["FCCD"]["ErrPos"],2), round(FCCDs_json[detector][source]["FCCD"]["ErrNeg"],2)
                        FCCD_correrr_up, FCCD_correrr_low, FCCD_uncorrerr_up, FCCD_uncorrerr_low = round(FCCDs_json[detector][source]["FCCD"]["ErrCorrPos"],2), round(FCCDs_json[detector][source]["FCCD"]["ErrCorrNeg"],2), round(FCCDs_json[detector][source]["FCCD"]["ErrUncorrPos"],2), round(FCCDs_json[detector][source]["FCCD"]["ErrUncorrNeg"],2)
                except KeyError:
                    print("no result for detector ", detector," and ", source)
                    FCCD, FCCD_err_up, FCCD_err_low = np.nan, np.nan, np.nan
                    FCCD_correrr_up, FCCD_correrr_low, FCCD_uncorrerr_up, FCCD_uncorrerr_low = np.nan, np.nan, np.nan, np.nan

                FCCD_source_all.append(FCCD)
                FCCD_source_err_up_all.append(FCCD_err_up)
                FCCD_source_err_low_all.append(FCCD_err_low)
                FCCD_source_correrr_up_all.append(FCCD_correrr_up)
                FCCD_source_correrr_low_all.append(FCCD_correrr_low)
                FCCD_source_uncorrerr_up_all.append(FCCD_uncorrerr_up)
                FCCD_source_uncorrerr_low_all.append(FCCD_uncorrerr_low)


    
        if source == "Ba133":
            FCCD_Ba_all, FCCD_Ba_err_up_all, FCCD_Ba_err_low_all = FCCD_source_all, FCCD_source_err_up_all, FCCD_source_err_low_all
            FCCD_Ba_correrr_up_all, FCCD_Ba_correrr_low_all, FCCD_Ba_uncorrerr_up_all, FCCD_Ba_uncorrerr_low_all = FCCD_source_correrr_up_all, FCCD_source_correrr_low_all, FCCD_source_uncorrerr_up_all, FCCD_source_uncorrerr_low_all
        else: 
            FCCD_Am_all, FCCD_Am_err_up_all, FCCD_Am_err_low_all = FCCD_source_all, FCCD_source_err_up_all, FCCD_source_err_low_all
            FCCD_Am_correrr_up_all, FCCD_Am_correrr_low_all, FCCD_Am_uncorrerr_up_all, FCCD_Am_uncorrerr_low_all = FCCD_source_correrr_up_all, FCCD_source_correrr_low_all, FCCD_source_uncorrerr_up_all, FCCD_source_uncorrerr_low_all
    
    abs_shift_pct_list = []
    for i, detector in enumerate(detectors_all):
        if not np.isnan(FCCD_Ba_all[i]) and not np.isnan(FCCD_Am_all[i]):
            abs_shift_pct = np.abs(FCCD_Am_all[i] - FCCD_Ba_all[i])/FCCD_Ba_all[i]
            abs_shift_pct_list.append(abs_shift_pct)
    
    av_abs_shift_pct = np.mean(abs_shift_pct_list)
    print(av_abs_shift_pct)

    fig, ax = plt.subplots(figsize=(7,6))
    plt.gca().set_aspect('equal')
    plt.errorbar(FCCD_Am_all, FCCD_Ba_all, xerr = (FCCD_Am_err_low_all, FCCD_Am_err_up_all), yerr = (FCCD_Ba_err_low_all, FCCD_Ba_err_up_all), fmt='o', linewidth=1.5, ms=1, label="Total Error")
    plt.errorbar(FCCD_Am_all, FCCD_Ba_all, xerr = (FCCD_Am_correrr_low_all, FCCD_Am_correrr_up_all), yerr = (FCCD_Ba_correrr_low_all, FCCD_Ba_correrr_up_all), fmt='o', linewidth=1.5, ms=1, label="Correlated Error")
    plt.errorbar(FCCD_Am_all, FCCD_Ba_all, xerr = (FCCD_Am_uncorrerr_low_all, FCCD_Am_uncorrerr_up_all), yerr = (FCCD_Ba_uncorrerr_low_all, FCCD_Ba_uncorrerr_up_all), fmt='o', linewidth=1.5, ms=1, label="Uncorrelated Error")
    plt.errorbar(FCCD_Am_all, FCCD_Ba_all, fmt='x', linewidth=0.75, ms=4, color="green")
    plt.xlabel(r'$^{241}$Am FCCD / mm')
    plt.ylabel(r'$^{133}$Ba FCCD / mm')
    plt.xlim(0.5,1.5)
    plt.ylim(0.5,1.5)
    info = [f"Mean abs relative shift = {av_abs_shift_pct*100:.1f}$\%$"]
    plt.legend(title="\n".join(info))
    plt.tight_layout()
    ax.axline((0, 0), slope=1, linestyle="dashed", color="grey")
    plt.savefig(CodePath+"/../resultsAll/FCCDshift.png", bbox_inches='tight')
    plt.show()
    



