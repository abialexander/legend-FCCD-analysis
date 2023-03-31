import json
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mc
import numpy as np
import colorsys

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
    

    with open(CodePath+"/../FCCDvalues_"+source+".json", "w") as outfile:
        json.dump(dict_all, outfile, indent=4)
    
    return dict_all


def plotFCCDs(order_list,source_list):
    """
    Plot all FCCDs for a given order_list and source_list
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

            json_path = CodePath+"/../FCCDvalues_"+source+".json"
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
                    FCCD_source, FCCD_err_up_source, FCCD_err_low_source = FCCDs_json[detector]["rounded"]["FCCD"], FCCDs_json[detector]["rounded"]["FCCD_err_total_up"], FCCDs_json[detector]["rounded"]["FCCD_err_total_low"] 
                    FCCDs_source.append(FCCD_source)
                    FCCD_err_ups_source.append(FCCD_err_up_source)
                    FCCD_err_lows_source.append(FCCD_err_low_source)
                    detectors_source.append(detector)
                except:
                    print("no FCCD result with ",source," for ", detector)

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

    ax.legend(loc='upper left', bbox_to_anchor=(0, 1), fontsize=10)
    ax2.legend(loc='upper left', bbox_to_anchor=(0, 0.77), fontsize=10)

    ax.tick_params(axis='x', labelrotation=45)
    ax.set_xlabel('Detector', fontsize=11)
    ax.set_ylabel('FCCD (mm)', fontsize=11)
    ax.grid(linestyle='dashed', linewidth=0.5)
    plt.tight_layout()
    ax.set_ylim(0.5,1.5)
    ax.set_title("FCCDs", fontsize=12)
    plt.savefig(CodePath+"/../FCCDs.png", bbox_inches='tight') 
    plt.show()

            




