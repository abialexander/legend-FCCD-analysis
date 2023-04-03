from src.calibration import *
from src.GammaLineCounting import *
from src.FCCD import *
from src.plotSpectra import *

def main():

    #-----------------------------------------------------
    # Get Relevant paths
    #-----------------------------------------------------
    CodePath=os.path.dirname(os.path.realpath(__file__))
    sim_folder_Ba133 = "/lfs/l1/legend/users/aalexander/legend-g4simple-simulation/simulations/" #Abis sims
    sim_folder_Am241_HS1 = "/lfs/l1/legend/detector_char/enr/hades/simulations/legend-g4simple-simulation/simulations/" #Vale sims
    sim_folder_Am241_HS6 = "" #TO DO
    data_folder_ICPC = "/lfs/l1/legend/legend-prodenv/prod-usr/ggmarsh-test-v03/gen/"
    data_folder_BEGe = "/lfs/l1/legend/legend-prodenv/prod-usr/ggmarsh-full_dl-v01/gen/" #gerda BEGe batch 1 and 2, order 0 and 1
   
    #====================================================
    # EDIT PROCESSING CONFIG BELOW
    #====================================================
    order_list = [2,4,5,7,8,9] #List of orders to process
    source = "Ba133" #"Ba133", "Am241_HS1" or "Am241_HS6"
    energy_filter="cuspEmax_ctc"
    cuts=True
    #-----------------------------------------------------
    Calibrate_Data = True  #Pre-reqs: needs dsp pygama data
    Gamma_line_count_data = False #Pre-reqs: needs calibration
    Gamma_line_count_MC = False #Pre-reqs: needs AV post processed MC for range of FCCDs
    Calculate_FCCD = False #Pre-reqs: needs gammaline counts for data and MC
    Gamma_line_count_MC_bestfitFCCD = False #Pre-reqs: needs AV postprocessed MC for best fit FCCD
    PlotSpectra = False #Pre-reqs: needs all above stages
    #====================================================

    if source == "Ba133":
        source_data = "ba_HS4" #convert source name into correct nomenclature
        sim_folder = sim_folder_Ba133
    elif source == "Am241_HS1":
        source_data = "am_HS1"
        sim_folder = sim_folder_Am241_HS1
    elif source == "Am241_HS6":
        source_data = "am_HS6"
        sim_folder = sim_folder_Am241_HS6

    #Get detector list
    detector_list = CodePath+"/detector_list.json"
    with open(detector_list) as json_file:
        detector_list_data = json.load(json_file)

    #Iterate through detectors in each order
    for order in order_list:
        detectors = detector_list_data["order_"+str(order)]["detectors"]
        detector_type = detector_list_data["order_"+str(order)]["detector_type"]
        runs = detector_list_data["order_"+str(order)]["runs"][source]
        MC_source_positions = detector_list_data["order_"+str(order)]["MC_source_positions"][source]

        for ind, detector in enumerate(detectors):

            if detector != "V04549A": ##or detector != "V08682B" or detector != "V09724A":
            # # if detector == "V07646A" or detector == "V07302A":
                continue
            print("")
            print("detector: ", detector)

            # print(runs)
            run = runs[ind]
            MC_source_position = MC_source_positions[ind] #=["top","0r","78z"]
            MC_source_pos_underscore = MC_source_position[0]+"_"+ MC_source_position[1]+"_"+MC_source_position[2] #="top_0r_78z"
            MC_source_pos_hyphon = MC_source_position[0]+"-"+ MC_source_position[1]+"-"+MC_source_position[2] #="top-0r-78z"

            #========Calibration - DATA==========
            if Calibrate_Data == True:

                detector_name = "I"+detector[1:] if order == 2 else detector  
                data_path = data_folder_ICPC+detector_name+"/tier2/"+source_data+"_top_dlt/" if detector_type == "ICPC" else data_folder_BEGe+detector_name+"/tier2/"+source_data+"_top_dlt/"
                perform_calibration(detector, source, data_path, energy_filter, cuts, run)

            #========Gamma line count - DATA==========
            if Gamma_line_count_data == True:
                spectra_type = "data"

                detector_name = "I"+detector[1:] if order == 2 else detector 
                data_path = data_folder_ICPC+detector_name+"/tier2/"+source_data+"_top_dlt/" if detector_type == "ICPC" else data_folder_BEGe+detector_name+"/tier2/"+source_data+"_top_dlt/"
                if cuts == False:
                    calibration = CodePath+"/results/data_calibration/"+detector+"/"+source+"/calibration_run"+str(run)+"_nocuts.json"
                else:
                    calibration = CodePath+"/results/data_calibration/"+detector+"/"+source+"/calibration_run"+str(run)+"_cuts.json"

                perform_gammaLineCounting(detector, source, spectra_type, data_path=data_path, calibration=calibration, energy_filter=energy_filter, cuts=cut, run=run)

            #========Gamma line count - MC==========
            if Gamma_line_count_MC == True:
                spectra_type = "MC"
                #normal paramaters:
                DLF_list=[1.0]
                smear="g"
                frac_FCCDbore=0.5
                TL_model="notl"
                # FCCD_list=[0.0] #ICPCs
                if source == "Ba133":
                    FCCD_list=[0.0,0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 3.0] #ICPCs
                    # FCCD_list=[0.0,0.25, 0.5, 0.75, 1.0, 1.25, 1.5,1.75,2.0, 3.0] #BEGes
                elif source == "Am241_HS1":
                    FCCD_list=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
                for FCCD in FCCD_list:
                    for DLF in DLF_list:
                        print("FCCD:", FCCD, ", DLF: ", DLF)
                        MC_id=detector+"-"+source_data+"-"+MC_source_pos_hyphon+"_"+smear+"_"+TL_model+"_FCCD"+str(FCCD)+"mm_DLF"+str(DLF)+"_fracFCCDbore"+str(frac_FCCDbore)
                        sim_path=sim_folder+detector+"/"+source_data+"/"+MC_source_pos_underscore+"/hdf5/AV_processed/"+MC_id+".hdf5"
                        perform_gammaLineCounting(detector, source, spectra_type,sim_path=sim_path, MC_id=MC_id)
                        print("")
            
            #========Calculate FCCD==========
            if Calculate_FCCD == True:

                MC_id=detector+"-"+source_data+"-"+MC_source_pos_hyphon
                smear="g"
                TL_model="notl"
                frac_FCCDbore=0.5
                calculateFCCD(detector, source, MC_id, smear, TL_model, frac_FCCDbore, energy_filter, cuts, run)

            #========Gamma line count -best fit MC==========
            if Gamma_line_count_MC_bestfitFCCD == True:
                spectra_type = "MC"
                #normal paramaters:
                DLF=1.0
                smear="g"
                frac_FCCDbore=0.5
                TL_model="notl"
                #get best fit FCCD
                if cuts == False:
                    with open(CodePath+"/results/FCCD/"+detector+"/"+source+"/FCCD_"+detector+"-"+source_data+"-"+MC_source_pos_hyphon+"_"+smear+"_"+TL_model+"_fracFCCDbore"+str(frac_FCCDbore)+"_"+energy_filter+"_run"+str(run)+"_nocuts.json") as json_file:
                        FCCD_data = json.load(json_file)
                else: #4sigma cuts default
                    with open(CodePath+"/results/FCCD/"+detector+"/"+source+"/FCCD_"+detector+"-"+source_data+"-"+MC_source_pos_hyphon+"_"+smear+"_"+TL_model+"_fracFCCDbore"+str(frac_FCCDbore)+"_"+energy_filter+"_run"+str(run)+"_cuts.json") as json_file:
                        FCCD_data = json.load(json_file)
                FCCD = round(FCCD_data["FCCD"],2)
                TL_model="l"
                MC_id = detector+"-"+source_data+"-"+MC_source_pos_hyphon+"_"+smear+"_"+TL_model+"_FCCD"+str(FCCD)+"mm_DLF"+str(DLF)+"_fracFCCDbore"+str(frac_FCCDbore)
                sim_path=sim_folder+detector+"/"+source_data+"/"+MC_source_pos_underscore+"/hdf5/AV_processed/"+MC_id+".hdf5"
                perform_gammaLineCounting(detector, source, spectra_type,sim_path=sim_path, MC_id=MC_id)
            
            #========Plot Spectra==========
            if PlotSpectra == True:
                
                #data args
                detector_name = "I"+detector[1:] if order == 2 else detector 
                data_path = data_folder_ICPC+detector_name+"/tier2/"+source_data+"_top_dlt/" if detector_type == "ICPC" else data_folder_BEGe+detector_name+"/tier2/"+source_data+"_top_dlt/"
                if cuts == False:
                    calibration = CodePath+"/results/data_calibration/"+detector+"/"+source+"/calibration_run"+str(run)+"_nocuts.json"
                else:
                    calibration = CodePath+"/results/data_calibration/"+detector+"/"+source+"/calibration_run"+str(run)+"_cuts.json"

                #MC args
                DLF=1.0
                smear="g"
                frac_FCCDbore=0.5
                TL_model="notl"
                if cuts == False:
                    with open(CodePath+"/results/FCCD/"+detector+"/"+source+"/FCCD_"+detector+"-"+source_data+"-"+MC_source_pos_hyphon+"_"+smear+"_"+TL_model+"_fracFCCDbore"+str(frac_FCCDbore)+"_"+energy_filter+"_run"+str(run)+"_nocuts.json") as json_file:
                        FCCD_data = json.load(json_file)
                else: #4sigma cuts default
                    with open(CodePath+"/results/FCCD/"+detector+"/"+source+"/FCCD_"+detector+"-"+source_data+"-"+MC_source_pos_hyphon+"_"+smear+"_"+TL_model+"_fracFCCDbore"+str(frac_FCCDbore)+"_"+energy_filter+"_run"+str(run)+"_cuts.json") as json_file:
                        FCCD_data = json.load(json_file)
                FCCD = round(FCCD_data["FCCD"],2)
                TL_model="l"
                MC_id = detector+"-"+source_data+"-"+MC_source_pos_hyphon+"_"+smear+"_"+TL_model+"_FCCD"+str(FCCD)+"mm_DLF"+str(DLF)+"_fracFCCDbore"+str(frac_FCCDbore)
                sim_path=sim_folder+detector+"/"+source_data+"/"+MC_source_pos_underscore+"/hdf5/AV_processed/"+MC_id+".hdf5"

                plotSpectra(detector, source, MC_id, sim_path, FCCD, DLF, data_path, calibration, energy_filter, cuts, run)


if __name__ == "__main__":
    main()