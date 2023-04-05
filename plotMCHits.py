from src.rawMCHits import *
import json

def main():

    #====================
    detector = "V04549A"
    source = "Ba133"
    #====================
    CodePath=os.path.dirname(os.path.realpath(__file__))
    sim_folder_Ba133 = "/lfs/l1/legend/users/aalexander/legend-g4simple-simulation/simulations/" #Abis sims
    sim_folder_Am241_HS1 = "/lfs/l1/legend/detector_char/enr/hades/simulations/legend-g4simple-simulation/simulations/" #Vale sims
    #Get detector list
    detector_list = CodePath+"/detector_list.json"
    with open(detector_list) as json_file:
        detector_list_data = json.load(json_file)
    #====================
    if source == "Ba133":
        source_data = "ba_HS4" #convert source name into correct nomenclature
        sim_folder = sim_folder_Ba133
    elif source == "Am241_HS1":
        source_data = "am_HS1"
        sim_folder = sim_folder_Am241_HS1
    order = detector[2]
    detectors = detector_list_data["order_"+str(order)]["detectors"]
    runs = detector_list_data["order_"+str(order)]["runs"][source]
    run = runs[detectors.index(detector)]
    MC_source_positions = detector_list_data["order_"+str(order)]["MC_source_positions"][source]
    MC_source_position = MC_source_positions[detectors.index(detector)]
    MC_source_pos_underscore = MC_source_position[0]+"_"+ MC_source_position[1]+"_"+MC_source_position[2] #="top_0r_78z"
    MC_source_pos_hyphon = MC_source_position[0]+"-"+ MC_source_position[1]+"-"+MC_source_position[2] #="top-0r-78z"
    MC_id=detector+"-"+source_data+"-"+MC_source_pos_hyphon
    MC_raw_path=sim_folder+detector+"/"+source_data+"/"+MC_source_pos_underscore+"/hdf5/"

    plotMCHits(detector, source, MC_raw_path, MC_id,reopen_saved=True)

if __name__ == "__main__":
    main()