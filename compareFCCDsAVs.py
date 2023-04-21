from src.calibration import *
from src.compareResults import *
from src.ActiveVolume import *

def main():

    #====================================================
    # Collate FCCDs into JSON file
    #====================================================
    # order_list = [2,4,5,7,8,9] #List of orders to process
    # source = "Ba133" #"Ba133", "Am241_HS1" or "Am241_HS6"
    # FCCDs_Ba133 = collateFCCDs(order_list, source)

    #====================================================
    # Plot FCCDs
    #====================================================
    # order_list = [2,4,5,7,8,9]
    # source_list = ["Ba133", "Am241_HS1"]
    # plotFCCDs(order_list, source_list)

    #====================================================
    # Make LaTeX Table Code
    #====================================================
    # order_list = [2,4,5,7,8,9]
    # source_list = ["Ba133", "Am241_HS1"]
    # makeLaTeXTable(order_list, source_list)

    #====================================================
    # Convert all FCCDs -> AVs
    #====================================================
    order_list = [2,4,5,7,8,9]
    metadata_folder = "/lfs/l1/legend/detector_char/enr/hades/simulations/legend-g4simple-simulation/tools/legend-metadata/hardware/detectors/"
    FCCDtoAV(order_list, metadata_folder)
    

if __name__ == "__main__":
    main()