from src.calibration import *
from src.compareResults import *

def main():

    #====================================================
    # Collate FCCDs into JSON file
    #====================================================
    order_list = [2,4] #List of orders to process
    source = "Ba133" #"Ba133", "Am241_HS1" or "Am241_HS6"
    FCCDs_Ba133 = collateFCCDs(order_list, source)

    #====================================================
    # Plot FCCDs
    #====================================================
    order_list = [2,4]
    source_list = ["Ba133"]
    plotFCCDs(order_list, source_list)
     


if __name__ == "__main__":
    main()