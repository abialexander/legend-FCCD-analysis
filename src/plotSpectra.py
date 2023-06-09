import os
import json
from matplotlib import gridspec
from src.calibration import *
from src.compareResults import lighten_color
from scipy.stats import poisson, norm
from matplotlib import patches

def computeDataMCratios(energy_bins_centres, counts_data, counts_MC, scaling_MC=None, residuals = False):
    "compute the ratio of data/MC for 2 histograms (shared energy bins, counts_data and counts_MC)"
    Data_MC_ratios = []
    Data_MC_ratios_err = []

    # for index, bin in enumerate(energy_bins[1:]):
    for index, bin in enumerate(energy_bins_centres):
        data = counts_data[index]
        MC = counts_MC[index] #This counts has already been scaled by weights
        if scaling_MC is not None:
            MC_unscaled = MC/scaling_MC #convert back to unscaled MC
        if MC == 0:
            ratio = np.nan
            error = np.nan
        else:
            try:
                if residuals == True:
                    ratio = (data-MC)/MC #return residuals instead of ratio
                else: 
                    ratio = data/MC
                try:
                    if scaling_MC is None: 
                        error = (data/MC)*np.sqrt(1/data + 1/MC)
                    else:
                        error = np.sqrt((data/(MC_unscaled**2*scaling_MC**2)) + (data**2/(MC_unscaled**3*scaling_MC**2)))
                except:
                    error = np.nan
            except:
                ratio = np.nan #if MC=0 and dividing by 0
        Data_MC_ratios.append(ratio)
        Data_MC_ratios_err.append(error)


    return Data_MC_ratios, Data_MC_ratios_err

def smallest_poisson_interval(cov, mu):

    res = [mu,mu]
    mode = math.floor(mu) #start from the mode=integer part of the mean
    l = mode
    u = mode
    prob = poisson.pmf(mode,mu)
    while (prob < cov):
        prob_u = poisson.pmf(u+1, mu)
        l_new=l-1 if l>0 else 0
        prob_l = poisson.pmf(l_new, mu)
        # we expand on the right if:
        #- the lower edge is already at zero
        #- the prob of the right point is higher than the left
        if (l == 0 or prob_u > prob_l):
            u += 1
            prob += prob_u
        #otherwhise we expand on the left
        elif (prob_u < prob_l):
            l-= 1
            prob += prob_l
        #if prob_u == prob_l we expand on both sides
        else:
            u += 1
            l -= 1
            prob += prob_u + prob_l
    l_n=0 if l==0 else l-0.5
    res = [l_n, u+0.5]
    return res

def draw_poisson_bands(mu, x_low, x_size, residuals = False):

    sig1 = smallest_poisson_interval(0.682, mu)
    sig2 = smallest_poisson_interval(0.954, mu)
    sig3 = smallest_poisson_interval(0.997, mu)

    if (residuals) :
        if (mu != 0):
            sig1[0] /= mu
            sig1[1] /= mu
            sig2[0] /= mu
            sig2[1] /= mu
            sig3[0] /= mu
            sig3[1] /= mu
        else:
            sig1[0] = sig1[1] = 1
            sig2[0] = sig2[1] = 1
            sig3[0] = sig3[1] = 1

    cent_b1 = (sig1[1] + sig1[0])/2
    cent_b2 = (sig2[1] + sig2[0])/2
    cent_b3 = (sig3[1] + sig3[0])/2

    xdw = x_low
    xup = x_low + x_size

    box_b1 = patches.Rectangle((xdw, sig1[0]), width=abs(xup-xdw),height=abs(sig1[1]-sig1[0]),color='yellowgreen')
    box_b2 = patches.Rectangle((xdw, sig2[0]), width=abs(xup-xdw),height=abs(sig2[1]-sig2[0]), color='gold')
    box_b3 = patches.Rectangle((xdw, sig3[0]), width=abs(xup-xdw),height=abs(sig3[1]-sig3[0]), color='orange')

    return box_b1, box_b2, box_b3


def plotSpectra(detector, source, MC_id, sim_path, FCCD, DLF, data_path, calibration, energy_filter, cuts, run):
    """
    Plot best fit FCCD MC spectra with data
    args: 
        - detector
        - source ("Ba133", "Am241_HS1")
        - MC_id ({detector}-ba_HS4-top-0r-78z_${smear}_${TL_model}_FCCD${FCCD}mm_DLF${DLF}_fracFCCDbore${frac_FCCDbore}
        - sim_path (path to AV processed sim, e.g. /lfs/l1/legend/users/aalexander/legend-g4simple-simulation/legend/simulations/${detector}/ba_HS4/top_0r_78z/hdf5/AV_processed/${MC_id}.hdf5
        - FCCD
        - DLF
        - data_path 
        - calibration (=path to data calibration coeffs)
        - energy_filter (cuspE_ctc)
        - cuts (True/False)
        - run (1,2,etc)
    """

    #initialise directories to save spectra
    dir=os.path.dirname(os.path.realpath(__file__))
    outputFolder = dir+"/../results/Spectra/"+detector+"/"+source+"/plots/"
    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)


    #==========================================================
    #LOAD ENERGY SPECTRA
    #==========================================================
    # GET DATA
    #load uncalibrated energy
    sigma_cuts = 4 #default
    energy_filter_data, failed_cuts = load_energy_data(data_path, energy_filter, cuts, run, sigma_cuts=sigma_cuts)
    #calibrate
    with open(calibration) as json_file:
        calibration_coefs = json.load(json_file)
    m = calibration_coefs[energy_filter]["calibration"][0]
    c = calibration_coefs[energy_filter]["calibration"][1]
    energy_data = energy_filter_data*m + c

    # GET MC
    df =  pd.read_hdf(sim_path, key="procdf")
    energy_MC = df['energy']

    #==========================================================
    #LOAD PEAK COUNTS FOR SCALING
    #==========================================================
    #Get peak counts from data
    PeakCounts_data = dir+"/../results/PeakCounts/"+detector+"/"+source+"/data/"+"PeakCounts_"+detector+"_"+energy_filter+"_run"+str(run)
    if cuts == False:
        PeakCounts_data = PeakCounts_data+"_nocuts.json"
    else:
        PeakCounts_data = PeakCounts_data+"_cuts.json"
    with open(PeakCounts_data) as json_file:
        PeakCounts = json.load(json_file)
    if source == "Ba133":
        C_scale_data = PeakCounts['counts'][2] #C_356
    else:
        print("No scaling") #NEED TO DO Am241 case
        sys.exit()
    
    PeakCounts_MC = dir+"/../results/PeakCounts/"+detector+"/"+source+"/MC/"+"PeakCounts_"+MC_id+".json"
    with open(PeakCounts_MC) as json_file:
        PeakCounts = json.load(json_file)
    if source == "Ba133":
        C_scale_MC = PeakCounts['counts'][2] #C_356
    else:
        print("No scaling") #NEED TO DO Am241 case
        sys.exit()

    #==========================================================
    #PLOT DATA + SCALED MC
    #==========================================================
    binwidth = 0.25 #keV
    xlo = 0
    xhi = 450 if source == "Ba133" else 120
    bins = np.arange(xlo,xhi,binwidth)
    bins_centres = (bins[:-1] + bins[1:])/2
    
    fig = plt.figure()
    gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1]) 
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1], sharex = ax0)

    linewidth = 0.35 if binwidth<0.2 else 0.5
    counts_data, bins, bars_data = ax0.hist(energy_data, bins=bins,  label = "Data", histtype = 'step', linewidth = linewidth)
    scaling_MC = C_scale_data/C_scale_MC #scaling of MC
    counts_MC, bins, bars = ax0.hist(energy_MC, bins = bins, weights=(scaling_MC)*np.ones_like(energy_MC), label = "MC: FCCD "+str(FCCD)+"mm (scaled)", histtype = 'step', linewidth = linewidth)

    #compute ratio of data/MC
    Data_MC_ratios, Data_MC_ratios_err = computeDataMCratios(bins_centres, counts_data, counts_MC, scaling_MC=scaling_MC, residuals=False)
    # ax1.errorbar(bins_centres, Data_MC_ratios, yerr=Data_MC_ratios_err, color="dimgrey", elinewidth = linewidth, fmt='o', ms = 1.0, mew = 1.0)
    # ax1.hlines(0, xlo, xhi, colors="gray", linestyles='dashed', linewidth=linewidth) 
    # ax1.fill_between(bins_centres, -3*np.array(Data_MC_ratios_err), 3*np.array(Data_MC_ratios_err), label=r'3$\sigma$', color=lighten_color("lightsalmon",0.75))
    # ax1.fill_between(bins_centres, -2*np.array(Data_MC_ratios_err), 2*np.array(Data_MC_ratios_err), label=r'2$\sigma$', color=lighten_color("lightyellow", 0.75))
    # ax1.fill_between(bins_centres, -1*np.array(Data_MC_ratios_err), Data_MC_ratios_err, label=r'1$\sigma$', color=lighten_color("lightgreen", 0.75))
    # ax1.plot(bins_centres, Data_MC_ratios, marker="o",color="black", linestyle='None', ms=0.5)
    # # ax1.set_ylabel(r'(Data-MC)/MC')
    # # ax1.set_ylim(-5,5)

    #NEW BANDS
    ax1.plot(bins_centres, Data_MC_ratios, marker="o",color="black", linestyle='None', ms=0.5)
    for c,b,l, in zip(counts_MC, bins, pgh.get_bin_widths(bins)):
        box1b, box2b, box3b = draw_poisson_bands(c,b,l,True)
        ax1.add_patch(box3b)
        ax1.add_patch(box2b)
        ax1.add_patch(box1b)
    ax1.set_yscale("log")
    ax1.set_ylabel("Data/MC")

    plt.xlabel("Energy [keV]")
    ax0.set_ylabel("Counts / "+str(binwidth)+" keV")
    ax0.set_yscale("log")
    ax0.legend(loc = "lower left", fontsize=9)
    ax0.set_title(detector+", "+source)
    ax1.set_xlim(xlo,xhi)
    ax0.set_xlim(xlo,xhi)
    
    # plt.subplots_adjust(hspace=.0)

    if cuts == False:
        plt.savefig(outputFolder+"DataMC_"+MC_id+"_"+energy_filter+"_run"+str(run)+"_nocuts.png")
    else:
        plt.savefig(outputFolder+"DataMC_"+MC_id+"_"+energy_filter+"_run"+str(run)+"_cuts.png")

    plt.show()
    print("done")


def plotSpectra_nosmear(detector, source, MC_id, sim_path):
    """
    Plot smear and no smear MC spectra
    args: 
        - detector
        - source ("Ba133", "Am241_HS1")
        - MC_id ({detector}-ba_HS4-top-0r-78z_${smear}_${TL_model}_FCCD${FCCD}mm_DLF${DLF}_fracFCCDbore${frac_FCCDbore}
        - sim_path (path to AV processed sim, e.g. /lfs/l1/legend/users/aalexander/legend-g4simple-simulation/legend/simulations/${detector}/ba_HS4/top_0r_78z/hdf5/AV_processed/${MC_id}.hdf5
    """

    #initialise directories to save spectra
    dir=os.path.dirname(os.path.realpath(__file__))
    outputFolder = dir+"/../results/Spectra/"+detector+"/"+source+"/plots/"
    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)


    #==========================================================
    #LOAD ENERGY SPECTRA
    #==========================================================

    #GET MC smear
    df =  pd.read_hdf(sim_path, key="procdf")
    energy_MC = df['energy']

    #GET MC no smear
    sim_path_nosmear = sim_path.replace("_g_", "_nosmear_")
    print(sim_path_nosmear)
    df_nosmear =  pd.read_hdf(sim_path_nosmear, key="procdf")
    energy_MC_nosmear = df_nosmear['energy']*1000

    #==========================================================
    #PLOT MC smear and no smear
    #==========================================================
    binwidth = 0.25 #keV
    xlo = 0
    xhi = 450 if source == "Ba133" else 120
    bins = np.arange(xlo,xhi,binwidth)
    bins_centres = (bins[:-1] + bins[1:])/2
     
    fig = plt.figure()

    linewidth = 1
    counts_MC_nosmear, bins, bars = plt.hist(energy_MC_nosmear, bins = bins, label = "no smear", histtype = 'step', linewidth = linewidth)
    counts_MC, bins, bars = plt.hist(energy_MC, bins = bins, label = "smear", histtype = 'step', linewidth = linewidth)
    plt.xlabel("Energy [keV]")
    plt.ylabel("Counts / "+str(binwidth)+" keV")
    plt.yscale("log")
    plt.legend(loc = "lower left")
    plt.title(detector+", "+source)
    plt.xlim(xlo,xhi)

    plt.savefig(outputFolder+MC_id+"_smearcomparison.png")

    plt.show()
    print("done")







    