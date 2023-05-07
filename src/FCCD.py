import json
import os
import numpy as np
from iminuit import Minuit
from iminuit.cost import LeastSquares
from scipy import optimize
from scipy import stats
import matplotlib.pyplot as plt

def error_Ba133_MC(O_Ba133, O_Ba133_err):
    "Returns the total, correlated and uncorrelated % errors on count ratio observable for the MC."

    #MC Systematics
    #values from Bjoern's thesis - Barium source, all percentages
    gamma_line=0.69 
    geant4=2.
    source_thickness=0.02
    source_material=0.01
    endcap_thickness=0.28
    detector_cup_thickness=0.07
    detector_cup_material=0.03

    #MC statistical
    MC_statistics = O_Ba133_err/O_Ba133*100

    #Total error: sum in quadrature of all contributions
    tot_error=np.sqrt(gamma_line**2+geant4**2+source_thickness**2+source_material**2+endcap_thickness**2+detector_cup_thickness**2+detector_cup_material**2+MC_statistics**2)

    correlated = [gamma_line, geant4, source_thickness, source_material, endcap_thickness, detector_cup_thickness, detector_cup_material]
    uncorrelated = [MC_statistics]

    #correlated error
    corr_error = np.sqrt(gamma_line**2+geant4**2+source_thickness**2+source_material**2+endcap_thickness**2+detector_cup_thickness**2+detector_cup_material**2)

    #uncorrelated error
    uncorr_error = MC_statistics
    
    return tot_error, corr_error, uncorr_error

def error_Am241_MC(O_Am241, O_Am241_err):
    #TO DO
    return 0

def exponential_decay(x, a, b ,c):
    f = a*np.exp(-b*x) + c
    return f

def invert_exponential(x,a,b,c):
    #for f=a*exp(-b*x) +c
    # x = the O_ba133 value
    return (1/b)*np.log(a/(x-c))

def iminuit_LS_fit(x,y,yerr,fit_func, guess, bounds=None,return_chi_sq=False):
    "perform least squares iminuit fit, requires yerr"
    least_squares = LeastSquares(x, y, yerr, fit_func)
    m = Minuit(least_squares, *guess) 
    if bounds is not None:
        m.limits = bounds
    m.migrad()  
    m.hesse()  
    coeff, coeff_err = m.values, m.errors
    if return_chi_sq == False:
        return coeff, coeff_err, m
    else:
        chi_sq = m.fval
        dof =  len(x) - m.nfit
        return coeff, coeff_err, m, chi_sq, dof

def scipy_LS_fit(x,y,fit_func,guess,bounds=None, yerr=None):
    "perform least squares scipy fit, does not require yerr"
    if bounds == None:
        coeff, coeff_cov = optimize.curve_fit(fit_func, x, y, p0=guess, maxfev = 10**7, method ="trf", sigma = yerr, absolute_sigma=False)
    else:
        coeff, coeff_cov = optimize.curve_fit(fit_func, x, y, p0=guess, maxfev = 10**7, method ="trf", sigma = yerr, absolute_sigma=False, bounds=bounds)
    coeff_err = np.sqrt(coeff_cov.diagonal())
    return coeff, coeff_err

    
def calculateFCCD(detector, source, MC_id, smear, TL_model, frac_FCCDbore, energy_filter, cuts, run, plot_all_error_bands = False):
    """
    Calculate FCCD of data
    args: 
        - detector
        - source ("Ba133", "Am241_HS1")
        - MC_id ({detector}-ba_HS4-top-0r-78z_${smear}_${TL_model}_FCCD${FCCD}mm_DLF${DLF}_fracFCCDbore${frac_FCCDbore}
        - smear (resolution smearing, e.g. gaussian: "g")
        - TL_model ("notl")
        - frac_FCCDbore (fractional thickness of FCCD around borehole, e.g. 0.5)
        - energy_filter (cuspE_ctc)
        - cuts (True/False)
        - run (1,2,etc)
        - plot_all_error_bands = True if you want a plot highlighting all the decomposed errors and how they propagate
    """

    #initialise directories for detectors to save 
    dir = os.path.dirname(os.path.realpath(__file__))
    outputFolder = dir+"/../results/FCCD/"+detector+"/"+source+"/plots/"
    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)

    if source == "Ba133":
        FCCD_list=[0.0,0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 3.0] #ICPCs
        # FCCD_list=[0.0,0.25, 0.5, 0.75, 1.0, 1.25, 1.5,1.75,2.0, 3.0] #BEGes
        if detector == "V07646A":
            # FCCD_list = [0.0,0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]
            FCCD_list = [0.0,0.1,0.2,0.3,0.4,0.5, 0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5]
    elif source == "Am241_HS1":
        FCCD_list=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
    
    countRatio_list = []
    countRatio_tot_err_pct_list = [] 
    countRatio_corr_err_pct_list = [] #corr is actually all systematic errors
    countRatio_uncorr_err_pct_list = [] #uncorr is actually only statistical error

    #=============================================
    # Get count ratio for simulations
    #=============================================
    DLF = 1.0 #considering 0 TL
    for FCCD in FCCD_list:
        with open(dir+"/../results/PeakCounts/"+detector+"/"+source+"/MC/PeakCounts_"+MC_id+"_"+smear+"_"+TL_model+"_FCCD"+str(FCCD)+'mm_DLF'+str(DLF)+'_fracFCCDbore'+str(frac_FCCDbore)+'.json') as json_file:
            PeakCounts = json.load(json_file)
            if source == "Ba133":
                C_356 = PeakCounts['counts'][2]
                C_79 = PeakCounts['counts'][0]
                C_81 = PeakCounts['counts'][1]
            elif source == "Am241_HS1":
                C_60 = PeakCounts['counts'][2]
                C_99 = PeakCounts['counts'][0]
                C_103 = PeakCounts['counts'][1]
            countRatio = PeakCounts["countRatio"]
            countRatio_err = PeakCounts["countRatio_err"] #stat fit error on MC
            countRatio_list.append(countRatio)

            #get errors:
            error_func = error_Ba133_MC if source == "Ba133" else error_Am241_MC
            countRatio_tot_err_pct, countRatio_corr_err_pct, countRatio_uncorr_err_pct = error_func(countRatio, countRatio_err) #stat and syst error on MC
            countRatio_tot_err_pct_list.append(countRatio_tot_err_pct)
            countRatio_corr_err_pct_list.append(countRatio_corr_err_pct)
            countRatio_uncorr_err_pct_list.append(countRatio_uncorr_err_pct)
    
    #=============================================
    # Get count ratio for data
    #=============================================
    if cuts == False:
        with open(dir+"/../results/PeakCounts/"+detector+"/"+source+"/data/PeakCounts_"+detector+"_"+energy_filter+"_run"+str(run)+"_nocuts.json") as json_file:
            PeakCounts = json.load(json_file)
    else:
        cuts_sigma = 4 #default =4, change by hand here if interested
        print("data cuts sigma: ", str(cuts_sigma))
        if cuts_sigma == 4:
            with open(dir+"/../results/PeakCounts/"+detector+"/"+source+"/data/PeakCounts_"+detector+"_"+energy_filter+"_run"+str(run)+"_cuts.json") as json_file:
                PeakCounts = json.load(json_file)
        else:
            with open(dir+"/../results/PeakCounts/"+detector+"/"+source+"/data/PeakCounts_"+detector+"_"+energy_filter+"_run"+str(run)+"_cuts"+str(cuts_sigma)+"sigma.json") as json_file:
                PeakCounts = json.load(json_file)

    if source == "Ba133":
        C_356 = PeakCounts['counts'][2]
        C_79 = PeakCounts['counts'][0]
        C_81 = PeakCounts['counts'][1]
    elif source == "Am241_HS1":
        C_60 = PeakCounts['counts'][2]
        C_99 = PeakCounts['counts'][0]
        C_103 = PeakCounts['counts'][1]
    countRatio_data = PeakCounts["countRatio"]
    countRatio_data_err = PeakCounts["countRatio_err"] #stat error on data

    #========= PLOTTING ===========
    plot_colors = {"MC":"black", "MC_fit": "black", "data": "red", "data_err_stat": "orange", "MC_err_stat": "green", "MC_err_syst": "pink", "MC_err_total": "green", "FCCD": "red", "FCCD_err_total": "blue", "FCCD_err_MCstatsyst": "red", "FCCD_err_MCstat": "purple", "FCCD_err_MCsyst": "pink", "FCCD_err_datastat": "green", "FCCD_err_statMCstatdata": "grey"}
    linewidth_err = 0.75
    linewidth_main = 1.0
    
    #plot and fit exp decay
    print("fitting exp decay")
    xdata, ydata = np.array(FCCD_list), np.array(countRatio_list)
    yerr = countRatio_tot_err_pct_list*ydata/100 #absolute total error
    aguess, bguess, cguess = max(ydata), 1,  min(ydata)
    p_guess = [aguess,bguess,cguess]
    coeff, coeff_err, m, chi_sq, dof = iminuit_LS_fit(xdata,ydata,yerr,exponential_decay, p_guess,return_chi_sq=True)
    print("valid fit: ", m.valid)
    print("r chi sq: ", chi_sq/dof)
    a,b,c = coeff[0], coeff[1], coeff[2]
    a_err,b_err,c_err = coeff_err[0], coeff_err[1], coeff_err[2]

    fig, ax = plt.subplots()
    plt.errorbar(xdata, ydata, xerr=0, yerr =yerr, label = "MC", color= plot_colors["MC"], fmt='x', ms = 3.0, mew = 3.0) #elinewidth = 1.0
    xfit = np.linspace(min(xdata), max(xdata), 1000)
    yfit = exponential_decay(xfit,*coeff)
    plt.plot(xfit, yfit, color=plot_colors["MC_fit"], linewidth=linewidth_main)

    #=====fit exp decay of error bars========

    # MC stat and syst
    print("fitting exp decay of total MC countRatio errors")
    y_uplim = ydata+yerr
    p_guess_up = [max(y_uplim), 1, min(y_uplim)]
    coeff_up, coeff_up_err = scipy_LS_fit(xdata,y_uplim,exponential_decay,p_guess_up)
    yfit_up = exponential_decay(xfit,*coeff_up)
    plt.plot(xfit, yfit_up, color=plot_colors["MC_err_total"], linestyle='dashed', linewidth=linewidth_main, label="MC total err (stat/uncorr + syst/corr)")
    y_lowlim = ydata-yerr
    p_guess_low = [max(y_lowlim), 1, min(y_lowlim)]
    coeff_low, coeff_low_err = scipy_LS_fit(xdata,y_lowlim,exponential_decay,p_guess_low)
    yfit_low = exponential_decay(xfit,*coeff_low)
    plt.plot(xfit, yfit_low, color=plot_colors["MC_err_total"], linestyle='dashed', linewidth=linewidth_main)
    a_up, b_up, c_up = coeff_up[0], coeff_up[1], coeff_up[2]
    a_low, b_low, c_low = coeff_low[0], coeff_low[1], coeff_low[2] 

    # MC corr
    print("fitting exp decay of MC stat") 
    yerr_corr = np.array(countRatio_corr_err_pct_list)*ydata/100
    y_uplim_corr = ydata+yerr_corr
    coeff_up_corr, coeff_up_corr_err = scipy_LS_fit(xdata,y_uplim_corr,exponential_decay,p_guess_up)
    yfit_up_corr = exponential_decay(xfit,*coeff_up_corr)
    if plot_all_error_bands == True:
        plt.plot(xfit, yfit_up_corr, color=plot_colors["FCCD_err_MCsyst"], linestyle='dashed', linewidth=1, label="MC stat/uncorr err")
    
    y_lowlim_corr = ydata-yerr_corr
    coeff_low_corr, coeff_low_corr_err = scipy_LS_fit(xdata,y_lowlim_corr,exponential_decay,p_guess_low)
    yfit_low_corr = exponential_decay(xfit,*coeff_low_corr)
    if plot_all_error_bands == True:
        plt.plot(xfit, yfit_low_corr, color=plot_colors["FCCD_err_MCsyst"],linestyle='dashed', linewidth=1)
    a_up_corr, b_up_corr, c_up_corr = coeff_up_corr[0], coeff_up_corr[1], coeff_up_corr[2]
    a_low_corr, b_low_corr, c_low_corr = coeff_low_corr[0], coeff_low_corr[1], coeff_low_corr[2] 

    # MC uncorr
    print("fitting exp decay of MC uncorr err")
    yerr_uncorr = countRatio_uncorr_err_pct_list*ydata/100
    y_uplim_uncorr = ydata+yerr_uncorr
    coeff_up_uncorr, coeff_up_uncorr_err = scipy_LS_fit(xdata,y_uplim_uncorr,exponential_decay,p_guess_up)
    yfit_up_uncorr = exponential_decay(xfit,*coeff_up_uncorr)
    if plot_all_error_bands == True:
        plt.plot(xfit, yfit_up_uncorr, color=plot_colors["FCCD_err_statMCstatdata"], linestyle='dashed', linewidth=1, label="MC syst/uncorr err")
    y_lowlim_uncorr = ydata-yerr_uncorr
    coeff_low_uncorr, coeff_low_uncorr_err = scipy_LS_fit(xdata,y_lowlim_uncorr,exponential_decay,p_guess_low)
    yfit_low_uncorr = exponential_decay(xfit,*coeff_low_uncorr)
    if plot_all_error_bands == True:
        plt.plot(xfit, yfit_low_uncorr, color=plot_colors["FCCD_err_statMCstatdata"],linestyle='dashed', linewidth=1)
    a_up_uncorr, b_up_uncorr, c_up_uncorr = coeff_up_uncorr[0], coeff_up_uncorr[1], coeff_up_uncorr[2]
    a_low_uncorr, b_low_uncorr, c_low_uncorr = coeff_low_uncorr[0], coeff_low_uncorr[1], coeff_low_uncorr[2] 

    #=========Compute FCCD and all errors ===============

    #calculate FCCD of data - invert eq
    FCCD_data = invert_exponential(countRatio_data,a,b,c)
    print('FCCD of data extrapolated: ')
    print(str(FCCD_data))

    #TOTAL ERROR = (statistical (uncorr) error on data, statistical (uncorr) and systematic (corr) on MC)
    FCCD_err_total_up = invert_exponential((countRatio_data-countRatio_data_err),a_up, b_up,c_up) - FCCD_data
    FCCD_err_total_low = FCCD_data - invert_exponential((countRatio_data+countRatio_data_err),a_low, b_low,c_low)
    print('Total error:  + '+ str(FCCD_err_total_up) +" - "+str(FCCD_err_total_low))

    #TOTAL CORR ERROR == syst/corr MC only
    FCCD_err_systMC_up = invert_exponential(countRatio_data, a_up_corr, b_up_corr, c_up_corr) -FCCD_data
    FCCD_err_systMC_low = FCCD_data - invert_exponential(countRatio_data, a_low_corr, b_low_corr, c_low_corr)
    print('TOTAL CORR ERROR == systematic/corr error on MC only')
    print("+ "+str(FCCD_err_systMC_up) +" - "+str(FCCD_err_systMC_low))
    FCCD_err_corr_up, FCCD_err_corr_low = FCCD_err_systMC_up, FCCD_err_systMC_low

    #TOTAL UNCORR ERROR == stat/uncorr MC + stat/uncorr data
    FCCD_err_statMCstatdata_up = invert_exponential(countRatio_data-countRatio_data_err, a_up_uncorr, b_up_uncorr, c_up_uncorr) -FCCD_data
    FCCD_err_statMCstatdata_low = FCCD_data - invert_exponential(countRatio_data+countRatio_data_err, a_low_uncorr, b_low_uncorr, c_low_uncorr)
    print('TOTAL UNCORR ERROR == stat/uncorr error on MC and stat/uncorr error on data')
    print("+ "+str(FCCD_err_statMCstatdata_up) +" - "+str(FCCD_err_statMCstatdata_low))
    FCCD_err_uncorr_up, FCCD_err_uncorr_low = FCCD_err_statMCstatdata_up, FCCD_err_statMCstatdata_low

    print("sum in quadrature of corr and uncorr error:")
    FCCD_err_corr_uncorr_quadrature_up = np.sqrt(FCCD_err_uncorr_up**2 + FCCD_err_corr_up)
    FCCD_err_corr_uncorr_quadrature_low = np.sqrt(FCCD_err_uncorr_low**2 + FCCD_err_corr_low)
    print("+ ", FCCD_err_corr_uncorr_quadrature_up, ", - ", FCCD_err_corr_uncorr_quadrature_low)

    print("linear sum of corr and uncorr error:")
    print("+ ", FCCD_err_corr_up + FCCD_err_uncorr_up,", - ", FCCD_err_corr_low + FCCD_err_uncorr_low)

    #=============Complete Plot===================

    info_str = '\n'.join((r'$\chi^2/dof=%.2f/%.0f$'%(chi_sq, dof), r'FCCD=$%.3f^{+%.2f}_{-%.2f}$ mm' % (FCCD_data, FCCD_err_total_up, FCCD_err_total_low)))
    plt.text(0.02, 0.98, info_str, transform=ax.transAxes, fontsize=8.5,verticalalignment='top')


    #plot horizontal data line and errors
    plt.hlines(countRatio_data+countRatio_data_err, 0, FCCD_list[-1], color=plot_colors["data_err_stat"], label = 'Data total err (stat/uncorr)', linewidth=linewidth_err, linestyle = 'dashed')
    plt.hlines(countRatio_data-countRatio_data_err, 0, FCCD_list[-1], color=plot_colors["data_err_stat"], linewidth=linewidth_err, linestyle = 'dashed')
    plt.hlines(countRatio_data, 0, FCCD_list[-1], color=plot_colors["data"], label = 'Data', linewidth=linewidth_main)
    
    #plot vertical FCCD lines
    plt.vlines(FCCD_data, 0, countRatio_data, color=plot_colors["FCCD"] , linestyles='dashed', linewidth=linewidth_main)

    #plot total error line
    plt.vlines(FCCD_data+FCCD_err_total_up, 0, countRatio_data-countRatio_data_err, color=plot_colors["FCCD_err_total"], linestyles='dashed', linewidths=linewidth_main, label="FCCD total err")
    plt.vlines(FCCD_data-FCCD_err_total_low, 0, countRatio_data+countRatio_data_err, color=plot_colors["FCCD_err_total"], linestyles='dashed', linewidths=linewidth_main)

    # plot syst MC/total corr error line
    plt.vlines(FCCD_data+FCCD_err_corr_up, 0, countRatio_data, color=plot_colors["FCCD_err_MCsyst"], linestyles='dashed', linewidths=linewidth_err, label = "FCCD corr err")
    plt.vlines(FCCD_data-FCCD_err_corr_low, 0, countRatio_data, color=plot_colors["FCCD_err_MCsyst"], linestyles='dashed', linewidths=linewidth_err)

    # #plot stat MC + stat data/ total uncorr:
    plt.vlines(FCCD_data+FCCD_err_uncorr_up, 0, countRatio_data-countRatio_data_err, color=plot_colors["FCCD_err_statMCstatdata"], linestyles='dashed', linewidths=linewidth_err, label = "FCCD uncorr err")
    plt.vlines(FCCD_data-FCCD_err_uncorr_low, 0, countRatio_data+countRatio_data_err, color=plot_colors["FCCD_err_statMCstatdata"], linestyles='dashed', linewidths=linewidth_err)

    if source == "Ba133":
        plt.ylabel(r'$O_{Ba133} = (C_{79.6keV}+C_{81keV})/C_{356keV}$')
        plt.ylim(0.0,1.5)
    plt.xlabel("FCCD (mm)")
    plt.xlim(0,FCCD_list[-1])
    plt.title(detector+", "+source)
    plt.legend(loc="upper right", fontsize=8)
    plt.tight_layout()

    if cuts == False:
        plt.savefig(outputFolder+"FCCD_"+MC_id+"_"+smear+"_"+TL_model+"_fracFCCDbore"+str(frac_FCCDbore)+"_"+energy_filter+"_run"+str(run)+"_nocuts.png")
    else:
        plt.savefig(outputFolder+"FCCD_"+MC_id+"_"+smear+"_"+TL_model+"_fracFCCDbore"+str(frac_FCCDbore)+"_"+energy_filter+"_run"+str(run)+"_cuts.png")

    plt.show()

    # Save interpolated fccd for data to a json file

    FCCD_data_dict = {
        "FCCD": FCCD_data,
        "FCCD_err_total_up": FCCD_err_total_up,
        "FCCD_err_total_low": FCCD_err_total_low,
        "FCCD_err_corr_up": FCCD_err_corr_up,
        "FCCD_err_corr_low": FCCD_err_corr_low,
        "FCCD_err_uncorr_up": FCCD_err_uncorr_up,
        "FCCD_err_uncorr_low": FCCD_err_uncorr_low,
        "FCCD_err_corr_uncorr_quadrature_up": FCCD_err_corr_uncorr_quadrature_up,
        "FCCD_err_corr_uncorr_quadrature_low": FCCD_err_corr_uncorr_quadrature_low,
        "countRatio_data": countRatio_data,
        "countRatio_data_err": countRatio_data_err,
        "exp_fit_params":{"a": a, "a_err": a_err, "b": b, "b_err": b_err, "c":c, "c_err": c_err, "a_up":a_up, "a_low":a_low, "b_up": b_up, "b_low":b_low}
    }

    if cuts == False:
        with open(outputFolder+"../FCCD_"+MC_id+"_"+smear+"_"+TL_model+"_fracFCCDbore"+str(frac_FCCDbore)+"_"+energy_filter+"_run"+str(run)+"_nocuts.json", "w") as outfile:
            json.dump(FCCD_data_dict, outfile, indent=4)
    else:
        if cuts_sigma ==4: #change manually if interested
            with open(outputFolder+"../FCCD_"+MC_id+"_"+smear+"_"+TL_model+"_fracFCCDbore"+str(frac_FCCDbore)+"_"+energy_filter+"_run"+str(run)+"_cuts.json", "w") as outfile:
                json.dump(FCCD_data_dict, outfile, indent=4)
        else:
            with open(outputFolder+"../FCCD_"+MC_id+"_"+smear+"_"+TL_model+"_fracFCCDbore"+str(frac_FCCDbore)+"_"+energy_filter+"_run"+str(run)+"_cuts"+str(cuts_sigma)+"sigma.json", "w") as outfile:
                json.dump(FCCD_data_dict, outfile, indent=4)

    print("done")







