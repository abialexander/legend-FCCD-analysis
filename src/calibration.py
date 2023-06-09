import os
import sys
import math
import numpy as np
import json
import fnmatch
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pygama.analysis.histograms as pgh
import pygama.analysis.calibration as cal
import pygama.analysis.peak_fitting as pgp
import pygama.lh5 as lh5
import pygama.genpar_tmp.cuts as cut


def read_all_dsp_lh5(t2_folder, cuts, cut_file_path=None, run="all", sigma=4):

    sto = lh5.Store()
    files = os.listdir(t2_folder)
    files = fnmatch.filter(files, "*lh5")
    if run != "all":
        files = fnmatch.filter(files, "*run000"+str(run)+"*")

    df_list = []

    if cuts == False:
        for file in files:

            #get data, no cuts
            tb = sto.read_object("raw",t2_folder+file)[0]
            df = lh5.Table.get_dataframe(tb)
            df_list.append(df)

        df_total = pd.concat(df_list, axis=0, ignore_index=True)
        return df_total

    else: #apply cuts
        files = [t2_folder+file for file in files] #get list of full paths
        lh5_group = "raw"
        df_total_cuts, failed_cuts = cut.load_df_with_cuts(files, lh5_group, cut_file = cut_file_path, cut_parameters= {'bl_mean':sigma,'bl_std':sigma}) #, verbose=True)

        return df_total_cuts, failed_cuts


def load_energy_data(data_path, energy_filter, cuts, run, plot_hist = False, sigma_cuts=4):

    if cuts == False:
        df_total_lh5 = read_all_dsp_lh5(data_path,cuts,run=run)
        failed_cuts = np.zeros(len(df_total_lh5[energy_filter]))
    else:
        print("sigma cuts: ", str(sigma_cuts))
        df_total_lh5, failed_cuts = read_all_dsp_lh5(data_path,cuts,run=run, sigma=sigma_cuts)
        failed_cuts = failed_cuts[energy_filter]
    energy_filter_data = df_total_lh5[energy_filter]

    if plot_hist == True:
        plt.hist(energy_filter_data, bins=5000, histtype="step")
        plt.xlim(0,5000)
        plt.xlabel(energy_filter, ha='right', x=1)
        plt.ylabel("Counts",ha='right', y=1)
        plt.yscale('log')
        plt.tight_layout()
        plt.show()


    return energy_filter_data, failed_cuts

def fwhm_slope(x, m0, m1, m2):
    """
    Fit the energy resolution curve
    """
    return np.sqrt(m0 + m1*x +(m2*(x**2)))

def invert_linear(pars):
    "invert y=a*x+b where pars=[a,b] and return new_pars=[a2,b2]"
    # x = y/a -b/a
    a,b = pars[0], pars[1]
    a2, b2 = 1/a, -b/a
    inv_pars = [a2,b2]
    return inv_pars

def plotCalibratedEnergy(ecal_pass, ecal_cut, xlo, xhi, binwidth, plot_title=None,plotPath=None):
    """ Plot the calibrated energy hist
        args: 
        - ecal_pass: calibrated energies passing data cuts
        - ecal_cut: calibrated energies failing data cuts
        - xlo: min x value, 0
        - xhi: max x value, i.e. 450 keV
        - binwidth: e.g. 0.1 keV
    """
    nb = int((xhi-xlo)/binwidth)
    hist_pass, bin_edges = np.histogram(ecal_pass, range=(xlo, xhi), bins=nb)
    bins_pass = pgh.get_bin_centers(bin_edges)
    hist_cut, bin_edges = np.histogram(ecal_cut, range=(xlo, xhi), bins=nb)
    bins_cut = pgh.get_bin_centers(bin_edges)
    plt.plot(bins_pass, hist_pass, label='QC pass', lw=1, c='b')
    plt.plot(bins_cut,  hist_cut,  label='QC fail', lw=1, c='r')
    plt.plot(bins_cut,  hist_cut+hist_pass,  label='no QC', lw=1)

    plt.xlabel("Energy (keV)", ha='right', x=1)
    plt.ylabel("Counts / "+str(binwidth)+" keV", ha='right', y=1)
    plt.title(plot_title)
    plt.yscale('log')
    plt.tight_layout()
    plt.legend(loc='upper right')

    if plotPath is not None:
        plt.savefig(plotPath)


def plotCalibrationCurve(calResults,xlo,xhi,plot_title=None,plotPath=None,residuals="raw"):
    """ Plot the calibration curve with residuals
        args: 
        - calResults: array from calibration routine
        - xlo: min x value, 0
        - xhi: max x value, i.e. 450 keV
        - residuals: "raw" = data-fit, "error" = data-fit/error(data)
    """
    pk_pars = calResults['pk_pars']
    uncalE_mus = np.array([pars[1] for pars in pk_pars]).astype(float)
    pk_covs = calResults['pk_covs']
    uncalE_mus_err = np.array([np.sqrt(cov[1][1]) for cov in pk_covs]).astype(float)
    truthEs_old = np.array(calResults["got_peaks_keV"])
    truthEs=[]
    for i, uncalE_mu in enumerate(uncalE_mus):
        truthE = calResults["got_peaks_keV"][i]
        truthEs.append(truthE)
    truthEs=np.array(truthEs)
    pk_cal_pars = np.array(calResults["pk_cal_pars"]) #pars from cal to uncal

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios":[2,1]})
    ax1.errorbar(truthEs,uncalE_mus,yerr=uncalE_mus_err, marker='o',linestyle='none', c='b', label="Data")
    fit = pgp.poly(truthEs, pk_cal_pars)
    ax1.plot(truthEs,fit,lw=1, c='g', linestyle="-", label=R"Linear fit")
    ax1.set_ylabel("Uncalibrated Energy (ADC)", ha='right', y=1, fontsize=10)
    ax1.legend()

    if residuals == "raw":
        residuals = uncalE_mus-fit
        # residual_lbl = "Data-Fit (ADC)"
        residual_lbl = "Residuals (ADC)"
    elif residuals == "error":
        residuals = (uncalE_mus-fit)/uncalE_mus_err
        residual_lbl = r"$(Data-Fit)/\sigma_{Data}$"
    ax2.plot(truthEs,residuals, marker='o', lw=1, c='b')
    ax2.set_ylabel(residual_lbl, ha='right', y=1, fontsize=10)
    ax2.axhline(0, 0, xhi, color="grey", linestyle="-", linewidth=0.5)
    ax2.set_xlabel("Calibrated Energy (keV)", ha='right', x=1, fontsize=10)
    fig.suptitle(plot_title)

    if plotPath is not None:
        plt.savefig(plotPath)

def plotResolutionCurve(calResults, calPars, xlo, xhi, plot_title=None, plotPath=None, residuals="raw"):
    """ Plot the resolution curve with residuals
        returns: 
        - fit_pars: resolution sqrt fit parameters
        args: 
        - calResults: array from calibration routine
        - calPars: linear calibration pars from calibration routine
        - xlo: min x value, 0
        - xhi: max x value, i.e. 450 keV
        - residuals: "raw" = data-fit
    """
    fitted_peaks = calResults['fitted_keV']
    pk_pars      = calResults['pk_pars']
    mus          = np.array([pars[1] for pars in pk_pars]).astype(float)
    fwhms        = np.array([pars[2] for pars in pk_pars]).astype(float)*calPars[0]*2.*math.sqrt(2.*math.log(2.))
    pk_covs      = calResults['pk_covs']
    dfwhms       = np.array([], dtype=np.float32)
    for i, covsi in enumerate(pk_covs):
        covsi    = np.asarray(covsi, dtype=float)
        parsigsi = np.sqrt(covsi.diagonal())
        dfwhms   = np.append(dfwhms,parsigsi[2]*calPars[0]*2.*math.sqrt(2.*math.log(2.)))
    fwhm_peaks   = np.array([], dtype=np.float32)
    for i,peak in enumerate(fitted_peaks):
        fwhm_peaks = np.append(fwhm_peaks,peak)

    param_guess  = [0.2,0.001,0.000001]
    param_bounds = (0, [10., 1. ,0.1])
    fit_pars, fit_covs = curve_fit(fwhm_slope, fwhm_peaks, fwhms, sigma=dfwhms, p0=param_guess, bounds=param_bounds)
    print('FWHM curve fit: ',fit_pars)
    fit_vals = fwhm_slope(fwhm_peaks,fit_pars[0],fit_pars[1],fit_pars[2])
    print('FWHM fit values: ',fit_vals)
    fit_qbb = fwhm_slope(2039.0,fit_pars[0],fit_pars[1],fit_pars[2])
    print('FWHM energy resolution at Qbb: %1.2f keV' % fit_qbb)

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios":[2,1]})
    ax1.errorbar(fwhm_peaks,fwhms,yerr=dfwhms, marker='o',linestyle='none', c='b', label="Data")
    ax1.plot(fwhm_peaks,fit_vals,lw=1, c='g', linestyle="-", label=R"Fit: $\sqrt{a+bE}$")
    ax1.set_ylabel("FWHM (keV)", ha='right', y=1, fontsize=10)
    fit_info = [r"FWHM @ $Q_{\beta\beta}$: %1.2f keV"%fit_qbb]
    ax1.legend(title="\n".join(fit_info),title_fontsize=9.5,prop={'size': 9.5})

    if residuals == "raw":
        residuals = (fwhms-fit_vals)
        # residual_lbl = "Data-Fit (keV)"
        residual_lbl = "Residuals (keV)"
    ax2.plot(fwhm_peaks,residuals, marker='o', lw=1, c='b')
    ax2.set_xlabel("Energy (keV)", ha='right', x=1, fontsize=10)
    ax2.set_ylabel(residual_lbl, ha='right', y=1, fontsize=10)
    ax2.set_ylim([-0.1,0.1])
    ax2.axhline(0, 0, xhi, color="grey", linestyle="-", linewidth=0.5)

    fig.suptitle(plot_title)

    if plotPath is not None:
        plt.savefig(plotPath)

    return fit_pars

def perform_calibration(detector, source, data_path, energy_filter, cuts, run, plot_calibration_curve=False):

    """
    Perform energy calibration on data spectra
    args: 
        - detector
        - source ("Ba133", "Am241_HS1")
        - data_path
        - energy_filter (cuspEmax_ctc)
        - cuts (True/False)
        - run (1,2,etc)
    """

    #initialise directories for detectors to save
    CodePath=os.path.dirname(os.path.realpath(__file__)) #i.e. /lfs/l1/legend/user/aalexander/legend-FCCD-analysis/src
    outputFolder = CodePath+"/../results/data_calibration/"+detector+"/"+source+"/"
    if not os.path.exists(outputFolder+"plots/"):
        os.makedirs(outputFolder+"plots/")

    #====Load data======
    energy_filter_data, failed_cuts = load_energy_data(data_path, energy_filter, cuts, run)

    #========Compute calibration coefficients===========
    print("Calibrating...")

    if source == "Ba133":
        glines    = [80.9979, 160.61, 223.24, 276.40, 302.85, 356.01, 383.85] # gamma lines used for calibration
        range_keV = [(1,1),(1.5,1.5),(2.5,2.5),(2.5,2.5),(3,3),(3,3),(3,3)] # side bands width
        funcs = [pgp.gauss_step,pgp.gauss_step,pgp.gauss_step,pgp.gauss_step,pgp.gauss_step,pgp.gauss_step,pgp.gauss_step]
        guess = 383/(energy_filter_data.quantile(0.9))

    elif source == "Am241_HS1" or source == "Am241_HS6":
        glines=[59.5409, 98.97, 102.98 ,123.05]
        range_keV=[(3., 3.),(1.5,1.5),(1.5,1.5),(1.5,1.5)]
        funcs = [pgp.gauss_step,pgp.gauss_step,pgp.gauss_step,pgp.gauss_step,pgp.gauss_step,pgp.gauss_step,pgp.gauss_step,pgp.gauss_step,pgp.gauss_step]
        if detector=='V02160A' or detector=='V05268A':
            guess= 0.1#0.045#0.1 #V02160A #0.057   0.032 if V07647A  #0.065 7298B
        elif detector=='V05266A'or detector=="B00035A":
            guess=0.08
        elif detector=='V05267B'or detector=='V04545A'or detector=='V09372A' or detector=="B00035B" :
            guess=0.07
        elif detector=='V08682B':
            guess=0.03
        elif detector=='V08682A'or detector=='V09374A':
            guess=0.053
        #elif detector=='V04549B': #only for am_HS6
        #    guess=0.1
        else:
            guess=0.045

    print("Find peaks and compute calibration curve...",end=' ')

    try:
        pars, cov, results = cal.hpge_E_calibration(energy_filter_data, glines, guess,deg=1,range_keV = range_keV,funcs = funcs, verbose=True)
        print("cal pars: ", pars)
        ecal_pass = pgp.poly(energy_filter_data, pars)
        ecal_cut  = pgp.poly(failed_cuts,  pars)
        calib_pars=True

    except IndexError: #problem for Am-241
        calibration="/lfs/l1/legend/legend-prodenv/prod-usr/ggmarsh-full_dl-v01/pargen/dsp_ecal/"+detector+".json"
        print("calibration with Th file:", calibration)
        with open(calibration) as json_file:
            calibration_coefs = json.load(json_file)
        m = calibration_coefs[energy_filter]["Calibration_pars"][0]
        c = calibration_coefs[energy_filter]["Calibration_pars"][1]
        ecal_pass = energy_filter_data*m + c
        ecal_cut = failed_cuts*m+c
        calib_pars=False

    #======Plots Info ================
    binwidth = 0.1
    xlo = 0
    xhi = 450 if source == "Ba133" else 130
    nb = int((xhi-xlo)/binwidth)
    plot_title = detector+' - run '+str(run)

    #======Plot calibrated energy=======
    if cuts == True:
        plotPath = outputFolder+"plots/calibrated_energy_"+energy_filter+"_run"+str(run)+"_cuts.png"
    else:
       plotPath = outputFolder+"plots/calibrated_energy_"+energy_filter+"_run"+str(run)+"_nocuts.png"

    plotCalibratedEnergy(ecal_pass, ecal_cut, xlo, xhi, binwidth, plot_title=plot_title,plotPath=plotPath)

    #======Plot calibration curve=======
    if plot_calibration_curve == True:
        if cuts == True:
            plotPath = outputFolder+"plots/calibration_curve_"+energy_filter+"_run"+str(run)+"_cuts.png"
        else:
            plotPath = outputFolder+"plots/calibration_curve_"+energy_filter+"_run"+str(run)+"_nocuts.png"

        plotCalibrationCurve(results,xlo,xhi,plot_title=plot_title,plotPath=plotPath)


    #=========Plot Resolution Curve and save all coefficients ===========

    if calib_pars == True:

        if cuts == False:
            plotPath = outputFolder+"plots/resolution_curve_"+energy_filter+"_run"+str(run)+"_nocuts.png"
        else:
            plotPath = outputFolder+"plots/resolution_curve_"+energy_filter+"_run"+str(run)+"_cuts.png"
        
        fit_pars = plotResolutionCurve(results, pars, xlo, xhi, plot_title=plot_title, plotPath=plotPath)

        #=========Save Calibration Coefficients==========
        dict = {energy_filter: {"resolution": list(fit_pars), "calibration": list(pars)}}
        if cuts == False:
            with open(outputFolder+"calibration_run"+str(run)+"_nocuts.json", "w") as outfile:
                json.dump(dict, outfile, indent=4)
        else:
            with open(outputFolder+"calibration_run"+str(run)+"_cuts.json", "w") as outfile:
                json.dump(dict, outfile, indent=4)

    plt.close("all")
    print("done")
    print("")



if __name__ == "__main__":
    main()
