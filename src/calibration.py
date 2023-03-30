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
        plt.xlabel(energy_filter,     ha='right', x=1)
        plt.ylabel("Counts",     ha='right', y=1)
        plt.yscale('log')
        plt.tight_layout()
        plt.show()


    return energy_filter_data, failed_cuts

def fwhm_slope(x, m0, m1, m2):
    """
    Fit the energy resolution curve
    """
    return np.sqrt(m0 + m1*x +(m2*(x**2)))


def perform_calibration(detector, source, data_path, energy_filter, cuts, run):

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
        range_keV = [(1,1),(1.5,1.5),(2,2),(2.5,2.5),(3,3),(3,3),(3,3)] # side bands width

        guess = 383/(energy_filter_data.quantile(0.9))

        funcs = [pgp.gauss_step,pgp.gauss_step,pgp.gauss_step,pgp.gauss_step,pgp.gauss_step,pgp.gauss_step,pgp.gauss_step]

    elif source == "Am241_HS1" or source == "Am241_HS6":
        glines=[59.5409, 98.97, 102.98 ,123.05]
        range_keV=[(3., 3.),(1.5,1.5),(1.5,1.5),(1.5,1.5)]

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

        funcs = [pgp.gauss_step,pgp.gauss_step,pgp.gauss_step,pgp.gauss_step,pgp.gauss_step,pgp.gauss_step,pgp.gauss_step,pgp.gauss_step,pgp.gauss_step]


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

    #======Plot calibrated energy=======

    xpb = 0.1
    xlo = 0
    xhi = 450 if source == "Ba133" else 120

    nb = int((xhi-xlo)/xpb)
    hist_pass, bin_edges = np.histogram(ecal_pass, range=(xlo, xhi), bins=nb)
    bins_pass = pgh.get_bin_centers(bin_edges)
    hist_cut, bin_edges = np.histogram(ecal_cut, range=(xlo, xhi), bins=nb)
    bins_cut = pgh.get_bin_centers(bin_edges)
    plot_title = detector+' - run '+str(run)
    plt.plot(bins_pass, hist_pass, label='QC pass', lw=1, c='b')
    plt.plot(bins_cut,  hist_cut,  label='QC fail', lw=1, c='r')
    plt.plot(bins_cut,  hist_cut+hist_pass,  label='no QC', lw=1)

    plt.xlabel("Energy (keV)",     ha='right', x=1)
    plt.ylabel("Counts / "+str(xpb)+" keV",     ha='right', y=1)
    plt.title(plot_title)
    plt.yscale('log')
    plt.tight_layout()
    plt.legend(loc='upper right')

    if cuts == True:
        plt.savefig(outputFolder+"plots/calibrated_energy_"+energy_filter+"_run"+str(run)+"_cuts.png")
    else:
        plt.savefig(outputFolder+"plots/calibrated_energy_"+energy_filter+"_run"+str(run)+"_nocuts.png")

    #=========Plot Calibration Curve===========

    if calib_pars == True:

        fitted_peaks = results['fitted_keV']
        pk_pars      = results['pk_pars']
        mus          = np.array([pars[1] for pars in pk_pars]).astype(float)
        fwhms        = np.array([pars[2] for pars in pk_pars]).astype(float)*pars[0]*2.*math.sqrt(2.*math.log(2.))
        pk_covs      = results['pk_covs']
        dfwhms       = np.array([], dtype=np.float32)
        for i, covsi in enumerate(pk_covs):
            covsi    = np.asarray(covsi, dtype=float)
            parsigsi = np.sqrt(covsi.diagonal())
            dfwhms   = np.append(dfwhms,parsigsi[2]*pars[0]*2.*math.sqrt(2.*math.log(2.)))
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

        fig, (ax1, ax2) = plt.subplots(2, 1, constrained_layout=True, sharex=True)
        ax1.errorbar(fwhm_peaks,fwhms,yerr=dfwhms, marker='o',lw=0, c='b')
        ax1.plot(fwhm_peaks,fit_vals,lw=1, c='g')
        ax1.set_ylabel("FWHM (keV)", ha='right', y=1)

        ax2.plot(fitted_peaks,pgp.poly(mus, pars)-fitted_peaks, marker='o', lw=1, c='b')
        ax2.set_xlabel("Energy (keV)",    ha='right', x=1)
        ax2.set_ylabel("Residuals (keV)", ha='right', y=1)

        fig.suptitle(plot_title)

        if cuts == False:
            plt.savefig(outputFolder+"plots/calibration_curve_"+energy_filter+"_run"+str(run)+"_nocuts.png")
        else:
            plt.savefig(outputFolder+"plots/calibration_curve_"+energy_filter+"_run"+str(run)+"_cuts.png")
        
        #=========Save Calibration Coefficients==========
        dict = {energy_filter: {"resolution": list(fit_pars), "calibration": list(pars)}}
        print(dict)
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
