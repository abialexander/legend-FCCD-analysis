import pandas as pd
import numpy as np
import h5py
import os
import matplotlib.pyplot as plt
import argparse
from scipy import optimize
from scipy import stats
import glob
import json
from datetime import datetime
from scipy.integrate import quad
import fnmatch

import pygama.lh5 as lh5
from pygama.analysis import histograms
from pygama.analysis import peak_fitting
import pygama.genpar_tmp.cuts as cut

from iminuit import Minuit
from iminuit.cost import LeastSquares

from src.calibration import *



def gauss_count(a,mu,sigma, a_err, bin_width, integration_method="numerical"):
    "count/integrate a gaussian peak"

    if integration_method == "analytical": #from -inf to +inf
        integral = a/bin_width
        integral_err = a_err/bin_width

    else: #numerical, from mu-3sigma to mu+3sigma
        integral_356_3sigma_list = quad(peak_fitting.gauss,mu-3*sigma, mu+3*sigma, args=(mu,sigma,a))
        integral = integral_356_3sigma_list[0]/bin_width
        integral_err = a_err/bin_width

    return integral, integral_err


def double_gauss_step_pdf(x, a, mu1, sigma1, bkg, s, mu2, sigma2):
    """
    Pdf for double Gaussian on a single step background (Ba133 79/81 double, or Am241 double)
    args: a, mu1, sigma1, bkg, s, mu2, sigma2; a mus, sigmas for the signals and bkg, s for the background
    """
                        
    peak1 = peak_fitting.gauss(x,mu1,sigma1,a*R_doublePeak)
    peak2 = peak_fitting.gauss(x,mu2,sigma2,a)
    step = peak_fitting.step(x,mu1,sigma1,bkg,s)
    f = peak1 + peak2 + step
    return f

def gauss_step_pdf(x,  a, mu, sigma, bkg, s):
    """
    Pdf for Gaussian on step background
    args: a mu, sigma for the signal and bkg, s for the background
    """
    peak = peak_fitting.gauss(x, mu, sigma, a)
    step = peak_fitting.step(x,mu,sigma,bkg,s)
    f = peak + step
    return f
    

def perform_gammaLineCounting(detector, source, spectra_type, data_path=None, calibration=None, energy_filter=None, cuts=None, run=None, sim_path=None, MC_id=None):
    "Perform the gammaline counting on relevant gammaline peaks in Ba or Am, data or MC spectra"
    
    dir = os.path.dirname(os.path.realpath(__file__))

    if spectra_type == "MC" and (sim_path is None or MC_id is None):
        print("Need input args of <sim_path> and/or <MC_id>")
        sys.exit()
    if spectra_type == "data" and (data_path is None or calibration is None or energy_filter is None or cuts is None or run is None):
        print("Need input args of <data_path>, <calibration>, <energy_filter>, <cuts>, <run> ")
        sys.exit()

    #==========================================================
    #LOAD ENERGY SPECTRA
    #==========================================================

    if spectra_type == "data":

        #initialise directories for detectors to save
        outputFolder = dir+"/../results/PeakCounts/"+detector+"/data/plots/"
        if not os.path.exists(outputFolder):
            os.makedirs(outputFolder)
        
        #load uncalibrated energy
        sigma_cuts = 4 #default
        energy_filter_data, failed_cuts = load_energy_data(data_path, energy_filter, cuts, run, sigma_cuts=sigma_cuts)

        #calibrate
        with open(calibration) as json_file:
            calibration_coefs = json.load(json_file)
        m = calibration_coefs[energy_filter]["calibration"][0]
        c = calibration_coefs[energy_filter]["calibration"][1]
        energies = energy_filter_data*m + c
    
    if spectra_type == "MC":

        #initialise directories for detectors to save
        outputFolder = dir+"/../results/PeakCounts/"+detector+"/MC/plots/"
        if not os.path.exists(outputFolder):
            os.makedirs(outputFolder)

        #Load energy
        df =  pd.read_hdf(sim_path, key="procdf")
        energies = df['energy']

    
    #get total pygama histogram
    binwidth = 0.1 #keV
    Emax = 450 if source == "Ba133" else 120
    bins = np.arange(0,Emax,binwidth)
    hist, bins, var = histograms.get_hist(energies, bins=bins)

    #==========================================================
    # PEAK FITTING
    #==========================================================
    peak_counts = []
    peak_counts_err = []
    
    if source == "Ba133":
        peak_ranges = [[77,84],[352, 359.5],[158,163],[221.5,225],[274,279],[300,306],[381,386.5]] #Rough by eye
        peaks = [81, 356, 161, 223, 276, 303, 383]
        global R_doublePeak
        R_doublePeak = 2.65/32.9 #intensity ratio for Ba-133 double peak
        
    for index, i in enumerate(peak_ranges):

        #prepare histogram
        print(str(peaks[index]), " keV")
        xmin, xmax = i[0], i[1]
        bins_peak = np.arange(xmin,xmax,binwidth)
        hist_peak, bins_peak, var_peak = histograms.get_hist(energies, bins=bins_peak)
        bins_centres_peak = histograms.get_bin_centers(bins_peak)
        yerr = np.sqrt(hist_peak)
        
        #remove any zeros
        zeros = (hist_peak == 0)
        mask = ~(zeros)
        hist_peak, bins_centres_peak, yerr  = hist_peak[mask], bins_centres_peak[mask], yerr[mask]

        #fit function initial guess
        if peaks[index] == 81:
            fitting_func = double_gauss_step_pdf
            mu_79_guess, sigma_79_guess, bkg_guess, s_guess = 79.6142, 0.5, min(hist_peak), min(hist_peak)
            mu_81_guess, sigma_81_guess, a_guess = 80.9979, 0.5, max(hist_peak)
            fitting_func_guess = [a_guess, mu_79_guess, sigma_79_guess, bkg_guess, s_guess, mu_81_guess, sigma_81_guess]
            bounds=[(0,np.inf), (0,np.inf), (0,np.inf), (-np.inf,np.inf), (0,np.inf), (0,np.inf), (0,np.inf)]
        
        else:
            fitting_func = gauss_step_pdf
            mu_guess, sigma_guess, a_guess, bkg_guess, s_guess = peaks[index], 1, max(hist_peak), min(hist_peak), min(hist_peak)
            fitting_func_guess = [a_guess, mu_guess, sigma_guess, bkg_guess, s_guess]
            bounds = [(0,np.inf), (0,np.inf), (0,np.inf), (-np.inf,np.inf), (0,np.inf)]
        
        
        #fiting using iminuit and a least squares cost function
        least_squares = LeastSquares(bins_centres_peak, hist_peak, yerr, fitting_func)
        m = Minuit(least_squares, *fitting_func_guess)  # starting values for gauss_step
        m.limits = bounds
        m.migrad()  # finds minimum of least_squares function
        m.hesse()   # accurately computes uncertainties

        #compute chi sq of fit
        chi_sq = m.fval
        dof =  len(bins_centres_peak) - m.nfit

        if chi_sq/dof > 50 or m.valid == False: #repeat fit with no bounds if bad
            print("refitting...")
            m = Minuit(least_squares, *fitting_func_guess)
            m.migrad(ncall=10**7, iterate=100) 
            m.hesse()
            # m.simplex()
            # m.migrad(ncall=10**7, iterate=1000)
            chi_sq = m.fval
            dof =  len(bins_centres_peak) - m.nfit
        
        print("valid fit: ", m.valid)
        print("r chi sq: ", chi_sq/dof)

        coeff, coeff_err = m.values, m.errors

        #Plot
        xfit = np.linspace(xmin, xmax, 1000)
        yfit = fitting_func(xfit, *coeff)
        if fitting_func == gauss_step_pdf:
            yfit_gauss = peak_fitting.gauss(xfit, coeff[1], coeff[2], coeff[0])
        else: #double gauss
            yfit_gauss = peak_fitting.gauss(xfit,coeff[1], coeff[2], R_doublePeak*coeff[0]) + peak_fitting.gauss(xfit,coeff[5], coeff[6], coeff[0])
        yfit_step = peak_fitting.step(xfit, *coeff[1:5]) #mu, sigma, bkg, s)

        fig, ax = plt.subplots()
        histograms.plot_hist(hist_peak, bins_peak, var=None, show_stats=False, label="Data")
        fit_str = "Fit: gauss+step" if fitting_func == gauss_step_pdf else "Fit: double gauss+step"
        plt.plot(xfit, yfit, label=fit_str)
        plt.plot(xfit, yfit_gauss, "--", label ="Fit: gauss")
        plt.plot(xfit, yfit_step, "--", label ="Fit: step")

        plt.xlim(xmin, xmax)
        plt.yscale("log")
        plt.xlabel("Energy (keV)", fontsize=10)
        plt.ylabel("Counts / "+str(binwidth)+" keV", fontsize=10)
        plt.ylim(0.5*min(hist_peak), 2*max(hist_peak))

        # display legend with some fit info
        fit_info = [f"$\\chi^2$ / $n_\\mathrm{{dof}}$ = {chi_sq:.1f} / {dof}",]
        plt.legend(title="\n".join(fit_info),title_fontsize=9,prop={'size': 9}, loc="upper left")
        fit_params = []
        for p, v, e in zip(m.parameters, m.values, m.errors):
            if p == "mu" or p=="sigma" or p == "mu1" or p=="sigma1" or p == "mu2" or p=="sigma2":
                p = "$\%s$"%p
            string = f"{p} = ${v:.2f} \\pm {e:.2f}$"
            fit_params.append(string)
        plt.text(0.75, 0.98, "\n".join(fit_params), transform=ax.transAxes, fontsize=8 ,verticalalignment='top') 

        #Save fig
        if spectra_type == "MC":
            ax.set_title(MC_id, fontsize=9)
            plt.savefig(outputFolder+MC_id+"_"+str(peaks[index])+'keV.png')

        if spectra_type == "data":
            ax.set_title("Data: "+detector, fontsize=10)
            if cuts == False:
                plt.savefig(outputFolder+detector+"_"+str(peaks[index])+'keV_'+energy_filter+'_run'+str(run)+'_nocuts.png')
            else:
                if sigma_cuts ==4:
                    plt.savefig(outputFolder+detector+"_"+str(peaks[index])+'keV_cuts_'+energy_filter+'_run'+str(run)+'_cuts.png')

        plt.close("all")

        #Counting - integrate gaussian signal part +/- 3 sigma
        if fitting_func == gauss_step_pdf:
            a, mu, sigma, a_err = coeff[0],coeff[1],coeff[2], coeff_err[0]
            C, C_err = gauss_count(a, mu, sigma, a_err, binwidth)
            print("peak counts = ", str(C)," +/- ", str(C_err))
            peak_counts.append(C)
            peak_counts_err.append(C_err)
            if peaks[index] == 356:
                C_356, C_356_err = C, C_err
        else: #double gauss
            a1, a1_err, a2, a2_err = R_doublePeak*coeff[0], R_doublePeak*coeff_err[0], coeff[0], coeff_err[0]
            mu1, sigma1, mu2, sigma2 = coeff[1],coeff[2], coeff[5], coeff[6]
            C1, C1_err = gauss_count(a1,mu1,sigma1, a1_err, binwidth)
            print("peak count 1 = ", str(C1)," +/- ", str(C1_err))
            peak_counts.append(C1)
            peak_counts_err.append(C1_err)
            C2, C2_err = gauss_count(a2,mu2,sigma2, a2_err, binwidth)
            print("peak count 2 = ", str(C2)," +/- ", str(C2_err))
            peak_counts.append(C2)
            peak_counts_err.append(C2_err)
            if peaks[index] == 81:
                C_79, C_79_err, C_81, C_81_err = C1, C1_err, C2, C2_err
    

    #Compute count ratio
    if source == "Ba133":
        if (C_356 == np.nan) or (C_79 == np.nan) or (C_81 == np.nan):
            countRatio, countRatio_err = np.nan, np.nan
        else:
            countRatio = (C_79 + C_81)/C_356
            countRatio_err = countRatio*np.sqrt((C_79_err**2 + C_81_err**2)/(C_79+C_81)**2 + (C_356_err/C_356)**2)
        
    print("countRatio = " , countRatio, " +/- ", countRatio_err)

    #==========================================================
    #Save count values to json file
    #==========================================================
    if source == "Ba133":
        peaks.insert(0,79)
    
    PeakCounts = {"peaks":peaks, "counts": peak_counts, "counts_err": peak_counts_err, "countRatio":countRatio,  "countRatio_err":countRatio_err}

    if spectra_type=="MC":
        with open(outputFolder+"../PeakCounts_"+MC_id+".json", "w") as outfile:
            json.dump(PeakCounts, outfile, indent=4)
    if spectra_type=="data":
        if cuts == False:
            with open(outputFolder+"../PeakCounts_"+detector+"_"+energy_filter+"_run"+str(run)+"_nocuts.json", "w") as outfile:
                json.dump(PeakCounts, outfile, indent=4)
        else:
            if sigma_cuts ==4:
                with open(outputFolder+"../PeakCounts_"+detector+"_"+energy_filter+"_run"+str(run)+"_cuts.json", "w") as outfile:
                    json.dump(PeakCounts, outfile, indent=4)
            else:
                with open(outputFolder+"../PeakCounts_"+detector+"_"+energy_filter+"_run"+str(run)+"_cuts"+str(sigma_cuts)+"sigma.json", "w") as outfile:
                    json.dump(PeakCounts, outfile, indent=4)

            








