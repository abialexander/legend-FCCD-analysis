import json
import sys
import os
import math
import numpy as np

class Vividict(dict):
    def __missing__(self, key):
        value = self[key] = type(self)()
        return value


def FCCDtoAV(order_list, metadata_folder):
    """
    Returns AVs from FCCD for Ba133 and Am241, for all detectors in listed orders
    args:
    - order_list: e.g. [2,4,5,7,8,9]
    - metadata_folder: path to folder containing the metadata containing detector geometry files
    """

    CodePath=os.path.dirname(os.path.realpath(__file__))

    FCCD_AV=Vividict()

    detector_list = CodePath+"/../detector_list.json"
    with open(detector_list) as json_file:
        data_detectorlist = json.load(json_file)

    for order in order_list:
        # print(order)
        detectors = data_detectorlist["order_"+str(order)]["detectors"]
        # print(detectors)
        for detName in detectors:
            # print(detector)


            #=============================================
            # Get FCCDs from combined json file
            #=============================================
            #Ba133:
            allFCCDs_Ba133_json = CodePath+"/../FCCDvalues_Ba133.json"
            with open(allFCCDs_Ba133_json) as json_file:
                data_ba = json.load(json_file)
            try:
                FCCD_ba = data_ba[detName]['unrounded']['FCCD']
                errPos_ba = data_ba[detName]['unrounded']['FCCD_err_total_up']
                errNeg_ba = data_ba[detName]['unrounded']['FCCD_err_total_low']
            except KeyError:
                print("no FCCD  for Ba133 and detector ",detName)
                # continue
                # print("Setting FCCD to 0 for Ba133 and detector ",detName)
                FCCD_ba = 0.
                errPos_ba = 0.
                errNeg_ba = 0.
            # if detector == "V07646A":
            #     print("Setting FCCD to 0 for Ba133 and detector ",detector)
            #     FCCD_ba = 0.
            #     errPos_ba = 0.
            #     errNeg_ba = 0.
            #Am241:
            allFCCDs_Am241_HS1_json = CodePath+"/../FCCDvalues_Am241_HS1.json"
            with open(allFCCDs_Am241_HS1_json) as json_file:
                data_am = json.load(json_file)
            try:
                FCCD_am = data_am[detName]['unrounded']['FCCD']
                errPos_am = data_am[detName]['unrounded']['FCCD_err_total_up']
                errNeg_am = data_am[detName]['unrounded']['FCCD_err_total_low']
            except KeyError:
                print("no FCCD  for Am241 and detector ",detName)
                # continue
                print("Setting FCCD to 0 for Am241 and detector ",detName)
                FCCD_am = 0.
                errPos_am = 0.
                errNeg_am = 0.

            #=============================================
            # Define detector geometry
            #=============================================
            # Get geoemtry file
            # geom_file = "/lfs/l1/legend/detector_char/enr/hades/simulations/legend-g4simple-simulation/tools/legend-metadata/hardware/detectors/" + detName + ".json"
            geom_file =  metadata_folder+detName+".json"
            with open(geom_file) as json_file:
                geom_json = json.load(json_file)
            geom = geom_json['geometry']

            #Define geometry
            H_c = height_in_mm = geom["height_in_mm"] # = crystal height
            R_c = radius_in_mm = geom["radius_in_mm"] #crystal main/bottom radius

            well_gap_in_mm = geom["well"]["gap_in_mm"] # deep well
            h_w = H_c - well_gap_in_mm #well height
            r_w = well_radius_in_mm = geom["well"]["radius_in_mm"] #radius well

            h_g = groove_height_in_mm =  geom["groove"]["depth_in_mm"]
            r_g_o = groove_outer_radius_in_mm =  geom["groove"]["outer_radius_in_mm"]
            r_g_i = groove_inner_radius_in_mm =  geom["contact"]["radius_in_mm"]

            h_o = taper_bottom_outer_height_in_mm = geom["taper"]["bottom"]["outer"]["height_in_mm"] #height of bottom conical part
            try :
                taper_bottom_outer_angle_in_deg = geom["taper"]["bottom"]["outer"]["angle_in_deg"]
            except KeyError:
                taper_bottom_outer_angle_in_deg = geom["taper"]["bottom"]["outer"]["radius_in_mm"]  #typo in the metafile
            r_o = R_c - h_o *math.tan(math.radians(taper_bottom_outer_angle_in_deg)) #radius of bottom crystal

            h_u = taper_top_outer_height_in_mm = geom["taper"]["top"]["outer"]["height_in_mm"] #height of top conical part
            taper_top_outer_angle_in_deg = geom["taper"]["top"]["outer"]["angle_in_deg"]
            r_u = R_c - h_u *math.tan(math.radians(taper_top_outer_angle_in_deg)) #radius of top crystal

            h_i = taper_top_inner_height = geom["taper"]["top"]["inner"]["height_in_mm"]
            taper_top_inner_angle_in_deg = geom["taper"]["top"]["inner"]["angle_in_deg"]
            #taper_top_inner_angle_in_deg=4.085#4.63 4.19
            r_i = taper_top_inner_radius = r_w+ h_i* math.tan(math.radians(taper_top_inner_angle_in_deg))

            h_topg = top_groove_height = geom["topgroove"]["depth_in_mm"]
            r_topg = top_groove_radius = geom["topgroove"]["radius_in_mm"]

            h_b = bottom_cyl_height = geom["bottom_cyl"]["height_in_mm"]
            r_b = bottom_cyl_radius = R_c - geom["bottom_cyl"]["radius_in_mm"]
            h_t = bottom_cyl_trantisiton_height = geom["bottom_cyl"]["transition_in_mm"]

            h_cr = geom["crack"]["height_in_mm"]
            r_cr = geom["crack"]["radius_in_mm"]

            #=============================================
            #create a detector object
            #=============================================
            detector = Detector( detName,
                                FCCD_am, errPos_am, errNeg_am,
                                FCCD_ba, errPos_ba, errNeg_ba,
                                H_c, R_c,
                                h_w, r_w,
                                h_g, r_g_o, r_g_i,
                                h_o, r_o,
                                h_u, r_u,
                                h_i, r_i,
                                h_topg, r_topg,
                                h_b, r_b, h_t,
                                h_cr, r_cr
                                )

            # if (FCCD_am!=0. and FCCD_ba!=0):
            #     FCCD, FCCD_errPos, FCCD_errNeg = detector.FCCD_collection()
            # else:
            #     FCCD, FCCD_errPos, FCCD_errNeg = FCCD_ba, errPos_ba, errNeg_ba

            for source in ["Ba133", "Am241_HS1"]:
                if source == "Ba133":
                    FCCD, FCCD_errPos, FCCD_errNeg = FCCD_ba, errPos_ba, errNeg_ba
                else:
                    FCCD, FCCD_errPos, FCCD_errNeg = FCCD_am, errPos_am, errNeg_am
                
                FCCD_bore = FCCD / 2.
                FCCD_bore_errPos = FCCD_errPos / 2.
                FCCD_bore_errNeg = FCCD_errNeg / 2.

                vol = detector.volume_comp()

                av  = detector.active_volume_comp(FCCD, FCCD_bore)
                av_errPos = detector.active_volume_comp(FCCD - FCCD_errNeg, FCCD_bore - FCCD_bore_errNeg) - av   # smaller DL bigger AV
                av_errNeg  = av - detector.active_volume_comp(FCCD + FCCD_errPos, FCCD_bore + FCCD_bore_errPos)        # bigger DL smaller AV

                ratio = detector.ratio(FCCD, FCCD_bore)
                ratio_errNeg = ratio - detector.ratio(FCCD + FCCD_errPos, FCCD_bore + FCCD_bore_errPos)
                ratio_errPos = detector.ratio(FCCD - FCCD_errNeg, FCCD_bore - FCCD_bore_errNeg) - ratio

                results_source = {
                    "FCCD" : {
                        "Central": FCCD,
                        "ErrPos" : FCCD_errPos,
                        "ErrNeg" : FCCD_errNeg
                        },
                    "ActiveVolume" : {
                        "Central": av,
                        "ErrPos" : av_errPos,
                        "ErrNeg" : av_errNeg
                        },
                    "AV/Volume" : {
                        "Central": av/vol,
                        "ErrPos" : ratio_errPos,
                        "ErrNeg" : ratio_errNeg
                    }
                }
                FCCD_AV[detName][source] = results_source

    AV_file = CodePath+"/../AVvalues.json"
    with open(AV_file, "w") as outfile:
        json.dump(FCCD_AV, outfile, indent=4)


class Detector():
    "detector"

    def __init__(self, detName,
                FCCD_am, errPos_am, errNeg_am,
                FCCD_ba, errPos_ba, errNeg_ba,
                H_c, R_c,
                h_w, r_w,
                h_g, r_g_o, r_g_i,
                h_o, r_o,
                h_u, r_u,
                h_i, r_i,
                h_topg, r_topg,
                h_b, r_b, h_t,
                h_cr, r_cr
                ):
        self.detName = detName
        self.FCCD_am = FCCD_am
        self.errPos_am = errPos_am
        self.errNeg_am = errNeg_am
        self.FCCD_ba = FCCD_ba
        self.errPos_ba = errPos_ba
        self.errNeg_ba = errNeg_ba
        self.H_c = H_c
        self.R_c = R_c
        self.h_w = h_w
        self.r_w = r_w
        self.h_g = h_g
        self.r_g_o = r_g_o
        self.r_g_i = r_g_i
        self.h_o = h_o
        self.r_o = r_o
        self.h_u = h_u
        self.r_u = r_u
        self.h_i = h_i
        self.r_i = r_i
        self.h_topg = h_topg
        self.r_topg = r_topg
        self.h_b = h_b
        self.r_b = r_b
        self.h_t = h_t
        self.h_cr = h_cr
        self.r_cr = r_cr

    #FCCD parameters
    def FCCD_collection(self):
        FCCD_sigma_am = (self.errPos_am + self.errNeg_am) / 2.
        FCCD_sigma_ba = (self.errPos_ba + self.errNeg_ba) / 2.
        w_am = 1. / (FCCD_sigma_am * FCCD_sigma_am)
        w_ba = 1. / (FCCD_sigma_ba * FCCD_sigma_ba)
        FCCD_coll = w_am * self.FCCD_am + w_ba * self.FCCD_ba
        w = w_am + w_ba
        FCCD_coll /= w
        FCCD_coll_err_pos=math.sqrt(self.errPos_am*self.errPos_am + self.errPos_ba*self.errPos_ba)
        FCCD_coll_err_neg=math.sqrt(self.errNeg_am*self.errNeg_am + self.errNeg_ba*self.errNeg_ba)

        return FCCD_coll, FCCD_coll_err_pos, FCCD_coll_err_neg


    #volume

    #cylinder volume
    def cylinder_volume(self):
        return math.pi * self.H_c *self.R_c * self.R_c
    @staticmethod
    def S_cylinder_volume(h, r):
        return  math.pi * h * r *r
    #_ANS = S_cylinder_volume.__func__(h, r)

    # groove volume
    def groove_volume(self):
        return math.pi * self.h_g *(self.r_g_o*self.r_g_o - self.r_g_i*self.r_g_i)
    @staticmethod
    def S_groove_volume(d, r, R):
        return math.pi * d * (R*R - r*r)
    #_ANS = S_groove_volume.__func__()

    #well volume
    def well_volume(self):
        return Detector.S_cylinder_volume(self.h_w, self.r_w)
    @staticmethod
    def S_well_volume(h, r):
        return Detector.S_cylinder_volume(h, r)
    #_ANS = S_well_volume.__func__()

    # volume normal detector
    def standard_volume(self):
        vol = self.cylinder_volume()
        vol -= self.groove_volume()
        vol -= self.well_volume()
        return vol
    @staticmethod
    def S_standard_volume(h, r, d, rg, Rg, hc, rc):
        vol = Detector.S_cylinder_volume(h, r)
        vol -= Detector.S_groove_volume(d, rg, Rg)
        vol -= Detector.S_well_volume(hc, rc)
        return vol
    #_ANS = S_standard_volume.__func__()

    # truncated cone volume
    def cut_cone_volume(self):
        return 1/3 * math.pi * self.h_u * (self.R_c * self.R_c + self.R_c*self.r_u + self.r_u*self.r_u)
    @staticmethod
    def S_cut_cone_volume(h, r, R):
        return 1/3 * math.pi * h * (r*r + r*R + R*R)
    #_ANS = S_cut_cone_volume.__func__()

    # crack volume
    def crack_volume(self):
        phi_cr = math.acos((self.R_c-self.r_cr)/self.R_c)
        vol = self.h_cr * self.r_cr*self.r_cr * (3*math.sin(phi_cr) - 3 * phi_cr* math.cos(phi_cr) - math.sin(phi_cr)*math.sin(phi_cr)) / (3 *(1-math.cos(phi_cr)))
        return vol
    @staticmethod
    def S_crack_volume(h, r, R):
        phi = math.acos((R-r)/R)
        vol = h * r*r * (3*math.sin(phi) - 3 * phi* math.cos(phi) - math.sin(phi)*math.sin(phi)) / (3*(1-math.cos(phi)))
        return vol

    #taper botttom outer volume
    def taper_bottom_outer_volume(self):
        vol = self.cut_cone_volume()
        return vol
    @staticmethod
    def S_taper_bottom_outer_volume(h, r, R):
        vol = Detector.S_cut_cone_volume(h, r, R)
        return vol
    #_ANS = S_taper_top_outer_volume.__func__()

    #taper top outer volume
    def taper_top_outer_volume(self):
        vol = self.cut_cone_volume()
        return vol
    @staticmethod
    def S_taper_top_outer_volume(h, r, R):
        vol = Detector.S_cut_cone_volume(h, r, R)
        return vol
    #_ANS = S_taper_top_outer_volume.__func__()

    #taper inner volume
    def taper_top_inner_volume(self):
        vol = Detector.S_cut_cone_volume(self.h_i, self.r_i, self.r_w)
        vol += Detector.S_well_volume(self.h_w-self.h_i, self.r_w)
        return vol
    @staticmethod
    def S_taper_top_inner_volume(h, r , H, R):
        vol = Detector.S_cut_cone_volume(h, r, R)
        vol += Detector.S_well_volume(H-h, R)
        return vol
    #_ANS = S_taper_top_inner_volume.__func__()

    #top grooove volume
    def topgroove_volume(self):
        vol = Detector.S_cylinder_volume(self.h_topg, self.r_topg)
        vol -= Detector.S_cylinder_volume(self.h_topg, self.r_w)
        return vol
    @staticmethod
    def S_topgroove_volume(h ,r, R):
        vol = Detector.S_cylinder_volume(h, r)
        vol -= Detector.S_cylinder_volume(h, R)
        return vol
    #_ANS = S_topgroove_volume.__func__()

    #multiradius volume
    def multiradius_volume(self):
        vol = Detector.S_cylinder_volume(self.h_b, self.r_b)
        vol += Detector.S_cut_cone_volume(self.h_t, self.r_b, self.R_c)
        return vol
    @staticmethod
    def S_multiradius_volume(h, ht, r, R):
        vol = Detector.S_cylinder_volume(h, r)
        vol += Detector.S_cut_cone_volume(ht, r, R)
        return vol
    #_ANS = S_multiradius_volume.__func__()

    def bottom_crack_volume(self):
        vol = self.crack_volume()
        return vol
    @staticmethod
    def S_bottom_crack_volume(h, r, R):
        vol = Detector.S_crack_volume(h, r, R)
        return vol


    #no bullatization situation

    def volume_comp(self):
        volume = self.standard_volume()

        #taper_bottom_outer
        if any(value!=0 for value in [self.h_o]):
            volume -= Detector.S_cylinder_volume(self.h_o, self.R_c)   #cut the bottom side of the cylinder
            volume += self.taper_bottom_outer_volume()   #add the taper bottom side of the cylinder

        #taper_top_outer
        if any(value!=0 for value in [self.h_u]):
            volume -= Detector.S_cylinder_volume(self.h_u, self.R_c)   #cut the upper side of the cylinder
            volume += self.taper_top_outer_volume()   #add the taper upper side of the cylinder

        #taper_top_inner
        if any(value!=0 for value in [self.h_i]):
            volume += self.well_volume() #add the well back
            volume -= self.taper_top_inner_volume()

        #topgroove
        if any(value!=0 for value in [self.h_topg, self.r_topg]):
            volume -= self.topgroove_volume()

        #multiradius
        if any(value!=0 for value in [self.h_b, self.h_t]):
            volume -= Detector.S_cylinder_volume(self.h_b + self.h_t, self.R_c)
            volume += self.multiradius_volume()

        if any(value!=0 for value in [self.h_cr, self.r_cr]):
            volume -= self.bottom_crack_volume()

        # calculate volume of detector in cm3
        volume = 0.001 * volume

        return volume


    #active volume

    # active volume of normal detector
    def active_volume_standard(self, FCCD, FCCD_bore):
        vol = Detector.S_cylinder_volume(self.H_c - 2*FCCD, self.R_c - FCCD)
        vol -=  Detector.S_well_volume(self.h_w + FCCD_bore - FCCD, self.r_w + FCCD_bore)
        vol +=  Detector.S_cylinder_volume(FCCD, self.r_g_o)
        vol -= self.groove_volume()
        return vol

    # active volume functions tapered
    @staticmethod
    def S_corner_radius_shift(h, r, R, FCCD):   #r=corner_radius, h=corner_height
        tth = (R - r) / h
        th = math.atan(tth)
        s = tth*(1/math.sin(th) - 1) * FCCD
        return r - s
    #_ANS = S_corner_radius_shift.__func__()

    @staticmethod
    def S_corner_height_shift(h, r, R, FCCD):
        tth = h / (R - r)
        th = math.atan(tth)
        s = tth*(1/math.sin(th) - 1) * FCCD
        return h - FCCD + s
    #_ANS = S_corner_height_shift.__func__()

    @staticmethod
    def S_inner_corner_radius_shift(h, r, R, FCCD, FCCD_bore):
        tth = (R - r) / h
        th = math.atan(tth)
        return tth * abs(FCCD - FCCD_bore/math.sin(th))
    #_ANS = S_inner_corner_radius_shift.__func__()

    @staticmethod
    def S_crack_corner_radius_shift(h, r, R, FCCD):   #r=corner_radius, h=corner_height
        tth = (R - r) / h
        th = math.atan(tth)
        s = tth*(1/math.sin(th) - 1) * FCCD
        return r - FCCD + s

    #taper_top_outer
    def active_volume_taper_bottom_outer(self, FCCD):
        h_o_new = Detector.S_corner_height_shift(self.h_o, self.r_o, self.R_c, FCCD)
        r_o_new = Detector.S_corner_radius_shift(self.h_o, self.r_o, self.R_c, FCCD)
        vol = Detector.S_taper_bottom_outer_volume(h_o_new, r_o_new, self.R_c - FCCD)
        return vol

    #taper_top_outer
    def active_volume_taper_top_outer(self, FCCD):
        h_u_new = Detector.S_corner_height_shift(self.h_u, self.r_u, self.R_c, FCCD)
        r_u_new = Detector.S_corner_radius_shift(self.h_u, self.r_u, self.R_c, FCCD)
        vol = Detector.S_taper_top_outer_volume(h_u_new, r_u_new, self.R_c - FCCD)
        return vol

    #taper_top_inner
    def active_volume_taper_top_inner(self, FCCD, FCCD_bore):
        r_i_new = Detector.S_inner_corner_radius_shift(self.h_i, self.r_w, self.r_i, FCCD, FCCD_bore)
        if (self.h_w != self.h_i):
            h_i_new = Detector.S_corner_height_shift(self.h_i, self.r_w, self.r_i, FCCD_bore)
            r_w_new = self.r_w + FCCD_bore
            h_w_new = self.h_w + FCCD_bore - FCCD
            vol = Detector.S_taper_top_inner_volume(h_i_new, r_i_new, h_w_new, r_w_new)
            vol += Detector.S_well_volume(self.h_w - h_i_new + FCCD_bore, r_w_new)
        else:
            h_i_new = self.h_i + FCCD_bore - FCCD
            r_w_new = Detector.S_corner_radius_shift(self.h_i, self.r_w, self.r_i, FCCD_bore)
            vol = Detector.S_taper_top_inner_volume(h_i_new, r_i_new, h_i_new, r_w_new)
        return vol


    #topgroove
    def active_volume_topgroove(self, FCCD, FCCD_bore):
        vol = Detector.S_topgroove_volume(self.h_topg, self.r_topg + FCCD, self.r_w + FCCD_bore)
        return vol

    #multiradius
    def active_volume_multiradius(self, FCCD):
        h_b_new = Detector.S_corner_height_shift(self.h_t, self.r_b, self.R_c, FCCD)
        vol = Detector.S_multiradius_volume(h_b_new, self.h_t, self.r_b - FCCD, self.R_c - FCCD)
        return vol

    def active_volume_bottom_crack(self, FCCD):
        h_cr_new = Detector.S_corner_height_shift(self.h_cr, self.r_cr, self.R_c, FCCD)
        r_cr_new = Detector.S_crack_corner_radius_shift(self.h_cr, self.r_cr, self.R_c, FCCD)
        vol = Detector.S_bottom_crack_volume(h_cr_new, r_cr_new, self.R_c - FCCD)
        return vol


    def active_volume_comp(self, FCCD, FCCD_bore):

        active_volume = self.active_volume_standard(FCCD, FCCD_bore)

        #taper_bottom_outer
        if any(value!=0 for value in [self.h_o]):
            h_o_new = Detector.S_corner_height_shift(self.h_o, self.r_o, self.R_c, FCCD)
            active_volume -= Detector.S_cylinder_volume(h_o_new, self.R_c - FCCD)   #cut the bottom side of the cylinder
            active_volume += self.active_volume_taper_bottom_outer(FCCD)   #add the taper bottom side of the cylinder

        #taper_top_outer
        if any(value!=0 for value in [self.h_u]):
            h_u_new = Detector.S_corner_height_shift(self.h_u, self.r_u, self.R_c, FCCD)
            active_volume -= Detector.S_cylinder_volume(h_u_new, self.R_c - FCCD)   #cut the upper side of the cylinder
            active_volume += self.active_volume_taper_top_outer(FCCD)   #add the taper upper side of the cylinder

        #taper_top_inner
        if any(value!=0 for value in [self.h_i]):
            active_volume += Detector.S_well_volume(self.h_w + FCCD_bore - FCCD, self.r_w + FCCD_bore) #add the well back
            active_volume -= self.active_volume_taper_top_inner(FCCD, FCCD_bore)

        #topgroove
        if any(value!=0 for value in [self.h_topg, self.r_topg]):
            active_volume -= self.active_volume_topgroove(FCCD, FCCD_bore)

        #multiradius
        if any(value!=0 for value in [self.h_b, self.h_t]):
            h_b_new = Detector.S_corner_height_shift(self.h_t, self.r_b, self.R_c, FCCD)
            active_volume -= Detector.S_cylinder_volume(h_b_new + self.h_t, self.R_c - FCCD)
            active_volume += self.active_volume_multiradius(FCCD)

        if any(value!=0 for value in [self.h_cr, self.r_cr]):
            active_volume -= self.active_volume_bottom_crack(FCCD)

        # calculate active volume of detector in cm3
        active_volume = 0.001 * active_volume
        return active_volume

        #calculate ratio
    def ratio(self,FCCD, FCCD_bore):
        ratio = self.active_volume_comp(FCCD, FCCD_bore) / self.volume_comp()
        return ratio