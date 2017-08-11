#!/usr/bin/env python

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import ephem
import sqlite3 as lite
import os
import argparse
import sys
from lsst.sims.utils import _hpid2RaDec, calcLmstLast, _raDec2Hpid
import healpy as hp
from astropy.coordinates import SkyCoord
from astropy import units as u


# Altitude and Azimuth of a single field at t (JD) in rad
def Fields_local_coordinate(Field_ra, Field_dec, t, Site):

    # date and time
    Site.date = t
    curr_obj = ephem.FixedBody()
    curr_obj._ra = Field_ra * np.pi / 180
    curr_obj._dec = Field_dec * np.pi / 180
    curr_obj.compute(Site)
    altitude = curr_obj.alt
    azimuth = curr_obj.az
    return altitude, azimuth

def stupidFast_RaDec2AltAz(ra, dec, mjd, lmst=None):
    """
    Convert Ra,Dec to Altitude and Azimuth.

    Coordinate transformation is killing performance. Just use simple equations to speed it up
    and ignore abberation, precesion, nutation, nutrition, etc.

    Parameters
    ----------
    ra : array_like
        RA, in radians.
    dec : array_like
        Dec, in radians. Must be same length as `ra`.
    lat : float
        Latitude of the observatory in radians.
    lon : float
        Longitude of the observatory in radians.
    mjd : float
        Modified Julian Date.

    Returns
    -------
    alt : numpy.array
        Altitude, same length as `ra` and `dec`. Radians.
    az : numpy.array
        Azimuth, same length as `ra` and `dec`. Radians.
    """
    lat=-0.517781017
    lon=-1.2320792

    if lmst is None:
        lmst, last = calcLmstLast(mjd, lon)
        lmst = lmst/12.*np.pi  # convert to rad
    ha = lmst-ra
    sindec = np.sin(dec)
    sinlat = np.sin(lat)
    coslat = np.cos(lat)
    sinalt = sindec*sinlat+np.cos(dec)*coslat*np.cos(ha)
    sinalt = inrange(sinalt)
    alt = np.arcsin(sinalt)
    cosaz = (sindec-np.sin(alt)*sinlat)/(np.cos(alt)*coslat)
    cosaz = inrange(cosaz)
    az = np.arccos(cosaz)
    signflip = np.where(np.sin(ha) > 0)
    az[signflip] = 2.*np.pi-az[signflip]
    return alt, az

def update_moon(t, Site):
    Moon = ephem.Moon()
    Site.date = t
    Moon.compute(Site)
    X, Y = AltAz2XY(Moon.alt, Moon.az)
    r = Moon.size / 3600 * np.pi / 180 * 2
    return X, Y, r, Moon.alt

def AltAz2XY(Alt, Az):
    X = np.cos(Alt) * np.cos(Az) * -1
    Y = np.cos(Alt) * np.sin(Az)
    #Y = Alt * 2/ np.pi
    #X = Az / (2*np.pi)
    return -1.*Y, -1.*X

def mutually_exclusive_regions(nside=64):
    indx = np.arange(hp.nside2npix(nside))
    all_true = np.ones(np.size(indx), dtype=bool)
    SCP_indx = is_SCP(nside)
    NES_indx = is_NES(nside)
    GP_indx = is_GP(nside)

    all_butWFD = SCP_indx + NES_indx + GP_indx
    GP_NES     = GP_indx + NES_indx

    WFD_indx = all_true - all_butWFD
    SCP_indx = SCP_indx - GP_NES*SCP_indx - NES_indx*SCP_indx
    NES_indx = NES_indx - GP_indx*SCP_indx

    return indx[SCP_indx], indx[NES_indx], indx[GP_indx], indx[WFD_indx]

def is_WFD(nside=64, dec_min=-60., dec_max=0.):
    """
    Define a wide fast deep region. Return a healpix map with WFD pixels as true.
    """
    ra, dec = ra_dec_hp_map(nside=nside)
    WFD_indx =((dec >= np.radians(dec_min)) & (dec <= np.radians(dec_max)))
    return WFD_indx


def is_SCP(nside=64, dec_max=-60.):
    """
    Define the South Celestial Pole region. Return a healpix map with SCP pixels as true.
    """
    ra, dec = ra_dec_hp_map(nside=nside)
    result = np.zeros(ra.size)
    good = np.where(dec < np.radians(dec_max))
    result[good] += 1

    SCP_indx = (result == 1)
    return SCP_indx


def is_NES(nside=64, width=15, dec_min=0., fill_gap=True):
    """
    Define the North Ecliptic Spur region. Return a healpix map with NES pixels as true.
    """
    ra, dec = ra_dec_hp_map(nside=nside)
    result = np.zeros(ra.size)
    coord = SkyCoord(ra=ra*u.rad, dec=dec*u.rad)
    eclip_lat = coord.barycentrictrueecliptic.lat.radian
    good = np.where((np.abs(eclip_lat) <= np.radians(width)) & (dec > dec_min))
    result[good] += 1

    if fill_gap:
        good = np.where((dec > np.radians(dec_min)) & (ra < np.radians(180)) &
                        (dec < np.radians(width)))
        result[good] = 1

    NES_indx = (result==1)

    return NES_indx


def is_GP(nside=64, center_width=10., end_width=4.,
                              gal_long1=70., gal_long2=290.):
    """
    Define the Galactic Plane region. Return a healpix map with GP pixels as true.
    """
    ra, dec = ra_dec_hp_map(nside=nside)
    result = np.zeros(ra.size)
    coord = SkyCoord(ra=ra*u.rad, dec=dec*u.rad)
    g_long, g_lat = coord.galactic.l.radian, coord.galactic.b.radian
    good = np.where((g_long < np.radians(gal_long1)) & (np.abs(g_lat) < np.radians(center_width)))
    result[good] += 1
    good = np.where((g_long > np.radians(gal_long2)) & (np.abs(g_lat) < np.radians(center_width)))
    result[good] += 1
    # Add tapers
    slope = -(np.radians(center_width)-np.radians(end_width))/(np.radians(gal_long1))
    lat_limit = slope*g_long+np.radians(center_width)
    outside = np.where((g_long < np.radians(gal_long1)) & (np.abs(g_lat) > np.abs(lat_limit)))
    result[outside] = 0
    slope = (np.radians(center_width)-np.radians(end_width))/(np.radians(360. - gal_long2))
    b = np.radians(center_width)-np.radians(360.)*slope
    lat_limit = slope*g_long+b
    outside = np.where((g_long > np.radians(gal_long2)) & (np.abs(g_lat) > np.abs(lat_limit)))
    result[outside] = 0

    GP_inx =(result==1)

    return GP_inx

def ra_dec_hp_map(nside=64):
    """
    Return all the RA,dec points for the centers of a healpix map
    """
    ra, dec = _hpid2RaDec(nside, np.arange(hp.nside2npix(nside)))

    return ra, dec
def inrange(inval, minimum=-1., maximum=1.):
    """
    Make sure values are within min/max
    """
    inval = np.array(inval)
    below = np.where(inval < minimum)
    inval[below] = minimum
    above = np.where(inval > maximum)
    inval[above] = maximum
    return inval

def RaDec2region(ra, dec, nside):
    SCP_indx, NES_indx, GP_indx, WFD_indx = mutually_exclusive_regions(nside)

    indices = _raDec2Hpid(nside, np.radians(ra), np.radians(dec))
    result = np.empty(np.size(indices), dtype = object)
    SCP = np.in1d(indices, SCP_indx)
    NES = np.in1d(indices,NES_indx)
    GP  = np.in1d(indices,GP_indx)
    WFD = np.in1d(indices,WFD_indx)

    result[SCP] = 'SCP'
    result[NES] = 'NES'
    result[GP]  = 'GP'
    result[WFD] = 'WFD'

    return result

def visualize(night, file_name, PlotID = 1,FPS = 15,Steps = 20,MP4_quality = 300, Name = "Visualization.mp4", showClouds = False, nside = 64, fancy_slow = False):

    Site            = ephem.Observer()
    Site.lon        = -1.2320792
    Site.lat        = -0.517781017
    Site.elevation  = 2650
    Site.pressure   = 0.
    Site.horizon    = 0.

    # create pix
    hpid = np.arange(hp.nside2npix(16))
    ra, dec = _hpid2RaDec(16, hpid)
    SCP_indx, NES_indx, GP_indx, WFD_indx = mutually_exclusive_regions(nside = 16)
    DD_indx = []

    #Initialize date and time
    last_night = night-1

    #Connect to the History data base
    con = lite.connect(file_name)
    cur = con.cursor()

    # Prepare to save in MP4 format
    FFMpegWriter = animation.writers['ffmpeg']
    metadata = dict(title='LSST Simulation', artist='Elahe', comment='Test')
    writer = FFMpegWriter(fps=FPS, metadata=metadata)

    # Initialize plot
    Fig = plt.figure()
    if PlotID == 1:
        ax = plt.subplot(111, axisbg = 'black')
    if PlotID == 2:
        ax = plt.subplot(211, axisbg = 'black')

    unobserved, Observed_lastN, Obseved_toN,\
    WFD, GP, NE, SE, DD,\
    ToN_History_line,\
    uu,gg,rr,ii,zz,yy,\
    last_10_History_line,\
    Horizon, airmass_horizon, S_Pole,\
    LSST,\
    Clouds\
        = ax.plot([], [], '*',[], [], '*',[], [], '*',
                  [], [], '*',[], [], '*',[], [], '*', [], [], '*', [], [], '*', # regions
                  [], [], '*',
                  [], [], '*',[], [], '*',[], [], '*',
                  [], [], '*',[], [], '*',[], [], '*',
                  [], [], '-',
                  [], [], '-',[], [], '-',[], [], 'D',
                  [], [], 'o',
                  [], [], 'o')

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal', adjustable = 'box')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Coloring
    Horizon.set_color('white'); airmass_horizon.set_color('red')
    S_Pole.set_markersize(3);   S_Pole.set_markerfacecolor('red')
    star_size = 4

    unobserved.set_color('dimgray');        unobserved.set_markersize(star_size)
    Observed_lastN.set_color('blue');       Observed_lastN.set_markersize(star_size)
    Obseved_toN.set_color('chartreuse');    Obseved_toN.set_markersize(0)

    # regions
    WFD.set_color('black');               WFD.set_alpha(0.3)
    SE.set_color('orange');                  SE.set_alpha(0.3)
    NE.set_color('orange');                   NE.set_alpha(0.3)
    GP.set_color('orange');                    GP.set_alpha(0.3)
    DD.set_color('orange');                  DD.set_alpha(0.3); DD.set_markersize(7)

    # filters
    uu.set_color('purple'); gg.set_color('green'); rr.set_color('red')
    ii.set_color('orange'); zz.set_color('pink');   yy.set_color('deeppink')

    # clouds
    Clouds.set_color('white');              Clouds.set_markersize(10)
    Clouds.set_alpha(0.2);                  Clouds.set_markeredgecolor(None)

    ToN_History_line.set_color('orange');   ToN_History_line.set_lw(.5)
    last_10_History_line.set_color('gray');  last_10_History_line.set_lw(.5)

    LSST.set_color('red'); LSST.set_markersize(8)

    if PlotID == 2:
        freqAX = plt.subplot(212)
        cur.execute('SELECT N_visit, Last_visit, Second_last_visit, Third_last_visit, Fourth_last_visit From FieldsStatistics')
        row = cur.fetchall()
        N_visit     = [x[0] for x in row]
        Last_visit   = [x[1] for x in row]
        Second_last_visit = [x[2] for x in row]
        Third_last_visit  = [x[3] for x in row]
        Fourth_last_visit = [x[4] for x in row]

        initHistoricalcoverage = N_visit
        for index, id in enumerate(All_Fields):
            if Last_visit[index] > toN_start:
                initHistoricalcoverage[index] -= 1
                if Second_last_visit[index] > toN_start:
                    initHistoricalcoverage[index] -= 1
                    if Third_last_visit > toN_start:
                        initHistoricalcoverage[index] -= 1



        covering,current_cover = freqAX.plot(All_Fields[0],initHistoricalcoverage,'-',[],[],'o')

        freqAX.set_xlim(0,N_Fields)
        freqAX.set_ylim(0,np.max(initHistoricalcoverage)+5)
        covering.set_color('chartreuse');   covering.set_markersize(2)
        current_cover.set_color('red');     current_cover.set_markersize(6)




    # Figure labels and fixed elements
    Phi = np.arange(0, 2* np.pi, 0.05)
    Horizon.set_data(1.01*np.cos(Phi), 1.01*np.sin(Phi))
    ax.text(-1.3, 0, 'East', color = 'white', fontsize = 7)
    ax.text(1.15, 0 ,'West', color = 'white', fontsize = 7)
    ax.text( 0, 1.1, 'North', color = 'white', fontsize = 7)
    airmass_horizon.set_data(np.cos(np.pi/4) * np.cos(Phi), np.cos(np.pi/4) *  np.sin(Phi))
    ax.text(-.3, 0.6, 'airmass horizon', color = 'white', fontsize = 5, fontweight = 'bold')
    Alt, Az = Fields_local_coordinate(180, -90, 59581.0381944435, Site)
    x, y = AltAz2XY(Alt,Az)
    S_Pole.set_data(x, y)
    ax.text(x+ .05, y, 'S-Pole', color = 'white', fontsize = 7)
    DD_indicator = ax.text(-1.4,1.3, 'Deep Drilling Observation', color = 'red', fontsize = 9, visible = False)
    WFD_indicator = ax.text(-1.4,1.3, 'Wide Fast Deep Observation', color = 'white', fontsize = 9, visible = False)
    GP_indicator = ax.text(-1.4,1.3, 'Galactic Plane Observation', color = 'white', fontsize = 9, visible = False)
    NES_indicator = ax.text(-1.4,1.3, 'Notrh Ecliptic Spur Observation', color = 'white', fontsize = 9, visible = False)
    SCP_indicator = ax.text(-1.4,1.3, 'South Celestial Pole Observation', color = 'white', fontsize = 9, visible = False)


    # Observed last night fields
    cur.execute('SELECT RA, dec, mjd, filter FROM SummaryAllProps WHERE night=%i' % (last_night))
    row = cur.fetchall()
    if row is not None:
        F1 = [x[0:2] for x in row]
    else:
        F1 = []

    # Tonight observation path
    cur.execute('SELECT RA, dec, mjd, filter FROM SummaryAllProps WHERE night=%i' % (night))
    row = cur.fetchall()
    if row[0][0] is not None:
        F2 = [x[0:2] for x in row]
        RA = np.asanyarray([x[0] for x in row]); DEC = np.asarray([x[1] for x in row])
        F2_timing = [x[2] for x in row]
        F2_filtering = [x[3] for x in row]
        F2_region = RaDec2region(RA, DEC, nside)
    else:
        F2 = []; F2_timing = []; F2_filtering = []; F2_region = []

    # Sky elements
    Moon = Circle((0, 0), 0, color = 'silver', zorder = 3)
    ax.add_patch(Moon)
    Moon_text = ax.text([], [], 'Moon', color = 'white', fontsize = 7)

    doff = ephem.Date(0)-ephem.Date('1858/11/17')
    with writer.saving(Fig, Name, MP4_quality) :
        for t in np.linspace(F2_timing[0], F2_timing[-1], num = Steps):

            # Find the index of the current time
            time_index = 0
            while t > F2_timing[time_index]:
                time_index += 1
            if showClouds:
                Slot_n = 0
                while t > Time_slots[Slot_n]:
                    Slot_n += 1

            visit_index = 0
            visited_field = 0
            visit_index_u = 0; visit_index_g = 0; visit_index_r = 0; visit_index_i = 0; visit_index_z = 0; visit_index_y = 0
            visit_filter  = 'r'


            # Object fields: F1)Observed last night F2)Observed tonight F3)Unobserved F4)Covered by clouds
            F1_X = []; F1_Y = []; F2_X = []; F2_Y = []; F3_X = []; F3_Y = []; F4_X = []; F4_Y = []
            # Filter coloring for tonight observation
            U_X = []; U_Y = []; G_X = []; G_Y = []; R_X = []; R_Y = []; I_X = []; I_Y = []; Z_X = []; Z_Y = []; Y_X = []; Y_Y = []
            # Coloring different proposals
            WFD_X = []; WFD_Y = []; NE_X = []; NE_Y = []; SE_X = []; SE_Y = []; GP_X = []; GP_Y = []; DD_X = []; DD_Y = []

            # F1  coordinate:
            for i in F1:
                Alt, Az = Fields_local_coordinate(i[0], i[1], t-doff, Site)
                if Alt > 0:
                    X, Y    = AltAz2XY(Alt,Az)
                    F1_X.append(X); F1_Y.append(Y)

            # F2  coordinate:
            for i,tau,filter in zip(F2, F2_timing, F2_filtering):
                Alt, Az = Fields_local_coordinate(i[0], i[1], t-doff, Site)
                if Alt > 0:
                    X, Y    = AltAz2XY(Alt,Az)
                    F2_X.append(X); F2_Y.append(Y)
                    if filter == 'u':
                        U_X.append(X); U_Y.append(Y)
                        if t >= tau:
                            visit_index_u = len(U_X) -1
                    elif filter == 'g':
                        G_X.append(X); G_Y.append(Y)
                        if t >= tau:
                            visit_index_g = len(G_Y) -1
                    elif filter == 'r':
                        R_X.append(X); R_Y.append(Y)
                        if t >= tau:
                            visit_index_r = len(R_Y) -1
                    elif filter == 'i':
                        I_X.append(X); I_Y.append(Y)
                        if t >= tau:
                            visit_index_i = len(I_Y) -1
                    elif filter == 'z':
                        Z_X.append(X); Z_Y.append(Y)
                        if t >= tau:
                            visit_index_z = len(Z_Y) -1
                    elif filter == 'y':
                        Y_X.append(X); Y_Y.append(Y)
                        if t >= tau:
                            visit_index_y = len(Y_Y) -1

                    if t >= tau:
                        visit_index = len(F2_X) -1
                        visited_field = i
                        visit_filter  = filter

            # F3  coordinate:
            if fancy_slow:
                Alt, Az = stupidFast_RaDec2AltAz(ra, dec, t)
                for al, az,i in zip(Alt,Az,hpid):
                    if al > 0:
                        X, Y    = AltAz2XY(al,az)
                        F3_X.append(X); F3_Y.append(Y)
                        if i in DD_indx:
                            DD_X.append(X); DD_Y.append(Y)
                        elif i in WFD_indx:
                            WFD_X.append(X); WFD_Y.append(Y)
                        elif i in GP_indx:
                            GP_X.append(X); GP_Y.append(Y)
                        elif i in NES_indx:
                            NE_X.append(X); NE_Y.append(Y)
                        elif i in SCP_indx:
                            SE_X.append(X); SE_Y.append(Y)

            # F4 coordinates
            if showClouds:
                for i in range(0,N_Fields):
                    if All_Cloud_cover[Slot_n,i] == 2 or All_Cloud_cover[Slot_n,i] == 1 or All_Cloud_cover[Slot_n,i] == -1:
                        Alt, Az = Fields_local_coordinate(i[0], i[1], t-doff, Site)
                    if Alt > 0:
                        X, Y    = AltAz2XY(Alt,Az)
                        F4_X.append(X); F4_Y.append(Y)


            # Update plot
            unobserved.set_data([F3_X,F3_Y])
            Observed_lastN.set_data([F1_X,F1_Y])
            Obseved_toN.set_data([F2_X[0:visit_index],F2_Y[0:visit_index]])

            # filters
            uu.set_data([U_X[0:visit_index_u],U_Y[0:visit_index_u]]); gg.set_data([G_X[0:visit_index_g],G_Y[0:visit_index_g]])
            rr.set_data([R_X[0:visit_index_r],R_Y[0:visit_index_r]]); ii.set_data([I_X[0:visit_index_i],I_Y[0:visit_index_i]])
            zz.set_data([Z_X[0:visit_index_z],Z_Y[0:visit_index_z]]); yy.set_data([Y_X[0:visit_index_y],Y_Y[0:visit_index_y]])

            ToN_History_line.set_data([F2_X[0:visit_index], F2_Y[0:visit_index]])
            last_10_History_line.set_data([F2_X[visit_index - 10: visit_index], F2_Y[visit_index - 10: visit_index]])

            # telescope position and color
            LSST.set_data([F2_X[visit_index],F2_Y[visit_index]])
            if visit_filter == 'u':
                LSST.set_color('purple')
            if visit_filter == 'g':
                LSST.set_color('green')
            if visit_filter == 'r':
                LSST.set_color('red')
            if visit_filter == 'i':
                LSST.set_color('orange')
            if visit_filter == 'z':
                LSST.set_color('pink')
            if visit_filter == 'y':
                LSST.set_color('deeppink')

            Clouds.set_data([F4_X,F4_Y])

            # regions
            WFD.set_data([WFD_X, WFD_Y]); DD.set_data([DD_X,DD_Y]); NE.set_data([NE_X, NE_Y]); SE.set_data([SE_X, SE_Y])
            GP.set_data([GP_X, GP_Y])


            # Update Moon
            X, Y, r, alt = update_moon(t, Site)
            Moon.center = X, Y
            Moon.radius = r
            if alt > 0:
                #Moon.set_visible(True)
                Moon_text.set_visible(True)
                Moon_text.set_x(X+.002); Moon_text.set_y(Y+.002)
            else :
                Moon.set_visible(False)
                Moon_text.set_visible(False)

            #Update coverage
            if PlotID == 2:
                Historicalcoverage = np.zeros(N_Fields)
                for i,tau in zip(F2, F2_timing):
                    if tau <= t:
                        Historicalcoverage[i -1] += 1
                    else:
                        break
                tot = Historicalcoverage + initHistoricalcoverage
                current_cover.set_data(visited_field -1,tot[visited_field -1])
                covering.set_data(All_Fields[0], tot)

            #Update indicators of the proposal
            if F2_region[time_index]== 'DD':
                DD_indicator.set_visible(True)
            else:
                DD_indicator.set_visible(False)
            if F2_region[time_index]== 'WFD':
                WFD_indicator.set_visible(True)
            else:
                WFD_indicator.set_visible(False)
            if F2_region[time_index]== 'GP':
                GP_indicator.set_visible(True)
            else:
                GP_indicator.set_visible(False)
            if F2_region[time_index]== 'NES':
                NES_indicator.set_visible(True)
            else:
                NES_indicator.set_visible(False)
            if F2_region[time_index]== 'SCP':
                SCP_indicator.set_visible(True)
            else:
                SCP_indicator.set_visible(False)

            #Observation statistics
            leg = plt.legend([Observed_lastN, Obseved_toN],
                       ['Visited last night', time_index])
            for l in leg.get_texts():
                l.set_fontsize(6)
            date = t
            Fig.suptitle('Top view of the LSST site on {}, GMT'.format(date))


            '''
            # progress
            perc= int(100*(t - t_start)/(t_end - t_start))
            if perc <= 100:
                print('{} %'.format(perc))
            else:
                print('100 %')
            '''
            #Save current frame
            writer.grab_frame()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Generate an animation of number of nights from a simulation database")
    parser.add_argument("file_name", type=str, help="sqlite database")
    parser.add_argument("--night", type=int, default=0, help="the night to start on")
    parser.add_argument("--n_nights", type=int, default=1, help="number of nights to animate")
    parser.add_argument("--fancy_slow", type=bool, default=False, help="shows sky regions but runs slowly")

    args = parser.parse_args()
    file_name = args.file_name

    n_nights = args.n_nights  # number of the nights to be scheduled starting from 1st Jan. 2021

    for i in range(args.night, n_nights + args.night):
        night = i

        # create animation
        FPS = 10            # Frame per second
        Steps = 100          # Simulation steps
        MP4_quality = 300   # MP4 size and quality

        PlotID = 1        # 1 for one Plot, 2 for including covering pattern
        visualize(night, file_name, PlotID ,FPS, Steps, MP4_quality, 'LSST1plot{}.mp4'.format(i + 1), showClouds= False, fancy_slow=args.fancy_slow)

