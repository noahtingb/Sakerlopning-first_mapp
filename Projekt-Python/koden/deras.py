from __future__ import absolute_import
import numpy as np


def Kside_veg_v2019a(radI, radD, radG, shadow, svfS, svfW, svfN, svfE, svfEveg, svfSveg, svfWveg, svfNveg, azimuth,
                     altitude, psi, t, albedo, F_sh, KupE, KupS, KupW, KupN, cyl, lv, ani, diffsh, rows, cols):
    # New reflection equation 2012-05-25
    vikttot = 4.4897
    aziE = azimuth + t
    aziS = azimuth - 90 + t
    aziW = azimuth - 180 + t
    aziN = azimuth - 270 + t
    deg2rad = np.pi / 180
    KsideD = np.zeros((rows, cols))

    # Direct radiation ###
    if cyl == 1:  # Kside with cylinder ###
        KsideI = shadow * radI * np.cos(altitude * deg2rad)
        KeastI = 0
        KsouthI = 0
        KwestI = 0
        KnorthI = 0
    else:  # Kside with weights ###
        if azimuth > (360 - t) or azimuth <= (180 - t):
            KeastI = radI * shadow * np.cos(altitude * deg2rad) * np.sin(aziE * deg2rad)
        else:
            KeastI = 0
        if azimuth > (90 - t) and azimuth <= (270 - t):
            KsouthI = radI * shadow * np.cos(altitude * deg2rad) * np.sin(aziS * deg2rad)
        else:
            KsouthI = 0
        if azimuth > (180 - t) and azimuth <= (360 - t):
            KwestI = radI * shadow * np.cos(altitude * deg2rad) * np.sin(aziW * deg2rad)
        else:
            KwestI = 0
        if azimuth <= (90 - t) or azimuth > (270 - t):
            KnorthI = radI * shadow * np.cos(altitude * deg2rad) * np.sin(aziN * deg2rad)
        else:
            KnorthI = 0

        KsideI = shadow * 0

    ### Diffuse and reflected radiation ###
    [viktveg, viktwall] = Kvikt_veg(svfE, svfEveg, vikttot)
    svfviktbuvegE = (viktwall + (viktveg) * (1 - psi))

    [viktveg, viktwall] = Kvikt_veg(svfS, svfSveg, vikttot)
    svfviktbuvegS = (viktwall + (viktveg) * (1 - psi))

    [viktveg, viktwall] = Kvikt_veg(svfW, svfWveg, vikttot)
    svfviktbuvegW = (viktwall + (viktveg) * (1 - psi))

    [viktveg, viktwall] = Kvikt_veg(svfN, svfNveg, vikttot)
    svfviktbuvegN = (viktwall + (viktveg) * (1 - psi))

    ### Anisotropic Diffuse Radiation after Perez et al. 1993 ###
    if ani == 1:

        aniAlt = lv[0][:, 0]
        aniAzi = lv[0][:, 1]
        aniLum = lv[0][:, 2]

        phiVar = np.zeros((145, 1))

        radTot = np.zeros(1)

        for ix in range(0, 145):  # Azimuth delta
            if ix < 60:
                aziDel = 12
            elif ix >= 60 and ix < 108:
                aziDel = 15
            elif ix >= 108 and ix < 126:
                aziDel = 20
            elif ix >= 126 and ix < 138:
                aziDel = 30
            elif ix >= 138 and ix < 144:
                aziDel = 60
            elif ix == 144:
                aziDel = 360

            phiVar[ix] = (aziDel * deg2rad) * (np.sin((aniAlt[ix] + 6) * deg2rad) - np.sin(
                (aniAlt[ix] - 6) * deg2rad))  # Solid angle / Steradian

            radTot = radTot + (
                        aniLum[ix] * phiVar[ix] * np.sin(aniAlt[ix] * deg2rad))  # Radiance fraction normalization

        lumChi = (aniLum * radD) / radTot  # Radiance fraction normalization

        if cyl == 1:
            for idx in range(0, 145):
                anglIncC = np.cos(aniAlt[idx] * deg2rad) * np.cos(0) * np.sin(np.pi / 2) + np.sin(
                    aniAlt[idx] * deg2rad) * np.cos(
                    np.pi / 2)  # Angle of incidence, np.cos(0) because cylinder - always perpendicular
                KsideD = KsideD + diffsh[idx] * lumChi[idx] * anglIncC * phiVar[idx]  # Diffuse vertical radiation
            Keast = (albedo * (svfviktbuvegE * (radG * (1 - F_sh) + radD * F_sh)) + KupE) * 0.5
            Ksouth = (albedo * (svfviktbuvegS * (radG * (1 - F_sh) + radD * F_sh)) + KupS) * 0.5
            Kwest = (albedo * (svfviktbuvegW * (radG * (1 - F_sh) + radD * F_sh)) + KupW) * 0.5
            Knorth = (albedo * (svfviktbuvegN * (radG * (1 - F_sh) + radD * F_sh)) + KupN) * 0.5
        else:  # Box
            diffRadE = np.zeros((rows, cols));
            diffRadS = np.zeros((rows, cols));
            diffRadW = np.zeros((rows, cols));
            diffRadN = np.zeros((rows, cols))

            for idx in range(0, 145):
                if aniAzi[idx] <= (180):
                    anglIncE = np.cos(aniAlt[idx] * deg2rad) * np.cos((90 - aniAzi[idx]) * deg2rad) * np.sin(
                        np.pi / 2) + np.sin(
                        aniAlt[idx] * deg2rad) * np.cos(np.pi / 2)
                    diffRadE = diffRadE + diffsh[idx] * lumChi[idx] * anglIncE * phiVar[idx]  # * 0.5

                if aniAzi[idx] > (90) and aniAzi[idx] <= (270):
                    anglIncS = np.cos(aniAlt[idx] * deg2rad) * np.cos((180 - aniAzi[idx]) * deg2rad) * np.sin(
                        np.pi / 2) + np.sin(
                        aniAlt[idx] * deg2rad) * np.cos(np.pi / 2)
                    diffRadS = diffRadS + diffsh[idx] * lumChi[idx] * anglIncS * phiVar[idx]  # * 0.5

                if aniAzi[idx] > (180) and aniAzi[idx] <= (360):
                    anglIncW = np.cos(aniAlt[idx] * deg2rad) * np.cos((270 - aniAzi[idx]) * deg2rad) * np.sin(
                        np.pi / 2) + np.sin(
                        aniAlt[idx] * deg2rad) * np.cos(np.pi / 2)
                    diffRadW = diffRadW + diffsh[idx] * lumChi[idx] * anglIncW * phiVar[idx]  # * 0.5

                if aniAzi[idx] > (270) or aniAzi[idx] <= (90):
                    anglIncN = np.cos(aniAlt[idx] * deg2rad) * np.cos((0 - aniAzi[idx]) * deg2rad) * np.sin(
                        np.pi / 2) + np.sin(
                        aniAlt[idx] * deg2rad) * np.cos(np.pi / 2)
                    diffRadN = diffRadN + diffsh[idx] * lumChi[idx] * anglIncN * phiVar[idx]  # * 0.5

            KeastDG = diffRadE + (albedo * (svfviktbuvegE * (radG * (1 - F_sh) + radD * F_sh)) + KupE) * 0.5
            Keast = KeastI + KeastDG

            KsouthDG = diffRadS + (albedo * (svfviktbuvegS * (radG * (1 - F_sh) + radD * F_sh)) + KupS) * 0.5
            Ksouth = KsouthI + KsouthDG

            KwestDG = diffRadW + (albedo * (svfviktbuvegW * (radG * (1 - F_sh) + radD * F_sh)) + KupW) * 0.5
            Kwest = KwestI + KwestDG

            KnorthDG = diffRadN + (albedo * (svfviktbuvegN * (radG * (1 - F_sh) + radD * F_sh)) + KupN) * 0.5
            Knorth = KnorthI + KnorthDG

    else:
        KeastDG = (radD * (1 - svfviktbuvegE) + albedo * (
                svfviktbuvegE * (radG * (1 - F_sh) + radD * F_sh)) + KupE) * 0.5
        Keast = KeastI + KeastDG

        KsouthDG = (radD * (1 - svfviktbuvegS) + albedo * (
                svfviktbuvegS * (radG * (1 - F_sh) + radD * F_sh)) + KupS) * 0.5
        Ksouth = KsouthI + KsouthDG

        KwestDG = (radD * (1 - svfviktbuvegW) + albedo * (
                svfviktbuvegW * (radG * (1 - F_sh) + radD * F_sh)) + KupW) * 0.5
        Kwest = KwestI + KwestDG

        KnorthDG = (radD * (1 - svfviktbuvegN) + albedo * (
                svfviktbuvegN * (radG * (1 - F_sh) + radD * F_sh)) + KupN) * 0.5
        Knorth = KnorthI + KnorthDG

    return Keast, Ksouth, Kwest, Knorth, KsideI, KsideD


def Kvikt_veg(svf,svfveg,vikttot):

    # Least
    viktwall=(vikttot-(63.227*svf**6-161.51*svf**5+156.91*svf**4-70.424*svf**3+16.773*svf**2-0.4863*svf))/vikttot
    
    svfvegbu=(svfveg+svf-1)  # Vegetation plus buildings
    viktveg=(vikttot-(63.227*svfvegbu**6-161.51*svfvegbu**5+156.91*svfvegbu**4-70.424*svfvegbu**3+16.773*svfvegbu**2-0.4863*svfvegbu))/vikttot
    viktveg=viktveg-viktwall
    
    return viktveg,viktwall


import numpy as np
from copy import deepcopy

# This function combines the patches from Tregenza (1987)/Robinson & Stone (2004) and the approach by Unsworth & Monteith
# to calculate emissivities of the different parts of the sky vault.

def Lcyl(esky,L_lv, Ta):

    SBC = 5.67051e-8                    # Stefan-Boltzmann's Constant

    deg2rad = np.pi / 180               # Degrees to radians

    skyalt, skyalt_c = np.unique(L_lv[0][:, 0], return_counts=True)   # Unique altitudes in lv, i.e. unique altitude for the patches
    skyzen = 90-skyalt                  # Unique zeniths for the patches

    cosskyzen = np.cos(skyzen * deg2rad)# Cosine of the zenith angles
    sinskyzen = np.sin(skyzen * deg2rad)  # Cosine of the zenith angles

    a_c = 0.67                          # Constant??
    b_c = 0.094                         # Constant??

    ln_u_prec = esky/b_c-a_c/b_c-0.5    # Natural log of the reduced depth of precipitable water
    u_prec = np.exp(ln_u_prec)          # Reduced depth of precipitable water

    owp = u_prec/cosskyzen              # Optical water depth

    log_owp = np.log(owp)               # Natural log of optical water depth

    esky_p = a_c+b_c*log_owp            # Emissivity of each zenith angle, i.e. the zenith angle of each patch

    lsky_p = esky_p * SBC * ((Ta + 273.15) ** 4)    # Longwave radiation of the sky at different altitudes

    p_alt = L_lv[0][:,0]                # Altitudes of the Robinson & Stone patches

    # Calculation of steradian for each patch
    sr = np.zeros((p_alt.shape[0]))
    for i in range(p_alt.shape[0]):
        if skyalt_c[skyalt == p_alt[i]] > 1:
            sr[i] = ((360 / skyalt_c[skyalt == p_alt[i]]) * deg2rad) * (np.sin((p_alt[i] + p_alt[0]) * deg2rad) \
            - np.sin((p_alt[i] - p_alt[0]) * deg2rad))  # Solid angle / Steradian
        else:
            sr[i] = ((360 / skyalt_c[skyalt == p_alt[i]]) * deg2rad) * (np.sin((p_alt[i]) * deg2rad) \
                - np.sin((p_alt[i-1] + p_alt[0]) * deg2rad))  # Solid angle / Steradian

    sr_h = sr * np.cos((90 - p_alt) * deg2rad)  # Horizontal
    sr_v = sr * np.sin((90 - p_alt) * deg2rad)  # Vertical

    sr_h_w = sr_h / np.sum(sr_h)                # Horizontal weight
    sr_v_w = sr_v / np.sum(sr_v)                # Vertical weight

    sr_w = sr / np.sum(sr)

    Lside = np.zeros((p_alt.shape[0]))
    Ldown = np.zeros((p_alt.shape[0]))
    Lsky_t = np.zeros((p_alt.shape[0]))

    # Estimating longwave radiation for each patch to a horizontal or vertical surface
    for idx in skyalt:
        Ltemp = lsky_p[skyalt == idx]
        Lsky_t[p_alt == idx] = (Ltemp / skyalt_c[skyalt == idx])
        Ldown[p_alt == idx] = Ltemp * sr_h_w[p_alt == idx]          # Longwave radiation from each patch on a horizontal surface
        Lside[p_alt == idx] = Ltemp * sr_v_w[p_alt == idx]          # Longwave radiation from each patch on a vertical surface

    Lsky = deepcopy(L_lv)
    Lsky[0][:,2] = Ldown

    test = 0

    return Ldown, Lside, Lsky

import numpy as np

def Lside_veg_v2015a(svfS,svfW,svfN,svfE,svfEveg,svfSveg,svfWveg,svfNveg,svfEaveg,svfSaveg,svfWaveg,svfNaveg,azimuth,altitude,Ta,Tw,SBC,ewall,Ldown,esky,t,F_sh,CI,LupE,LupS,LupW,LupN):

    # This m-file is the current one that estimates L from the four cardinal points 20100414
    
    #Building height angle from svf
    svfalfaE=np.arcsin(np.exp((np.log(1-svfE))/2))
    svfalfaS=np.arcsin(np.exp((np.log(1-svfS))/2))
    svfalfaW=np.arcsin(np.exp((np.log(1-svfW))/2))
    svfalfaN=np.arcsin(np.exp((np.log(1-svfN))/2))
    
    vikttot=4.4897
    aziW=azimuth+t
    aziN=azimuth-90+t
    aziE=azimuth-180+t
    aziS=azimuth-270+t
    
    F_sh = 2*F_sh-1  #(cylindric_wedge scaled 0-1)
    
    c=1-CI
    Lsky_allsky = esky*SBC*((Ta+273.15)**4)*(1-c)+c*SBC*((Ta+273.15)**4)
    
    ## Least
    [viktveg, viktwall, viktsky, viktrefl] = Lvikt_veg(svfE, svfEveg, svfEaveg, vikttot)
    
    if altitude > 0:  # daytime
        alfaB=np.arctan(svfalfaE)
        betaB=np.arctan(np.tan((svfalfaE)*F_sh))
        betasun=((alfaB-betaB)/2)+betaB
        # betasun = np.arctan(0.5*np.tan(svfalfaE)*(1+F_sh)) #TODO This should be considered in future versions
        if (azimuth > (180-t))  and  (azimuth <= (360-t)):
            Lwallsun=SBC*ewall*((Ta+273.15+Tw*np.sin(aziE*(np.pi/180)))**4)*\
                viktwall*(1-F_sh)*np.cos(betasun)*0.5
            Lwallsh=SBC*ewall*((Ta+273.15)**4)*viktwall*F_sh*0.5
        else:
            Lwallsun=0
            Lwallsh=SBC*ewall*((Ta+273.15)**4)*viktwall*0.5
    else: #nighttime
        Lwallsun=0
        Lwallsh=SBC*ewall*((Ta+273.15)**4)*viktwall*0.5
    
    Lsky=((svfE+svfEveg-1)*Lsky_allsky)*viktsky*0.5
    Lveg=SBC*ewall*((Ta+273.15)**4)*viktveg*0.5
    Lground=LupE*0.5
    Lrefl=(Ldown+LupE)*(viktrefl)*(1-ewall)*0.5
    Least=Lsky+Lwallsun+Lwallsh+Lveg+Lground+Lrefl
    
    # clear alfaB betaB betasun Lsky Lwallsh Lwallsun Lveg Lground Lrefl viktveg viktwall viktsky
    
    ## Lsouth
    [viktveg,viktwall,viktsky,viktrefl]=Lvikt_veg(svfS,svfSveg,svfSaveg,vikttot)
    
    if altitude>0: # daytime
        alfaB=np.arctan(svfalfaS)
        betaB=np.arctan(np.tan((svfalfaS)*F_sh))
        betasun=((alfaB-betaB)/2)+betaB
        # betasun = np.arctan(0.5*np.tan(svfalfaS)*(1+F_sh))
        if (azimuth <= (90-t))  or  (azimuth > (270-t)):
            Lwallsun=SBC*ewall*((Ta+273.15+Tw*np.sin(aziS*(np.pi/180)))**4)*\
                viktwall*(1-F_sh)*np.cos(betasun)*0.5
            Lwallsh=SBC*ewall*((Ta+273.15)**4)*viktwall*F_sh*0.5
        else:
            Lwallsun=0
            Lwallsh=SBC*ewall*((Ta+273.15)**4)*viktwall*0.5
    else: #nighttime
        Lwallsun=0
        Lwallsh=SBC*ewall*((Ta+273.15)**4)*viktwall*0.5

    Lsky=((svfS+svfSveg-1)*Lsky_allsky)*viktsky*0.5
    Lveg=SBC*ewall*((Ta+273.15)**4)*viktveg*0.5
    Lground=LupS*0.5
    Lrefl=(Ldown+LupS)*(viktrefl)*(1-ewall)*0.5
    Lsouth=Lsky+Lwallsun+Lwallsh+Lveg+Lground+Lrefl
    
    # clear alfaB betaB betasun Lsky Lwallsh Lwallsun Lveg Lground Lrefl viktveg viktwall viktsky
    
    ## Lwest
    [viktveg,viktwall,viktsky,viktrefl]=Lvikt_veg(svfW,svfWveg,svfWaveg,vikttot)
    
    if altitude>0: # daytime
        alfaB=np.arctan(svfalfaW)
        betaB=np.arctan(np.tan((svfalfaW)*F_sh))
        betasun=((alfaB-betaB)/2)+betaB
        # betasun = np.arctan(0.5*np.tan(svfalfaW)*(1+F_sh))
        if (azimuth > (360-t))  or  (azimuth <= (180-t)):
            Lwallsun=SBC*ewall*((Ta+273.15+Tw*np.sin(aziW*(np.pi/180)))**4)*\
                viktwall*(1-F_sh)*np.cos(betasun)*0.5
            Lwallsh=SBC*ewall*((Ta+273.15)**4)*viktwall*F_sh*0.5
        else:
            Lwallsun=0
            Lwallsh=SBC*ewall*((Ta+273.15)**4)*viktwall*0.5
    else: #nighttime
        Lwallsun=0
        Lwallsh=SBC*ewall*((Ta+273.15)**4)*viktwall*0.5

    Lsky=((svfW+svfWveg-1)*Lsky_allsky)*viktsky*0.5
    Lveg=SBC*ewall*((Ta+273.15)**4)*viktveg*0.5
    Lground=LupW*0.5
    Lrefl=(Ldown+LupW)*(viktrefl)*(1-ewall)*0.5
    Lwest=Lsky+Lwallsun+Lwallsh+Lveg+Lground+Lrefl
    
    # clear alfaB betaB betasun Lsky Lwallsh Lwallsun Lveg Lground Lrefl viktveg viktwall viktsky
    
    ## Lnorth
    [viktveg,viktwall,viktsky,viktrefl]=Lvikt_veg(svfN,svfNveg,svfNaveg,vikttot)
    
    if altitude>0: # daytime
        alfaB=np.arctan(svfalfaN)
        betaB=np.arctan(np.tan((svfalfaN)*F_sh))
        betasun=((alfaB-betaB)/2)+betaB
        # betasun = np.arctan(0.5*np.tan(svfalfaN)*(1+F_sh))
        if (azimuth > (90-t))  and  (azimuth <= (270-t)):
            Lwallsun=SBC*ewall*((Ta+273.15+Tw*np.sin(aziN*(np.pi/180)))**4)*\
                viktwall*(1-F_sh)*np.cos(betasun)*0.5
            Lwallsh=SBC*ewall*((Ta+273.15)**4)*viktwall*F_sh*0.5
        else:
            Lwallsun=0
            Lwallsh=SBC*ewall*((Ta+273.15)**4)*viktwall*0.5
    else: #nighttime
        Lwallsun=0
        Lwallsh=SBC*ewall*((Ta+273.15)**4)*viktwall*0.5

    Lsky=((svfN+svfNveg-1)*Lsky_allsky)*viktsky*0.5
    Lveg=SBC*ewall*((Ta+273.15)**4)*viktveg*0.5
    Lground=LupN*0.5
    Lrefl=(Ldown+LupN)*(viktrefl)*(1-ewall)*0.5
    Lnorth=Lsky+Lwallsun+Lwallsh+Lveg+Lground+Lrefl

    # clear alfaB betaB betasun Lsky Lwallsh Lwallsun Lveg Lground Lrefl viktveg viktwall viktsky
    
    return Least,Lsouth,Lwest,Lnorth

import numpy as np

def Lside_veg_v2020a(svfS,svfW,svfN,svfE,svfEveg,svfSveg,svfWveg,svfNveg,svfEaveg,svfSaveg,svfWaveg,svfNaveg,azimuth,altitude,Ta,Tw,SBC,ewall,Ldown,esky,t,F_sh,CI,LupE,LupS,LupW,LupN,L_ani,Ldown_i):

    # This m-file is the current one that estimates L from the four cardinal points 20100414
    
    #Building height angle from svf
    svfalfaE=np.arcsin(np.exp((np.log(1-svfE))/2))
    svfalfaS=np.arcsin(np.exp((np.log(1-svfS))/2))
    svfalfaW=np.arcsin(np.exp((np.log(1-svfW))/2))
    svfalfaN=np.arcsin(np.exp((np.log(1-svfN))/2))
    
    vikttot=4.4897
    aziW=azimuth+t
    aziN=azimuth-90+t
    aziE=azimuth-180+t
    aziS=azimuth-270+t
    
    F_sh = 2*F_sh-1  #(cylindric_wedge scaled 0-1)
    
    c=1-CI
    Lsky_allsky = esky*SBC*((Ta+273.15)**4)*(1-c)+c*SBC*((Ta+273.15)**4)
    
    ## Least
    [viktveg, viktwall, viktsky, viktrefl] = Lvikt_veg(svfE, svfEveg, svfEaveg, vikttot)

    if altitude > 0:  # daytime
        alfaB=np.arctan(svfalfaE)
        betaB=np.arctan(np.tan((svfalfaE)*F_sh))
        betasun=((alfaB-betaB)/2)+betaB
        # betasun = np.arctan(0.5*np.tan(svfalfaE)*(1+F_sh)) #TODO This should be considered in future versions
        if (azimuth > (180-t))  and  (azimuth <= (360-t)):
            Lwallsun=SBC*ewall*((Ta+273.15+Tw*np.sin(aziE*(np.pi/180)))**4)*\
                viktwall*(1-F_sh)*np.cos(betasun)*0.5
            Lwallsh=SBC*ewall*((Ta+273.15)**4)*viktwall*F_sh*0.5
        else:
            Lwallsun=0
            Lwallsh=SBC*ewall*((Ta+273.15)**4)*viktwall*0.5
    else: #nighttime
        Lwallsun=0
        Lwallsh=SBC*ewall*((Ta+273.15)**4)*viktwall*0.5

    if L_ani == 1:
        #Lsky = Ldown * 0.5
        Lveg=SBC*ewall*((Ta+273.15)**4)*viktveg*0.5
        Lground=LupE*0.5
        Lrefl=(Ldown_i+LupE)*(viktrefl)*(1-ewall)*0.5
        Least=Lwallsun+Lwallsh+Lveg+Lground+Lrefl
        #Least = Lsky + Lwallsun + Lwallsh + Lveg + Lground + Lrefl
    else:
        Lsky = ((svfE + svfEveg - 1) * Lsky_allsky) * viktsky * 0.5
        Lveg = SBC * ewall * ((Ta + 273.15) ** 4) * viktveg * 0.5
        Lground = LupE * 0.5
        Lrefl = (Ldown + LupE) * (viktrefl) * (1 - ewall) * 0.5
        Least = Lsky + Lwallsun + Lwallsh + Lveg + Lground + Lrefl

    # print('East',Lveg[0,59], Lground[0,59], Lrefl[0,59], Least[0,59])

    Lsky = ((svfE + svfEveg - 1) * Lsky_allsky) * viktsky * 0.5
    Lveg = SBC * ewall * ((Ta + 273.15) ** 4) * viktveg * 0.5
    Lground = LupE * 0.5
    Lrefl = (Ldown_i + LupE) * (viktrefl) * (1 - ewall) * 0.5
    Least_i = Lsky + Lwallsun + Lwallsh + Lveg + Lground + Lrefl

    # print(Lsky[0,59],Lveg[0, 59], Lground[0, 59], Lrefl[0, 59], Least_i[0, 59])

    Lsky_t = Lsky

    # clear alfaB betaB betasun Lsky Lwallsh Lwallsun Lveg Lground Lrefl viktveg viktwall viktsky
    
    ## Lsouth
    [viktveg,viktwall,viktsky,viktrefl]=Lvikt_veg(svfS,svfSveg,svfSaveg,vikttot)
    
    if altitude>0: # daytime
        alfaB=np.arctan(svfalfaS)
        betaB=np.arctan(np.tan((svfalfaS)*F_sh))
        betasun=((alfaB-betaB)/2)+betaB
        # betasun = np.arctan(0.5*np.tan(svfalfaS)*(1+F_sh))
        if (azimuth <= (90-t))  or  (azimuth > (270-t)):
            Lwallsun=SBC*ewall*((Ta+273.15+Tw*np.sin(aziS*(np.pi/180)))**4)*\
                viktwall*(1-F_sh)*np.cos(betasun)*0.5
            Lwallsh=SBC*ewall*((Ta+273.15)**4)*viktwall*F_sh*0.5
        else:
            Lwallsun=0
            Lwallsh=SBC*ewall*((Ta+273.15)**4)*viktwall*0.5
    else: #nighttime
        Lwallsun=0
        Lwallsh=SBC*ewall*((Ta+273.15)**4)*viktwall*0.5

    if L_ani == 1:
        #Lsky = Ldown * 0.5
        Lveg=SBC*ewall*((Ta+273.15)**4)*viktveg*0.5
        Lground=LupS*0.5
        Lrefl=(Ldown_i+LupS)*(viktrefl)*(1-ewall)*0.5
        Lsouth=Lwallsun+Lwallsh+Lveg+Lground+Lrefl
        #Lsouth = Lsky + Lwallsun + Lwallsh + Lveg + Lground + Lrefl
    else:
        Lsky = ((svfS + svfSveg - 1) * Lsky_allsky) * viktsky * 0.5
        Lveg = SBC * ewall * ((Ta + 273.15) ** 4) * viktveg * 0.5
        Lground = LupS * 0.5
        Lrefl = (Ldown + LupS) * (viktrefl) * (1 - ewall) * 0.5
        Lsouth = Lsky + Lwallsun + Lwallsh + Lveg + Lground + Lrefl

    # print('South',Lveg[0, 59], Lground[0, 59], Lrefl[0, 59], Lsouth[0, 59])

    Lsky = ((svfS + svfSveg - 1) * Lsky_allsky) * viktsky * 0.5
    Lveg = SBC * ewall * ((Ta + 273.15) ** 4) * viktveg * 0.5
    Lground = LupS * 0.5
    Lrefl = (Ldown_i + LupS) * (viktrefl) * (1 - ewall) * 0.5
    Lsouth_i = Lsky + Lwallsun + Lwallsh + Lveg + Lground + Lrefl

    # print(Lsky[0,59],Lveg[0, 59], Lground[0, 59], Lrefl[0, 59], Lsouth_i[0, 59])

    Lsky_t = Lsky_t + Lsky

    # clear alfaB betaB betasun Lsky Lwallsh Lwallsun Lveg Lground Lrefl viktveg viktwall viktsky
    
    ## Lwest
    [viktveg,viktwall,viktsky,viktrefl]=Lvikt_veg(svfW,svfWveg,svfWaveg,vikttot)
    
    if altitude>0: # daytime
        alfaB=np.arctan(svfalfaW)
        betaB=np.arctan(np.tan((svfalfaW)*F_sh))
        betasun=((alfaB-betaB)/2)+betaB
        # betasun = np.arctan(0.5*np.tan(svfalfaW)*(1+F_sh))
        if (azimuth > (360-t))  or  (azimuth <= (180-t)):
            Lwallsun=SBC*ewall*((Ta+273.15+Tw*np.sin(aziW*(np.pi/180)))**4)*\
                viktwall*(1-F_sh)*np.cos(betasun)*0.5
            Lwallsh=SBC*ewall*((Ta+273.15)**4)*viktwall*F_sh*0.5
        else:
            Lwallsun=0
            Lwallsh=SBC*ewall*((Ta+273.15)**4)*viktwall*0.5
    else: #nighttime
        Lwallsun=0
        Lwallsh=SBC*ewall*((Ta+273.15)**4)*viktwall*0.5

    if L_ani == 1:
        #Lsky = Ldown * 0.5
        Lveg=SBC*ewall*((Ta+273.15)**4)*viktveg*0.5
        Lground=LupW*0.5
        Lrefl=(Ldown_i+LupW)*(viktrefl)*(1-ewall)*0.5
        Lwest=Lwallsun+Lwallsh+Lveg+Lground+Lrefl
        #Lwest = Lsky + Lwallsun + Lwallsh + Lveg + Lground + Lrefl
    else:
        Lsky = ((svfW + svfWveg - 1) * Lsky_allsky) * viktsky * 0.5
        Lveg = SBC * ewall * ((Ta + 273.15) ** 4) * viktveg * 0.5
        Lground = LupW * 0.5
        Lrefl = (Ldown + LupW) * (viktrefl) * (1 - ewall) * 0.5
        Lwest = Lsky + Lwallsun + Lwallsh + Lveg + Lground + Lrefl

    # print('West', Lveg[0, 59], Lground[0, 59], Lrefl[0, 59], Lwest[0, 59])

    Lsky = ((svfW + svfWveg - 1) * Lsky_allsky) * viktsky * 0.5
    Lveg = SBC * ewall * ((Ta + 273.15) ** 4) * viktveg * 0.5
    Lground = LupW * 0.5
    Lrefl = (Ldown_i + LupW) * (viktrefl) * (1 - ewall) * 0.5
    Lwest_i = Lsky + Lwallsun + Lwallsh + Lveg + Lground + Lrefl

    # print(Lsky[0,59],Lveg[0, 59], Lground[0, 59], Lrefl[0, 59], Lwest_i[0, 59])

    Lsky_t = Lsky_t + Lsky

    # clear alfaB betaB betasun Lsky Lwallsh Lwallsun Lveg Lground Lrefl viktveg viktwall viktsky
    
    ## Lnorth
    [viktveg,viktwall,viktsky,viktrefl]=Lvikt_veg(svfN,svfNveg,svfNaveg,vikttot)
    
    if altitude>0: # daytime
        alfaB=np.arctan(svfalfaN)
        betaB=np.arctan(np.tan((svfalfaN)*F_sh))
        betasun=((alfaB-betaB)/2)+betaB
        # betasun = np.arctan(0.5*np.tan(svfalfaN)*(1+F_sh))
        if (azimuth > (90-t))  and  (azimuth <= (270-t)):
            Lwallsun=SBC*ewall*((Ta+273.15+Tw*np.sin(aziN*(np.pi/180)))**4)*\
                viktwall*(1-F_sh)*np.cos(betasun)*0.5
            Lwallsh=SBC*ewall*((Ta+273.15)**4)*viktwall*F_sh*0.5
        else:
            Lwallsun=0
            Lwallsh=SBC*ewall*((Ta+273.15)**4)*viktwall*0.5
    else: #nighttime
        Lwallsun=0
        Lwallsh=SBC*ewall*((Ta+273.15)**4)*viktwall*0.5

    if L_ani == 1:
        #Lsky = Ldown * 0.5
        Lveg=SBC*ewall*((Ta+273.15)**4)*viktveg*0.5
        Lground=LupN*0.5
        Lrefl=(Ldown_i+LupN)*(viktrefl)*(1-ewall)*0.5
        Lnorth=Lwallsun+Lwallsh+Lveg+Lground+Lrefl
        #Lnorth = Lsky + Lwallsun + Lwallsh + Lveg + Lground + Lrefl
    else:
        Lsky = ((svfN + svfNveg - 1) * Lsky_allsky) * viktsky * 0.5
        Lveg = SBC * ewall * ((Ta + 273.15) ** 4) * viktveg * 0.5
        Lground = LupN * 0.5
        Lrefl = (Ldown + LupN) * (viktrefl) * (1 - ewall) * 0.5
        Lnorth = Lsky + Lwallsun + Lwallsh + Lveg + Lground + Lrefl

    # print('North', Lveg[0, 59], Lground[0, 59], Lrefl[0, 59], Lnorth[0, 59])

    Lsky = ((svfN + svfNveg - 1) * Lsky_allsky) * viktsky * 0.5
    Lveg = SBC * ewall * ((Ta + 273.15) ** 4) * viktveg * 0.5
    Lground = LupN * 0.5
    Lrefl = (Ldown_i + LupN) * (viktrefl) * (1 - ewall) * 0.5
    Lnorth_i = Lsky + Lwallsun + Lwallsh + Lveg + Lground + Lrefl

    # print(Lsky[0,59],Lveg[0, 59], Lground[0, 59], Lrefl[0, 59], Lnorth_i[0, 59])

    Lsky_t = Lsky_t + Lsky

    # clear alfaB betaB betasun Lsky Lwallsh Lwallsun Lveg Lground Lrefl viktveg viktwall viktsky
    
    return Least,Lsouth,Lwest,Lnorth,Least_i,Lsouth_i,Lwest_i,Lnorth_i, Lsky_t




def Lvikt_veg(svf,svfveg,svfaveg,vikttot):

    # Least
    viktonlywall=(vikttot-(63.227*svf**6-161.51*svf**5+156.91*svf**4-70.424*svf**3+16.773*svf**2-0.4863*svf))/vikttot

    viktaveg=(vikttot-(63.227*svfaveg**6-161.51*svfaveg**5+156.91*svfaveg**4-70.424*svfaveg**3+16.773*svfaveg**2-0.4863*svfaveg))/vikttot

    viktwall=viktonlywall-viktaveg

    svfvegbu=(svfveg+svf-1)  # Vegetation plus buildings
    viktsky=(63.227*svfvegbu**6-161.51*svfvegbu**5+156.91*svfvegbu**4-70.424*svfvegbu**3+16.773*svfvegbu**2-0.4863*svfvegbu)/vikttot
    viktrefl=(vikttot-(63.227*svfvegbu**6-161.51*svfvegbu**5+156.91*svfvegbu**4-70.424*svfvegbu**3+16.773*svfvegbu**2-0.4863*svfvegbu))/vikttot
    viktveg=(vikttot-(63.227*svfvegbu**6-161.51*svfvegbu**5+156.91*svfvegbu**4-70.424*svfvegbu**3+16.773*svfvegbu**2-0.4863*svfvegbu))/vikttot
    viktveg=viktveg-viktwall

    return viktveg,viktwall,viktsky,viktrefl


"""Adjusted code by Fredrik Lindberg, orginally from Dag Wastberg, Rambolls"""

import numpy as np
from collections import namedtuple

PET_person=namedtuple("PET_person","mbody age height activity sex clo")


def calculate_PET_index(Ta, Pa, Tmrt, va, pet):
    mbody=pet.mbody
    age=pet.age
    height=pet.height
    activity=pet.activity
    sex=pet.sex
    clo=pet.clo
    pet_index=np.zeros_like(Tmrt)
    for x in range(pet_index.shape[0]):
        for y in range(pet_index.shape[1]):
            pet_index[x][y]=_PET(Ta[x],Pa[x],Tmrt[x][y],va[x][y],mbody,age,height,activity,clo,sex)

def calculate_PET_index_vec(Ta, Pa, Tmrt, va,pet):
    mbody=pet.mbody
    age=pet.age
    height=pet.height
    activity=pet.activity
    sex=pet.sex
    clo=pet.clo

    pet_index=_PET(Ta,Pa,Tmrt,va,mbody,age,height,activity,clo,sex)

def _PET(ta,RH,tmrt,v,mbody,age,ht,work,icl,sex):
    """
    Args:
        ta: air temperature
        RH: relative humidity
        tmrt: Mean Radiant temperature
        v: wind at pedestrian heigh
        mbody: body masss (kg)
        age: person's age (years)
        ht: height (meters)
        work: activity level (W)
        icl: clothing amount (0-5)
        sex: 1=male 2=female
    Returns:
    """

    # humidity conversion
    vps = 6.107 * (10. ** (7.5 * ta / (238. + ta)))
    vpa = RH * vps / 100  # water vapour presure, kPa

    po = 1013.25  # Pressure
    p = 1013.25  # Pressure
    rob = 1.06
    cb = 3.64 * 1000
    food = 0
    emsk = 0.99
    emcl = 0.95
    evap = 2.42e6
    sigma = 5.67e-8
    cair = 1.01 * 1000

    eta = 0  # No idea what eta is

    c_1 = 0.
    c_2 = 0.
    c_3 = 0.
    c_4 = 0.
    c_5 = 0.
    c_6 = 0.
    c_7 = 0.
    c_8 = 0.
    c_9 = 0.
    c_10 = 0.
    c_11 = 0.

    # INBODY
    metbf = 3.19 * mbody ** (3 / 4) * (1 + 0.004 * (30 - age) + 0.018 * ((ht * 100 / (mbody ** (1 / 3))) - 42.1))
    metbm = 3.45 * mbody ** (3 / 4) * (1 + 0.004 * (30 - age) + 0.010 * ((ht * 100 / (mbody ** (1 / 3))) - 43.4))
    if sex == 1:
        met = metbm + work
    else:
        met = metbf + work

    h = met * (1 - eta)
    rtv = 1.44e-6 * met

    # sensible respiration energy
    tex = 0.47 * ta + 21.0
    eres = cair * (ta - tex) * rtv

    # latent respiration energy
    vpex = 6.11 * 10 ** (7.45 * tex / (235 + tex))
    erel = 0.623 * evap / p * (vpa - vpex) * rtv
    # sum of the results
    ere = eres + erel

    # calcul constants
    feff = 0.725
    adu = 0.203 * mbody ** 0.425 * ht ** 0.725
    facl = (-2.36 + 173.51 * icl - 100.76 * icl * icl + 19.28 * (icl ** 3)) / 100
    if facl > 1:
        facl = 1
    rcl = (icl / 6.45) / facl
    y = 1

    # should these be else if statements?
    if icl < 2:
        y = (ht-0.2) / ht
    if icl <= 0.6:
        y = 0.5
    if icl <= 0.3:
        y = 0.1

    fcl = 1 + 0.15 * icl
    r2 = adu * (fcl - 1. + facl) / (2 * 3.14 * ht * y)
    r1 = facl * adu / (2 * 3.14 * ht * y)
    di = r2 - r1
    acl = adu * facl + adu * (fcl - 1)

    tcore = [0] * 8

    wetsk = 0
    hc = 2.67 + 6.5 * v ** 0.67
    hc = hc * (p / po) ** 0.55
    c_1 = h + ere
    he = 0.633 * hc / (p * cair)
    fec = 1 / (1 + 0.92 * hc * rcl)
    htcl = 6.28 * ht * y * di / (rcl * np.log(r2 / r1) * acl)
    aeff = adu * feff
    c_2 = adu * rob * cb
    c_5 = 0.0208 * c_2
    c_6 = 0.76075 * c_2
    rdsk = 0.79 * 10 ** 7
    rdcl = 0

    count2 = 0
    j = 1

    while count2 == 0 and j < 7:
        tsk = 34
        count1 = 0
        tcl = (ta + tmrt + tsk) / 3
        count3 = 1
        enbal2 = 0

        while count1 <= 3:
            enbal = 0
            while (enbal*enbal2) >= 0 and count3 < 200:
                enbal2 = enbal
                # 20
                rclo2 = emcl * sigma * ((tcl + 273.2) ** 4 - (tmrt + 273.2) ** 4) * feff
                tsk = 1 / htcl * (hc * (tcl - ta) + rclo2) + tcl

                # radiation balance
                rbare = aeff * (1 - facl) * emsk * sigma * ((tmrt + 273.2) ** 4 - (tsk + 273.2) ** 4)
                rclo = feff * acl * emcl * sigma * ((tmrt + 273.2) ** 4 - (tcl + 273.2) ** 4)
                rsum = rbare + rclo

                # convection
                cbare = hc * (ta - tsk) * adu * (1 - facl)
                cclo = hc * (ta - tcl) * acl
                csum = cbare + cclo

                # core temperature
                c_3 = 18 - 0.5 * tsk
                c_4 = 5.28 * adu * c_3
                c_7 = c_4 - c_6 - tsk * c_5
                c_8 = -c_1 * c_3 - tsk * c_4 + tsk * c_6
                c_9 = c_7 * c_7 - 4. * c_5 * c_8
                c_10 = 5.28 * adu - c_6 - c_5 * tsk
                c_11 = c_10 * c_10 - 4 * c_5 * (c_6 * tsk - c_1 - 5.28 * adu * tsk)
                # tsk[tsk==36]=36.01
                if tsk == 36:
                    tsk = 36.01

                tcore[7] = c_1 / (5.28 * adu + c_2 * 6.3 / 3600) + tsk
                tcore[3] = c_1 / (5.28 * adu + (c_2 * 6.3 / 3600) / (1 + 0.5 * (34 - tsk))) + tsk
                if c_11 >= 0:
                    tcore[6] = (-c_10-c_11 ** 0.5) / (2 * c_5)
                if c_11 >= 0:
                    tcore[1] = (-c_10+c_11 ** 0.5) / (2 * c_5)
                if c_9 >= 0:
                    tcore[2] = (-c_7+abs(c_9) ** 0.5) / (2 * c_5)
                if c_9 >= 0:
                    tcore[5] = (-c_7-abs(c_9) ** 0.5) / (2 * c_5)
                tcore[4] = c_1 / (5.28 * adu + c_2 * 1 / 40) + tsk

                # transpiration
                tbody = 0.1 * tsk + 0.9 * tcore[j]
                sw = 304.94 * (tbody - 36.6) * adu / 3600000
                vpts = 6.11 * 10 ** (7.45 * tsk / (235. + tsk))
                if tbody <= 36.6:
                    sw = 0
                if sex == 2:
                    sw = 0.7 * sw
                eswphy = -sw * evap

                eswpot = he * (vpa - vpts) * adu * evap * fec
                wetsk = eswphy / eswpot
                if wetsk > 1:
                    wetsk = 1
                eswdif = eswphy - eswpot
                if eswdif <= 0:
                    esw = eswpot
                else:
                    esw = eswphy
                if esw > 0:
                    esw = 0

                # diffusion
                ed = evap / (rdsk + rdcl) * adu * (1 - wetsk) * (vpa - vpts)

                # MAX VB
                vb1 = 34 - tsk
                vb2 = tcore[j] - 36.6
                if vb2 < 0:
                    vb2 = 0
                if vb1 < 0:
                    vb1 = 0
                vb = (6.3 + 75 * vb2) / (1 + 0.5 * vb1)

                # energy balance
                enbal = h + ed + ere + esw + csum + rsum + food

                # clothing's temperature
                if count1 == 0:
                    xx = 1
                if count1 == 1:
                    xx = 0.1
                if count1 == 2:
                    xx = 0.01
                if count1 == 3:
                    xx = 0.001
                if enbal > 0:
                    tcl = tcl + xx
                else:
                    tcl = tcl - xx

                count3 = count3 + 1
            count1 = count1 + 1
            enbal2 = 0

        if j == 2 or j == 5:
            if c_9 >= 0:
                if tcore[j] >= 36.6 and tsk <= 34.050:
                    if (j != 4 and vb >= 91) or (j == 4 and vb < 89):
                        pass
                    else:
                        if vb > 90:
                            vb = 90
                        count2 = 1

        if j == 6 or j == 1:
            if c_11 > 0:
                if tcore[j] >= 36.6 and tsk > 33.850:
                    if (j != 4 and vb >= 91) or (j == 4 and vb < 89):
                        pass
                    else:
                        if vb > 90:
                            vb = 90
                        count2 = 1

        if j == 3:
            if tcore[j] < 36.6 and tsk <= 34.000:
                if (j != 4 and vb >= 91) or (j == 4 and vb < 89):
                    pass
                else:
                    if vb > 90:
                        vb = 90
                    count2 = 1

        if j == 7:
            if tcore[j] < 36.6 and tsk > 34.000:
                if (j != 4 and vb >= 91) or (j == 4 and vb < 89):
                    pass
                else:
                    if vb > 90:
                        vb = 90
                    count2 = 1

        if j == 4:
            if (j != 4 and vb >= 91) or (j == 4 and vb < 89):
                pass
            else:
                if vb > 90:
                    vb = 90
                count2 = 1

        j = j + 1

    # PET_cal
    tx = ta
    enbal2 = 0
    count1 = 0
    enbal = 0

    hc = 2.67 + 6.5 * 0.1 ** 0.67
    hc = hc * (p / po) ** 0.55

    while count1 <= 3:
        while (enbal * enbal2) >= 0:
            enbal2 = enbal

            # radiation balance
            rbare = aeff * (1 - facl) * emsk * sigma * ((tx + 273.2) ** 4 - (tsk + 273.2) ** 4)
            rclo = feff * acl * emcl * sigma * ((tx + 273.2) ** 4 - (tcl + 273.2) ** 4)
            rsum = rbare + rclo

            # convection
            cbare = hc * (tx - tsk) * adu * (1 - facl)
            cclo = hc * (tx - tcl) * acl
            csum = cbare + cclo

            # diffusion
            ed = evap / (rdsk + rdcl) * adu * (1 - wetsk) * (12 - vpts)

            # respiration
            tex = 0.47 * tx + 21
            eres = cair * (tx - tex) * rtv
            vpex = 6.11 * 10 ** (7.45 * tex / (235 + tex))
            erel = 0.623 * evap / p * (12 - vpex) * rtv
            ere = eres + erel

            # energy balance
            enbal = h + ed + ere + esw + csum + rsum

            # iteration concerning Tx
            if count1 == 0:
                xx = 1
            if count1 == 1:
                xx = 0.1
            if count1 == 2:
                xx = 0.01
            if count1 == 3:
                xx = 0.001
            if enbal > 0:
                tx = tx - xx
            if enbal < 0:
                tx = tx + xx
        count1 = count1 + 1
        enbal2 = 0

    return tx

from __future__ import division
import numpy as np


def Perez_v3(zen, azimuth, radD, radI, jday, patchchoice):
    """
    This function calculates distribution of luminance on the skyvault based on
    Perez luminince distribution model.
    
    Created by:
    Fredrik Lindberg 20120527, fredrikl@gvc.gu.se
    Gothenburg University, Sweden
    Urban Climte Group
    
    Input parameters:
     - zen:     Zenith angle of the Sun (in degrees)
     - azimuth: Azimuth angle of the Sun (in degrees)
     - radD:    Horizontal diffuse radiation (W m-2)
     - radI:    Direct radiation perpendicular to the Sun beam (W m-2)
     - jday:    Day of year
    
    Output parameters:
     - lv:   Relative luminance map (same dimensions as theta. gamma)
    

    acoeff=[1.353 -0.258 -0.269 -1.437
           -1.222 -0.773 1.415 1.102
           -1.100 -0.252 0.895 0.016
           -0.585 -0.665 -0.267 0.712
           -0.600 -0.347 -2.500 2.323
           -1.016 -0.367 1.008 1.405
           -1.000 0.021 0.503 -0.512
           -1.050 0.029 0.426 0.359];
    
    bcoeff=[-0.767 0.001 1.273 -0.123
            -0.205 0.037 -3.913 0.916
             0.278 -0.181 -4.500 1.177
             0.723 -0.622 -5.681 2.630
             0.294 0.049 -5.681 1.842
             0.288 -0.533 -3.850 3.375
            -0.300 0.192 0.702 -1.632
            -0.325 0.116 0.778 0.003];
    
    ccoeff=[2.800 0.600 1.238 1.000
            6.975 0.177 6.448 -0.124
            24.22 -13.08 -37.70 34.84
            33.34 -18.30 -62.25 52.08
            21.00 -4.766 -21.59 7.249
            14.00 -0.999 -7.14 7.547
            19.00 -5.000 1.243 -1.91
            31.06 -14.50 -46.11 55.37];
    
    dcoeff=[1.874 0.630 0.974 0.281
           -1.580 -0.508 -1.781 0.108
           -5.00 1.522 3.923 -2.62
           -3.50 0.002 1.148 0.106
           -3.50 -0.155 1.406 0.399
           -3.40 -0.108 -1.075 1.57
           -4.00 0.025 0.384 0.266
           -7.23 0.405 13.35 0.623];
    
    ecoeff=[0.035 -0.125 -0.572 0.994
            0.262 0.067 -0.219 -0.428
           -0.016 0.160 0.420 -0.556
            0.466 -0.33 -0.088 -0.033
            0.003 0.077 -0.066 -0.129
           -0.067 0.402 0.302 -0.484
            1.047 -0.379 -2.452 1.466
            1.500 -0.643 1.856 0.564];

    :param zen:
    :param azimuth:
    :param radD:
    :param radI:
    :param jday:
    :param patchchoice:
    :return:
    """

    m_a1 = np.array([1.3525, -1.2219, -1.1000, -0.5484, -0.6000, -1.0156, -1.0000, -1.0500])
    m_a2 = np.array([-0.2576, -0.7730, -0.2515, -0.6654, -0.3566, -0.3670, 0.0211, 0.0289])
    m_a3 = np.array([-0.2690, 1.4148, 0.8952, -0.2672, -2.5000, 1.0078, 0.5025, 0.4260])
    m_a4 = np.array([-1.4366, 1.1016, 0.0156, 0.7117, 2.3250, 1.4051, -0.5119, 0.3590])
    m_b1 = np.array([-0.7670, -0.2054, 0.2782, 0.7234, 0.2937, 0.2875, -0.3000, -0.3250])
    m_b2 = np.array([0.0007, 0.0367, -0.1812, -0.6219, 0.0496, -0.5328, 0.1922, 0.1156])
    m_b3 = np.array([1.2734, -3.9128, -4.5000, -5.6812, -5.6812, -3.8500, 0.7023, 0.7781])
    m_b4 = np.array([-0.1233, 0.9156, 1.1766, 2.6297, 1.8415, 3.3750, -1.6317, 0.0025])
    m_c1 = np.array([2.8000, 6.9750, 24.7219, 33.3389, 21.0000, 14.0000, 19.0000, 31.0625])
    m_c2 = np.array([0.6004, 0.1774, -13.0812, -18.3000, -4.7656, -0.9999, -5.0000, -14.5000])
    m_c3 = np.array([1.2375, 6.4477, -37.7000, -62.2500, -21.5906, -7.1406, 1.2438, -46.1148])
    m_c4 = np.array([1.0000, -0.1239, 34.8438, 52.0781, 7.2492, 7.5469, -1.9094, 55.3750])
    m_d1 = np.array([1.8734, -1.5798, -5.0000, -3.5000, -3.5000, -3.4000, -4.0000, -7.2312])
    m_d2 = np.array([0.6297, -0.5081, 1.5218, 0.0016, -0.1554, -0.1078, 0.0250, 0.4050])
    m_d3 = np.array([0.9738, -1.7812, 3.9229, 1.1477, 1.4062, -1.0750, 0.3844, 13.3500])
    m_d4 = np.array([0.2809, 0.1080, -2.6204, 0.1062, 0.3988, 1.5702, 0.2656, 0.6234])
    m_e1 = np.array([0.0356, 0.2624, -0.0156, 0.4659, 0.0032, -0.0672, 1.0468, 1.5000])
    m_e2 = np.array([-0.1246, 0.0672, 0.1597, -0.3296, 0.0766, 0.4016, -0.3788, -0.6426])
    m_e3 = np.array([-0.5718, -0.2190, 0.4199, -0.0876, -0.0656, 0.3017, -2.4517, 1.8564])
    m_e4 = np.array([0.9938, -0.4285, -0.5562, -0.0329, -0.1294, -0.4844, 1.4656, 0.5636])
    
    acoeff = np.transpose(np.atleast_2d([m_a1, m_a2, m_a3, m_a4]))
    bcoeff = np.transpose(np.atleast_2d([m_b1, m_b2, m_b3, m_b4]))
    ccoeff = np.transpose(np.atleast_2d([m_c1, m_c2, m_c3, m_c4]))
    dcoeff = np.transpose(np.atleast_2d([m_d1, m_d2, m_d3, m_d4]))
    ecoeff = np.transpose(np.atleast_2d([m_e1, m_e2, m_e3, m_e4]))

    deg2rad = np.pi/180
    rad2deg = 180/np.pi
    altitude = 90-zen
    zen = zen * deg2rad
    azimuth = azimuth * deg2rad
    altitude = altitude * deg2rad
    Idh = radD
    # Ibh = radI/sin(altitude)
    Ibn = radI

    # Skyclearness
    PerezClearness = ((Idh+Ibn)/(Idh+1.041*np.power(zen, 3)))/(1+1.041*np.power(zen, 3))
    # Extra terrestrial radiation
    day_angle = jday*2*np.pi/365
    #I0=1367*(1+0.033*np.cos((2*np.pi*jday)/365))
    I0 = 1367*(1.00011+0.034221*np.cos(day_angle) + 0.00128*np.sin(day_angle)+0.000719 *
               np.cos(2*day_angle)+0.000077*np.sin(2*day_angle))    # New from robinsson

    # Optical air mass
    # m=1/altitude; old
    if altitude >= 10*deg2rad:
        AirMass = 1/np.sin(altitude)
    elif altitude < 0:   # below equation becomes complex
        AirMass = 1/np.sin(altitude)+0.50572*np.power(180*complex(altitude)/np.pi+6.07995, -1.6364)
    else:
        AirMass = 1/np.sin(altitude)+0.50572*np.power(180*altitude/np.pi+6.07995, -1.6364)

    # Skybrightness
    # if altitude*rad2deg+6.07995>=0
    PerezBrightness = (AirMass*radD)/I0
    if Idh <= 10:
        # m_a=0;m_b=0;m_c=0;m_d=0;m_e=0;
        PerezBrightness = 0
    if altitude < 0:
        print("Airmass")
        print(AirMass)
        print(PerezBrightness)
    # sky clearness bins
    if PerezClearness < 1.065:
        intClearness = 0
    elif PerezClearness < 1.230:
        intClearness = 1
    elif PerezClearness < 1.500:
        intClearness = 2
    elif PerezClearness < 1.950:
        intClearness = 3
    elif PerezClearness < 2.800:
        intClearness = 4
    elif PerezClearness < 4.500:
        intClearness = 5
    elif PerezClearness < 6.200:
        intClearness = 6
    elif PerezClearness > 6.200:
        intClearness = 7
    else:
        raise ValueError('No valid PerezClearness, are inputs NaN?')

    m_a = acoeff[intClearness,  0] + acoeff[intClearness,  1] * zen + PerezBrightness * (acoeff[intClearness,  2] + acoeff[intClearness,  3] * zen)
    m_b = bcoeff[intClearness,  0] + bcoeff[intClearness,  1] * zen + PerezBrightness * (bcoeff[intClearness,  2] + bcoeff[intClearness,  3] * zen)
    m_e = ecoeff[intClearness,  0] + ecoeff[intClearness,  1] * zen + PerezBrightness * (ecoeff[intClearness,  2] + ecoeff[intClearness,  3] * zen)

    if intClearness > 0:
        m_c = ccoeff[intClearness, 0] + ccoeff[intClearness, 1] * zen + PerezBrightness * (ccoeff[intClearness, 2] + ccoeff[intClearness, 3] * zen)
        m_d = dcoeff[intClearness, 0] + dcoeff[intClearness, 1] * zen + PerezBrightness * (dcoeff[intClearness, 2] + dcoeff[intClearness, 3] * zen)
    else:
        # different equations for c & d in clearness bin no. 1,  from Robinsson
        m_c = np.exp(np.power(PerezBrightness * (ccoeff[intClearness, 0] + ccoeff[intClearness, 1] * zen), ccoeff[intClearness, 2]))-1
        m_d = -np.exp(PerezBrightness * (dcoeff[intClearness, 0] + dcoeff[intClearness, 1] * zen)) + dcoeff[intClearness, 2] + \
            PerezBrightness * dcoeff[intClearness, 3] * PerezBrightness

    # print 'a = ', m_a
    # print 'b = ', m_b
    # print 'e = ', m_e
    # print 'c = ', m_c
    # print 'd = ', m_d

    skyvaultalt = np.atleast_2d([])
    skyvaultazi = np.atleast_2d([])
    if patchchoice == 2:
        # Creating skyvault at one degree intervals
        skyvaultalt = np.ones([90, 361])*90
        skyvaultazi = np.empty((90, 361))
        for j in range(90):
            skyvaultalt[j, :] = 91-j
            skyvaultazi[j, :] = range(361)
            
    elif patchchoice == 1:
        # Creating skyvault of patches of constant radians (Tregeneza and Sharples, 1993)
        skyvaultaltint = [6, 18, 30, 42, 54, 66, 78]
        skyvaultaziint = [12, 12, 15, 15, 20, 30, 60]
        for j in range(7):
            for k in range(1, int(360/skyvaultaziint[j]) + 1):
                skyvaultalt = np.append(skyvaultalt, skyvaultaltint[j])
                skyvaultazi = np.append(skyvaultazi, k*skyvaultaziint[j])

        skyvaultalt = np.append(skyvaultalt, 90)
        skyvaultazi = np.append(skyvaultazi, 360)

    skyvaultzen = (90 - skyvaultalt) * deg2rad
    skyvaultalt = skyvaultalt * deg2rad
    skyvaultazi = skyvaultazi * deg2rad

    # Angular distance from the sun from Robinsson
    cosSkySunAngle = np.sin(skyvaultalt) * np.sin(altitude) + \
                     np.cos(altitude) * np.cos(skyvaultalt) * np.cos(np.abs(skyvaultazi-azimuth))

    # Main equation
    lv = (1 + m_a * np.exp(m_b / np.cos(skyvaultzen))) * ((1 + m_c * np.exp(m_d * np.arccos(cosSkySunAngle)) +
                                                           m_e * cosSkySunAngle * cosSkySunAngle))

    # Normalisation
    lv = lv / np.sum(lv)

    # plotting
    # axesm('stereo','Origin',[90 180],'MapLatLimit',[0 90],'Aspect','transverse')
    # framem off; gridm on; mlabel off; plabel off;axis on;
    # setm(gca,'MLabelParallel',-20)
    # geoshow(skyvaultalt*rad2deg,skyvaultazi*rad2deg,lv,'DisplayType','texture');
    # colorbar
    # set(gcf,'Color',[1 1 1])
    # pause(1)

    if patchchoice == 1:
        #x = np.atleast_2d([])
        #lv = np.transpose(np.append(np.append(np.append(x, skyvaultalt*rad2deg), skyvaultazi*rad2deg), lv))
        x = np.transpose(np.atleast_2d(skyvaultalt*rad2deg))
        y = np.transpose(np.atleast_2d(skyvaultazi*rad2deg))
        z = np.transpose(np.atleast_2d(lv))
        lv = np.append(np.append(x, y, axis=1), z, axis=1)
    return lv, PerezClearness, PerezBrightness


from __future__ import absolute_import
import numpy as np
# from .TsWaveDelay_2015a import TsWaveDelay_2015a
# from .Kup_veg_2015a import Kup_veg_2015a

def Solweig1D_2019a_calc(svf, svfveg, svfaveg, sh, vegsh,  albedo_b, absK, absL, ewall, Fside, Fup, Fcyl, altitude, azimuth, zen, jday,
                         onlyglobal, location, dectime, altmax, cyl, elvis, Ta, RH, radG, radD, radI, P,
                         Twater, TgK, Tstart, albedo_g, eground, TgK_wall, Tstart_wall, TmaxLST, TmaxLST_wall,
                         svfalfa, CI, ani, diffsh, trans):

    # This is the core function of the SOLWEIG1D model, 2019-Jun-21
    # Fredrik Lindberg, fredrikl@gvc.gu.se, Goteborg Urban Climate Group, Gothenburg University, Sweden

    svfE = svf
    svfW = svf
    svfN = svf
    svfS = svf
    svfEveg = svfveg
    svfSveg = svfveg
    svfWveg = svfveg
    svfNveg = svfveg
    svfEaveg = svfaveg
    svfSaveg = svfaveg
    svfWaveg = svfaveg
    svfNaveg = svfaveg
    psi = trans

    # Instrument offset in degrees
    t = 0.

    # Stefan Bolzmans Constant
    SBC = 5.67051e-8

    # Find sunrise decimal hour - new from 2014a
    _, _, _, SNUP = daylen(jday, location['latitude'])

    shadow = sh - (1 - vegsh) * (1 - psi)

    # Vapor pressure
    ea = 6.107 * 10 ** ((7.5 * Ta) / (237.3 + Ta)) * (RH / 100.)

    # Determination of clear - sky emissivity from Prata (1996)
    msteg = 46.5 * (ea / (Ta + 273.15))
    esky = (1 - (1 + msteg) * np.exp(-((1.2 + 3.0 * msteg) ** 0.5))) + elvis  # -0.04 old error from Jonsson et al.2006

    if altitude > 0: # # # # # # DAYTIME # # # # # #
        # Clearness Index on Earth's surface after Crawford and Dunchon (1999) with a correction
        #  factor for low sun elevations after Lindberg et al.(2008)
        I0, CI, Kt, I0et, CIuncorr = clearnessindex_2013b(zen, jday, Ta, RH / 100., radG, location, P)
        if (CI > 1) or (CI == np.inf):
            CI = 1

        # Estimation of radD and radI if not measured after Reindl et al.(1990)
        if onlyglobal == 1:
            I0, CI, Kt, I0et, CIuncorr = clearnessindex_2013b(zen, jday, Ta, RH / 100., radG, location, P)
            if (CI > 1) or (CI == np.inf):
                CI = 1

            radI, radD = diffusefraction(radG, altitude, Kt, Ta, RH)

        # Diffuse Radiation
        # Anisotropic Diffuse Radiation after Perez et al. 1993
        if ani == 1:
            patchchoice = 1
            zenDeg = zen*(180/np.pi)
            lv = Perez_v3(zenDeg, azimuth, radD, radI, jday, patchchoice)   # Relative luminance

            aniLum = 0.
            for idx in range(0, 145):
                aniLum = aniLum + diffsh[idx] * lv[0][idx][2]     # Total relative luminance from sky into each cell

            dRad = aniLum * radD   # Total diffuse radiation from sky into each cell

        else:
            dRad = radD * svf
            lv = 0

        # # # Surface temperature parameterisation during daytime # # # #
        # new using max sun alt.instead of  dfm
        Tgamp = (TgK * altmax - Tstart) + Tstart
        Tgampwall = (TgK_wall * altmax - (Tstart_wall)) + (Tstart_wall)
        Tg = Tgamp * np.sin((((dectime - np.floor(dectime)) - SNUP / 24) / (TmaxLST / 24 - SNUP / 24)) * np.pi / 2) + Tstart # 2015 a, based on max sun altitude
        Tgwall = Tgampwall * np.sin((((dectime - np.floor(dectime)) - SNUP / 24) / (TmaxLST_wall / 24 - SNUP / 24)) * np.pi / 2) + (Tstart_wall) # 2015a, based on max sun altitude

        if Tgwall < 0:  # temporary for removing low Tg during morning 20130205
            # Tg = 0
            Tgwall = 0

        # New estimation of Tg reduction for non - clear situation based on Reindl et al.1990
        radI0, _ = diffusefraction(I0, altitude, 1., Ta, RH)
        corr = 0.1473 * np.log(90 - (zen / np.pi * 180)) + 0.3454  # 20070329 correction of lat, Lindberg et al. 2008
        CI_Tg = (radI / radI0) + (1 - corr)
        if (CI_Tg > 1) or (CI_Tg == np.inf):
            CI_Tg = 1
        Tg = Tg * CI_Tg  # new estimation
        Tgwall = Tgwall * CI_Tg

        if Tg < 0.:
            Tg = 0.
        #Tg[Tg < 0] = 0  # temporary for removing low Tg during morning 20130205

        gvf = shadow

        Lup = SBC * eground * ((gvf + Ta + Tg + 273.15) ** 4)
        LupE = Lup
        LupS = Lup
        LupW = Lup
        LupN = Lup

        # Building height angle from svf
        F_sh = cylindric_wedge(zen, svfalfa, 1, 1)  # Fraction shadow on building walls based on sun alt and svf
        F_sh[np.isnan(F_sh)] = 0.5

        # # # # # # # Calculation of shortwave daytime radiative fluxes # # # # # # #
        Kdown = radI * shadow * np.sin(altitude * (np.pi / 180)) + dRad + albedo_b * (1 - svf) * \
                            (radG * (1 - F_sh) + radD * F_sh) # *sin(altitude(i) * (pi / 180))

        #Kdown = radI * shadow * np.sin(altitude * (np.pi / 180)) + radD * svfbuveg + albedo_b * (1 - svfbuveg) * \
                            #(radG * (1 - F_sh) + radD * F_sh) # *sin(altitude(i) * (pi / 180))

        Kup = albedo_g * (shadow * radI * np.sin(altitude * (np.pi / 180.))) + radD * svf + \
              albedo_b * (1 - svf) * (radG * (1 - F_sh) + radD * F_sh)

        # Kup, KupE, KupS, KupW, KupN = Kup_veg_2015a(radI, radD, radG, altitude, svf, albedo_b, F_sh, gvfalb,
        #             gvfalbE, gvfalbS, gvfalbW, gvfalbN, gvfalbnosh, gvfalbnoshE, gvfalbnoshS, gvfalbnoshW, gvfalbnoshN)

        Keast, Ksouth, Kwest, Knorth, KsideI, KsideD = Kside_veg_v2019a(radI, radD, radG, shadow, svfS, svfW, svfN, svfE,
                    svfEveg, svfSveg, svfWveg, svfNveg, azimuth, altitude, psi, t, albedo_b, F_sh, Kup, Kup, Kup,
                    Kup, cyl, lv, ani, diffsh, 1, 1)

        firstdaytime = 0

    else:  # # # # # # # NIGHTTIME # # # # # # # #

        Tgwall = 0
        # CI_Tg = -999  # F_sh = []

        # Nocturnal K fluxes set to 0
        Knight = 0.
        Kdown = 0.
        Kwest = 0.
        Kup = 0.
        Keast = 0.
        Ksouth = 0.
        Knorth = 0.
        KsideI = 0.
        KsideD = 0.
        F_sh = 0.
        Tg = 0.
        shadow = 0.

        # # # # Lup # # # #
        Lup = SBC * eground * ((Knight + Ta + Tg + 273.15) ** 4)
        # if landcover == 1:
        #     Lup[lc_grid == 3] = SBC * 0.98 * (Twater + 273.15) ** 4  # nocturnal Water temp
        LupE = Lup
        LupS = Lup
        LupW = Lup
        LupN = Lup

        # # For Tg output in POIs
        # TgOut = Ta + Tg

        I0 = 0
        # timeadd = 0
        # firstdaytime = 1

    # # # # Ldown # # # #
    Ldown = svf * esky * SBC * ((Ta + 273.15) ** 4) + (1 - svf) * ewall * SBC * ((Ta + 273.15 + Tgwall) ** 4) + \
            (1 - svf) * (1 - ewall) * esky * SBC * ((Ta + 273.15) ** 4)  # Jonsson et al.(2006)

    if CI < 0.95:  # non - clear conditions
        c = 1 - CI
        Ldown = Ldown * (1 - c) + \
                c * (svf * SBC * ((Ta + 273.15) ** 4) + (1 - svf) * ewall * SBC * ((Ta + 273.15 + Tgwall) ** 4) +
                     (1 - svf) * (1 - ewall) * esky * SBC * ((Ta + 273.15) ** 4))

    # # # # Lside # # # #
    Least, Lsouth, Lwest, Lnorth = Lside_veg_v2015a(svfS, svfW, svfN, svfE, svfEveg, svfSveg, svfWveg, svfNveg,
                    svfEaveg, svfSaveg, svfWaveg, svfNaveg, azimuth, altitude, Ta, Tgwall, SBC, ewall, Ldown,
                                                      esky, t, F_sh, CI, LupE, LupS, LupW, LupN)

    # # # # Calculation of radiant flux density and Tmrt # # # #
    if cyl == 1 and ani == 1:  # Human body considered as a cylinder with Perez et al. (1993)
        Sstr = absK * ((KsideI + KsideD) * Fcyl + (Kdown + Kup) * Fup + (Knorth + Keast + Ksouth + Kwest) * Fside) + absL * \
                        (Ldown * Fup + Lup * Fup + Lnorth * Fside + Least * Fside + Lsouth * Fside + Lwest * Fside)
    elif cyl == 1 and ani == 0: # Human body considered as a cylinder with isotropic all-sky diffuse
        Sstr = absK * (KsideI * Fcyl + (Kdown + Kup) * Fup + (Knorth + Keast + Ksouth + Kwest) * Fside) + absL * \
                        (Ldown * Fup + Lup * Fup + Lnorth * Fside + Least * Fside + Lsouth * Fside + Lwest * Fside)
    else: # Human body considered as a standing cube
        Sstr = absK * ((Kdown + Kup) * Fup + (Knorth + Keast + Ksouth + Kwest) * Fside) +absL * \
                        (Ldown * Fup + Lup * Fup + Lnorth * Fside + Least * Fside + Lsouth * Fside + Lwest * Fside)

    Tmrt = np.sqrt(np.sqrt((Sstr / (absL * SBC)))) - 273.2

    return Tmrt, Kdown, Kup, Ldown, Lup, Tg, ea, esky, I0, CI, Keast, Ksouth, Kwest, Knorth, Least, Lsouth, Lwest, \
           Lnorth, KsideI, radI, radD, shadow



from __future__ import absolute_import
import numpy as np
# from .TsWaveDelay_2015a import TsWaveDelay_2015a
# from .Kup_veg_2015a import Kup_veg_2015a

import matplotlib.pylab as plt

def Solweig1D_2020a_calc(svf, svfveg, svfaveg, sh, vegsh,  albedo_b, absK, absL, ewall, Fside, Fup, Fcyl, altitude, azimuth, zen, jday,
                         onlyglobal, location, dectime, altmax, cyl, elvis, Ta, RH, radG, radD, radI, P,
                         Twater, TgK, Tstart, albedo_g, eground, TgK_wall, Tstart_wall, TmaxLST, TmaxLST_wall,
                         svfalfa,  svfbuveg, CI, anisdiff, diffsh, trans, L_ani):

    # This is the core function of the SOLWEIG1D model, 2019-Jun-21
    # Fredrik Lindberg, fredrikl@gvc.gu.se, Goteborg Urban Climate Group, Gothenburg University, Sweden

    svfE = svf
    svfW = svf
    svfN = svf
    svfS = svf
    svfEveg = svfveg
    svfSveg = svfveg
    svfWveg = svfveg
    svfNveg = svfveg
    svfEaveg = svfaveg
    svfSaveg = svfaveg
    svfWaveg = svfaveg
    svfNaveg = svfaveg
    psi = trans

    # Instrument offset in degrees
    t = 0.

    # Stefan Bolzmans Constant
    SBC = 5.67051e-8

    # Find sunrise decimal hour - new from 2014a
    _, _, _, SNUP = daylen(jday, location['latitude'])

    shadow = sh - (1 - vegsh) * (1 - psi)

    # Vapor pressure
    ea = 6.107 * 10 ** ((7.5 * Ta) / (237.3 + Ta)) * (RH / 100.)

    # Determination of clear - sky emissivity from Prata (1996)
    msteg = 46.5 * (ea / (Ta + 273.15))
    esky = (1 - (1 + msteg) * np.exp(-((1.2 + 3.0 * msteg) ** 0.5))) + elvis  # -0.04 old error from Jonsson et al.2006

    # Anisotrophic longwave radiation
    if L_ani == 1:
        patchchoice = 1
        zenDeg = zen * (180 / np.pi)
        lv = Perez_v3(zenDeg, azimuth, radD, radI, jday, patchchoice)  # Relative luminance

        # if L_ani == 1:
        Ldown, Lside, Lsky = Lcyl(esky, lv, Ta)

        #test
        #Ldown[:] = Ldown.sum() / 145.
        #Lside[:] = Lside.sum() / 145.

        Ldown_a = 0.0
        Lside_a = 0.0
        for idx in range(0, 145):
            Ldown_a = Ldown_a + diffsh[idx] * Ldown[idx]
            Lside_a = Lside_a + diffsh[idx] * Lside[idx]

            # print(Ldown_a)

        Ldown_i = (svf + svfveg - 1) * esky * SBC * ((Ta + 273.15) ** 4)

    if altitude > 0: # # # # # # DAYTIME # # # # # #
        # Clearness Index on Earth's surface after Crawford and Dunchon (1999) with a correction
        #  factor for low sun elevations after Lindberg et al.(2008)
        I0, CI, Kt, I0et, CIuncorr = clearnessindex_2013b(zen, jday, Ta, RH / 100., radG, location, P)
        if (CI > 1) or (CI == np.inf):
            CI = 1

        # Estimation of radD and radI if not measured after Reindl et al.(1990)
        if onlyglobal == 1:
            I0, CI, Kt, I0et, CIuncorr = clearnessindex_2013b(zen, jday, Ta, RH / 100., radG, location, P)
            if (CI > 1) or (CI == np.inf):
                CI = 1

            radI, radD = diffusefraction(radG, altitude, Kt, Ta, RH)

        # Diffuse Radiation
        # Anisotropic Diffuse Radiation after Perez et al. 1993
        if anisdiff == 1:
            patchchoice = 1
            zenDeg = zen*(180/np.pi)
            lv = Perez_v3(zenDeg, azimuth, radD, radI, jday, patchchoice)   # Relative luminance

            aniLum = 0.
            for idx in range(0, 145):
                aniLum = aniLum + diffsh[idx] * lv[0][idx][2]     # Total relative luminance from sky into each cell

            dRad = aniLum * radD   # Total diffuse radiation from sky into each cell

        else:
            dRad = radD * svfbuveg
            lv = 0

        # # # Surface temperature parameterisation during daytime # # # #
        # new using max sun alt.instead of  dfm
        Tgamp = (TgK * altmax - Tstart) + Tstart
        Tgampwall = (TgK_wall * altmax - (Tstart_wall)) + (Tstart_wall)
        Tg = Tgamp * np.sin((((dectime - np.floor(dectime)) - SNUP / 24) / (TmaxLST / 24 - SNUP / 24)) * np.pi / 2) + Tstart # 2015 a, based on max sun altitude
        Tgwall = Tgampwall * np.sin((((dectime - np.floor(dectime)) - SNUP / 24) / (TmaxLST_wall / 24 - SNUP / 24)) * np.pi / 2) + (Tstart_wall) # 2015a, based on max sun altitude

        if Tgwall < 0:  # temporary for removing low Tg during morning 20130205
            # Tg = 0
            Tgwall = 0

        # New estimation of Tg reduction for non - clear situation based on Reindl et al.1990
        radI0, _ = diffusefraction(I0, altitude, 1., Ta, RH)
        corr = 0.1473 * np.log(90 - (zen / np.pi * 180)) + 0.3454  # 20070329 correction of lat, Lindberg et al. 2008
        CI_Tg = (radI / radI0) + (1 - corr)
        if (CI_Tg > 1) or (CI_Tg == np.inf):
            CI_Tg = 1
        Tg = Tg * CI_Tg  # new estimation
        Tgwall = Tgwall * CI_Tg

        if Tg < 0.:
            Tg = 0.
        #Tg[Tg < 0] = 0  # temporary for removing low Tg during morning 20130205

        gvf = shadow

        Lup = SBC * eground * ((gvf + Ta + Tg + 273.15) ** 4)
        LupE = Lup
        LupS = Lup
        LupW = Lup
        LupN = Lup

        # Building height angle from svfs
        F_sh = float(cylindric_wedge(zen, svfalfa, 1, 1))  # Fraction shadow on building walls based on sun alt and svf
        #F_sh[np.isnan(F_sh)] = 0.5

        # # # # # # # Calculation of shortwave daytime radiative fluxes # # # # # # #
        Kdown = radI * shadow * np.sin(altitude * (np.pi / 180)) + dRad + albedo_b * (1 - svfbuveg) * \
                            (radG * (1 - F_sh) + radD * F_sh) # *sin(altitude(i) * (pi / 180))
        
        Kup = albedo_g * (shadow * radI * np.sin(altitude * (np.pi / 180.))) + dRad + \
              albedo_b * (1 - svfbuveg) * (radG * (1 - F_sh) + radD * F_sh)

        # Kup, KupE, KupS, KupW, KupN = Kup_veg_2015a(radI, radD, radG, altitude, svf, albedo_b, F_sh, gvfalb,
        #             gvfalbE, gvfalbS, gvfalbW, gvfalbN, gvfalbnosh, gvfalbnoshE, gvfalbnoshS, gvfalbnoshW, gvfalbnoshN)

        Keast, Ksouth, Kwest, Knorth, KsideI, KsideD = Kside_veg_v2019a(radI, radD, radG, shadow, svfS, svfW, svfN, svfE,
                    svfEveg, svfSveg, svfWveg, svfNveg, azimuth, altitude, psi, t, albedo_b, F_sh, Kup, Kup, Kup,
                    Kup, cyl, lv, anisdiff, diffsh, 1, 1)

    else:  # # # # # # # NIGHTTIME # # # # # # # #

        Tgwall = 0

        # Nocturnal K fluxes set to 0
        Knight = 0.
        Kdown = 0.
        Kwest = 0.
        Kup = 0.
        Keast = 0.
        Ksouth = 0.
        Knorth = 0.
        KsideI = 0.
        KsideD = 0.
        F_sh = 0.
        Tg = 0.
        shadow = 0.

        # # # # Lup # # # #
        Lup = SBC * eground * ((Knight + Ta + Tg + 273.15) ** 4)
        # if landcover == 1:
        #     Lup[lc_grid == 3] = SBC * 0.98 * (Twater + 273.15) ** 4  # nocturnal Water temp
        LupE = Lup
        LupS = Lup
        LupW = Lup
        LupN = Lup

        I0 = 0

    # # # # Ldown # # # #
    if L_ani == 1:
        Ldown = Ldown_a + \
            (2 - svfveg - svfaveg) * ewall * SBC * ((Ta + 273.15) ** 4) + \
            (svfaveg - svf) * ewall * SBC * ((Ta + 273.15 + Tgwall) ** 4) + \
            (2 - svf - svfveg) * (1 - ewall) * esky * SBC * ((Ta + 273.15) ** 4)  # Jonsson et al.(2006)  
    else:
        Ldown = (svf + svfveg - 1) * esky * SBC * ((Ta + 273.15) ** 4) + \
            (2 - svfveg - svfaveg) * ewall * SBC * ((Ta + 273.15) ** 4) + \
            (svfaveg - svf) * ewall * SBC * ((Ta + 273.15 + Tgwall) ** 4) + \
            (2 - svf - svfveg) * (1 - ewall) * esky * SBC * ((Ta + 273.15) ** 4)  # Jonsson et al.(2006)
        # Ldown = svf * esky * SBC * ((Ta + 273.15) ** 4) + \ 
        #     (1 - svf) * ewall * SBC * ((Ta + 273.15 + Tgwall) ** 4) + \
        #     (1 - svf) * (1 - ewall) * esky * SBC * ((Ta + 273.15) ** 4)  # Jonsson et al.(2006)

    if CI < 0.95:  # non - clear conditions
        c = 1 - CI
        Ldown = Ldown * (1 - c) + \
                c * (svf * SBC * ((Ta + 273.15) ** 4) + (1 - svf) * ewall * SBC * ((Ta + 273.15 + Tgwall) ** 4) +
                     (1 - svf) * (1 - ewall) * esky * SBC * ((Ta + 273.15) ** 4))

    Ldown_i = (svf + svfveg - 1) * esky * SBC * ((Ta + 273.15) ** 4) + (2 - svfveg - svfaveg) * ewall * SBC * \
              ((Ta + 273.15) ** 4) + (svfaveg - svf) * ewall * SBC * ((Ta + 273.15 + Tgwall) ** 4) + \
              (2 - svf - svfveg) * (1 - ewall) * esky * SBC * ((Ta + 273.15) ** 4)  # Jonsson et al.(2006) # temporary for comparrison

    # # # # Lside # # # #
    Least, Lsouth, Lwest, Lnorth,Least_i,Lsouth_i,Lwest_i,Lnorth_i,Lsky_t = Lside_veg_v2020a(svfS, svfW, svfN, svfE, svfEveg, svfSveg, svfWveg, svfNveg,
                    svfEaveg, svfSaveg, svfWaveg, svfNaveg, azimuth, altitude, Ta, Tgwall, SBC, ewall, Ldown,
                                                      esky, t, F_sh, CI, LupE, LupS, LupW, LupN, L_ani, Ldown_i)
    # Least, Lsouth, Lwest, Lnorth = Lside_veg_v2015a(svfS, svfW, svfN, svfE, svfEveg, svfSveg, svfWveg, svfNveg,
    #                 svfEaveg, svfSaveg, svfWaveg, svfNaveg, azimuth, altitude, Ta, Tgwall, SBC, ewall, Ldown,
    #                                                   esky, t, F_sh, CI, LupE, LupS, LupW, LupN)

    # # # # Calculation of radiant flux density and Tmrt # # # #
    # if cyl == 1 and ani == 1:  # Human body considered as a cylinder with Perez et al. (1993)
    #     Sstr = absK * ((KsideI + KsideD) * Fcyl + (Kdown + Kup) * Fup + (Knorth + Keast + Ksouth + Kwest) * Fside) + absL * \
    #                     (Ldown * Fup + Lup * Fup + Lnorth * Fside + Least * Fside + Lsouth * Fside + Lwest * Fside)
    # elif cyl == 1 and ani == 0: # Human body considered as a cylinder with isotropic all-sky diffuse
    #     Sstr = absK * (KsideI * Fcyl + (Kdown + Kup) * Fup + (Knorth + Keast + Ksouth + Kwest) * Fside) + absL * \
    #                     (Ldown * Fup + Lup * Fup + Lnorth * Fside + Least * Fside + Lsouth * Fside + Lwest * Fside)
    # else: # Human body considered as a standing cube
    #     Sstr = absK * ((Kdown + Kup) * Fup + (Knorth + Keast + Ksouth + Kwest) * Fside) +absL * \
    #                     (Ldown * Fup + Lup * Fup + Lnorth * Fside + Least * Fside + Lsouth * Fside + Lwest * Fside)

    # # # # Calculation of radiant flux density and Tmrt # # # #
    if cyl == 1 and anisdiff == 1 and L_ani == 0:  # Human body considered as a cylinder with Perez et al. (1993)
        Sstr = absK * ((KsideI + KsideD) * Fcyl + (Kdown + Kup) * Fup + (Knorth + Keast + Ksouth + Kwest) * Fside) \
                + absL * ((Ldown + Lup) * Fup + (Lnorth + Least + Lsouth + Lwest) * Fside)
    elif cyl == 1 and anisdiff == 1 and L_ani == 1: # Human body considered as a cylinder with anisotrophic L and K)
        Sstr = absK * ((KsideI + KsideD) * Fcyl + (Kdown + Kup) * Fup + (Knorth + Keast + Ksouth + Kwest) * Fside) \
                + absL * ((Ldown + Lup) * Fup + Lside_a * Fcyl + (Lnorth + Least + Lsouth + Lwest) * Fside)
    elif cyl == 1 and anisdiff == 0: # Human body considered as a cylinder with isotropic all-sky diffuse
        Sstr = absK * (KsideI * Fcyl + (Kdown + Kup) * Fup + (Knorth + Keast + Ksouth + Kwest) * Fside) \
                + absL * ((Ldown + Lup) * Fup + (Lnorth + Least + Lsouth + Lwest) * Fside)
    else: # Human body considered as a standing cube
        Sstr = absK * ((Kdown + Kup) * Fup + (Knorth + Keast + Ksouth + Kwest) * Fside)  \
                + absL * ((Ldown + Lup) * Fup + (Lnorth + Least + Lsouth + Lwest) * Fside)
    

    Sstr_Li = absK * ((KsideI + KsideD) * Fcyl + (Kdown + Kup) * Fup + (Knorth + Keast + Ksouth + Kwest) * Fside) \
                + absL * ((Ldown + Lup) * Fup + (Lnorth_i + Least_i + Lsouth_i + Lwest_i) * Fside)
    # print("Sstr=" + str(Sstr))


    Tmrt = float(np.sqrt(np.sqrt((Sstr / (absL * SBC)))) - 273.2)

    return Tmrt, Kdown, Kup, Ldown, Lup, Tg, ea, esky, I0, CI, Keast, Ksouth, Kwest, Knorth, Least, Lsouth, Lwest, \
           Lnorth, KsideI, radI, radD, shadow


# from importdata import importdata
import numpy as np
import datetime
import calendar

def Solweig_2015a_metdata_noload(inputdata, location, UTC):
    """
    This function is used to process the input meteorological file.
    It also calculates Sun position based on the time specified in the met-file

    :param inputdata:
    :param location:
    :param UTC:
    :return:
    """

    # Newdata1, _, _ = importdata(inputdata, '\t', 1)
    # met = Newdata1['data']
    # met_header = Newdata1['textdata']
    met = inputdata
    data_len = len(met[:, 0])
    # dectime = np.empty(shape=(1, data_len))
    dectime = met[:, 1]+met[:, 2] / 24 + met[:, 3] / (60*24.)
    dectimemin = met[:, 3] / (60*24.)
    if data_len == 1:
        halftimestepdec = 0
    else:
        halftimestepdec = (dectime[1] - dectime[0]) / 2.
    time = dict()
    # time['min'] = 30
    time['sec'] = 0
    # (sunposition one halfhour before metdata)
    time['UTC'] = UTC
    sunmaximum = 0.
    leafon1 = 97  # this should change
    leafoff1 = 300  # this should change

    # initialize matrices
    # data_len = len(Newdata1['data'][:, 0])
    # data_len = len(met[:, 0])
    altitude = np.empty(shape=(1, data_len))
    azimuth = np.empty(shape=(1, data_len))
    zen = np.empty(shape=(1, data_len))
    jday = np.empty(shape=(1, data_len))
    YYYY = np.empty(shape=(1, data_len))
    leafon = np.empty(shape=(1, data_len))
    altmax = np.empty(shape=(1, data_len))

    sunmax = dict()

    for i, row in enumerate(met[:, 0]):
        YMD = datetime.datetime(int(met[i, 0]), 1, 1) + datetime.timedelta(int(met[i, 1]) - 1)
        # Finding maximum altitude in 15 min intervals (20141027)
        if (i == 0) or (np.mod(dectime[i], np.floor(dectime[i])) == 0):
            fifteen = 0.
            sunmaximum = -90.
            # sunmax.zenith = 90.
            sunmax['zenith'] = 90.
            while sunmaximum <= 90. - sunmax['zenith']:
                sunmaximum = 90. - sunmax['zenith']
                fifteen = fifteen + 15. / 1440.
                HM = datetime.timedelta(days=(60*10)/1440.0 + fifteen)
                YMDHM = YMD + HM
                time['year'] = YMDHM.year
                time['month'] = YMDHM.month
                time['day'] = YMDHM.day
                time['hour'] = YMDHM.hour
                time['min'] = YMDHM.minute
                # [time.year,time.month,time.day,time.hour,time.min,time.sec]=datevec(datenum([met[i,0],1,0])+np.floor(dectime(i,1))+(10*60)/1440+fifteen)
                sunmax = sun_position(time,location)
        altmax[0, i] = sunmaximum

        # time['year'] = float(met[i, 0])
        # time['month'] = float(met[i, 1])
        # time['day'] = float(met[i, 2])

        half = datetime.timedelta(days=halftimestepdec)
        H = datetime.timedelta(hours=met[i, 2])
        M = datetime.timedelta(minutes=met[i, 3])
        YMDHM = YMD + H + M - half
        time['year'] = YMDHM.year
        time['month'] = YMDHM.month
        time['day'] = YMDHM.day
        time['hour'] = YMDHM.hour
        time['min'] = YMDHM.minute
        sun = sun_position(time, location)
        altitude[0, i] = 90. - sun['zenith']
        azimuth[0, i] = sun['azimuth']
        zen[0, i] = sun['zenith'] * (np.pi/180.)

        # day of year and check for leap year
        if calendar.isleap(time['year']):
            dayspermonth = np.atleast_2d([31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
        else:
            dayspermonth = np.atleast_2d([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
        jday[0, i] = np.sum(dayspermonth[0, 0:time['month']-1]) + time['day']
        YYYY[0, i] = met[i, 0]
        doy = YMD.timetuple().tm_yday
        if (doy > leafon1) | (doy < leafoff1):
            leafon[0, i] = 1
        else:
            leafon[0, i] = 0

    # if time.year<2000, year=time.year-1900 ; else year=time.year-2000; end
    # [met,met_header,YYYY,altitude,azimuth,zen,jday,leafon,dectime,altmax]
    return YYYY, altitude, azimuth, zen, jday, leafon, dectime, altmax


# def isleapyear(year):
#     if (year % 4) == 0:
#         if (year % 100) == 0:
#             if (year % 400) == 0:
#                 return True
#     return False

import numpy as np

def utci_calculator(Ta, RH, Tmrt, va10m):
    # Program for calculating UTCI Temperature (UTCI)
    # released for public use after termination of COST Action 730

    # Translated from fortran by Fredrik Lindberg, Göteborg Urban Climate Group, Sweden
    # UTCI, Version a 0.002, October 2009
    # Copyright (C) 2009  Peter Broede

    if Ta <= -999 or RH <= -999 or va10m <= -999 or Tmrt <= -999:
        UTCI_approx = -999
    else:
        # saturation vapour pressure (es)
        g = np.array([-2.8365744E3, - 6.028076559E3, 1.954263612E1, - 2.737830188E-2,
                      1.6261698E-5, 7.0229056E-10, - 1.8680009E-13, 2.7150305])

        tk = Ta + 273.15  # ! air temp in K
        es = g[7] * np.log(tk)
        for i in range(0, 7):
            es = es + g[i] * tk ** (i + 1 - 3.)

        es = np.exp(es) * 0.01

        ehPa = es * RH / 100.

        D_Tmrt = Tmrt - Ta
        Pa = ehPa / 10.0  # use vapour pressure in kPa
        va = va10m

        # calculate 6th order polynomial as approximation
        UTCI_approx = Ta + \
        (6.07562052E-01) + \
        (-2.27712343E-02) * Ta + \
        (8.06470249E-04) * Ta * Ta + \
        (-1.54271372E-04) * Ta * Ta * Ta + \
        (-3.24651735E-06) * Ta * Ta * Ta * Ta + \
        (7.32602852E-08) * Ta * Ta * Ta * Ta * Ta + \
        (1.35959073E-09) * Ta * Ta * Ta * Ta * Ta * Ta + \
        (-2.25836520E+00) * va + \
        (8.80326035E-02) * Ta * va + \
        (2.16844454E-03) * Ta * Ta * va + \
        (-1.53347087E-05) * Ta * Ta * Ta * va + \
        (-5.72983704E-07) * Ta * Ta * Ta * Ta * va + \
        (-2.55090145E-09) * Ta * Ta * Ta * Ta * Ta * va + \
        (-7.51269505E-01) * va * va + \
        (-4.08350271E-03) * Ta * va * va + \
        (-5.21670675E-05) * Ta * Ta * va * va + \
        (1.94544667E-06) * Ta * Ta * Ta * va * va + \
        (1.14099531E-08) * Ta * Ta * Ta * Ta * va * va + \
        (1.58137256E-01) * va * va * va + \
        (-6.57263143E-05) * Ta * va * va * va + \
        (2.22697524E-07) * Ta * Ta * va * va * va + \
        (-4.16117031E-08) * Ta * Ta * Ta * va * va * va + \
        (-1.27762753E-02) * va * va * va * va + \
        (9.66891875E-06) * Ta * va * va * va * va + \
        (2.52785852E-09) * Ta * Ta * va * va * va * va + \
        (4.56306672E-04) * va * va * va * va * va + \
        (-1.74202546E-07) * Ta * va * va * va * va * va + \
        (-5.91491269E-06) * va * va * va * va * va * va + \
        (3.98374029E-01) * D_Tmrt + \
        (1.83945314E-04) * Ta * D_Tmrt + \
        (-1.73754510E-04) * Ta * Ta * D_Tmrt + \
        (-7.60781159E-07) * Ta * Ta * Ta * D_Tmrt + \
        (3.77830287E-08) * Ta * Ta * Ta * Ta * D_Tmrt + \
        (5.43079673E-10) * Ta * Ta * Ta * Ta * Ta * D_Tmrt + \
        (-2.00518269E-02) * va * D_Tmrt + \
        (8.92859837E-04) * Ta * va * D_Tmrt + \
        (3.45433048E-06) * Ta * Ta * va * D_Tmrt + \
        (-3.77925774E-07) * Ta * Ta * Ta * va * D_Tmrt + \
        (-1.69699377E-09) * Ta * Ta * Ta * Ta * va * D_Tmrt + \
        (1.69992415E-04) * va * va * D_Tmrt + \
        (-4.99204314E-05) * Ta * va * va * D_Tmrt + \
        (2.47417178E-07) * Ta * Ta * va * va * D_Tmrt + \
        (1.07596466E-08) * Ta * Ta * Ta * va * va * D_Tmrt + \
        (8.49242932E-05) * va * va * va * D_Tmrt + \
        (1.35191328E-06) * Ta * va * va * va * D_Tmrt + \
        (-6.21531254E-09) * Ta * Ta * va * va * va * D_Tmrt + \
        (-4.99410301E-06) * va * va * va * va * D_Tmrt + \
        (-1.89489258E-08) * Ta * va * va * va * va * D_Tmrt + \
        (8.15300114E-08) * va * va * va * va * va * D_Tmrt + \
        (7.55043090E-04) * D_Tmrt * D_Tmrt + \
        (-5.65095215E-05) * Ta * D_Tmrt * D_Tmrt + \
        (-4.52166564E-07) * Ta * Ta * D_Tmrt * D_Tmrt + \
        (2.46688878E-08) * Ta * Ta * Ta * D_Tmrt * D_Tmrt + \
        (2.42674348E-10) * Ta * Ta * Ta * Ta * D_Tmrt * D_Tmrt + \
        (1.54547250E-04) * va * D_Tmrt * D_Tmrt + \
        (5.24110970E-06) * Ta * va * D_Tmrt * D_Tmrt + \
        (-8.75874982E-08) * Ta * Ta * va * D_Tmrt * D_Tmrt + \
        (-1.50743064E-09) * Ta * Ta * Ta * va * D_Tmrt * D_Tmrt + \
        (-1.56236307E-05) * va * va * D_Tmrt * D_Tmrt + \
        (-1.33895614E-07) * Ta * va * va * D_Tmrt * D_Tmrt + \
        (2.49709824E-09) * Ta * Ta * va * va * D_Tmrt * D_Tmrt + \
        (6.51711721E-07) * va * va * va * D_Tmrt * D_Tmrt + \
        (1.94960053E-09) * Ta * va * va * va * D_Tmrt * D_Tmrt + \
        (-1.00361113E-08) * va * va * va * va * D_Tmrt * D_Tmrt + \
        (-1.21206673E-05) * D_Tmrt * D_Tmrt * D_Tmrt + \
        (-2.18203660E-07) * Ta * D_Tmrt * D_Tmrt * D_Tmrt + \
        (7.51269482E-09) * Ta * Ta * D_Tmrt * D_Tmrt * D_Tmrt + \
        (9.79063848E-11) * Ta * Ta * Ta * D_Tmrt * D_Tmrt * D_Tmrt + \
        (1.25006734E-06) * va * D_Tmrt * D_Tmrt * D_Tmrt + \
        (-1.81584736E-09) * Ta * va * D_Tmrt * D_Tmrt * D_Tmrt + \
        (-3.52197671E-10) * Ta * Ta * va * D_Tmrt * D_Tmrt * D_Tmrt + \
        (-3.36514630E-08) * va * va * D_Tmrt * D_Tmrt * D_Tmrt + \
        (1.35908359E-10) * Ta * va * va * D_Tmrt * D_Tmrt * D_Tmrt + \
        (4.17032620E-10) * va * va * va * D_Tmrt * D_Tmrt * D_Tmrt + \
        (-1.30369025E-09) * D_Tmrt * D_Tmrt * D_Tmrt * D_Tmrt + \
        (4.13908461E-10) * Ta * D_Tmrt * D_Tmrt * D_Tmrt * D_Tmrt + \
        (9.22652254E-12) * Ta * Ta * D_Tmrt * D_Tmrt * D_Tmrt * D_Tmrt + \
        (-5.08220384E-09) * va * D_Tmrt * D_Tmrt * D_Tmrt * D_Tmrt + \
        (-2.24730961E-11) * Ta * va * D_Tmrt * D_Tmrt * D_Tmrt * D_Tmrt + \
        (1.17139133E-10) * va * va * D_Tmrt * D_Tmrt * D_Tmrt * D_Tmrt + \
        (6.62154879E-10) * D_Tmrt * D_Tmrt * D_Tmrt * D_Tmrt * D_Tmrt + \
        (4.03863260E-13) * Ta * D_Tmrt * D_Tmrt * D_Tmrt * D_Tmrt * D_Tmrt + \
        (1.95087203E-12) * va * D_Tmrt * D_Tmrt * D_Tmrt * D_Tmrt * D_Tmrt + \
        (-4.73602469E-12) * D_Tmrt * D_Tmrt * D_Tmrt * D_Tmrt * D_Tmrt * D_Tmrt + \
        (5.12733497E+00) * Pa + \
        (-3.12788561E-01) * Ta * Pa + \
        (-1.96701861E-02) * Ta * Ta * Pa + \
        (9.99690870E-04) * Ta * Ta * Ta * Pa + \
        (9.51738512E-06) * Ta * Ta * Ta * Ta * Pa + \
        (-4.66426341E-07) * Ta * Ta * Ta * Ta * Ta * Pa + \
        (5.48050612E-01) * va * Pa + \
        (-3.30552823E-03) * Ta * va * Pa + \
        (-1.64119440E-03) * Ta * Ta * va * Pa + \
        (-5.16670694E-06) * Ta * Ta * Ta * va * Pa + \
        (9.52692432E-07) * Ta * Ta * Ta * Ta * va * Pa + \
        (-4.29223622E-02) * va * va * Pa + \
        (5.00845667E-03) * Ta * va * va * Pa + \
        (1.00601257E-06) * Ta * Ta * va * va * Pa + \
        (-1.81748644E-06) * Ta * Ta * Ta * va * va * Pa + \
        (-1.25813502E-03) * va * va * va * Pa + \
        (-1.79330391E-04) * Ta * va * va * va * Pa + \
        (2.34994441E-06) * Ta * Ta * va * va * va * Pa + \
        (1.29735808E-04) * va * va * va * va * Pa + \
        (1.29064870E-06) * Ta * va * va * va * va * Pa + \
        (-2.28558686E-06) * va * va * va * va * va * Pa + \
        (-3.69476348E-02) * D_Tmrt * Pa + \
        (1.62325322E-03) * Ta * D_Tmrt * Pa + \
        (-3.14279680E-05) * Ta * Ta * D_Tmrt * Pa + \
        (2.59835559E-06) * Ta * Ta * Ta * D_Tmrt * Pa + \
        (-4.77136523E-08) * Ta * Ta * Ta * Ta * D_Tmrt * Pa + \
        (8.64203390E-03) * va * D_Tmrt * Pa + \
        (-6.87405181E-04) * Ta * va * D_Tmrt * Pa + \
        (-9.13863872E-06) * Ta * Ta * va * D_Tmrt * Pa + \
        (5.15916806E-07) * Ta * Ta * Ta * va * D_Tmrt * Pa + \
        (-3.59217476E-05) * va * va * D_Tmrt * Pa + \
        (3.28696511E-05) * Ta * va * va * D_Tmrt * Pa + \
        (-7.10542454E-07) * Ta * Ta * va * va * D_Tmrt * Pa + \
        (-1.24382300E-05) * va * va * va * D_Tmrt * Pa + \
        (-7.38584400E-09) * Ta * va * va * va * D_Tmrt * Pa + \
        (2.20609296E-07) * va * va * va * va * D_Tmrt * Pa + \
        (-7.32469180E-04) * D_Tmrt * D_Tmrt * Pa + \
        (-1.87381964E-05) * Ta * D_Tmrt * D_Tmrt * Pa + \
        (4.80925239E-06) * Ta * Ta * D_Tmrt * D_Tmrt * Pa + \
        (-8.75492040E-08) * Ta * Ta * Ta * D_Tmrt * D_Tmrt * Pa + \
        (2.77862930E-05) * va * D_Tmrt * D_Tmrt * Pa + \
        (-5.06004592E-06) * Ta * va * D_Tmrt * D_Tmrt * Pa + \
        (1.14325367E-07) * Ta * Ta * va * D_Tmrt * D_Tmrt * Pa + \
        (2.53016723E-06) * va * va * D_Tmrt * D_Tmrt * Pa + \
        (-1.72857035E-08) * Ta * va * va * D_Tmrt * D_Tmrt * Pa + \
        (-3.95079398E-08) * va * va * va * D_Tmrt * D_Tmrt * Pa + \
        (-3.59413173E-07) * D_Tmrt * D_Tmrt * D_Tmrt * Pa + \
        (7.04388046E-07) * Ta * D_Tmrt * D_Tmrt * D_Tmrt * Pa + \
        (-1.89309167E-08) * Ta * Ta * D_Tmrt * D_Tmrt * D_Tmrt * Pa + \
        (-4.79768731E-07) * va * D_Tmrt * D_Tmrt * D_Tmrt * Pa + \
        (7.96079978E-09) * Ta * va * D_Tmrt * D_Tmrt * D_Tmrt * Pa + \
        (1.62897058E-09) * va * va * D_Tmrt * D_Tmrt * D_Tmrt * Pa + \
        (3.94367674E-08) * D_Tmrt * D_Tmrt * D_Tmrt * D_Tmrt * Pa + \
        (-1.18566247E-09) * Ta * D_Tmrt * D_Tmrt * D_Tmrt * D_Tmrt * Pa + \
        (3.34678041E-10) * va * D_Tmrt * D_Tmrt * D_Tmrt * D_Tmrt * Pa + \
        (-1.15606447E-10) * D_Tmrt * D_Tmrt * D_Tmrt * D_Tmrt * D_Tmrt * Pa + \
        (-2.80626406E+00) * Pa * Pa + \
        (5.48712484E-01) * Ta * Pa * Pa + \
        (-3.99428410E-03) * Ta * Ta * Pa * Pa + \
        (-9.54009191E-04) * Ta * Ta * Ta * Pa * Pa + \
        (1.93090978E-05) * Ta * Ta * Ta * Ta * Pa * Pa + \
        (-3.08806365E-01) * va * Pa * Pa + \
        (1.16952364E-02) * Ta * va * Pa * Pa + \
        (4.95271903E-04) * Ta * Ta * va * Pa * Pa + \
        (-1.90710882E-05) * Ta * Ta * Ta * va * Pa * Pa + \
        (2.10787756E-03) * va * va * Pa * Pa + \
        (-6.98445738E-04) * Ta * va * va * Pa * Pa + \
        (2.30109073E-05) * Ta * Ta * va * va * Pa * Pa + \
        (4.17856590E-04) * va * va * va * Pa * Pa + \
        (-1.27043871E-05) * Ta * va * va * va * Pa * Pa + \
        (-3.04620472E-06) * va * va * va * va * Pa * Pa + \
        (5.14507424E-02) * D_Tmrt * Pa * Pa + \
        (-4.32510997E-03) * Ta * D_Tmrt * Pa * Pa + \
        (8.99281156E-05) * Ta * Ta * D_Tmrt * Pa * Pa + \
        (-7.14663943E-07) * Ta * Ta * Ta * D_Tmrt * Pa * Pa + \
        (-2.66016305E-04) * va * D_Tmrt * Pa * Pa + \
        (2.63789586E-04) * Ta * va * D_Tmrt * Pa * Pa + \
        (-7.01199003E-06) * Ta * Ta * va * D_Tmrt * Pa * Pa + \
        (-1.06823306E-04) * va * va * D_Tmrt * Pa * Pa + \
        (3.61341136E-06) * Ta * va * va * D_Tmrt * Pa * Pa + \
        (2.29748967E-07) * va * va * va * D_Tmrt * Pa * Pa + \
        (3.04788893E-04) * D_Tmrt * D_Tmrt * Pa * Pa + \
        (-6.42070836E-05) * Ta * D_Tmrt * D_Tmrt * Pa * Pa + \
        (1.16257971E-06) * Ta * Ta * D_Tmrt * D_Tmrt * Pa * Pa + \
        (7.68023384E-06) * va * D_Tmrt * D_Tmrt * Pa * Pa + \
        (-5.47446896E-07) * Ta * va * D_Tmrt * D_Tmrt * Pa * Pa + \
        (-3.59937910E-08) * va * va * D_Tmrt * D_Tmrt * Pa * Pa + \
        (-4.36497725E-06) * D_Tmrt * D_Tmrt * D_Tmrt * Pa * Pa + \
        (1.68737969E-07) * Ta * D_Tmrt * D_Tmrt * D_Tmrt * Pa * Pa + \
        (2.67489271E-08) * va * D_Tmrt * D_Tmrt * D_Tmrt * Pa * Pa + \
        (3.23926897E-09) * D_Tmrt * D_Tmrt * D_Tmrt * D_Tmrt * Pa * Pa + \
        (-3.53874123E-02) * Pa * Pa * Pa + \
        (-2.21201190E-01) * Ta * Pa * Pa * Pa + \
        (1.55126038E-02) * Ta * Ta * Pa * Pa * Pa + \
        (-2.63917279E-04) * Ta * Ta * Ta * Pa * Pa * Pa + \
        (4.53433455E-02) * va * Pa * Pa * Pa + \
        (-4.32943862E-03) * Ta * va * Pa * Pa * Pa + \
        (1.45389826E-04) * Ta * Ta * va * Pa * Pa * Pa + \
        (2.17508610E-04) * va * va * Pa * Pa * Pa + \
        (-6.66724702E-05) * Ta * va * va * Pa * Pa * Pa + \
        (3.33217140E-05) * va * va * va * Pa * Pa * Pa + \
        (-2.26921615E-03) * D_Tmrt * Pa * Pa * Pa + \
        (3.80261982E-04) * Ta * D_Tmrt * Pa * Pa * Pa + \
        (-5.45314314E-09) * Ta * Ta * D_Tmrt * Pa * Pa * Pa + \
        (-7.96355448E-04) * va * D_Tmrt * Pa * Pa * Pa + \
        (2.53458034E-05) * Ta * va * D_Tmrt * Pa * Pa * Pa + \
        (-6.31223658E-06) * va * va * D_Tmrt * Pa * Pa * Pa + \
        (3.02122035E-04) * D_Tmrt * D_Tmrt * Pa * Pa * Pa + \
        (-4.77403547E-06) * Ta * D_Tmrt * D_Tmrt * Pa * Pa * Pa + \
        (1.73825715E-06) * va * D_Tmrt * D_Tmrt * Pa * Pa * Pa + \
        (-4.09087898E-07) * D_Tmrt * D_Tmrt * D_Tmrt * Pa * Pa * Pa + \
        (6.14155345E-01) * Pa * Pa * Pa * Pa + \
        (-6.16755931E-02) * Ta * Pa * Pa * Pa * Pa + \
        (1.33374846E-03) * Ta * Ta * Pa * Pa * Pa * Pa + \
        (3.55375387E-03) * va * Pa * Pa * Pa * Pa + \
        (-5.13027851E-04) * Ta * va * Pa * Pa * Pa * Pa + \
        (1.02449757E-04) * va * va * Pa * Pa * Pa * Pa + \
        (-1.48526421E-03) * D_Tmrt * Pa * Pa * Pa * Pa + \
        (-4.11469183E-05) * Ta * D_Tmrt * Pa * Pa * Pa * Pa + \
        (-6.80434415E-06) * va * D_Tmrt * Pa * Pa * Pa * Pa + \
        (-9.77675906E-06) * D_Tmrt * D_Tmrt * Pa * Pa * Pa * Pa + \
        (8.82773108E-02) * Pa * Pa * Pa * Pa * Pa + \
        (-3.01859306E-03) * Ta * Pa * Pa * Pa * Pa * Pa + \
        (1.04452989E-03) * va * Pa * Pa * Pa * Pa * Pa + \
        (2.47090539E-04) * D_Tmrt * Pa * Pa * Pa * Pa * Pa + \
        (1.48348065E-03) * Pa * Pa * Pa * Pa * Pa * Pa

    return UTCI_approx

author = 'xlinfr'

import numpy as np
import math

def clearnessindex_2013b(zen, jday, Ta, RH, radG, location, P):

    """ Clearness Index at the Earth's surface calculated from Crawford and Duchon 1999

    :param zen: zenith angle in radians
    :param jday: day of year
    :param Ta: air temperature
    :param RH: relative humidity
    :param radG: global shortwave radiation
    :param location: distionary including lat, lon and alt
    :param P: pressure
    :return:
    """

    if P == -999.0:
        p = 1013.  # Pressure in millibars
    if p < 500.:
        p = p*10.  # Convert from hPa to millibars

    Itoa = 1370.0  # Effective solar constant
    D = sun_distance.sun_distance(jday)  # irradiance differences due to Sun-Earth distances
    m = 35. * np.cos(zen) * ((1224. * (np.cos(zen)**2) + 1) ** (-1/2.))     # optical air mass at p=1013
    Trpg = 1.021-0.084*(m*(0.000949*p+0.051))**0.5  # Transmission coefficient for Rayliegh scattering and permanent gases

    # empirical constant depending on latitude
    if location['latitude'] < 10.:
        G = [3.37, 2.85, 2.80, 2.64]
    elif location['latitude'] >= 10. and location['latitude'] < 20.:
        G = [2.99, 3.02, 2.70, 2.93]
    elif location['latitude'] >= 20. and location['latitude'] <30.:
        G = [3.60, 3.00, 2.98, 2.93]
    elif location['latitude'] >= 30. and location['latitude'] <40.:
        G = [3.04, 3.11, 2.92, 2.94]
    elif location['latitude'] >= 40. and location['latitude'] <50.:
        G = [2.70, 2.95, 2.77, 2.71]
    elif location['latitude'] >= 50. and location['latitude'] <60.:
        G = [2.52, 3.07, 2.67, 2.93]
    elif location['latitude'] >= 60. and location['latitude'] <70.:
        G = [1.76, 2.69, 2.61, 2.61]
    elif location['latitude'] >= 70. and location['latitude'] <80.:
        G = [1.60, 1.67, 2.24, 2.63]
    elif location['latitude'] >= 80. and location['latitude'] <90.:
        G = [1.11, 1.44, 1.94, 2.02]

    if jday > 335 or jday <= 60:
        G = G[0]
    elif jday > 60 and jday <= 152:
        G = G[1]
    elif jday > 152 and jday <= 244:
        G = G[2]
    elif jday > 244 and jday <= 335:
        G = G[3]

    # dewpoint calculation
    a2 = 17.27
    b2 = 237.7
    Td = (b2*(((a2*Ta)/(b2+Ta))+np.log(RH)))/(a2-(((a2*Ta)/(b2+Ta))+np.log(RH)))
    Td = (Td*1.8)+32  # Dewpoint (F)
    u = np.exp(0.1133-np.log(G+1)+0.0393*Td)  # Precipitable water
    Tw = 1-0.077*((u*m)**0.3)  # Transmission coefficient for water vapor
    Tar = 0.935**m  # Transmission coefficient for aerosols

    I0=Itoa*np.cos(zen)*Trpg*Tw*D*Tar
    if abs(zen)>np.pi/2:
        I0 = 0
    # b=I0==abs(zen)>np.pi/2
    # I0(b==1)=0
    # clear b;
    if not(np.isreal(I0)):
        I0 = 0

    corr=0.1473*np.log(90-(zen/np.pi*180))+0.3454  # 20070329

    CIuncorr = radG / I0
    CI = CIuncorr + (1-corr)
    I0et = Itoa*np.cos(zen)*D  # extra terrestial solar radiation
    Kt = radG / I0et
    if math.isnan(CI):
        CI = float('Inf')

    return I0, CI, Kt, I0et, CIuncorr


import numpy as np

def cylindric_wedge(zen, svfalfa, rows, cols):

    # Fraction of sunlit walls based on sun altitude and svf wieghted building angles
    # input: 
    # sun zenith angle "beta"
    # svf related angle "alfa"

    beta=zen
    alfa=np.zeros((rows, cols)) + svfalfa
    
    # measure the size of the image
    # sizex=size(svfalfa,2)
    # sizey=size(svfalfa,1)
    
    xa=1-2./(np.tan(alfa)*np.tan(beta))
    ha=2./(np.tan(alfa)*np.tan(beta))
    ba=(1./np.tan(alfa))
    hkil=2.*ba*ha
    
    qa=np.zeros((rows, cols))
    # qa(length(svfalfa),length(svfalfa))=0;
    qa[xa<0]=np.tan(beta)/2
    
    Za=np.zeros((rows, cols))
    # Za(length(svfalfa),length(svfalfa))=0;
    Za[xa<0]=((ba[xa<0]**2)-((qa[xa<0]**2)/4))**0.5
    
    phi=np.zeros((rows, cols))
    #phi(length(svfalfa),length(svfalfa))=0;
    phi[xa<0]=np.arctan(Za[xa<0]/qa[xa<0])
    
    A=np.zeros((rows, cols))
    # A(length(svfalfa),length(svfalfa))=0;
    A[xa<0]=(np.sin(phi[xa<0])-phi[xa<0]*np.cos(phi[xa<0]))/(1-np.cos(phi[xa<0]))
    
    ukil=np.zeros((rows, cols))
    # ukil(length(svfalfa),length(svfalfa))=0
    ukil[xa<0]=2*ba[xa<0]*xa[xa<0]*A[xa<0]
    
    Ssurf=hkil+ukil
    
    F_sh=(2*np.pi*ba-Ssurf)/(2*np.pi*ba)#Xa
    
    return F_sh



import numpy as np

def daylen(DOY, XLAT):
    # Calculation of declination of sun (Eqn. 16). Amplitude= +/-23.45
    # deg. Minimum = DOY 355 (DEC 21), maximum = DOY 172.5 (JUN 21/22).
    # Sun angles.  SOC limited for latitudes above polar circles.
    # Calculate daylength, sunrise and sunset (Eqn. 17)

    RAD=np.pi/180.0

    DEC = -23.45 * np.cos(2.0*np.pi*(DOY+10.0)/365.0)

    SOC = np.tan(RAD*DEC) * np.tan(RAD*XLAT)
    SOC = min(max(SOC,-1.0),1.0)
    # SOC=alt

    DAYL = 12.0 + 24.0*np.arcsin(SOC)/np.pi
    SNUP = 12.0 - DAYL/2.0
    SNDN = 12.0 + DAYL/2.0

    return DAYL, DEC, SNDN, SNUP

from __future__ import division
import numpy as np

def diffusefraction(radG,altitude,Kt,Ta,RH):
    """
    This function estimates diffuse and directbeam radiation according to
    Reindl et al (1990), Solar Energy 45:1

    :param radG:
    :param altitude:
    :param Kt:
    :param Ta:
    :param RH:
    :return:
    """

    alfa = altitude*(np.pi/180)

    if Ta <= -999.00 or RH <= -999.00 or np.isnan(Ta) or np.isnan(RH):
        if Kt <= 0.3:
            radD = radG*(1.020-0.248*Kt)
        elif Kt > 0.3 and Kt < 0.78:
            radD = radG*(1.45-1.67*Kt)
        else:
            radD = radG*0.147
    else:
        RH = RH/100
        if Kt <= 0.3:
            radD = radG*(1 - 0.232 * Kt + 0.0239 * np.sin(alfa) - 0.000682 * Ta + 0.0195 * RH)
        elif Kt > 0.3 and Kt < 0.78:
            radD = radG*(1.329- 1.716 * Kt + 0.267 * np.sin(alfa) - 0.00357 * Ta + 0.106 * RH)
        else:
            radD = radG*(0.426 * Kt - 0.256 * np.sin(alfa) + 0.00349 * Ta + 0.0734 * RH)

    radI = (radG - radD)/(np.sin(alfa))

    # Corrections for low sun altitudes (20130307)
    if radI < 0:
        radI = 0

    if altitude < 1 and radI > radG:
        radI=radG

    if radD > radG:
        radD = radG

    return radI, radD



from flask import Flask, render_template, request, session
import numpy as np
import clearnessindex_2013b as ci
import requests
import json
import base64
import pandas as pd


app = Flask(__name__)
app.config["DEBUG"] = True
app.config["SECRET_KEY"] = "klefiedoedfoiefnnoefnveodf"

calcresult = []
result = []

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/petresult', methods=["GET", "POST"])
def petresult():
    return render_template("petresult.html", result=result)

@app.route('/prognoseerror', methods=["GET", "POST"])
def prognoseerror():
    return render_template("prognoseerror.html", result=result)

@app.route('/prognose', methods=["GET", "POST"])
def prognose():
    if request.method == "GET":
        return render_template("prognose.html")

    if request.method == "POST":
        city = request.form["city"]
        if city == 'Gothenburg, Sweden':
            url = 'https://api.github.com/repos/David-Rayner-GVC/pet_data/contents/json/Gothenburg.json'
            lat = 57.7
            lon = 12.0
            UTC = 1
        else:
            return render_template("prognoseerror.html", result='Prognose for ' + city + ', not found.')

        # download data from github
        req = requests.get(url)
        if req.status_code == requests.codes.ok:
            req = req.json()  # the response is a JSON
            content = base64.b64decode(req['content'])
        else:
            return render_template("petprognoseresult.html", result='Content was not found.')

        dict_loaded = json.loads(content)

        for key, value in dict_loaded['data_vars'].items():
            dict_loaded['data_vars'][key]['data'] = [np.nan if isinstance(x,str) else x for x in value['data'] ]

        timestamp = dict_loaded['coords']['time']['data']

        # putting data in separate vectors
        veclen = timestamp.__len__()
        year = np.empty(veclen, int)
        month = np.empty(veclen, int)
        day = np.empty(veclen, int)
        hour = np.empty(veclen, int)
        minu = np.empty(veclen, int)
        year = np.empty(veclen, int)
        Ta = np.empty(veclen, float)
        RH = np.empty(veclen, float)
        radD = np.empty(veclen, float)
        radI = np.empty(veclen, float)
        radG = np.empty(veclen, float)
        Ws = np.empty(veclen, float)

        for i in range(0, veclen):
            year[i] = int(timestamp[i][0:4])
            month[i] = int(timestamp[i][5:7])
            day[i] = int(timestamp[i][8:10])
            hour[i] = int(timestamp[i][11:13])
            minu[i] = int(timestamp[i][14:16])
            Ta[i] = float(dict_loaded['data_vars']['air_temperature']['data'][i])
            RH[i] = float(dict_loaded['data_vars']['relative_humidity']['data'][i])
            radD[i] = float(dict_loaded['data_vars']['downward_diffuse']['data'][i])
            radI[i] = float(dict_loaded['data_vars']['downward_direct']['data'][i])
            Ws[i] = np.sqrt(float(dict_loaded['data_vars']['eastward_wind']['data'][i])**2 + float(dict_loaded['data_vars']['northward_wind']['data'][i])**2)
            
        with np.errstate(invalid='ignore'):
          radI[radI < 0.] = 0.
          radD[radD < 0.] = 0.
        radG = radD + radI

        # re-create xarray Dataset
        #x_loaded = Dataset.from_dict(dict_loaded)

        # putting data in separate vectors
        #year = np.empty((x_loaded.air_temperature.time.shape[0]), int)

        #uResponse = requests.get(uri)
        #try:
        #    uResponse = requests.get(uri)
        #except requests.ConnectionError:
        #    return "Connection Error"

        #Jresponse = uResponse.text
        #result = json.loads(Jresponse)

        #with urllib.request.urlopen(uri) as url:
        #    result = json.loads(url.read().decode())

        #return '''
        #        <html>
        #            <body>
        #                <div class="container">
        ##                    <p>{result}</p>
        #                    <p><a href="/">Click here to calculate again</a>
        #                <div>
        #            </body>
        #        </html>
        #    '''.format(result=city)

        poi_save = petcalcprognose(Ta, RH, Ws, radG, radD, radI, year, month, day, hour, minu, lat, lon, UTC)

        tab = pd.DataFrame(poi_save[1:,[1,2,22,24,26,33]])
        tab.columns = ['Day of Year', 'Hour','T_air','RH','Tmrt', 'PET']

        tabhtml = tab.to_html(classes='data', header="true")

        doy = poi_save[1:, 1]
        hour = poi_save[1:, 2]
        petres = poi_save[:,26]
        #petres = str(round(poi_save[:,26], 1))

        return render_template("petprognoseresult.html", result1=doy, result2=hour, result3=tabhtml)


@app.route('/petprognoseresult', methods=["GET", "POST"])
def petprognoseresult():
    if request.method == "GET":
        return render_template("petprognoseresult.html", result1=result1, result2=result2, result3=result3)

    if request.method == "POST":
        return render_template("petprognoseresult.html", result1=result1, result2=result2, result3=result3)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template("main_page.html", calcresult=calcresult)

    if request.method == "POST":
        try:
            month = int(request.form["month"])
        except:
            month = -999
        try:
            day = int(request.form["day"])
        except:
            day = -999
        try:
            hour = int(request.form["hour"])
        except:
            hour = -999
        year = 2019
        minu = 30
        try:
            Ta = float(request.form["Ta"])
        except:
            Ta = -999
        try:
            RH = float(request.form["RH"])
        except:
            RH = -999
        try:
            Ws = float(request.form["Ws"])
        except:
            Ws = -999
        #try:
        #    radG = float(request.form["radG"])
        #except:
        #    errors += "<p>{!r} is not a number.</p>\n".format(request.form["radG"])
        sky = request.form["sky"]

        if month > 12 or month < 0:
            return render_template("petresult.html", result="Incorrect month filled in")
        if day > 31 or day < 0:
            return render_template("petresult.html", result="Incorrect day filled in")
        if hour > 23 or hour < 0:
            return render_template("petresult.html", result="Incorrect hour filled in")
        if Ta > 60 or Ta < -60:
            return render_template("petresult.html", result="Unreasonable air temperature filled in")
        if RH > 100 or RH < 0:
            return render_template("petresult.html", result="Unreasonable relative humidity filled in")
        if Ws > 100 or Ws < 0:
            return render_template("petresult.html", result="Unreasonable Wind speed filled in")

        #day of year
        if (year % 4) == 0:
            if (year % 100) == 0:
                if (year % 400) == 0:
                    leapyear = 1
                else:
                    leapyear = 0
            else:
                leapyear = 1
        else:
            leapyear = 0

        if leapyear == 1:
            dayspermonth = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        else:
            dayspermonth = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

        doy = np.sum(dayspermonth[0:month - 1]) + day

        # Currently looked to Gothenburg
        location = {'longitude': 12.0, 'latitude': 57.7, 'altitude': 3.}
        UTC = 1

        # Radiation
        P = -999.
        radG = 40.

        metdata = np.zeros((1, 24)) - 999.
        metdata[0, 0] = year
        metdata[0, 1] = doy
        metdata[0, 2] = hour
        metdata[0, 3] = minu
        metdata[0, 11] = Ta
        metdata[0, 10] = RH

        YYYY, altitude, azimuth, zen, jday, leafon, dectime, altmax = Solweig_2015a_metdata_noload(metdata, location, UTC)
        if altitude > 0.:
            I0, _, Kt, _, _ = ci.clearnessindex_2013b(zen, jday, Ta, RH / 100., radG, location, P)

            if sky == "Clear (100%)":
                radG = I0
            elif sky == "Semi-cloudy (80%)":
                radG = I0 * 0.8
            elif sky == "Cloudy (60%)":
                radG = I0 * 0.6
            else:
                radG = I0 * 0.4

            I0, _, Kt, _, _ = ci.clearnessindex_2013b(zen, jday, Ta, RH / 100., radG, location, P)
        else:
            radG = 0.

        # Main calculation
        if Ta is not None and RH is not None and Ws is not None and radG is not None:
            Tmrt, resultPET, resultUTCI = petcalc(Ta, RH, Ws, radG, year, month, day, hour, minu)
            result = str(round(resultPET, 1))
            return render_template("petresult.html", result=result)

            #'''
            #    <html>
            #        <body>
            #            <div class="container">
            #                <p><font size="14">{result}</font></p>
            #                <p><a href="/"><font size="10">Click here to calculate again</font></a>
            #            <div>
            #        </body>
            #    </html>
            #'''.format(result=testout)

#
#
#from flask import Flask, request, session
#
#from processing import calculate_mode
#
#app = Flask(__name__)
#app.config["DEBUG"] = True
#app.config["SECRET_KEY"] = "klefiedoedfoiefnnoefnveodf"
#
##inputs = []
#
#@app.route("/", methods=["GET", "POST"])
#def mode_page():
#    if "inputs" not in session:
#        session["inputs"] = []
#    errors = ""
#    if request.method == "POST":
#        try:
#            #inputs.append(float(request.form["number"]))
#            session["inputs"].append(float(request.form["number"]))
#            session.modified = True
#        except:
#            errors += "<p>{!r} is not a number.</p>\n".format(request.form["number"])
#
#        if request.form["action"] == "Calculate number":
#            #result = calculate_mode(inputs)
#            result = calculate_mode(session["inputs"])
#            #inputs.clear()
#            session["inputs"].clear()
#            session.modified = True
#            return '''
#                <html>
#                    <body>
#                        <p>{result}</p>
#                        <p><a href="/">Click here to calculate again</a>
#                    </body>
#                </html>
#            '''.format(result=result)
#
##    if len(inputs) == 0:
#    if len(session["inputs"]) == 0:
#        numbers_so_far = ""
#    else:
#        numbers_so_far = "<p>Numbers so far:</p>"
#        for number in session["inputs"]:
#        #for number in inputs:
#            numbers_so_far += "<p>{}</p>".format(number)
#
#    return '''
#        <html>
#            <body>
#                {numbers_so_far}
#                {errors}
#                <p>Enter your number:</p>
#                <form method="post" action=".">
#                    <p><input name="number" /></p>
#                    <p><input type="submit" name="action" value="Add another" /></p>
#                    <p><input type="submit" name="action" value="Calculate number" /></p>
#                </form>
#            </body>
#        </html>
#    '''.format(numbers_so_far=numbers_so_far, errors=errors)
#



#from flask import Flask, request
#from processing import do_calculation
#
#app = Flask(__name__)
#app.config["DEBUG"] = True
#
#@app.route("/", methods=["GET", "POST"])
#def adder_page():
#    errors = ""
#    if request.method == "POST":
#        number1 = None
#        number2 = None
#        try:
#            number1 = float(request.form["number1"])
#        except:
#            errors += "<p>{!r} is not a number.</p>\n".format(request.form["number1"])
#        try:
#            number2 = float(request.form["number2"])
#        except:
#            errors += "<p>{!r} is not a number.</p>\n".format(request.form["number2"])
#        if number1 is not None and number2 is not None:
#            result = do_calculation(number1, number2)
#            return '''
#                <html>
#                   <body>
#                        <p>The result is {result}</p>
#                        <p><a href="/">Click here to calculate again</a>
#                    </body>
#                </html>
#            '''.format(result=result)
#
#    return '''
#        <html>
#            <body>
#                {errors}
#                <p>Enter your numbers:</p>
#                <form method="post" action=".">
#                    <p><input name="number1" /></p>
#                    <p><input name="number2" /></p>
#                    <p><input type="submit" value="Do calculation" /></p>
#                </form>
#            </body>
#        </html>
#    '''.format(errors=errors)


import numpy as np
import clearnessindex_2013b as ci
#import diffusefraction as df
import Solweig1D_2019a_calc as so
import PET_calculations as p
import UTCI_calculations as utci

def petcalc(Ta, RH, Ws, radG, year, month, day, hour, minu):
    sh = 1.  # 0 if shadowed by building
    vegsh = 1.  # 0 if shadowed by tree
    svfveg = 1.
    svfaveg = 1.
    trans = 1.
    elvis = 0

    # Location and time settings. Should be moved out later on
    UTC = 1
    lat = 57.7
    lon = 12.0

    if lon > 180.:
        lon = lon - 180.

    # Human parameter data. Should maybe be move out later on
    absK = 0.70
    absL = 0.95
    pos = 0
    mbody = 75.
    ht = 180 / 100.
    clo = 0.9
    age = 35
    activity = 80.
    sex = 1

    if pos == 0:
        Fside = 0.22
        Fup = 0.06
        height = 1.1
        Fcyl = 0.28
    else:
        Fside = 0.166666
        Fup = 0.166666
        height = 0.75
        Fcyl = 0.2

    cyl = 1
    ani = 1

    # Environmental data. Should maybe bo moved out later on.
    albedo_b = 0.2
    albedo_g = 0.15
    ewall = 0.9
    eground = 0.95
    svf = 0.6

    # Meteorological data, Should maybe be move out later on.
    sensorheight = 10.0
    onlyglobal = 1

    #metfileexist = 0
    #PathMet = None
    metdata = np.zeros((1, 24)) - 999.

    #date = self.calendarWidget.selectedDate()
    #year = date.year()
    #month = date.month()
    #day = date.day()
    #time = self.spinBoxTimeEdit.time()
    #hour = time.hour()
    #minu = time.minute()
    doy = day_of_year(year, month, day)

    #Ta = self.doubleSpinBoxTa.value()
    #RH = self.doubleSpinBoxRH.value()
    #radG = self.doubleSpinBoxradG.value()
    radD = -999.
    radI = -999.
    #Ws = self.doubleSpinBoxWs.value()

    metdata[0, 0] = year
    metdata[0, 1] = doy
    metdata[0, 2] = hour
    metdata[0, 3] = minu
    metdata[0, 11] = Ta
    metdata[0, 10] = RH
    metdata[0, 14] = radG
    metdata[0, 21] = radD
    metdata[0, 22] = radI
    metdata[0, 9] = Ws

    location = {'longitude': lon, 'latitude': lat, 'altitude': 3.}
    YYYY, altitude, azimuth, zen, jday, leafon, dectime, altmax = Solweig_2015a_metdata_noload(metdata, location, UTC)

    svfalfa = np.arcsin(np.exp((np.log((1.-svf))/2.)))

    # %Creating vectors from meteorological input
    DOY = metdata[:, 1]
    hours = metdata[:, 2]
    minu = metdata[:, 3]
    Ta = metdata[:, 11]
    RH = metdata[:, 10]
    radG = metdata[:, 14]
    radD = metdata[:, 21]
    radI = metdata[:, 22]
    P = metdata[:, 12]
    Ws = metdata[:, 9]

    TgK = 0.37
    Tstart = -3.41
    TmaxLST = 15
    TgK_wall = 0.58
    Tstart_wall = -3.41
    TmaxLST_wall = 15

    # If metfile starts at night
    CI = 1.

    if ani == 1:
        skyvaultalt = np.atleast_2d([])
        skyvaultazi = np.atleast_2d([])
        skyvaultaltint = [6, 18, 30, 42, 54, 66, 78]
        skyvaultaziint = [12, 12, 15, 15, 20, 30, 60]
        for j in range(7):
            for k in range(1, int(360/skyvaultaziint[j]) + 1):
                skyvaultalt = np.append(skyvaultalt, skyvaultaltint[j])

        skyvaultalt = np.append(skyvaultalt, 90)

        diffsh = np.zeros((145))
        svfalfadeg = svfalfa / (np.pi / 180.)
        for k in range(0, 145):
            if skyvaultalt[k] > svfalfadeg:
                diffsh[k] = 1
    else:
        diffsh = []

    #numformat = '%3d %2d %3d %2d %6.5f ' + '%6.2f ' * 29
    poi_save = np.zeros((1, 34))

    # main loop
    for i in np.arange(0, Ta.__len__()):
        # Daily water body temperature
        if (dectime[i] - np.floor(dectime[i])) == 0 or (i == 0):
            Twater = np.mean(Ta[jday[0] == np.floor(dectime[i])])

        # Nocturnal cloudfraction from Offerle et al. 2003
        if (dectime[i] - np.floor(dectime[i])) == 0:
            daylines = np.where(np.floor(dectime) == dectime[i])
            alt = altitude[0][daylines]
            alt2 = np.where(alt > 1)
            rise = alt2[0][0]
            [_, CI, _, _, _] = ci.clearnessindex_2013b(zen[0, i + rise + 1], jday[0, i + rise + 1],
                                                    Ta[i + rise + 1],
                                                    RH[i + rise + 1] / 100., radG[i + rise + 1], location,
                                                    P[i + rise + 1])
            if (CI > 1) or (CI == np.inf):
                CI = 1

        Tmrt, Kdown, Kup, Ldown, Lup, Tg, ea, esky, I0, CI, Keast, Ksouth, Kwest, Knorth, Least, Lsouth, Lwest, \
        Lnorth, KsideI, radIo, radDo, shadow = so.Solweig1D_2019a_calc(svf, svfveg, svfaveg, sh, vegsh,  albedo_b, absK, absL, ewall,
                                                            Fside, Fup, Fcyl,
                                                            altitude[0][i], azimuth[0][i], zen[0][i], jday[0][i],
                                                            onlyglobal, location, dectime[i], altmax[0][i], cyl, elvis,
                                                            Ta[i], RH[i], radG[i], radD[i], radI[i], P[i],
                                                            Twater, TgK, Tstart, albedo_g, eground, TgK_wall, Tstart_wall,
                                                            TmaxLST, TmaxLST_wall, svfalfa, CI, ani, diffsh, trans)

        # Write to array
        poi_save[0, 0] = YYYY[0][i]
        poi_save[0, 1] = jday[0][i]
        poi_save[0, 2] = hours[i]
        poi_save[0, 3] = minu[i]
        poi_save[0, 4] = dectime[i]
        poi_save[0, 5] = altitude[0][i]
        poi_save[0, 6] = azimuth[0][i]
        poi_save[0, 7] = radIo
        poi_save[0, 8] = radDo
        poi_save[0, 9] = radG[i]
        poi_save[0, 10] = Kdown
        poi_save[0, 11] = Kup
        poi_save[0, 12] = Keast
        poi_save[0, 13] = Ksouth
        poi_save[0, 14] = Kwest
        poi_save[0, 15] = Knorth
        poi_save[0, 16] = Ldown
        poi_save[0, 17] = Lup
        poi_save[0, 18] = Least
        poi_save[0, 19] = Lsouth
        poi_save[0, 20] = Lwest
        poi_save[0, 21] = Lnorth
        poi_save[0, 22] = Ta[i]
        poi_save[0, 23] = Tg + Ta[i]
        poi_save[0, 24] = RH[i]
        poi_save[0, 25] = esky
        poi_save[0, 26] = Tmrt
        poi_save[0, 27] = I0
        poi_save[0, 28] = CI
        poi_save[0, 29] = shadow
        poi_save[0, 30] = svf
        poi_save[0, 31] = KsideI


        # Recalculating wind speed based on pwerlaw
        WsPET = (1.1 / sensorheight) ** 0.2 * Ws[i]
        WsUTCI = (10. / sensorheight) ** 0.2 * Ws[i]
        resultPET = p._PET(Ta[i], RH[i], Tmrt[0][i], WsPET, mbody, age, ht, activity, clo, sex)
        poi_save[0, 32] = resultPET
        resultUTCI = utci.utci_calculator(Ta[i], RH[i], Tmrt[0][i], WsUTCI)
        poi_save[0, 33] = resultUTCI

    return Tmrt[0][0], resultPET, resultUTCI

def day_of_year(yyyy, month, day):
        if (yyyy % 4) == 0:
            if (yyyy % 100) == 0:
                if (yyyy % 400) == 0:
                    leapyear = 1
                else:
                    leapyear = 0
            else:
                leapyear = 1
        else:
            leapyear = 0

        if leapyear == 1:
            dayspermonth = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        else:
            dayspermonth = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

        doy = np.sum(dayspermonth[0:month - 1]) + day

        return doy


import numpy as np
import clearnessindex_2013b as ci
#import diffusefraction as df
import Solweig1D_2020a_calc as so
import PET_calculations as p
import UTCI_calculations as utci

def petcalc(Ta, RH, Ws, radG, year, month, day, hour, minu):
    sh = 1.  # 0 if shadowed by building
    vegsh = 1.  # 0 if shadowed by tree
    svfveg = 1.
    svfaveg = 1.
    trans = 1.
    elvis = 0

    anisdiff = 1
    L_ani = 0

    # Location and time settings. Should be moved out later on
    UTC = 1
    lat = 57.7
    lon = 12.0

    if lon > 180.:
        lon = lon - 180.

    # Human parameter data. Should maybe be move out later on
    absK = 0.70
    absL = 0.95
    pos = 0
    mbody = 75.
    ht = 180 / 100.
    clo = 0.9
    age = 35
    activity = 80.
    sex = 1

    if pos == 0:
        Fside = 0.22
        Fup = 0.06
        height = 1.1
        Fcyl = 0.28
    else:
        Fside = 0.166666
        Fup = 0.166666
        height = 0.75
        Fcyl = 0.2

    cyl = 1
    ani = 1

    # Environmental data. Should maybe bo moved out later on.
    albedo_b = 0.2
    albedo_g = 0.15
    ewall = 0.9
    eground = 0.95
    svf = 0.6

    # Meteorological data, Should maybe be move out later on.
    sensorheight = 10.0
    onlyglobal = 1

    #metfileexist = 0
    #PathMet = None
    metdata = np.zeros((1, 24)) - 999.

    #date = self.calendarWidget.selectedDate()
    #year = date.year()
    #month = date.month()
    #day = date.day()
    #time = self.spinBoxTimeEdit.time()
    #hour = time.hour()
    #minu = time.minute()
    doy = day_of_year(year, month, day)

    #Ta = self.doubleSpinBoxTa.value()
    #RH = self.doubleSpinBoxRH.value()
    #radG = self.doubleSpinBoxradG.value()
    radD = -999.
    radI = -999.
    #Ws = self.doubleSpinBoxWs.value()

    metdata[0, 0] = year
    metdata[0, 1] = doy
    metdata[0, 2] = hour
    metdata[0, 3] = minu
    metdata[0, 11] = Ta
    metdata[0, 10] = RH
    metdata[0, 14] = radG
    metdata[0, 21] = radD
    metdata[0, 22] = radI
    metdata[0, 9] = Ws

    location = {'longitude': lon, 'latitude': lat, 'altitude': 3.}
    YYYY, altitude, azimuth, zen, jday, leafon, dectime, altmax = Solweig_2015a_metdata_noload(metdata, location, UTC)

    svfalfa = np.arcsin(np.exp((np.log((1.-svf))/2.)))

    # %Creating vectors from meteorological input
    DOY = metdata[:, 1]
    hours = metdata[:, 2]
    minu = metdata[:, 3]
    Ta = metdata[:, 11]
    RH = metdata[:, 10]
    radG = metdata[:, 14]
    radD = metdata[:, 21]
    radI = metdata[:, 22]
    P = metdata[:, 12]
    Ws = metdata[:, 9]

    TgK = 0.37
    Tstart = -3.41
    TmaxLST = 15
    TgK_wall = 0.58
    Tstart_wall = -3.41
    TmaxLST_wall = 15

    # If metfile starts at night
    CI = 1.

    if ani == 1:
        skyvaultalt = np.atleast_2d([])
        skyvaultazi = np.atleast_2d([])
        skyvaultaltint = [6, 18, 30, 42, 54, 66, 78]
        skyvaultaziint = [12, 12, 15, 15, 20, 30, 60]
        for j in range(7):
            for k in range(1, int(360/skyvaultaziint[j]) + 1):
                skyvaultalt = np.append(skyvaultalt, skyvaultaltint[j])

        skyvaultalt = np.append(skyvaultalt, 90)

        diffsh = np.zeros((145))
        svfalfadeg = svfalfa / (np.pi / 180.)
        for k in range(0, 145):
            if skyvaultalt[k] > svfalfadeg:
                diffsh[k] = 1
    else:
        diffsh = []

    #numformat = '%3d %2d %3d %2d %6.5f ' + '%6.2f ' * 29
    poi_save = np.zeros((1, 34))

    # main loop
    for i in np.arange(0, Ta.__len__()):
        # Daily water body temperature
        if (dectime[i] - np.floor(dectime[i])) == 0 or (i == 0):
            Twater = np.mean(Ta[jday[0] == np.floor(dectime[i])])

        # Nocturnal cloudfraction from Offerle et al. 2003
        if (dectime[i] - np.floor(dectime[i])) == 0:
            daylines = np.where(np.floor(dectime) == dectime[i])
            alt = altitude[0][daylines]
            alt2 = np.where(alt > 1)
            rise = alt2[0][0]
            [_, CI, _, _, _] = ci.clearnessindex_2013b(zen[0, i + rise + 1], jday[0, i + rise + 1],
                                                    Ta[i + rise + 1],
                                                    RH[i + rise + 1] / 100., radG[i + rise + 1], location,
                                                    P[i + rise + 1])
            if (CI > 1) or (CI == np.inf):
                CI = 1

        Tmrt, Kdown, Kup, Ldown, Lup, Tg, ea, esky, I0, CI, Keast, Ksouth, Kwest, Knorth, Least, Lsouth, Lwest, \
        Lnorth, KsideI, radIo, radDo, shadow = so.Solweig1D_2020a_calc(svf, svfveg, svfaveg, sh, vegsh,  albedo_b, absK, absL, ewall,
                                                            Fside, Fup, Fcyl,
                                                            altitude[0][i], azimuth[0][i], zen[0][i], jday[0][i],
                                                            onlyglobal, location, dectime[i], altmax[0][i], cyl, elvis,
                                                            Ta[i], RH[i], radG[i], radD[i], radI[i], P[i],
                                                            Twater, TgK, Tstart, albedo_g, eground, TgK_wall, Tstart_wall,
                                                            TmaxLST, TmaxLST_wall, svfalfa, CI, ani, anisdiff, diffsh, trans, L_ani)

        # Write to array
        poi_save[0, 0] = YYYY[0][i]
        poi_save[0, 1] = jday[0][i]
        poi_save[0, 2] = hours[i]
        poi_save[0, 3] = minu[i]
        poi_save[0, 4] = dectime[i]
        poi_save[0, 5] = altitude[0][i]
        poi_save[0, 6] = azimuth[0][i]
        poi_save[0, 7] = radIo
        poi_save[0, 8] = radDo
        poi_save[0, 9] = radG[i]
        poi_save[0, 10] = Kdown
        poi_save[0, 11] = Kup
        poi_save[0, 12] = Keast
        poi_save[0, 13] = Ksouth
        poi_save[0, 14] = Kwest
        poi_save[0, 15] = Knorth
        poi_save[0, 16] = Ldown
        poi_save[0, 17] = Lup
        poi_save[0, 18] = Least
        poi_save[0, 19] = Lsouth
        poi_save[0, 20] = Lwest
        poi_save[0, 21] = Lnorth
        poi_save[0, 22] = Ta[i]
        poi_save[0, 23] = Tg + Ta[i]
        poi_save[0, 24] = RH[i]
        poi_save[0, 25] = esky
        poi_save[0, 26] = Tmrt
        poi_save[0, 27] = I0
        poi_save[0, 28] = CI
        poi_save[0, 29] = shadow
        poi_save[0, 30] = svf
        poi_save[0, 31] = KsideI


        # Recalculating wind speed based on pwerlaw
        WsPET = (1.1 / sensorheight) ** 0.2 * Ws[i]
        WsUTCI = (10. / sensorheight) ** 0.2 * Ws[i]
        resultPET = p._PET(Ta[i], RH[i], Tmrt, WsPET, mbody, age, ht, activity, clo, sex)
        poi_save[0, 32] = resultPET
        resultUTCI = utci.utci_calculator(Ta[i], RH[i], Tmrt, WsUTCI)
        poi_save[0, 33] = resultUTCI

    return Tmrt, resultPET, resultUTCI

def day_of_year(yyyy, month, day):
        if (yyyy % 4) == 0:
            if (yyyy % 100) == 0:
                if (yyyy % 400) == 0:
                    leapyear = 1
                else:
                    leapyear = 0
            else:
                leapyear = 1
        else:
            leapyear = 0

        if leapyear == 1:
            dayspermonth = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        else:
            dayspermonth = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

        doy = np.sum(dayspermonth[0:month - 1]) + day

        return doy


import numpy as np
import clearnessindex_2013b as ci
#import diffusefraction as df
import Solweig1D_2020a_calc as so
import PET_calculations as p
import UTCI_calculations as utci
#import matplotlib.pylab as plt

def petcalcprognose(Ta, RH, Ws, radG, radD, radI, year, month, day, hour, minu, lat, lon, UTC, elev=3.):

    elvis = 0
    cyl = 1
    anisdiff = 1
    L_ani = 0

    # Location and time settings
    # UTC = 0
    # lat = 57.7
    # lon = 12.0

    if lon > 180.:
        lon = lon - 180.

    # Human parameter data
    absK = 0.70
    absL = 0.95
    pos = 0
    mbody = 75.
    ht = 180 / 100.
    clo = 0.9
    age = 35
    activity = 80.
    sex = 1

    if pos == 0:
        Fside = 0.22
        Fup = 0.06
        height = 1.1
        Fcyl = 0.28
    else:
        Fside = 0.166666
        Fup = 0.166666
        height = 0.75
        Fcyl = 0.2


    # ani = 1

    # Environmental data
    albedo_b = 0.2
    albedo_g = 0.15
    ewall = 0.9
    eground = 0.95
    svf = 0.6
    svfalfa = np.arcsin(np.exp((np.log((1.-svf))/2.)))
    sh = 1.  # 0 if shadowed by building
    vegsh = 1.  # 0 if shadowed by tree
    svfveg = 1.
    svfaveg = 1.
    trans = 1.
    svfbuveg = (svf - (1. - svfveg) * (1. - trans))

    # Meteorological data
    sensorheight = 10.0
    onlyglobal = 0

    metdata = np.zeros((Ta.__len__(), 24)) - 999.

    #numformat = '%3d %2d %3d %2d %6.5f ' + '%6.2f ' * 29
    poi_save = np.zeros((Ta.__len__(), 34))*np.NaN

    # doy = day_of_year(year, month, day)

    metdata[:, 0] = year
    for i in range(0,  Ta.__len__()):
        metdata[i, 1] = day_of_year(year[i], month[i], day[i])
    metdata[:, 2] = hour
    metdata[:, 3] = minu
    # metdata[0, 11] = Ta
    # metdata[0, 10] = RH
    # metdata[0, 14] = radG
    # metdata[0, 21] = radD
    # metdata[0, 22] = radI
    # metdata[0, 9] = Ws

    location = {'longitude': lon, 'latitude': lat, 'altitude': elev}
    YYYY, altitude, azimuth, zen, jday, leafon, dectime, altmax = Solweig_2015a_metdata_noload(metdata, location, UTC)

    radI = (radG - radD)/(np.sin(altitude[0][:]*(np.pi/180)))
    with np.errstate(invalid='ignore'):
      radI[radI < 0] = 0.
    for i in range(0,  Ta.__len__()):
        if altitude[0][i] < 0.:
           radG[i] = 0.
        if altitude[0][i] < 1 and radI[i] > radG[i]:
            radI[i]=radG[i]
        if radD[i] > radG[i]:
            radD[i] = radG[i]

    # %Creating vectors from meteorological input
    # DOY = metdata[:, 1]
    # hour = metdata[:, 2]
    # minu = metdata[:, 3]
    # Ta = metdata[:, 11]
    # RH = metdata[:, 10]
    # radG = metdata[:, 14]
    # radD = metdata[:, 21]
    # radI = metdata[:, 22]
    P = metdata[:, 12]
    # Ws = metdata[:, 9]
    Twater = []

    TgK = 0.37
    Tstart = -3.41
    TmaxLST = 15
    TgK_wall = 0.58
    Tstart_wall = -3.41
    TmaxLST_wall = 15

    # If metfile starts at night
    CI = 1.

    if anisdiff == 1:
        skyvaultalt = np.atleast_2d([])
        # skyvaultazi = np.atleast_2d([])
        skyvaultaltint = [6, 18, 30, 42, 54, 66, 78]
        skyvaultaziint = [12, 12, 15, 15, 20, 30, 60]
        for j in range(7):
            for k in range(1, int(360/skyvaultaziint[j]) + 1):
                skyvaultalt = np.append(skyvaultalt, skyvaultaltint[j])

        skyvaultalt = np.append(skyvaultalt, 90)

        diffsh = np.zeros((145))
        svfalfadeg = svfalfa / (np.pi / 180.)
        for k in range(0, 145):
            if skyvaultalt[k] > svfalfadeg:
                diffsh[k] = 1
    else:
        diffsh = []



    # main loop
    for i in np.arange(1, Ta.__len__()): # starting from 1 as rad[0] is nan
        # print(i)
        
        # what we can save without calculation
        poi_save[i, 0] = YYYY[0][i]
        poi_save[i, 1] = jday[0][i]
        poi_save[i, 2] = hour[i]
        poi_save[i, 3] = minu[i]
        poi_save[i, 4] = dectime[i]
        poi_save[i, 5] = altitude[0][i]
        poi_save[i, 6] = azimuth[0][i]
        poi_save[i, 9] = radG[i]
        poi_save[i, 22] = Ta[i]
        poi_save[i, 24] = RH[i]
        poi_save[i, 30] = svf

        # Daily water body temperature
        if (dectime[i] - np.floor(dectime[i])) == 0 or (i == 0):
            Twater = np.mean(Ta[jday[0] == np.floor(dectime[i])])

        # Nocturnal cloudfraction from Offerle et al. 2003
        # Last CI from previous day is used until midnight
        # after which we swap to first CI of following day. 
        if (dectime[i] - np.floor(dectime[i])) == 0:
            daylines = np.where(np.floor(dectime) == dectime[i])
            alt = altitude[0][daylines]
            alt2 = np.where(alt > 1)
            try:
              rise = alt2[0][0]
              [_, CI, _, _, _] = ci.clearnessindex_2013b(zen[0, i + rise + 1], 
                jday[0, i + rise + 1], Ta[i + rise + 1],
                RH[i + rise + 1] / 100., radG[i + rise + 1], location,
                P[i + rise + 1])
            except IndexError as error:
              # there was no hour after sunrise for following day. 
              # Just keep the last CI
              pass
            if (CI > 1) or (CI == np.inf):
                CI = 1

        try:
          Tmrt, Kdown, Kup, Ldown, Lup, Tg, ea, esky, I0, CI, Keast, Ksouth, Kwest, Knorth, Least, Lsouth, Lwest, \
          Lnorth, KsideI, radIo, radDo, shadow = so.Solweig1D_2020a_calc(svf, svfveg, svfaveg, sh, vegsh,  albedo_b, 
                        absK, absL, ewall, Fside, Fup, Fcyl,
                        altitude[0][i], azimuth[0][i], zen[0][i], jday[0][i],
                        onlyglobal, location, dectime[i], altmax[0][i], cyl, elvis,
                        Ta[i], RH[i], radG[i], radD[i], radI[i], P[i],
                        Twater, TgK, Tstart, albedo_g, eground, TgK_wall, Tstart_wall,
                        TmaxLST, TmaxLST_wall, svfalfa, svfbuveg, CI, anisdiff, diffsh, trans, L_ani)
        except ValueError:
          # presumably NaNs
          #print('NaNs?')
          continue
        except:
          #print('Other error?')
          raise 

        # Write to array
        poi_save[i, 0] = YYYY[0][i]
        poi_save[i, 1] = jday[0][i]
        poi_save[i, 2] = hour[i]
        poi_save[i, 3] = minu[i]
        poi_save[i, 4] = dectime[i]
        poi_save[i, 5] = altitude[0][i]
        poi_save[i, 6] = azimuth[0][i]
        poi_save[i, 7] = radIo
        poi_save[i, 8] = radDo
        poi_save[i, 9] = radG[i]
        poi_save[i, 10] = Kdown
        poi_save[i, 11] = Kup
        poi_save[i, 12] = Keast
        poi_save[i, 13] = Ksouth
        poi_save[i, 14] = Kwest
        poi_save[i, 15] = Knorth
        poi_save[i, 16] = Ldown
        poi_save[i, 17] = Lup
        poi_save[i, 18] = Least
        poi_save[i, 19] = Lsouth
        poi_save[i, 20] = Lwest
        poi_save[i, 21] = Lnorth
        poi_save[i, 22] = Ta[i]
        poi_save[i, 23] = Tg + Ta[i]
        poi_save[i, 24] = RH[i]
        poi_save[i, 25] = esky
        poi_save[i, 26] = Tmrt
        poi_save[i, 27] = I0
        poi_save[i, 28] = CI
        poi_save[i, 29] = shadow
        poi_save[i, 30] = svf
        poi_save[i, 31] = KsideI


        # Recalculating wind speed based on pwerlaw
        WsPET = (1.1 / sensorheight) ** 0.2 * Ws[i]
        WsUTCI = (10. / sensorheight) ** 0.2 * Ws[i]
        resultPET = p._PET(Ta[i], RH[i], Tmrt, WsPET, mbody, age, ht, activity, clo, sex)
        poi_save[i, 32] = resultPET
        resultUTCI = utci.utci_calculator(Ta[i], RH[i], Tmrt, WsUTCI)
        poi_save[i, 33] = resultUTCI

    return poi_save

def day_of_year(yyyy, month, day):
        if (yyyy % 4) == 0:
            if (yyyy % 100) == 0:
                if (yyyy % 400) == 0:
                    leapyear = 1
                else:
                    leapyear = 0
            else:
                leapyear = 1
        else:
            leapyear = 0

        if leapyear == 1:
            dayspermonth = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        else:
            dayspermonth = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

        doy = np.sum(dayspermonth[0:month - 1]) + day

        return doy


__author__ = 'xlinfr'
import numpy as np


def sun_distance(jday):
    """

    #% Calculatesrelative earth sun distance
    #% with day of year as input.
    #% Partridge and Platt, 1975
    """
    b = 2.*np.pi*jday/365.
    D = np.sqrt((1.00011+np.dot(0.034221, np.cos(b))+np.dot(0.001280, np.sin(b))+np.dot(0.000719,
                                        np.cos((2.*b)))+np.dot(0.000077, np.sin((2.*b)))))
    return D


# -*- coding: utf-8 -*-
from __future__ import division
import datetime
import numpy as np


def sun_position(time, location):
    """
    % sun = sun_position(time, location)
    %
    % This function compute the sun position (zenith and azimuth angle at the observer
    % location) as a function of the observer local time and position.
    %
    % It is an implementation of the algorithm presented by Reda et Andreas in:
    %   Reda, I., Andreas, A. (2003) Solar position algorithm for solar
    %   radiation application. National Renewable Energy Laboratory (NREL)
    %   Technical report NREL/TP-560-34302.
    % This document is avalaible at www.osti.gov/bridge
    %
    % This algorithm is based on numerical approximation of the exact equations.
    % The authors of the original paper state that this algorithm should be
    % precise at +/- 0.0003 degrees. I have compared it to NOAA solar table
    % (http://www.srrb.noaa.gov/highlights/sunrise/azel.html) and to USNO solar
    % table (http://aa.usno.navy.mil/data/docs/AltAz.html) and found very good
    % correspondance (up to the precision of those tables), except for large
    % zenith angle, where the refraction by the atmosphere is significant
    % (difference of about 1 degree). Note that in this code the correction
    % for refraction in the atmosphere as been implemented for a temperature
    % of 10C (283 kelvins) and a pressure of 1010 mbar. See the subfunction
    % �sun_topocentric_zenith_angle_calculation� for a possible modification
    % to explicitely model the effect of temperature and pressure as describe
    % in Reda & Andreas (2003).
    %
    % Input parameters:
    %   time: a structure that specify the time when the sun position is
    %   calculated.
    %       time.year: year. Valid for [-2000, 6000]
    %       time.month: month [1-12]
    %       time.day: calendar day [1-31]
    %       time.hour: local hour [0-23]
    %       time.min: minute [0-59]
    %       time.sec: second [0-59]
    %       time.UTC: offset hour from UTC. Local time = Greenwich time + time.UTC
    %   This input can also be passed using the Matlab time format ('dd-mmm-yyyy HH:MM:SS').
    %   In that case, the time has to be specified as UTC time (time.UTC = 0)
    %
    %   location: a structure that specify the location of the observer
    %       location.latitude: latitude (in degrees, north of equator is
    %       positive)
    %       location.longitude: longitude (in degrees, positive for east of
    %       Greenwich)
    %       location.altitude: altitude above mean sea level (in meters)
    %
    % Output parameters
    %   sun: a structure with the calculated sun position
    %       sun.zenith = zenith angle in degrees (angle from the vertical)
    %       sun.azimuth = azimuth angle in degrees, eastward from the north.
    % Only the sun zenith and azimuth angles are returned as output, but a lot
    % of other parameters are calculated that could also extracted as output of
    % this function.
    %
    % Exemple of use
    %
    % location.longitude = -105.1786;
    % location.latitude = 39.742476;
    % location.altitude = 1830.14;
    % time.year = 2005;
    % time.month = 10;
    % time.day = 17;
    % time.hour = 6;
    % time.min = 30;
    % time.sec = 30;
    % time.UTC = -7;
    % %
    % location.longitude = 11.94;
    % location.latitude = 57.70;
    % location.altitude = 3.0;
    % time.UTC = 1;
    % sun = sun_position(time, location);
    %
    % sun =
    %
    %      zenith: 50.1080438859849
    %      azimuth: 194.341174010338
    %
    % History
    %   09/03/2004  Original creation by Vincent Roy (vincent.roy@drdc-rddc.gc.ca)
    %   10/03/2004  Fixed a bug in julian_calculation subfunction (was
    %               incorrect for year 1582 only), Vincent Roy
    %   18/03/2004  Correction to the header (help display) only. No changes to
    %               the code. (changed the �elevation� field in �location� structure
    %               information to �altitude�), Vincent Roy
    %   13/04/2004  Following a suggestion from Jody Klymak (jklymak@ucsd.edu),
    %               allowed the 'time' input to be passed as a Matlab time string.
    %   22/08/2005  Following a bug report from Bruce Bowler
    %               (bbowler@bigelow.org), modified the julian_calculation function. Bug
    %               was 'MATLAB has allowed structure assignment  to a non-empty non-structure
    %               to overwrite the previous value.  This behavior will continue in this release,
    %               but will be an error in a  future version of MATLAB.  For advice on how to
    %               write code that  will both avoid this warning and work in future versions of
    %               MATLAB,  see R14SP2 Release Notes'. Script should now be
    %               compliant with futher release of Matlab...
    """

    # 1. Calculate the Julian Day, and Century. Julian Ephemeris day, century
    # and millenium are calculated using a mean delta_t of 33.184 seconds.
    julian = julian_calculation(time)
    #print(julian)

    # 2. Calculate the Earth heliocentric longitude, latitude, and radius
    # vector (L, B, and R)
    earth_heliocentric_position = earth_heliocentric_position_calculation(julian)

    # 3. Calculate the geocentric longitude and latitude
    sun_geocentric_position = sun_geocentric_position_calculation(earth_heliocentric_position)

    # 4. Calculate the nutation in longitude and obliquity (in degrees).
    nutation = nutation_calculation(julian)

    # 5. Calculate the true obliquity of the ecliptic (in degrees).
    true_obliquity = true_obliquity_calculation(julian, nutation)

    # 6. Calculate the aberration correction (in degrees)
    aberration_correction = abberation_correction_calculation(earth_heliocentric_position)

    # 7. Calculate the apparent sun longitude in degrees)
    apparent_sun_longitude = apparent_sun_longitude_calculation(sun_geocentric_position, nutation, aberration_correction)

    # 8. Calculate the apparent sideral time at Greenwich (in degrees)
    apparent_stime_at_greenwich = apparent_stime_at_greenwich_calculation(julian, nutation, true_obliquity)

    # 9. Calculate the sun rigth ascension (in degrees)
    sun_rigth_ascension = sun_rigth_ascension_calculation(apparent_sun_longitude, true_obliquity, sun_geocentric_position)

    # 10. Calculate the geocentric sun declination (in degrees). Positive or
    # negative if the sun is north or south of the celestial equator.
    sun_geocentric_declination = sun_geocentric_declination_calculation(apparent_sun_longitude, true_obliquity,
                                                                        sun_geocentric_position)

    # 11. Calculate the observer local hour angle (in degrees, westward from south).
    observer_local_hour = observer_local_hour_calculation(apparent_stime_at_greenwich, location, sun_rigth_ascension)

    # 12. Calculate the topocentric sun position (rigth ascension, declination and
    # rigth ascension parallax in degrees)
    topocentric_sun_position = topocentric_sun_position_calculate(earth_heliocentric_position, location,
                                                                  observer_local_hour, sun_rigth_ascension,
                                                                  sun_geocentric_declination)

    # 13. Calculate the topocentric local hour angle (in degrees)
    topocentric_local_hour = topocentric_local_hour_calculate(observer_local_hour, topocentric_sun_position)

    # 14. Calculate the topocentric zenith and azimuth angle (in degrees)
    sun = sun_topocentric_zenith_angle_calculate(location, topocentric_sun_position, topocentric_local_hour)

    return sun


def julian_calculation(t_input):
    """
    % This function compute the julian day and julian century from the local
    % time and timezone information. Ephemeris are calculated with a delta_t=0
    % seconds.

    % If time input is a Matlab time string, extract the information from
    % this string and create the structure as defined in the main header of
    % this script.
    """
    if not isinstance(t_input, dict):
        # tt = datetime.datetime.strptime(t_input, "%Y-%m-%d %H:%M:%S.%f")    # if t_input is a string of this format
        # t_input should be a datetime object
        time = dict()
        time['UTC'] = 0
        time['year'] = t_input.year
        time['month'] = t_input.month
        time['day'] = t_input.day
        time['hour'] = t_input.hour
        time['min'] = t_input.minute
        time['sec'] = t_input.second
    else:
        time = t_input

    if time['month'] == 1 or time['month'] == 2:
        Y = time['year'] - 1
        M = time['month'] + 12
    else:
        Y = time['year']
        M = time['month']

    ut_time = ((time['hour'] - time['UTC'])/24) + (time['min']/(60*24)) + (time['sec']/(60*60*24))   # time of day in UT time.
    D = time['day'] + ut_time   # Day of month in decimal time, ex. 2sd day of month at 12:30:30UT, D=2.521180556

    # In 1582, the gregorian calendar was adopted
    if time['year'] == 1582:
        if time['month'] == 10:
            if time['day'] <= 4:   # The Julian calendar ended on October 4, 1582
                B = (0)
            elif time['day'] >= 15:   # The Gregorian calendar started on October 15, 1582
                A = np.floor(Y/100)
                B = 2 - A + np.floor(A/4)
            else:
                print('This date never existed!. Date automatically set to October 4, 1582')
                time['month'] = 10
                time['day'] = 4
                B = 0
        elif time['month'] < 10:   # Julian calendar
            B = 0
        else: # Gregorian calendar
            A = np.floor(Y/100)
            B = 2 - A + np.floor(A/4)
    elif time['year'] < 1582:   # Julian calendar
        B = 0
    else:
        A = np.floor(Y/100)    # Gregorian calendar
        B = 2 - A + np.floor(A/4)

    julian = dict()
    julian['day'] = D + B + np.floor(365.25*(Y+4716)) + np.floor(30.6001*(M+1)) - 1524.5

    delta_t = 0   # 33.184;
    julian['ephemeris_day'] = (julian['day']) + (delta_t/86400)
    julian['century'] = (julian['day'] - 2451545) / 36525
    julian['ephemeris_century'] = (julian['ephemeris_day'] - 2451545) / 36525
    julian['ephemeris_millenium'] = (julian['ephemeris_century']) / 10

    return julian


def earth_heliocentric_position_calculation(julian):
    """
    % This function compute the earth position relative to the sun, using
    % tabulated values. 
    
    % Tabulated values for the longitude calculation
    % L terms  from the original code.
    """
    # Tabulated values for the longitude calculation
    # L terms  from the original code. 
    L0_terms = np.array([[175347046.0, 0, 0],
                        [3341656.0, 4.6692568, 6283.07585],
                        [34894.0, 4.6261, 12566.1517],
                        [3497.0, 2.7441, 5753.3849],
                        [3418.0, 2.8289, 3.5231],
                        [3136.0, 3.6277, 77713.7715],
                        [2676.0, 4.4181, 7860.4194],
                        [2343.0, 6.1352, 3930.2097],
                        [1324.0, 0.7425, 11506.7698],
                        [1273.0, 2.0371, 529.691],
                        [1199.0, 1.1096, 1577.3435],
                        [990, 5.233, 5884.927],
                        [902, 2.045, 26.298],
                        [857, 3.508, 398.149],
                        [780, 1.179, 5223.694],
                        [753, 2.533, 5507.553],
                        [505, 4.583, 18849.228],
                        [492, 4.205, 775.523],
                        [357, 2.92, 0.067],
                        [317, 5.849, 11790.629],
                        [284, 1.899, 796.298],
                        [271, 0.315, 10977.079],
                        [243, 0.345, 5486.778],
                        [206, 4.806, 2544.314],
                        [205, 1.869, 5573.143],
                        [202, 2.4458, 6069.777],
                        [156, 0.833, 213.299],
                        [132, 3.411, 2942.463],
                        [126, 1.083, 20.775],
                        [115, 0.645, 0.98],
                        [103, 0.636, 4694.003],
                        [102, 0.976, 15720.839],
                        [102, 4.267, 7.114],
                        [99, 6.21, 2146.17],
                        [98, 0.68, 155.42],
                        [86, 5.98, 161000.69],
                        [85, 1.3, 6275.96],
                        [85, 3.67, 71430.7],
                        [80, 1.81, 17260.15],
                        [79, 3.04, 12036.46],
                        [71, 1.76, 5088.63],
                        [74, 3.5, 3154.69],
                        [74, 4.68, 801.82],
                        [70, 0.83, 9437.76],
                        [62, 3.98, 8827.39],
                        [61, 1.82, 7084.9],
                        [57, 2.78, 6286.6],
                        [56, 4.39, 14143.5],
                        [56, 3.47, 6279.55],
                        [52, 0.19, 12139.55],
                        [52, 1.33, 1748.02],
                        [51, 0.28, 5856.48],
                        [49, 0.49, 1194.45],
                        [41, 5.37, 8429.24],
                        [41, 2.4, 19651.05],
                        [39, 6.17, 10447.39],
                        [37, 6.04, 10213.29],
                        [37, 2.57, 1059.38],
                        [36, 1.71, 2352.87],
                        [36, 1.78, 6812.77],
                        [33, 0.59, 17789.85],
                        [30, 0.44, 83996.85],
                        [30, 2.74, 1349.87],
                        [25, 3.16, 4690.48]])

    L1_terms = np.array([[628331966747.0, 0, 0],
                        [206059.0, 2.678235, 6283.07585],
                        [4303.0, 2.6351, 12566.1517],
                        [425.0, 1.59, 3.523],
                        [119.0, 5.796, 26.298],
                        [109.0, 2.966, 1577.344],
                        [93, 2.59, 18849.23],
                        [72, 1.14, 529.69],
                        [68, 1.87, 398.15],
                        [67, 4.41, 5507.55],
                        [59, 2.89, 5223.69],
                        [56, 2.17, 155.42],
                        [45, 0.4, 796.3],
                        [36, 0.47, 775.52],
                        [29, 2.65, 7.11],
                        [21, 5.34, 0.98],
                        [19, 1.85, 5486.78],
                        [19, 4.97, 213.3],
                        [17, 2.99, 6275.96],
                        [16, 0.03, 2544.31],
                        [16, 1.43, 2146.17],
                        [15, 1.21, 10977.08],
                        [12, 2.83, 1748.02],
                        [12, 3.26, 5088.63],
                        [12, 5.27, 1194.45],
                        [12, 2.08, 4694],
                        [11, 0.77, 553.57],
                        [10, 1.3, 3286.6],
                        [10, 4.24, 1349.87],
                        [9, 2.7, 242.73],
                        [9, 5.64, 951.72],
                        [8, 5.3, 2352.87],
                        [6, 2.65, 9437.76],
                        [6, 4.67, 4690.48]])

    L2_terms = np.array([[52919.0, 0, 0],
                        [8720.0, 1.0721, 6283.0758],
                        [309.0, 0.867, 12566.152],
                        [27, 0.05, 3.52],
                        [16, 5.19, 26.3],
                        [16, 3.68, 155.42],
                        [10, 0.76, 18849.23],
                        [9, 2.06, 77713.77],
                        [7, 0.83, 775.52],
                        [5, 4.66, 1577.34],
                        [4, 1.03, 7.11],
                        [4, 3.44, 5573.14],
                        [3, 5.14, 796.3],
                        [3, 6.05, 5507.55],
                        [3, 1.19, 242.73],
                        [3, 6.12, 529.69],
                        [3, 0.31, 398.15],
                        [3, 2.28, 553.57],
                        [2, 4.38, 5223.69],
                        [2, 3.75, 0.98]])

    L3_terms = np.array([[289.0, 5.844, 6283.076],
                        [35, 0, 0],
                        [17, 5.49, 12566.15],
                        [3, 5.2, 155.42],
                        [1, 4.72, 3.52],
                        [1, 5.3, 18849.23],
                        [1, 5.97, 242.73]])
    L4_terms = np.array([[114.0, 3.142, 0],
                        [8, 4.13, 6283.08],
                        [1, 3.84, 12566.15]])

    L5_terms = np.array([1, 3.14, 0])
    L5_terms = np.atleast_2d(L5_terms)    # since L5_terms is 1D, we have to convert it to 2D to avoid indexErrors

    A0 = L0_terms[:, 0]
    B0 = L0_terms[:, 1]
    C0 = L0_terms[:, 2]

    A1 = L1_terms[:, 0]
    B1 = L1_terms[:, 1]
    C1 = L1_terms[:, 2]

    A2 = L2_terms[:, 0]
    B2 = L2_terms[:, 1]
    C2 = L2_terms[:, 2]

    A3 = L3_terms[:, 0]
    B3 = L3_terms[:, 1]
    C3 = L3_terms[:, 2]

    A4 = L4_terms[:, 0]
    B4 = L4_terms[:, 1]
    C4 = L4_terms[:, 2]

    A5 = L5_terms[:, 0]
    B5 = L5_terms[:, 1]
    C5 = L5_terms[:, 2]

    JME = julian['ephemeris_millenium']

    # Compute the Earth Heliochentric longitude from the tabulated values.
    L0 = np.sum(A0 * np.cos(B0 + (C0 * JME)))
    L1 = np.sum(A1 * np.cos(B1 + (C1 * JME)))
    L2 = np.sum(A2 * np.cos(B2 + (C2 * JME)))
    L3 = np.sum(A3 * np.cos(B3 + (C3 * JME)))
    L4 = np.sum(A4 * np.cos(B4 + (C4 * JME)))
    L5 = A5 * np.cos(B5 + (C5 * JME))

    earth_heliocentric_position = dict()
    earth_heliocentric_position['longitude'] = (L0 + (L1 * JME) + (L2 * np.power(JME, 2)) +
                                                          (L3 * np.power(JME, 3)) +
                                                          (L4 * np.power(JME, 4)) +
                                                          (L5 * np.power(JME, 5))) / 1e8
    # Convert the longitude to degrees.
    earth_heliocentric_position['longitude'] = earth_heliocentric_position['longitude'] * 180/np.pi

    # Limit the range to [0,360]
    earth_heliocentric_position['longitude'] = set_to_range(earth_heliocentric_position['longitude'], 0, 360)

    # Tabulated values for the earth heliocentric latitude. 
    # B terms  from the original code. 
    B0_terms = np.array([[280.0, 3.199, 84334.662],
                        [102.0, 5.422, 5507.553],
                        [80, 3.88, 5223.69],
                        [44, 3.7, 2352.87],
                        [32, 4, 1577.34]])

    B1_terms = np.array([[9, 3.9, 5507.55],
                         [6, 1.73, 5223.69]])

    A0 = B0_terms[:, 0]
    B0 = B0_terms[:, 1]
    C0 = B0_terms[:, 2]
    
    A1 = B1_terms[:, 0]
    B1 = B1_terms[:, 1]
    C1 = B1_terms[:, 2]
    
    L0 = np.sum(A0 * np.cos(B0 + (C0 * JME)))
    L1 = np.sum(A1 * np.cos(B1 + (C1 * JME)))

    earth_heliocentric_position['latitude'] = (L0 + (L1 * JME)) / 1e8

    # Convert the latitude to degrees. 
    earth_heliocentric_position['latitude'] = earth_heliocentric_position['latitude'] * 180/np.pi

    # Limit the range to [0,360];
    earth_heliocentric_position['latitude'] = set_to_range(earth_heliocentric_position['latitude'], 0, 360)

    # Tabulated values for radius vector. 
    # R terms from the original code
    R0_terms = np.array([[100013989.0, 0, 0],
                        [1670700.0, 3.0984635, 6283.07585],
                        [13956.0, 3.05525, 12566.1517],
                        [3084.0, 5.1985, 77713.7715],
                        [1628.0, 1.1739, 5753.3849],
                        [1576.0, 2.8469, 7860.4194],
                        [925.0, 5.453, 11506.77],
                        [542.0, 4.564, 3930.21],
                        [472.0, 3.661, 5884.927],
                        [346.0, 0.964, 5507.553],
                        [329.0, 5.9, 5223.694],
                        [307.0, 0.299, 5573.143],
                        [243.0, 4.273, 11790.629],
                        [212.0, 5.847, 1577.344],
                        [186.0, 5.022, 10977.079],
                        [175.0, 3.012, 18849.228],
                        [110.0, 5.055, 5486.778],
                        [98, 0.89, 6069.78],
                        [86, 5.69, 15720.84],
                        [86, 1.27, 161000.69],
                        [85, 0.27, 17260.15],
                        [63, 0.92, 529.69],
                        [57, 2.01, 83996.85],
                        [56, 5.24, 71430.7],
                        [49, 3.25, 2544.31],
                        [47, 2.58, 775.52],
                        [45, 5.54, 9437.76],
                        [43, 6.01, 6275.96],
                        [39, 5.36, 4694],
                        [38, 2.39, 8827.39],
                        [37, 0.83, 19651.05],
                        [37, 4.9, 12139.55],
                        [36, 1.67, 12036.46],
                        [35, 1.84, 2942.46],
                        [33, 0.24, 7084.9],
                        [32, 0.18, 5088.63],
                        [32, 1.78, 398.15],
                        [28, 1.21, 6286.6],
                        [28, 1.9, 6279.55],
                        [26, 4.59, 10447.39]])

    R1_terms = np.array([[103019.0, 1.10749, 6283.07585],
                        [1721.0, 1.0644, 12566.1517],
                        [702.0, 3.142, 0],
                        [32, 1.02, 18849.23],
                        [31, 2.84, 5507.55],
                        [25, 1.32, 5223.69],
                        [18, 1.42, 1577.34],
                        [10, 5.91, 10977.08],
                        [9, 1.42, 6275.96],
                        [9, 0.27, 5486.78]])

    R2_terms = np.array([[4359.0, 5.7846, 6283.0758],
                        [124.0, 5.579, 12566.152],
                        [12, 3.14, 0],
                        [9, 3.63, 77713.77],
                        [6, 1.87, 5573.14],
                        [3, 5.47, 18849]])

    R3_terms = np.array([[145.0, 4.273, 6283.076],
                        [7, 3.92, 12566.15]])
    
    R4_terms = [4, 2.56, 6283.08]
    R4_terms = np.atleast_2d(R4_terms)    # since L5_terms is 1D, we have to convert it to 2D to avoid indexErrors

    A0 = R0_terms[:, 0]
    B0 = R0_terms[:, 1]
    C0 = R0_terms[:, 2]
    
    A1 = R1_terms[:, 0]
    B1 = R1_terms[:, 1]
    C1 = R1_terms[:, 2]
    
    A2 = R2_terms[:, 0]
    B2 = R2_terms[:, 1]
    C2 = R2_terms[:, 2]
    
    A3 = R3_terms[:, 0]
    B3 = R3_terms[:, 1]
    C3 = R3_terms[:, 2]
    
    A4 = R4_terms[:, 0]
    B4 = R4_terms[:, 1]
    C4 = R4_terms[:, 2]

    # Compute the Earth heliocentric radius vector
    L0 = np.sum(A0 * np.cos(B0 + (C0 * JME)))
    L1 = np.sum(A1 * np.cos(B1 + (C1 * JME)))
    L2 = np.sum(A2 * np.cos(B2 + (C2 * JME)))
    L3 = np.sum(A3 * np.cos(B3 + (C3 * JME)))
    L4 = A4 * np.cos(B4 + (C4 * JME))

    # Units are in AU
    earth_heliocentric_position['radius'] = (L0 + (L1 * JME) + (L2 * np.power(JME, 2)) +
                                             (L3 * np.power(JME, 3)) +
                                             (L4 * np.power(JME, 4))) / 1e8

    return earth_heliocentric_position


def sun_geocentric_position_calculation(earth_heliocentric_position):
    """
    % This function compute the sun position relative to the earth.
    """
    sun_geocentric_position = dict()
    sun_geocentric_position['longitude'] = earth_heliocentric_position['longitude'] + 180
    # Limit the range to [0,360];
    sun_geocentric_position['longitude'] = set_to_range(sun_geocentric_position['longitude'], 0, 360)

    sun_geocentric_position['latitude'] = -earth_heliocentric_position['latitude']
    # Limit the range to [0,360]
    sun_geocentric_position['latitude'] = set_to_range(sun_geocentric_position['latitude'], 0, 360)
    return sun_geocentric_position


def nutation_calculation(julian):
    """
    % This function compute the nutation in longtitude and in obliquity, in
    % degrees.
    :param julian:
    :return: nutation
    """

    # All Xi are in degrees.
    JCE = julian['ephemeris_century']

    # 1. Mean elongation of the moon from the sun
    p = np.atleast_2d([(1/189474), -0.0019142, 445267.11148, 297.85036])

    # X0 = polyval(p, JCE);
    X0 = p[0, 0] * np.power(JCE, 3) + p[0, 1] * np.power(JCE, 2) + p[0, 2] * JCE + p[0, 3]   # This is faster than polyval...

    # 2. Mean anomaly of the sun (earth)
    p = np.atleast_2d([-(1/300000), -0.0001603, 35999.05034, 357.52772])

    # X1 = polyval(p, JCE)
    X1 = p[0, 0] * np.power(JCE, 3) + p[0, 1] * np.power(JCE, 2) + p[0, 2] * JCE + p[0, 3]

    # 3. Mean anomaly of the moon
    p = np.atleast_2d([(1/56250), 0.0086972, 477198.867398, 134.96298])
    
    # X2 = polyval(p, JCE);
    X2 = p[0, 0] * np.power(JCE, 3) + p[0, 1] * np.power(JCE, 2) + p[0, 2] * JCE + p[0, 3]

    # 4. Moon argument of latitude
    p = np.atleast_2d([(1/327270), -0.0036825, 483202.017538, 93.27191])

    # X3 = polyval(p, JCE)
    X3 = p[0, 0] * np.power(JCE, 3) + p[0, 1] * np.power(JCE, 2) + p[0, 2] * JCE + p[0, 3]

    # 5. Longitude of the ascending node of the moon's mean orbit on the
    # ecliptic, measured from the mean equinox of the date
    p = np.atleast_2d([(1/450000), 0.0020708, -1934.136261, 125.04452])

    # X4 = polyval(p, JCE);
    X4 = p[0, 0] * np.power(JCE, 3) + p[0, 1] * np.power(JCE, 2) + p[0, 2] * JCE + p[0, 3]

    # Y tabulated terms from the original code
    Y_terms = np.array([[0, 0, 0, 0, 1],
                        [-2, 0, 0, 2, 2],
                        [0, 0, 0, 2, 2],
                        [0, 0, 0, 0, 2],
                        [0, 1, 0, 0, 0],
                        [0, 0, 1, 0, 0],
                        [-2, 1, 0, 2, 2],
                        [0, 0, 0, 2, 1],
                        [0, 0, 1, 2, 2],
                        [-2, -1, 0, 2, 2],
                        [-2, 0, 1, 0, 0],
                        [-2, 0, 0, 2, 1],
                        [0, 0, -1, 2, 2],
                        [2, 0, 0, 0, 0],
                        [0, 0, 1, 0, 1],
                        [2, 0, -1, 2, 2],
                        [0, 0, -1, 0, 1],
                        [0, 0, 1, 2, 1],
                        [-2, 0, 2, 0, 0],
                        [0, 0, -2, 2, 1],
                        [2, 0, 0, 2, 2],
                        [0, 0, 2, 2, 2],
                        [0, 0, 2, 0, 0],
                        [-2, 0, 1, 2, 2],
                        [0, 0, 0, 2, 0],
                        [-2, 0, 0, 2, 0],
                        [0, 0, -1, 2, 1],
                        [0, 2, 0, 0, 0],
                        [2, 0, -1, 0, 1],
                        [-2, 2, 0, 2, 2],
                        [0, 1, 0, 0, 1],
                        [-2, 0, 1, 0, 1],
                        [0, -1, 0, 0, 1],
                        [0, 0, 2, -2, 0],
                        [2, 0, -1, 2, 1],
                        [2, 0, 1, 2, 2],
                        [0, 1, 0, 2, 2],
                        [-2, 1, 1, 0, 0],
                        [0, -1, 0, 2, 2],
                        [2, 0, 0, 2, 1],
                        [2, 0, 1, 0, 0],
                        [-2, 0, 2, 2, 2],
                        [-2, 0, 1, 2, 1],
                        [2, 0, -2, 0, 1],
                        [2, 0, 0, 0, 1],
                        [0, -1, 1, 0, 0],
                        [-2, -1, 0, 2, 1],
                        [-2, 0, 0, 0, 1],
                        [0, 0, 2, 2, 1],
                        [-2, 0, 2, 0, 1],
                        [-2, 1, 0, 2, 1],
                        [0, 0, 1, -2, 0],
                        [-1, 0, 1, 0, 0],
                        [-2, 1, 0, 0, 0],
                        [1, 0, 0, 0, 0],
                        [0, 0, 1, 2, 0],
                        [0, 0, -2, 2, 2],
                        [-1, -1, 1, 0, 0],
                        [0, 1, 1, 0, 0],
                        [0, -1, 1, 2, 2],
                        [2, -1, -1, 2, 2],
                        [0, 0, 3, 2, 2],
                        [2, -1, 0, 2, 2]])

    nutation_terms = np.array([[-171996, -174.2, 92025, 8.9],
                                [-13187, -1.6, 5736, -3.1],
                                [-2274, -0.2, 977, -0.5],
                                [2062, 0.2, -895, 0.5],
                                [1426, -3.4, 54, -0.1],
                                [712, 0.1, -7, 0],
                                [-517, 1.2, 224, -0.6],
                                [-386, -0.4, 200, 0],
                                [-301, 0, 129, -0.1],
                                [217, -0.5, -95, 0.3],
                                [-158, 0, 0, 0],
                                [129, 0.1, -70, 0],
                                [123, 0, -53, 0],
                                [63, 0, 0, 0],
                                [63, 0.1, -33, 0],
                                [-59, 0, 26, 0],
                                [-58, -0.1, 32, 0],
                                [-51, 0, 27, 0],
                                [48, 0, 0, 0],
                                [46, 0, -24, 0],
                                [-38, 0, 16, 0],
                                [-31, 0, 13, 0],
                                [29, 0, 0, 0],
                                [29, 0, -12, 0],
                                [26, 0, 0, 0],
                                [-22, 0, 0, 0],
                                [21, 0, -10, 0],
                                [17, -0.1, 0, 0],
                                [16, 0, -8, 0],
                                [-16, 0.1, 7, 0],
                                [-15, 0, 9, 0],
                                [-13, 0, 7, 0],
                                [-12, 0, 6, 0],
                                [11, 0, 0, 0],
                                [-10, 0, 5, 0],
                                [-8, 0, 3, 0],
                                [7, 0, -3, 0],
                                [-7, 0, 0, 0],
                                [-7, 0, 3, 0],
                                [-7, 0, 3, 0],
                                [6, 0, 0, 0],
                                [6, 0, -3, 0],
                                [6, 0, -3, 0],
                                [-6, 0, 3, 0],
                                [-6, 0, 3, 0],
                                [5, 0, 0, 0],
                                [-5, 0, 3, 0],
                                [-5, 0, 3, 0],
                                [-5, 0, 3, 0],
                                [4, 0, 0, 0],
                                [4, 0, 0, 0],
                                [4, 0, 0, 0],
                                [-4, 0, 0, 0],
                                [-4, 0, 0, 0],
                                [-4, 0, 0, 0],
                                [3, 0, 0, 0],
                                [-3, 0, 0, 0],
                                [-3, 0, 0, 0],
                                [-3, 0, 0, 0],
                                [-3, 0, 0, 0],
                                [-3, 0, 0, 0],
                                [-3, 0, 0, 0],
                                [-3, 0, 0, 0]])

    # Using the tabulated values, compute the delta_longitude and
    # delta_obliquity.
    Xi = np.array([X0, X1, X2, X3, X4])    # a col mat in octave

    tabulated_argument = Y_terms.dot(np.transpose(Xi)) * (np.pi/180)

    delta_longitude = (nutation_terms[:, 0] + (nutation_terms[:, 1] * JCE)) * np.sin(tabulated_argument)
    delta_obliquity = (nutation_terms[:, 2] + (nutation_terms[:, 3] * JCE)) * np.cos(tabulated_argument)

    nutation = dict()    # init nutation dictionary
    # Nutation in longitude
    nutation['longitude'] = np.sum(delta_longitude) / 36000000

    # Nutation in obliquity
    nutation['obliquity'] = np.sum(delta_obliquity) / 36000000

    return nutation


def true_obliquity_calculation(julian, nutation):
    """
    This function compute the true obliquity of the ecliptic.

    :param julian:
    :param nutation:
    :return:
    """

    p = np.atleast_2d([2.45, 5.79, 27.87, 7.12, -39.05, -249.67, -51.38, 1999.25, -1.55, -4680.93, 84381.448])

    # mean_obliquity = polyval(p, julian.ephemeris_millenium/10);
    U = julian['ephemeris_millenium'] / 10
    mean_obliquity = p[0, 0] * np.power(U, 10) + p[0, 1] * np.power(U, 9) + \
                     p[0, 2] * np.power(U, 8) + p[0, 3] * np.power(U, 7) + \
                     p[0, 4] * np.power(U, 6) + p[0, 5] * np.power(U, 5) + \
                     p[0, 6] * np.power(U, 4) + p[0, 7] * np.power(U, 3) + \
                     p[0, 8] * np.power(U, 2) + p[0, 9] * U + p[0, 10]

    true_obliquity = (mean_obliquity/3600) + nutation['obliquity']

    return true_obliquity


def abberation_correction_calculation(earth_heliocentric_position):
    """
    This function compute the aberration_correction, as a function of the
    earth-sun distance.

    :param earth_heliocentric_position:
    :return:
    """
    aberration_correction = -20.4898/(3600*earth_heliocentric_position['radius'])
    return aberration_correction


def apparent_sun_longitude_calculation(sun_geocentric_position, nutation, aberration_correction):
    """
    This function compute the sun apparent longitude

    :param sun_geocentric_position:
    :param nutation:
    :param aberration_correction:
    :return:
    """
    apparent_sun_longitude = sun_geocentric_position['longitude'] + nutation['longitude'] + aberration_correction
    return apparent_sun_longitude


def apparent_stime_at_greenwich_calculation(julian, nutation, true_obliquity):
    """
    This function compute the apparent sideral time at Greenwich.

    :param julian:
    :param nutation:
    :param true_obliquity:
    :return:
    """

    JD = julian['day']
    JC = julian['century']

    # Mean sideral time, in degrees
    mean_stime = 280.46061837 + (360.98564736629*(JD-2451545)) + \
                 (0.000387933*np.power(JC, 2)) - \
                 (np.power(JC, 3)/38710000)

    # Limit the range to [0-360];
    mean_stime = set_to_range(mean_stime, 0, 360)

    apparent_stime_at_greenwich = mean_stime + (nutation['longitude'] * np.cos(true_obliquity * np.pi/180))
    return apparent_stime_at_greenwich


def sun_rigth_ascension_calculation(apparent_sun_longitude, true_obliquity, sun_geocentric_position):
    """
    This function compute the sun rigth ascension.
    :param apparent_sun_longitude:
    :param true_obliquity:
    :param sun_geocentric_position:
    :return:
    """

    argument_numerator = (np.sin(apparent_sun_longitude * np.pi/180) * np.cos(true_obliquity * np.pi/180)) - \
        (np.tan(sun_geocentric_position['latitude'] * np.pi/180) * np.sin(true_obliquity * np.pi/180))
    argument_denominator = np.cos(apparent_sun_longitude * np.pi/180);

    sun_rigth_ascension = np.arctan2(argument_numerator, argument_denominator) * 180/np.pi
    # Limit the range to [0,360];
    sun_rigth_ascension = set_to_range(sun_rigth_ascension, 0, 360)
    return sun_rigth_ascension


def sun_geocentric_declination_calculation(apparent_sun_longitude, true_obliquity, sun_geocentric_position):
    """

    :param apparent_sun_longitude:
    :param true_obliquity:
    :param sun_geocentric_position:
    :return:
    """

    argument = (np.sin(sun_geocentric_position['latitude'] * np.pi/180) * np.cos(true_obliquity * np.pi/180)) + \
        (np.cos(sun_geocentric_position['latitude'] * np.pi/180) * np.sin(true_obliquity * np.pi/180) *
         np.sin(apparent_sun_longitude * np.pi/180))

    sun_geocentric_declination = np.arcsin(argument) * 180/np.pi
    return sun_geocentric_declination


def observer_local_hour_calculation(apparent_stime_at_greenwich, location, sun_rigth_ascension):
    """
    This function computes observer local hour.

    :param apparent_stime_at_greenwich:
    :param location:
    :param sun_rigth_ascension:
    :return:
    """

    observer_local_hour = apparent_stime_at_greenwich + location['longitude'] - sun_rigth_ascension
    # Set the range to [0-360]
    observer_local_hour = set_to_range(observer_local_hour, 0, 360)
    return observer_local_hour


def topocentric_sun_position_calculate(earth_heliocentric_position, location,
                                       observer_local_hour, sun_rigth_ascension, sun_geocentric_declination):
    """
    This function compute the sun position (rigth ascension and declination)
    with respect to the observer local position at the Earth surface.

    :param earth_heliocentric_position:
    :param location:
    :param observer_local_hour:
    :param sun_rigth_ascension:
    :param sun_geocentric_declination:
    :return:
    """

    # Equatorial horizontal parallax of the sun in degrees
    eq_horizontal_parallax = 8.794 / (3600 * earth_heliocentric_position['radius'])

    # Term u, used in the following calculations (in radians)
    u = np.arctan(0.99664719 * np.tan(location['latitude'] * np.pi/180))

    # Term x, used in the following calculations
    x = np.cos(u) + ((location['altitude']/6378140) * np.cos(location['latitude'] * np.pi/180))

    # Term y, used in the following calculations
    y = (0.99664719 * np.sin(u)) + ((location['altitude']/6378140) * np.sin(location['latitude'] * np.pi/180))

    # Parallax in the sun rigth ascension (in radians)
    nominator = -x * np.sin(eq_horizontal_parallax * np.pi/180) * np.sin(observer_local_hour * np.pi/180)
    denominator = np.cos(sun_geocentric_declination * np.pi/180) - (x * np.sin(eq_horizontal_parallax * np.pi/180) *
                                                                    np.cos(observer_local_hour * np.pi/180))
    sun_rigth_ascension_parallax = np.arctan2(nominator, denominator)
    # Conversion to degrees.
    topocentric_sun_position = dict()
    topocentric_sun_position['rigth_ascension_parallax'] = sun_rigth_ascension_parallax * 180/np.pi

    # Topocentric sun rigth ascension (in degrees)
    topocentric_sun_position['rigth_ascension'] = sun_rigth_ascension + (sun_rigth_ascension_parallax * 180/np.pi)

    # Topocentric sun declination (in degrees)
    nominator = (np.sin(sun_geocentric_declination * np.pi/180) - (y*np.sin(eq_horizontal_parallax * np.pi/180))) * \
                np.cos(sun_rigth_ascension_parallax)
    denominator = np.cos(sun_geocentric_declination * np.pi/180) - (y*np.sin(eq_horizontal_parallax * np.pi/180)) * \
                                                                   np.cos(observer_local_hour * np.pi/180)
    topocentric_sun_position['declination'] = np.arctan2(nominator, denominator) * 180/np.pi
    return topocentric_sun_position


def topocentric_local_hour_calculate(observer_local_hour, topocentric_sun_position):
    """
    This function compute the topocentric local jour angle in degrees

    :param observer_local_hour:
    :param topocentric_sun_position:
    :return:
    """

    topocentric_local_hour = observer_local_hour - topocentric_sun_position['rigth_ascension_parallax']
    return topocentric_local_hour


def sun_topocentric_zenith_angle_calculate(location, topocentric_sun_position, topocentric_local_hour):
    """
    This function compute the sun zenith angle, taking into account the
    atmospheric refraction. A default temperature of 283K and a
    default pressure of 1010 mbar are used.

    :param location:
    :param topocentric_sun_position:
    :param topocentric_local_hour:
    :return:
    """

    # Topocentric elevation, without atmospheric refraction
    argument = (np.sin(location['latitude'] * np.pi/180) * np.sin(topocentric_sun_position['declination'] * np.pi/180)) + \
    (np.cos(location['latitude'] * np.pi/180) * np.cos(topocentric_sun_position['declination'] * np.pi/180) *
     np.cos(topocentric_local_hour * np.pi/180))
    true_elevation = np.arcsin(argument) * 180/np.pi

    # Atmospheric refraction correction (in degrees)
    argument = true_elevation + (10.3/(true_elevation + 5.11))
    refraction_corr = 1.02 / (60 * np.tan(argument * np.pi/180))

    # For exact pressure and temperature correction, use this,
    # with P the pressure in mbar amd T the temperature in Kelvins:
    # refraction_corr = (P/1010) * (283/T) * 1.02 / (60 * tan(argument * pi/180));

    # Apparent elevation
    apparent_elevation = true_elevation + refraction_corr

    sun = dict()
    sun['zenith'] = 90 - apparent_elevation

    # Topocentric azimuth angle. The +180 conversion is to pass from astronomer
    # notation (westward from south) to navigation notation (eastward from
    # north);
    nominator = np.sin(topocentric_local_hour * np.pi/180)
    denominator = (np.cos(topocentric_local_hour * np.pi/180) * np.sin(location['latitude'] * np.pi/180)) - \
    (np.tan(topocentric_sun_position['declination'] * np.pi/180) * np.cos(location['latitude'] * np.pi/180))
    sun['azimuth'] = (np.arctan2(nominator, denominator) * 180/np.pi) + 180

    # Set the range to [0-360]
    sun['azimuth'] = set_to_range(sun['azimuth'], 0, 360)
    return sun


def set_to_range(var, min_interval, max_interval):
    """
    Sets a variable in range min_interval and max_interval

    :param var:
    :param min_interval:
    :param max_interval:
    :return:
    """
    var = var - max_interval * np.floor(var/max_interval)

    if var < min_interval:
        var = var + max_interval
    return var


