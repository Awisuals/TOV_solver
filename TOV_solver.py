# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 09:42:03 2022

@author: anter
"""
from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d

"""

Tämän ohjelma ratkaisee yhdistetyn epälineaarisen
differentiaaliryhmän. Yhtälöt ovat ns. TOV-yhtälöt, jotka kuvaavat yhdessä
tähden rakenteen.

Ohjelmalla voi määrittää jollekkin tähtityypille massa-säde - relaation
mallintamalla useaa tähteä annetuilla parametreilla sekä joillakin
tilanyhtälöillä ja varioimalla esim. tähden keskipisteen energiatiheyttä. 

Notaatio:
    WD = White Dwarf, NS = Neutron Star

Valitaan yksiköiksi geometrisoidut yksiköt:
    G = c = 1

Yhtälöt:
    
    dmdr = 4*np.pi*rho*r**2                               
    dpdr = -(rho+p)*(m + 4*np.pi*r**3*p)/(r*(r-2*m))   # Relativistinen
    dpdr = -(m*rho)/(r**2)                             # Ei-Relativistinen
    
Näiden lisäksi tarvitaan tilanyhtälö. Valkoiselle kääpiölle valitaan
paineen ja energiatiheyden relatoiva polytrooppimalli:
    
    p = Kappa*rho**Gamma
    
Neutronitähdelle valitaan sopiva(t) malli(t) paperista 
"A unified equation of state of dense matter and neutron star
structure".:
    
    Taulukot 3. ja 5.
    
Ohjelmalla voi lakea avaruuden kaarevuusskalaarin (R) - Riccin skalaari -
ja piirtää tästä kuvaajan.:
    
    R = -8*np.pi*G*(rho - 3*p)

"""

#%% 

"""

Yleisiä funktioita hyötykäyttöön.

"""

def find_duplicates(arr):
    """
    Käy annetun taulukon läpi ja etsii siitä dublikaatit.
    Löytäessään duplikaatteja funktio summaa pienen luvun
    jälkimmäiseen alkioon.

    Parameters
    ----------
    arr : Array
        Taulukko.

    Returns
    -------
    None.

    """
    # TODO: KORJAA DUPLIKAATIN ETSINTÄ JA MUUTTO
    
    i = 1
    value = arr[0]
    while i < len(arr):
        if arr[i] == value:
            print("Duplikaatti löytynyt!")
            print("indeksi:" + str(i-1) + "ja arvo" + str(arr[i]))
            arr[i] = arr[i]+1e-3
        value = arr[i]
        i+=1
    
    return

def graph(x, y, style, label, xlabel, ylabel):

    
    """
    Generates graph with given parameters.

    Parameters
    ----------
    x : table
        values for x-axis.
    y : table
        values for y-axis.
    label : string
        label for legend.
    xlabel : string
        label for x-axis.
    ylabel : string
        label for y-axis.

    Returns
    -------
    None.

    """
    
    plt.figure()
    style(x, y, label=label)
    plt.axhline(y=0, color='r', linestyle='--')     # Piirtää suoran y = 0.
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.yscale('log')
    plt.xscale('log')
    plt.legend()
    plt.show()       

def unit_conversion(SYS, VAR, VAL, DIR):
    """
    Changes units on desired direction. Handles
    geometrized, natural and SI-units.
    
    Unit system:
        0 for geometrizied units
        1 for natural units
    
    Possible conversion apply for:
        M, P and RHO.
        
    Directions:
        1: from geom. to SI.
        -1: from SI to geom.

    Parameters
    ----------
    VAR : String
        Variable that needs unit change.
    VAL : Float
        Value of said variable.
    DIR : Int
        Direction of desired conversion.

    Returns
    -------
    None.

    """
    VAR_TAU = ["M", "P", "RHO"]
    SYS_CON = [[1.3466e27, 1.2102e44, 1.2102e44],
               [1.7827e-27, 2.0852e37, 2.0852e37]]
    VAR_CON = SYS_CON[SYS]
    for i, var in enumerate(VAR_TAU):
        if VAR == var and DIR == 1:
            VAL = VAL*VAR_CON[i]
        elif VAR == var and DIR == -1:
            VAL = VAL/VAR_CON[i]
    return VAL

def polytrope_constants(r0, rho_c, n):
    """
    Returns constant of proportionality that 
    emerges in astrophysical polytrope equation.
    
    Parameters
    ----------
    r0 : Float
        Approximate radius. (Meters)
    rho_c : Float
        Energy density in core. (in geom.-units)
    n : Float
        polytrope index. (Goes from 0.5 to inf, 
                          depends on the model)

    Returns
    -------
    Float
        constant of proportionality

    """
    K = (r0**2*4*np.pi*rho_c**(1+1/n))/(n+1)
    G = (n+1)/n
    return K, G

def find_radius(p_t, r_t, raja=0):
    """
    Etsii tähden rajaa vastaavan säteen.
    Raja voidaan määritellä funktiokutsussa tai
    funktion määrittelyssä.

    Parameters
    ----------
    p_t : Array
        Paineen taulukko.
    r_t : Array
        Säteen taulukko.
    raja : Float, määritelty raja
        Määrää tähden rajan. The default is 1e-10.

    Returns
    -------
    R_raja : Float
        Tähden rajaa vastaavan säteen.

    """
    p = p_t[0]
    R_raja = r_t[0]
    p_raja = raja*p
    i = 0
    while i < len(p_t):
        p = p_t[i]
        R_raja = r_t[i] 
        i += 1
        if p < p_raja:
            break
    print(R_raja)
    return R_raja

def find_mass_in_radius(m_t, r_t, r_raja):
    """
    Palauttaa tähden säteen sisällä olevan massan.

    Parameters
    ----------
    m_t : Array
        Massan taulukko.
    r_t : Array
        DESCRIPTION.
    r_raja : Float
        Tähden raja.

    Returns
    -------
    m : Float
        Tähden säteen sisällä oleva massa.

    """
    r = r_t[0]
    m = m_t[0]
    i = 0
    while i < len(r_t):
        r = r_t[i]
        m = m_t[i]
        i += 1
        if r > r_raja:
            break
    print(m)
    print("==========")
    return m

def set_initial_conditions(rmin, G, K, rho0=0, p0=0, a=0):
    """
    Utility routine to set initial data. Can be given either
    pressure or energy density at core. Value a tells
    which is first.

    Parameters
    ----------
    rmin : Float
        Lower limit for integration.
    G : Float
        Polytrope constant.
    K : Float
        Polytrope constant of proportionality.
    rho0 : Float, optional
        Given energy density at core. The default is 0.
    p0 : Float, optional
        Given pressure at core. The default is 0.
    a : Int, optional
        Choice for given initial value. Another is then 0 and
        calculated from EoS. Can take both also.
        
        Choice:
            a = 0 is for given rho0.
            a = 1 is for given p0.
            a = 2 is for given rho0 and p0.
        
        The default is 0.

    Returns
    -------
    m : Float
        Mass inside radius rmin.
    p : Float
        pressure at r ~ 0.
    rho : Float
        Energy density at r ~ 0.

    """
    rho_values0 = [rho0, EoS_p2r(p0, G, K), rho0]
    p_values0 = [EoS_r2p(rho0, G, K), p0, p0]
    if a == 0:
        rho = rho_values0[a]
        p = p_values0[a]
    elif a == 1:
        p = p_values0[a]
        rho = rho_values0[a]
    else:
        rho = rho
        p = p0
    m = 4./3.*np.pi*rho*rmin**3
    return m, p, rho

# %%

"""

Määritellään polytrooppi tilanyhtälö energiatiheydelle ja paineelle.

Määritetään Tolman-Oppenheimer-Volkoff - yhtälöt, jotka ratkaisemalla
saadaan kuvattua tähden rakenne.

"""

def EoS_r2p(rho, Gamma, Kappa):
    """
    Tilanyhtälö // Equation of state, EoS. 
    
    Annettaessa energiatiheyden ja vakiot palauttaa paineen.

    Parameters
    ----------
    rho : float
        Energiatiheys.
    Gamma : float
        Vakio.
    Kappa : float
        Vakio.

    Returns
    -------
    p : float
        Paine.

    """
    p = Kappa*rho**Gamma
    return p

def EoS_p2r(p, Gamma, Kappa):
    """
    Tilanyhtälö // Equation of state, EoS. 
    
    Annettaessa energiatiheyden ja vakiot palauttaa paineen.


    Parameters
    ----------
    p : float
        Paine.
    Gamma : float
        Vakio.
    Kappa : float
        Vakio.

    Returns
    -------
    rho : float
        Energiatiheys.

    """
    rho = (p/Kappa)**(1/Gamma)
    return rho

def TOV(r, y, K, G):
    """
    Määritellään TOV-yhtälöt ja palautetaan ne taulukossa.

    Parameters
    ----------
    y : Array
        Alkuarvot.
    r : Array
        Integrointirajat.

    Returns
    -------
    dy : Array
        Integroitavat funktiot.

    """
    m = y[0]                            # Asetetaan muuttujat taulukkoon
    p = y[1]    
    # print(p)
    # G = Gamma
    # K = Kappa
    # rho_eos = [EoS_p2r(p, G, K), NS_EoS_P2rho(p)]
    rho = EoS_p2r(p, G, K)      # WD:n energiatiheys rho_eos[0]
    # rho = NS_EoS_P2rho(p)             # NS:n energiatiheys
    
    dy = np.empty_like(y)
    dy[0] = 4*np.pi*rho*r**2                            # Massa säteen sisällä                               
    dy[1] = -(rho+p)*(m + 4*np.pi*r**3*p)/(r*(r-2*m))   # Paine - REL
    # dy[1] = -(m*rho)/(r**2)                           # Paine - EI-REL
    
    return dy

def found_radius(t, y, dump1, dump2): # 
    """
    Event function: Zero of pressure 
    ODE integration stops when this function returns True
    """
    dump1, dump2 = dump1, dump2
    return y[1].real

found_radius.terminal = True
found_radius.direction = -1       


#%%

# ============================================================
# PARAMETRIEN ARVOJA

"""

konversio J/m3 -> 1.1036e-26 g/cm3

Valkoinen kääpiö (geom. units):
    R0 = ~R_earth = 6e6
    rho_c = ~1e-10

REL (TOV):
    n = 3
    Gamma = 4/3

    
EI-REL (NEWT.):
    n = 1.5
    Gamma = 5/3
    
    
"""

#%%

"""

Valitaan alkuarvaukset ja ratkaistaan tähden rakennetta
kuvaavat TOV - yhtälöt STIFF - integraattorilla.

Metodi:
    solve_ivp().

"""

# VAKIOITA
LCGS = 1.476701332464468e+05
CONV_jmc2gcmc = 1.1036e-26      # Konversiokerroin 1 J/m3 = 1.1036e-26 g/cm3
M_sun = 2e30                  # kg
R_earth = 6e6
WD_rho_c = 1e-10
c = 1
G = 1

# Asetetaan integrointiparametrit
rmin, rmax = 1e-3, np.inf
N = 200
rspan = np.linspace(rmin, rmax, N)

# Initiaalirajat säteelle
r0, rf = rmin, rmax               # Otetaan rspan ääriarvot.

def SOLVE_TOV(R_body, n, a=0, rho_c=0, p_c=0):
    
    Kappa, Gamma = polytrope_constants(R_body, rho_c, n)
    m, p, rho = set_initial_conditions(r0, Gamma, Kappa, rho_c, p_c, a)
    y0 = m, p
    
    print("Tulostetaan alkuarvot. \n Kappa ja Gamma:" + str(Kappa) + " ja " + str(Gamma) + 
          "\n Asetetut alkuarvot (m, p ja rho):" + str(y0) + "\n \n")
    
    soln = solve_ivp(TOV, (r0, rf), y0, method='BDF',
                     dense_output=True, events=found_radius, 
                     args = (Kappa, Gamma))
    
    print("Solverin parametreja:")
    print(soln.nfev, 'evaluations required')
    print(soln.t_events)
    print(soln.y_events)
    print("\n \n")

    # Määritellään muuttujat taulukkoon. 
    # Ratkaisut yksiköissä [p] = g/cm**2 ja [rho] = g/cm**3 
    # TODO korjaa rhon yksiköt ja tarkista säteen yksiköt
    r = soln.t #* LCGS * 1e-6 # km(?)
    m = unit_conversion(0, "M", soln.y[0].real, 1)/M_sun
    p = unit_conversion(0, "P", soln.y[1].real, 1)*CONV_jmc2gcmc
    rho = unit_conversion(0, "RHO", EoS_p2r(p, Gamma, Kappa), 1)*CONV_jmc2gcmc

    print("Saadut TOV ratkaisut: \n")
    print("Säde: \n" + str(r) + "\n Massa: \n" + str(m) + 
          "\n Paine: \n" + str(p) + "\n Energiatiheys: \n" + str(rho))
    print("\n \n")
    
    # Piirretään ratkaisun malli kuvaajiin.
    graph(r, m, plt.plot, "massa", "säde, r", "massa, m")
    graph(r, p, plt.plot, "paine", "säde, r", "paine, p")
    graph(r, rho, plt.plot, 
          f'energiatiheys, \n rho_c = {rho_c.real} \n Kappa = {Kappa.real} \n Gamma = {Gamma}' 
          , "säde, r", "energiatiheys, rho")
    
    return r, m, p, rho

# Ratkaistaan TOV valkoisen kääpiön alkuarvoille:
# SOLVE_TOV(R_earth, 3, rho_c = 1e-9+0j)



#%%

"""

Ratkaistaan massa-säde relaatio. Etsitään TOV-yhtälöiden ratkaisuja
jollakin rhospan-alueella. Ratkaistaan yhtälöitä siis eri tähden keskipisteen
energiatiheyksien arvoilla.

Etsitään tähden raja (find_radius) paineen ratkaisusta ja sitä vastaava
massa massan kuvaajasta. Tallennetaan nämä arvot taulukkoon ja piirretään
kuvaaja.

Mallinnetaan nyt useaa tähteä ja piirretään
Massa-Säde - relaatio.

"""

# TODO korjaa

def MR_relaatio(rho_min, rho_max):
    
    # Build 200 star models
    
    rhospan = np.linspace(rho_min, rho_max, 100)
    R = []
    M = []
    for rho0 in rhospan:
        r, m, p, rho = SOLVE_TOV(R_earth, 3, rho_c = rho0)
        r_boundary = find_radius(p, r, raja=0)
        m_boundary = find_mass_in_radius(m, r, r_boundary)
        R.append(r_boundary)
        M.append(m_boundary)
    print("Tulostetaan ratkaistut massat ja niitä vastaavat säteet: \n")
    print("Säteet: \n " + str(R) + "\n Massat: \n" + str(M))
    graph(R, M, plt.scatter, "Massa-säde - relaatio", "Säde", "Massa") 
    graph(R, M, plt.plot, "Massa-säde - relaatio", "Säde", "Massa") 
    
    return R, M

# MR_relaatio(1e-20+0j, 1e-8+0j)

#%%

"""

Määritellään Riccin skalaari ja ratkaistaan se
annetuilla parametreilla.

"""


def Ricci_scalar(p, rho, r):
    """
    Laskee avaruuden kaarevuusskalaarin - Riccin skalaari.

    Parameters
    ----------
    p : Array
        Paineen ratkaisu.
    rho : Array
        Energiatiheyden ratkaisu.

    Returns
    -------
    None.

    """
    R_scalar = 8*np.pi*(rho - 3*p)      # G = 1
    graph(r, R_scalar, plt.plot, 
          "Avaruuden kaarevuus", "Säde, r", "Riccin skalaari, R")

#%%

"""

Rakennetaan neutronitähden malli paperista "A unified equation 
of state of dense matter and neutron star structure" saadulla datalla
sisemmän kuoren tilanyhtälöstä ja ytimen tilanyhtälöstä.

Tilanyhtälöt:
    Ulompi kuori  -> Polytrooppi tilanyhtälö Gamma = 4/3
    Sisempi kuori -> Data paperin taulukosta 3.
    Ydin          -> Data paperin taulukosta 5.
    
Tilanyhtälöiden muuttujat datasta:
    n_b, rho, P, Gamma.

"""

# Neutronitähden sisemmän kuoren tilanyhtälön
# ratkaistuja parametreja.
NS_Eos_ic = pd.read_csv(
    'NT_EOS_inner_crust.txt', sep=";", header=None)

NS_EoS__ic_n_b      = NS_Eos_ic[0].values
NS_EoS__ic_rho      = NS_Eos_ic[1].values
NS_EoS__ic_P        = NS_Eos_ic[2].values
NS_EoS__ic_Gamma    = NS_Eos_ic[3].values

# Neutronitähden ytimen tilanyhtälön
# ratkaistuja parametreja.
NS_Eos_core = pd.read_csv(
    'NT_EOS_core.txt', sep=";", header=None)

NS_EoS__core_n_b    = NS_Eos_core[0].values
NS_EoS__core_rho    = NS_Eos_core[1].values 
NS_EoS__core_P      = NS_Eos_core[2].values
NS_EoS__core_Gamma  = NS_Eos_core[3].values

# Yhdistetään sisemmän kuoren ja ytimen
# energiatiheys ja paine. Käännetään taulukot myös
# alkamaan ytimestä.

# Energiatiheys
NS_EoS_ic_core_rho = np.flip(np.append(NS_EoS__ic_rho, NS_EoS__core_rho) * 7.4261e-25, -1)
# print("energiatiheys")
# print(NS_EoS_ic_core_rho + len(NS_EoS_ic_core_rho))
# find_duplicates(NS_EoS_ic_core_rho)
# print(NS_EoS_ic_core_rho + len(NS_EoS_ic_core_rho))

# Kerrotaan painetta konversiokertoimella 1.1036e-27,
# jotta saadaan energiatiheydelle ja paineelle samat yksiköt g/cm**3

# Paine, MUUTETAAN DUPLIKAATIN DATA TEKSTITIEDOSTOSSA
# TODO: MUUTA DATA OIKEAKSI
# MUUTETTU INDEKSI ON NT_EOS_CORE.txt ENSIMMÄINEN PAINE
NS_EoS_ic_core_P = np.flip(np.append(NS_EoS__ic_P, NS_EoS__core_P) * 8.2627e-46, -1)
# print("Paine")
# print(NS_EoS_ic_core_P + len(NS_EoS_ic_core_P))
# find_duplicates(NS_EoS_ic_core_P)
# print(NS_EoS_ic_core_P + len(NS_EoS_ic_core_P))

# Plotataan paine ja energiatiheys kuvaaja (rho, P) tutkimuspaperista.
graph(NS_EoS_ic_core_P, NS_EoS_ic_core_rho, plt.scatter, "NS EoS, (P, rho)", 
      "Paine, P", "Energiatiheys, rho")

# Määritetään interpoloitu funktio alkuperäiselle datalle
f = interp1d(NS_EoS_ic_core_P, NS_EoS_ic_core_rho, kind='cubic')

def NS_EoS_P2rho(P_point):
    """
    Neutronitähden tilanyhtälö. Palauttaa energiatiheyden
    annettaessa paineen.

    Parameters
    ----------
    P_point : float OR array of floats
        Paine.

    Returns
    -------
    float OR array of floats
        Palauttaa energiatiheyden annetussa pistejoukossa.

    """
    return f(P_point)

# Määritetään x-akselin paineen arvoille uusi tiheys
NS_EoS_ic_core_P_new = np.logspace(
    NS_EoS_ic_core_P[0], NS_EoS_ic_core_P[-1], 10000, base=1e-15)

# Testataan interpolointia yhdellä arvolla
# print("testataan yksi interpolaation arvo, tästä saadaan painetta vastaava energiatiheys")
# print(NS_EoS_P2rho(NS_EoS_ic_core_P_new[100]))

# Piirretään interpoloidut datapisteet.
graph(NS_EoS_ic_core_P_new, f(NS_EoS_ic_core_P_new), plt.scatter, 
        "NS EoS, (rho, P) interpolate", "Paine, P", "Energiatiheys, rho")

# Tulostetaan NS:n paine ja energiatiheys konsoliin.
print("Neutronitähden energiatiheys ja paine ytimestä")
print(NS_EoS_P2rho(NS_EoS_ic_core_P_new), NS_EoS_ic_core_P_new)

# Määrätään alkuarvot TOV:n ratkaisulle
NS_p0 = NS_EoS_ic_core_P_new[0]
NS_rho_c = NS_EoS_ic_core_rho[0]
NS_m0 = 4./3.*np.pi*NS_rho_c*rmin**3
NS_y0 = NS_m0, NS_p0

# Ratkaistaan TOV ja mallinnetaan NS:n rakenne.
sol_NS = solve_ivp(TOV, (r0, rf), NS_y0, method='BDF', dense_output=True, 
                    events=found_radius)

# Ratkaisut
NS_r_stiff = sol_NS.t * LCGS * 1e-6 # km(?)
NS_m = sol_NS.y[0].real
NS_p = sol_NS.y[1].real
NS_rho = NS_EoS_P2rho(NS_p)

# NS:n malli kuvaajissa
graph(NS_m, NS_r_stiff, plt.plot, "Massa säteen funktiona", "Säde", "Massa")
graph(NS_p, NS_r_stiff, plt.plot, "Paine säteen funktiona", "Säde", "Paine")
graph(NS_m, NS_r_stiff, plt.plot, "Energiatiheys säteen funktiona", "Säde", "Eenrgiatiheys")

