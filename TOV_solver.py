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

# %%
"""

Esimerkkejä koodin ajamisesta konsolissa:
    TODO WD - rakenne
    TODO WD Massa-säde - relaatio
    TODO WD avaruuden kaarevuus
    TODO NS - rakenne
    TODO NS avaruuden kaarevuus
    TODO kivieksoplaneetta

Parametrien arvojen kokoluokka:

Valkoinen kääpiö (geom. units):
    R0 = ~R_earth = 6e6
    rho_c = ~1e-10

Neutronitähti (geom. units):
    R0 = ~10km = 10000m

REL (TOV):
    n = 3
    Gamma = 4/3

EI-REL (NEWT.):
    n = 1.5
    Gamma = 5/3

Määrätään vakioita ja integrointirajat.

"""

# VAKIOITA
M_sun = 2e30              # kg
R_WD0 = 6e6               # m
R_NS0 = 10000             # m

# Geometrisoidut yksiköt. (Luonnollisia yksiköitä)
c = 1
G = 1

# Asetetaan integrointiparametrit. 
# Integraattori adaptiivinen, lopettaa integroinnin tähden rajalla.
rmin, rmax = 1e-3, np.inf
N = 200
rspan = np.linspace(rmin, rmax, N)

# Initiaalirajat säteelle
r0, rf = rmin, rmax               # Otetaan rspan ääriarvot.

# %%
"""

Yleisiä funktioita hyötykäyttöön.

"""


def graph(x, y, style, label, xlabel, ylabel, scale, otsikko):
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
    plt.xscale(scale)
    plt.yscale(scale)
    plt.title(otsikko)
    plt.legend()
    plt.show()


def unit_conversion(SYS, VAR, VAL, DIR):
    """
    Changes units on desired direction. Handles
    geometrized, natural and SI-units.

    Unit system:
        0 for geometrizied units
        1 for natural units
        2 for [p]=erg/cm**3 and [rho]=g/cm**3 -> GEOM.

    Possible conversion apply for:
        M, P and RHO.

    Directions:
        1: from SYS. to SI. MULTIPLY
        -1: from SI to SYS. DIVIDE

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
               [1.7827e-27, 2.0852e37, 2.0852e37],
               [1, 8.2627e-46, 7.4261e-25]]
    VAR_CON = SYS_CON[SYS]
    for i, var in enumerate(VAR_TAU):
        if VAR == var and DIR == 1:
            VAL = VAL*VAR_CON[i]
        elif VAR == var and DIR == -1:
            VAL = VAL/VAR_CON[i]
    return VAL


def gamma_from_n(n):
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
    return (n+1)/n


def kappa_from_r0rho0n(r0, rho0, n):
    return (r0**2*4*np.pi*rho0**(1+1/n))/(n+1)


def kappa_from_p0rho0(p0, rho0, G):
    return (p0)/(rho0**G)


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
        rho = rho0
        p = p0
    m = 4./3.*np.pi*rho*rmin**3
    return m, p, rho


# %%
"""

Määritellään polytrooppi tilanyhtälö energiatiheydelle ja paineelle.

Määritetään Tolman-Oppenheimer-Volkoff - yhtälöt, jotka ratkaisemalla
saadaan kuvattua tähden rakenne.

"""


def EoS_CUSTOMDATA_p2rho(interpolation, p_point):
    return interpolation(p_point)


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


def EoS_choiser(choise, interpolation, p, Gamma, Kappa):
    """
    

    Parameters
    ----------
    choise : TYPE
        DESCRIPTION.
    interpolation : TYPE
        DESCRIPTION.
    p : TYPE
        DESCRIPTION.
    Gamma : TYPE
        DESCRIPTION.
    Kappa : TYPE
        DESCRIPTION.

    Returns
    -------
    rho : TYPE
        DESCRIPTION.

    """

    if choise == 0:
        rho = EoS_p2r(p, Gamma, Kappa)
    elif choise == 1:
        rho = EoS_CUSTOMDATA_p2rho(interpolation, p)

    return rho


def TOV(r, y, K, G, interpolation, rho_func):
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
    # Asetetaan muuttujat taulukkoon
    # Energiatiheys valitaan valitsin-funktiossa.
    m = y[0]                            
    p = y[1]
    rho = EoS_choiser(rho_func, interpolation, p, G, K)

    # Ratkaistavat yhtälöt
    dy = np.empty_like(y)
    dy[0] = 4*np.pi*rho*r**2                            # Massa säteen sisällä
    dy[1] = -(rho+p)*(m + 4*np.pi*r**3*p)/(r*(r-2*m))   # Paine - REL
    # dy[1] = -(m*rho)/(r**2)                           # Paine - EI-REL

    return dy


def found_radius(t, y, d1, d2, d3, d4):
    """
    Event function: Zero of pressure
    ODE integration stops when this function returns True
    """
    d1, d2, d3, d4 = d1, d2, d3, d4
    return y[1].real


found_radius.terminal = True
found_radius.direction = -1

# %%


def SOLVE_TOV(n, R_body=0, kappa_choise=0, rho_K=0, p_K=0,
              rho_c=0, p_c=0, a=0, interpolation=0, rho_func=0, kappale=""):
    """
    

    Parameters
    ----------
    n : TYPE
        DESCRIPTION.
    R_body : TYPE, optional
        DESCRIPTION. The default is 0..
    rho_K : TYPE, optional
        DESCRIPTION. The default is 0..
    p_K : TYPE, optional
        DESCRIPTION. The default is 0..
    rho_c : TYPE, optional
        DESCRIPTION. The default is 0..
    p_c : TYPE, optional
        DESCRIPTION. The default is 0..
    a : TYPE, optional
        DESCRIPTION. The default is 0..
    interpolation : TYPE, optional
        DESCRIPTION. The default is 0..
    rho_func : TYPE, optional
        DESCRIPTION. The default is 0..

    Returns
    -------
    r : TYPE
        DESCRIPTION.
    m : TYPE
        DESCRIPTION.
    p : TYPE
        DESCRIPTION.
    rho : TYPE
        DESCRIPTION.

    """

    Gamma = gamma_from_n(n)
    Kappa_VAL = [kappa_from_p0rho0(p_K, rho_K, Gamma), 
                 kappa_from_r0rho0n(R_body, rho_K, n)]
    Kappa = Kappa_VAL[kappa_choise]
    m, p, rho = set_initial_conditions(r0, Gamma, Kappa, rho_c, p_c, a)
    y0 = m, p

    print("Tulostetaan alkuarvot. \n Kappa ja Gamma:" + str(Kappa) +
          " ja " + str(Gamma) + "\n Asetetut alkuarvot (m, p ja rho):"
          + str(y0) + "\n \n")

    soln = solve_ivp(TOV, (r0, rf), y0, method='BDF',
                     dense_output=True, events=found_radius,
                     args=(Kappa, Gamma, interpolation, rho_func))

    print("Solverin parametreja:")
    print(soln.nfev, 'evaluations required')
    print(soln.t_events)
    print(soln.y_events)
    print("\n \n")

    # Määritellään muuttujat taulukkoon.
    # Ratkaisut yksiköissä [m] = kg, [p] = m**-2 ja [rho] = m**-2
    r = soln.t
    m = soln.y[0].real
    p = soln.y[1].real
    rho = EoS_p2r(p, Gamma, Kappa)

    print("Saadut TOV ratkaisut: \n")
    print("Säde: \n" + str(r.real) + "\n Massa: \n" + str(m.real) +
          "\n Paine: \n" + str(p.real) + "\n Energiatiheys: \n" + str(rho.real))
    print("\n \n")

    # Piirretään ratkaisun malli kuvaajiin.
    # [m] = kg, [p] = erg/cm**m ja [rho] = g/cm**3 
    graph(r, unit_conversion(0, "M", m, 1),
          plt.plot, "massa", "säde, r (m)", "massa, m (kg)", 'linear',
          kappale + " " + "massa säteen funktiona")
    graph(r, unit_conversion(2, "P", p, -1),
          plt.plot, "paine", "säde, r (m)", "p (erg/cm^3)", 'linear', 
          kappale + " " + "paine säteen funktiona")
    graph(r, unit_conversion(2, "RHO", rho, -1), plt.plot,
          fr'$\rho_c$ = {rho_c.real}' '\n'
          fr'$K$ = {Kappa.real}' '\n' 
          fr'$\Gamma$ = {Gamma}',
          "säde, r", "energiatiheys, rho (g/cm^3)", 'linear',
          kappale + " " + "energiatiheys säteen funktiona")

    return r, m, p, rho


# Ratkaistaan TOV valkoisen kääpiön alkuarvoille:
# SOLVE_TOV(3, R_body=6e6, rho_K=1e-10+0j, rho_c=1e-10+0j, a=0, rho_func=0)


# %%
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
        r, m, p, rho = SOLVE_TOV(R_WD0, 3, rho_cK=1e-11+0j, rho_c=rho0)
        r_boundary = find_radius(p, r, raja=0)
        m_boundary = find_mass_in_radius(m, r, r_boundary)
        R.append(r_boundary)
        M.append(m_boundary)
    print("Tulostetaan ratkaistut massat ja niitä vastaavat säteet: \n")
    print("Säteet: \n " + str(R) + "\n Massat: \n" + str(M))
    graph(R, M, plt.scatter, "Massa-säde - relaatio", "Säde",
          "Massa", 'linear', "Massa-säde")
    graph(R, M, plt.plot, "Massa-säde - relaatio", "Säde",
          "Massa", 'linear', "Massa-säde")

    return R, M


# MR_relaatio(1e-16+0j, 1e-8+0j)


# %%
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
    R_scalar = -8*np.pi*(rho - 3*p)
    graph(r, R_scalar, plt.plot,
          "Avaruuden kaarevuus", "Säde, r", "Riccin skalaari, R", 
          'linear', "Avaruuden kaarevuusskalaari")


# %%
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

# Neutronitähden ytimen tilanyhtälön
# ratkaistuja parametreja.
NS_Eos_core = pd.read_csv(
    'NT_EOS_core.txt', sep=";", header=None)

NS_EoS_core_n_b = NS_Eos_core[0].values
NS_EoS_core_rho = NS_Eos_core[1].values
NS_EoS_core_P = NS_Eos_core[2].values
NS_EoS_core_Gamma = NS_Eos_core[3].values

# Neutronitähden sisemmän kuoren tilanyhtälön
# ratkaistuja parametreja.
NS_Eos_ic = pd.read_csv(
    'NT_EOS_inner_crust.txt', sep=";", header=None)

NS_EoS_ic_n_b = NS_Eos_ic[0].values
NS_EoS_ic_rho = NS_Eos_ic[1].values
NS_EoS_ic_P = NS_Eos_ic[2].values
NS_EoS_ic_Gamma = NS_Eos_ic[3].values

# Neutronitähden ulomman kuoren ratkaistu tilanyhtälö
# paperista otetuilla alkuarvoilla

NS_EoS_oc_r, NS_EoS_oc_m, NS_EoS_oc_P, NS_EoS_oc_RHO = SOLVE_TOV(
    3, 
    R_body=R_NS0,
    kappa_choise=0, 
    rho_K=2.5955e-13+0j,
    p_K=5.13527e-16, a=1, 
    p_c=5.13527e-16,
    kappale="Valkoisen kääpiön (NS:n ulomman kuoren) \n")

# Yhdistetään sisemmän kuoren ja ytimen
# energiatiheys ja paine. Käännetään taulukot myös
# alkamaan ytimestä. Muutetaan paine ja energiatiheyden 
# yksiköt: [p] = [rho] = m**-2

# Energiatiheys
NS_EoS_ic_core_rho = unit_conversion(2, "RHO", np.flip(np.append(
        NS_EoS_ic_rho, NS_EoS_core_rho), -1), 1)

# TODO: MUUTA DATA OIKEAKSI - MUUTETAAN DUPLIKAATIN DATA TEKSTITIEDOSTOSSA
# MUUTETTU INDEKSI ON NT_EOS_CORE.txt ENSIMMÄINEN PAINE
NS_EoS_ic_core_P = unit_conversion(2, "P", np.flip(np.append(
        NS_EoS_ic_P, NS_EoS_core_P), -1), 1)

# Plotataan paine ja energiatiheys kuvaaja (rho, P) tutkimuspaperista.
graph(NS_EoS_ic_core_P, NS_EoS_ic_core_rho, plt.scatter,
      "NS EoS, (P, rho) - ic-core",
      "Paine, P (m^-2)", "Energiatiheys, rho (m^-2)", 'log',
      "NS:n energiatiheys paineen ftiona ic-core")

# Yhdistetään paperin data ja ratkaistu polytrooppi NS:n
# ulommaksi kuoreksi. Tulostetaan sitten.
NS_EoS_P = np.flip(np.unique(np.delete(
    np.append(NS_EoS_ic_core_P.real, NS_EoS_oc_P.real), -1)), -1)
NS_EoS_RHO = np.flip(np.unique(np.delete(
    np.append(NS_EoS_ic_core_rho.real, NS_EoS_oc_RHO.real), -1)), -1)

graph(NS_EoS_P, NS_EoS_RHO, plt.scatter, "NS EoS, (P, rho)",
      "Paine, P (m^-2)", "Energiatiheys, rho (m^-2)", 'log',
      "NS:n Energiatiheys paineen ftiona")

# Määritetään interpoloitu funktio NS:n (p, rho)-datalle.
NS_EoS_interpolate = interp1d(NS_EoS_P, NS_EoS_RHO,kind='cubic',
                              bounds_error=False,
                              fill_value=(2.89938e-19, 3.00742e-09))

# Määritetään x-akselin paineen arvoille uusi tiheys
NS_EoS_P_new = np.logspace(np.log10(NS_EoS_P[0]), np.log10(NS_EoS_P[-1]), 1000)

# Piirretään interpoloidut datapisteet.
graph(NS_EoS_P_new, NS_EoS_interpolate(NS_EoS_P_new), plt.plot,
      "NS EoS, (rho, P) interpolate", "Paine, P (m^-2)", 
      "Energiatiheys, rho (m^-2)", 'log', "NS:n interpoloitu energiatiheys paineen ftiona")

# Tulostetaan NS:n paine ja energiatiheys konsoliin.
# print("Neutronitähden energiatiheys ja paine ytimestä: \n")
# print(NS_EoS_P_new, NS_EoS_interpolate(NS_EoS_P_new))

NS_r, NS_m, NS_p, NS_rho = SOLVE_TOV(
    3, R_body=R_NS0, kappa_choise=0,
    rho_K=NS_EoS_interpolate(NS_EoS_P_new[2])+0j,
    p_K=NS_EoS_P_new[2],
    rho_c=NS_EoS_interpolate(NS_EoS_P_new[2])+0j,
    p_c=NS_EoS_P_new[2],
    a=2,
    interpolation=NS_EoS_interpolate,
    rho_func=1,
    kappale="Neutronitähden")

Ricci_scalar(NS_p, NS_rho, NS_r)



