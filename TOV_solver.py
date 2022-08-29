# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 09:42:03 2022

@author: Antero
"""
from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
"""

Yleisiä funktioita hyötykäyttöön. // General functions for utility use.

"""


def graph(x, y, style, label, xlabel, ylabel, scale, title):
    """
    Generates graph with given parameters.

    Parameters
    ----------
    x : Array
        Values for x-axis.
    y : Array
        Values for y-axis.
    style :
        Choose plt.scatter or plt.plot.
    label : String
        Label for legend.
    xlabel : String
        Label for x-axis.
    ylabel : String
        Label for y-axis.
    scale : String
        Scaling for x- and y-axis. Example 'log' or 'linear'.    
    title : String
        Title for graph.

    Returns
    -------
    None.

    """
    plt.figure()
    style(x, y, label=label)
    
    # Piirtää suoran y = 0. // Draws horizontal line y = 0.
    plt.axhline(y=0, color='r', linestyle='--')
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xscale(scale)
    plt.yscale(scale)
    plt.title(title)
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
    SYS : Int
        Choose unit system.
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
    Polytrope power that
    emerges in astrophysical polytrope equation.

    Parameters
    ----------
    n : Float
        polytrope index. (Goes from 0 to inf,
                          depends on the model)

    Returns
    -------
    Float
        Polytrope power

    """
    return (n+1)/n


def kappa_from_r0rho0n(r0, rho0, n):
    """
    Calculates constant of proportionality that
    emergences in polytrope equation.

    Parameters
    ----------
    r0 : Int
        Approximate radius of astrophysical body.
    rho0 : Float
        Center energy density.
    n : Float
        Polytrope index.

    Returns
    -------
    Float
        Constant of proportionality.

    """
    return (r0**2*4*np.pi*rho0**(1+1/n))/(n+1)


def kappa_from_p0rho0(p0, rho0, G):
    """
    Calculates constant of proportionality that
    emergences in polytrope equation.

    Parameters
    ----------
    p0 : Float
        Pressure at some radius r.
    rho0 : Float
        Energy density at r.
    G : Float
        Polytrope power.

    Returns
    -------
    Float
        Constant of proportionality.

    """
    return (p0)/(rho0**G)


def find_radius(p_t, r_t, raja=0):
    """
    Finds the radius corresponding to the boundary of the star.
    The limit can be defined in the function call or
    in the function definition.

    Parameters
    ----------
    p_t : Array
        Pressure values.
    r_t : Array
        Radius values.
    raja : Float 
        Defined boundary of a star. The default is 0.

    Returns
    -------
    R_raja : Float
        A radius corresponding to the star's boundary.

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
    Finds the mass within the radius of the star.

    Parameters
    ----------
    m_t : Array
        Mass values.
    r_t : Array
        Radius values.
    r_raja : Float
        Boundary of the star.

    Returns
    -------
    m : Float
        Mass within the radius of the star.

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
        Polytrope constant power.
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


def Ricci_scalar(p, rho, r):
    """
    Computes the curvature scalar of space - the Ricci scalar
    and plots it.

    Parameters
    ----------
    p : Array
        Pressure solution.
    rho : Array
        Energy density solution.

    Returns
    -------
    None.

    """
    R_scalar = -8*np.pi*(rho - 3*p)
    graph(r, R_scalar, plt.plot,
          "Avaruuden kaarevuus", "Säde, r", "Riccin skalaari, R", 
          'linear', "Avaruuden kaarevuusskalaari")


# Määritetään Tolman-Oppenheimer-Volkoff - yhtälöt (m, p, EoS), 
# jotka ratkaisemalla saadaan kuvattua tähden rakenne.
# //
# Determine the Tolman-Oppenheimer-Volkoff equations (m, p, EoS),
# by solving which the structure of the star can be described.


def EoS_cd_p2rho(interpolation, p_point):
    """
    Equation of state from custom interpolated (P, RHO)-data.

    Parameters
    ----------
    interpolation : Interpolate function.
        Returns rho when given p.
    p_point : Float, Array
        Pressure of a astrophysical body.

    Returns
    -------
    Float, Array
        Returns RHO as a function of P.

    """
    return interpolation(p_point)


def EoS_r2p(rho, Gamma, Kappa):
    """
    Equation of state, EoS.

    Given the energy density and constants returns the pressure.

    Parameters
    ----------
    rho : Float
        Energy density.
    Gamma : Float
        Polytrope constant.
    Kappa : Float
        Constant of proportionality.

    Returns
    -------
    p : Float
        Pressure.

    """
    p = Kappa*rho**Gamma
    return p


def EoS_p2r(p, Gamma, Kappa):
    """
    Equation of state, EoS.

    Given pressure and constants returns the energy density.

    Parameters
    ----------
    p : Float
        Pressure.
    Gamma : Float
        Polytrope constant.
    Kappa : Float
        Constant of proportionality.

    Returns
    -------
    rho : Float
        Energy density.

    """
    rho = (p/Kappa)**(1/Gamma)
    return rho


def EoS_choiser(choise, interpolation, p, Gamma, Kappa):
    """
    Chooses wanted EoS for DE-group as rho and returns it
    with given parameters.

    Parameters
    ----------
    choise : Int
        Chooses between defined EoS to return.
        Can be 0 or 1.
    interpolation : args
    p : args
    Gamma : argsz
    Kappa : args

    Returns
    -------
    rho : Float, Array
        Energy density with calculated with wanted EoS and given params.

    """
    if choise == 0:
        rho = EoS_p2r(p, Gamma, Kappa)
    elif choise == 1:
        rho = EoS_cd_p2rho(interpolation, p)

    return rho


def TOV(r, y, K, G, interpolation, rho_func):
    """
    Let's define the TOV equations and return them in an array.

    Parameters
    ----------
    y : Array
        Initial values.
    r : Array
        Integration params.

    Returns
    -------
    dy : Array
        Integroitavat funktiot.

    """
    # Asetetaan muuttujat taulukkoon
    # Energiatiheys valitaan valitsin-funktiossa.
    # //
    # Let's set the variables in the table. 
    # The energy density is selected in the selector function.
    m = y[0]                            
    p = y[1]
    rho = EoS_choiser(rho_func, interpolation, p, G, K)

    # Ratkaistavat yhtälöt // The equations to be solved
    dy = np.empty_like(y)
    # Massa säteen sisällä // Mass inside some radius
    dy[0] = 4*np.pi*rho*r**2
    # TODO tee valitsin paineelle
    # Paine - REL // Pressure - REL
    dy[1] = -(rho+p)*(m + 4*np.pi*r**3*p)/(r*(r-2*m))
    # Paine - EI-REL // Pressure - NON-REL
    # dy[1] = -(m*rho)/(r**2)
    return dy


def found_radius(t, y, d1, d2, d3, d4):
    """
    Event function: Zero of pressure
    ODE integration stops when this function returns True

    Parameters
    ----------
    t : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.
    d1, d2, d3, d4 : args
        Dummy params.

    Returns
    -------
    None
        Checks when pressure reaches zero.

    """
    d1, d2, d3, d4 = d1, d2, d3, d4
    return y[1].real


def main(model, args=[]):
    
    model_choise = ["EP", "NS", "WD_NREL", 
                    "WD_REL", "MSS_RADZONE", "SS", "GC"]
    
    model_params = [[3, 10000, 0, 2.5955e-13+0j, 5.13527e-16, 1, 5.13527e-16,
    "Valkoisen kääpiön (NS:n ulomman kuoren) \n"], 
                    [], 
                    []]
    
    if model == "CUSTOM":
        n               =args[0]
        R_body          =args[1]
        kappa_choise    =args[2]
        rho_K           =args[3]
        p_K             =args[4]
        rho_c           =args[5]
        p_c             =args[6]
        a               =args[7]
        rho_func        =args[8]
        interpolation   =args[9]
        body            =args[10]
    else:
        for i, m in enumerate(model_choise):
            if m == model_choise[i]:
                n               =model_params[i][0]
                R_body          =model_params[i][1] 
                kappa_choise    =model_params[i][2]
                rho_K           =model_params[i][3]
                p_K             =model_params[i][4]
                rho_c           =model_params[i][5]
                p_c             =model_params[i][6]
                a               =model_params[i][7]
                rho_func        =model_params[i][8]
                interpolation   =model_params[i][9]
                body            =model_params[i][10]
                
    print("Model of your choise and semi-realistic params for it: \n")
    print("Model = "        + model)
    print("n = "            + str(n))
    print("R_body = "       + str(R_body))
    print("kappa_choise = " + str(kappa_choise))
    print("rho_K = "        + str(rho_K))
    print("p_K = "          + str(p_K))
    print("rho_c = "        + str(rho_c))
    print("p_c = "          + str(p_c))
    print("a = "            + str(a))
    print("rho_func = "     + str(rho_func))
    print("interpolate = "  + str(interpolation))
    print("body = "         + body)
    
    # Määrätään vakioita.
    # //
    # Determine constants.
    M_sun = 2e30              # kg
    R_WD0 = 6e6               # m
    
    # Geometrisoidut yksiköt (Luonnollisia yksiköitä) 
    # // 
    # Geometrizied units (Natural units)
    c = 1
    G = 1
    
    # Asetetaan integrointiparametrit.
    # Integraattori adaptiivinen, lopettaa integroinnin tähden rajalla.
    # //
    # Let's set the integration parameters.
    # Integrator adaptive, stops the integration at the star boundary.
    rmin, rmax = 0.1, np.inf
    N = 500
    rspan = np.linspace(rmin, rmax, N)
    
    # Initiaalirajat säteelle. // Initial limits for the radius.
    r0, rf = rmin, rmax
    
    
    # Ode-ratkaisijan lopettaminen ehdon täyttyessä.
    # //
    # Termination of ode solver when met with condition.
    found_radius.terminal = True
    found_radius.direction = -1
    
    # Määritellään funktio TOV-yhtälöiden ratkaisemiseksi ja koodin ajon
    # helpottamiseksi. Funktiolle annetaan kasa parametreja ja se ratkaisee
    # aijemmin määritellyt yhtälöt.
    # //
    # Let's define a function to solve the TOV equations and to help run the code
    # easier. The function is given a bunch of parameters and it solves
    # previously defined equations.
    def SOLVE_TOV(n, R_body=0, kappa_choise=0, rho_K=0, p_K=0,
                  rho_c=0, p_c=0, a=0, rho_func=0, interpolation=0, body=""):
        """
        Appropriate initial values ​​and equations are chosen. Solves TOV equations
        in this case for the corresponding astrophysical body. As a solution
        the mass, pressure, energy density and radius of an astrophysical body 
        are obtained.
        
        Solver works in geometrizied units so be sure to input values
        in those units! Function returns solutions in geometrizied units also.
    
        Parameters
        ----------
        n : Float
            Polytrope index.
        R_body : Float, optional
            Approximate the radius of the astrophysical body
            to be modeled . The default is 0..
        kappa_choise : Int, optional
            Choise for which way Kappa is computed:
                0=kappa_from_p0rho0 (needs corresponding p and rho)
                1=kappa_from_r0rho0n (needs approximate radius and CENTRAL rho)
        rho_K : Float, optional
            Energy density for which we want calculate 
            the corresponding constant of proportionality. The default is 0..
        p_K : Float, optional
            Pressure for which we want calculate 
            the corresponding constant of proportionality.
            ONLY NEEDED WHEN kappa_choise=0. Has to be corresponding 
            to rho_K. The default is 0..
        rho_c : Float, optional
            Central energy density. Used to compute initial values.
            The default is 0..
        p_c : Float, optional
            Central pressure. Used to compute initial values. The default is 0..
        a : Int, optional
            Choice for given initial value. Another is then 0 and
            calculated from EoS. Can take both also.
            Choice:
                a = 0 is for given rho0.
                a = 1 is for given p0.
                a = 2 is for given rho0 and p0.
            The default is 0..
        rho_func : Int, optional
            Choise for what EoS is used to compute energy density.
            Choise:
                0=Polytrope EoS.
                1=Interpolated EoS from data.
            The default is 0..
        interpolation : interpolate, optional
            Has to be given if choise rho_func=1. Otherwise can be ignored.
            The default is 0..
        body : String
            Changes title for graphs. Input depending what is modeled.
    
        Returns
        -------
        r : Array
            Radius solution for modeled body.
        m : Array
            Mass solution for modeled body.
        p : Array
            Pressure solution for modeled body.
        rho : Array
            Energy density solution for modeled body.
    
        """
        # Asetetaan alkuarvot // Set initial values
        Gamma = gamma_from_n(n)
        if kappa_choise == 0:
            Kappa = kappa_from_p0rho0(p_K, rho_K, Gamma)
        elif kappa_choise == 1:
            Kappa = kappa_from_r0rho0n(R_body, rho_K, n)
        
        m, p, rho = set_initial_conditions(r0, Gamma, Kappa, rho_c, p_c, a)
        y0 = m, p
        
        print("Tulostetaan alkuarvot. \n Kappa ja Gamma:" + str(Kappa) +
              " ja " + str(Gamma) + "\n Asetetut alkuarvot (m, p ja rho):"
              + str(y0) + "\n \n")
        
        # Ratkaistaan TOV annetuilla parametreilla 
        # // 
        # Let's solve the TOV with the given parameters
        soln = solve_ivp(TOV, (r0, rf), y0, method='BDF',
                         dense_output=True, events=found_radius,
                         args=(Kappa, Gamma, interpolation, rho_func))
    
        print("Solverin parametreja:")
        print(soln.nfev, 'evaluations required')
        print(soln.t_events)
        print(soln.y_events)
        print("\n \n")
    
        # TOV ratkaisut // TOV solutions
        # Ratkaisut yksiköissä // Solutions in units:
        # [m] = kg, [p] = m**-2 ja [rho] = m**-2
        r = soln.t
        m = soln.y[0].real
        p = soln.y[1].real
        rho = EoS_p2r(p, Gamma, Kappa)
    
        print("Saadut TOV ratkaisut: \n")
        print("Säde: \n" + str(r.real) + "\n Massa: \n" + str(m.real) +
              "\n Paine: \n" + str(p.real) + "\n Energiatiheys: \n" + str(rho.real))
        print("\n \n")
    
        # Piirretään ratkaisun malli kuvaajiin yksiköissä:
        # //
        # Let's plot the model of the solution on graphs in units:
        # [m] = kg, [p] = erg/cm**m ja [rho] = g/cm**3 
        graph(r, unit_conversion(0, "M", m, 1),
              plt.plot, "massa", "säde, r (m)", "massa, m (kg)", 'linear',
              body + " " + "massa säteen funktiona")
        graph(r, unit_conversion(2, "P", p, -1),
              plt.plot, "paine", "säde, r (m)", "p (erg/cm^3)", 'linear', 
              body + " " + "paine säteen funktiona")
        graph(r, unit_conversion(2, "RHO", rho, -1), plt.plot,
              fr'$\rho_c$ = {rho_c.real}' '\n'
              fr'$K$ = {Kappa.real}' '\n' 
              fr'$\Gamma$ = {Gamma}',
              "säde, r", "energiatiheys, rho (g/cm^3)", 'linear',
              body + " " + "energiatiheys säteen funktiona")
        return r.real, m.real, p.real, rho.real
    
    
    r_sol, m_sol, p_sol, rho_sol = SOLVE_TOV(n, R_body, kappa_choise, rho_K, p_K, rho_c, p_c, 
                             a, rho_func, interpolation, body)
    
    # Ratkaistaan TOV valkoisen kääpiön alkuarvoille:
    # SOLVE_TOV(3, R_body=6e6, rho_K=1e-10+0j, rho_c=1e-10+0j, a=0, rho_func=0)
    
    
    

    
    # Ratkaistaan massa-säde relaatio. Etsitään TOV-yhtälöiden ratkaisuja
    # jollakin rhospan-alueella. Ratkaistaan yhtälöitä siis tähden keskipisteen eri
    # energiatiheyksien arvoilla.
    
    # Etsitään tähden raja (find_radius) paineen ratkaisusta ja sitä vastaava
    # massa massan kuvaajasta. Tallennetaan nämä arvot taulukkoon ja piirretään
    # kuvaaja.
    
    # Mallinnetaan nyt useaa tähteä ja piirretään
    # Massa-Säde - relaatio.
    # //
    # Let's solve the mass-radius relation. We are looking for solutions to the
    # TOV equations in some rhospan area. So let's solve the equations
    # from the center of the star with varying values ​​of energy densities.
    
    # Let's find the limit of the star (find_radius) from the pressure solution 
    # and its equivalent mass from the mass solution. Let's save these values ​​in
    # an array and plot them.
    
    # Now let's model several stars and plot them
    # Mass-Radius - relation.
    
    # TODO korjaa
    def MR_relaatio(rho_min, rho_max):
        """
        Solves mass-radius - relation.
    
        Parameters
        ----------
        rho_min : Float
            Lower limit of central energy densities.
        rho_max : Float
            Higher limit of central energy densities.
    
        Returns
        -------
        R : Array
            Radiuses of star sequense.
        M : Array
            Masses of star sequense.
    
        """
    
        # Build 200 star models
        rhospan = np.linspace(rho_min, rho_max, 100)
        R = []
        M = []
        # Ratkaise TOV jokaiselle rho0:lle rhospan alueessa.
        # //
        # Solve the TOV for each rho0 in the range of rhospan.
        for rho0 in rhospan:
            r, m, p, rho = SOLVE_TOV(R_WD0, 3, rho_cK=1e-11+0j, rho_c=rho0)
            r_boundary = find_radius(p, r, raja=0)
            m_boundary = find_mass_in_radius(m, r, r_boundary)
            R.append(r_boundary)
            M.append(m_boundary)
        # Printtaa ja plottaa massa-säde - relaation. 
        # //
        # Print and plot the mass-radius relation.
        print("Tulostetaan ratkaistut massat ja niitä vastaavat säteet: \n")
        print("Säteet: \n " + str(R) + "\n Massat: \n" + str(M))
        graph(R, M, plt.scatter, "Massa-säde - relaatio", "Säde",
              "Massa", 'linear', "Massa-säde")
        graph(R, M, plt.plot, "Massa-säde - relaatio", "Säde",
              "Massa", 'linear', "Massa-säde")
        return R, M
    
    return r_sol, m_sol, p_sol, rho_sol
    
    # MR_relaatio(1e-16+0j, 1e-8+0j)

    
    
    
    
# Rakennetaan neutronitähden malli paperista "A unified equation
# of state of dense matter and neutron star structure" saadulla datalla
# sisemmän kuoren tilanyhtälöstä ja ytimen tilanyhtälöstä.
# Tilanyhtälöt:
#     Ulompi kuori  -> Polytrooppi tilanyhtälö Gamma = 4/3
#     Sisempi kuori -> Data paperin taulukosta 3.
#     Ydin          -> Data paperin taulukosta 5.
# //
# Let's build a neutron star model from the paper "A unified equation
# of state of dense matter and Neutron star structure" with the obtained data
# from the equation of state of the inner shell and the equation of state of 
# the core
# Equation of states:
#     Outer crust     -> Polytrope with Gamma = 4/3
#     Inner crust     -> Data from paper array 3.
#     Core            -> Data from paper array 5.

# Tilanyhtälöiden muuttujat datasta // Variables of state equations from data:
#     n_b, rho, P, Gamma.
def NS_MODEL():
    """
    A semi-accurate model of a neutron star.

    Returns
    -------
    NS_r : Array
        Solution for radius.
    NS_m : Array
        Solution for mass.
    NS_p : Array
        Solution for pressure.
    NS_rho : Array
        Solution for energy density.

    """
    # Vakioita // Constants
    R_NS0 = 10000             # m
    
    # Neutronitähden ytimen tilanyhtälön ratkaistuja parametreja.
    # //
    # Solved parameters of the neutron star core equation of state
    NS_Eos_core = pd.read_csv(
        'NT_EOS_core.txt', sep=";", header=None)
    
    NS_EoS_core_n_b = NS_Eos_core[0].values
    NS_EoS_core_rho = NS_Eos_core[1].values
    NS_EoS_core_P = NS_Eos_core[2].values
    NS_EoS_core_Gamma = NS_Eos_core[3].values
    
    # Neutronitähden sisemmän kuoren tilanyhtälön ratkaistuja parametreja.
    # //
    # The solved parameters of the neutron star's inner crust equation of state.
    NS_Eos_ic = pd.read_csv(
        'NT_EOS_inner_crust.txt', sep=";", header=None)
    
    NS_EoS_ic_n_b = NS_Eos_ic[0].values
    NS_EoS_ic_rho = NS_Eos_ic[1].values
    NS_EoS_ic_P = NS_Eos_ic[2].values
    NS_EoS_ic_Gamma = NS_Eos_ic[3].values
    
    # Neutronitähden ulomman kuoren ratkaistu tilanyhtälö
    # paperista otetuilla alkuarvoilla
    # //
    # The solved equation of state for the outer crust of a neutron star
    # with initial values ​​taken from the paper
    NS_EoS_oc_r, NS_EoS_oc_m, NS_EoS_oc_P, NS_EoS_oc_RHO = main("CUSTOM", (
        3, 
        R_NS0,
        0, 
        2.5955e-13+0j,
        5.13527e-16,
        0,
        5.13527e-16,
        1, 
        0,
        0,
        "Valkoisen kääpiön (NS:n ulomman kuoren) \n"))
    
    # Yhdistetään sisemmän kuoren ja ytimen energiatiheys ja paine. 
    # Käännetään taulukot myös alkamaan ytimestä. 
    # Muutetaan paine ja energiatiheyden yksiköt:
    # //
    # Connect the inner shell and core energy density and pressure. 
    # Let's turn the tables too to start from the core. 
    # Let's change the pressure and energy density units:
        # [p] = [rho] = m**-2
    
    # Energiatiheys // Energy density
    NS_EoS_ic_core_rho = unit_conversion(2, "RHO", np.flip(np.append(
            NS_EoS_ic_rho, NS_EoS_core_rho), -1), 1)
    # Paine // Pressure
    NS_EoS_ic_core_P = unit_conversion(2, "P", np.flip(np.append(
            NS_EoS_ic_P, NS_EoS_core_P), -1), 1)
    
    # Plotataan paine ja energiatiheys kuvaaja (rho, P) tutkimuspaperista.
    # //
    # Let's plot the pressure and energy density (rho, P) from the research paper
    graph(NS_EoS_ic_core_P, NS_EoS_ic_core_rho, plt.scatter,
          "NS EoS, (P, rho) - ic-core",
          "Paine, P (m^-2)", "Energiatiheys, rho (m^-2)", 'log',
          "NS:n energiatiheys paineen ftiona ic-core")
    
    # Yhdistetään paperin data ja ratkaistu ulomman kuoren malli.
    # //
    # The paper data and the solved model of the outer shell are combined.
    NS_EoS_P = np.flip(np.unique(np.delete(
        np.append(NS_EoS_ic_core_P.real, NS_EoS_oc_P.real), -1)), -1)
    NS_EoS_RHO = np.flip(np.unique(np.delete(
        np.append(NS_EoS_ic_core_rho.real, NS_EoS_oc_RHO.real), -1)), -1)
    
    # Plot
    graph(NS_EoS_P, NS_EoS_RHO, plt.scatter, "NS EoS, (P, rho)",
          "Paine, P (m^-2)", "Energiatiheys, rho (m^-2)", 'log',
          "NS:n Energiatiheys paineen ftiona")
    
    # Määritetään interpoloitu funktio NS:n (p, rho)-datalle.
    # //
    # Let's define an interpolated function for NS (p, rho) data.
    NS_EoS_interpolate = interp1d(NS_EoS_P, NS_EoS_RHO,kind='cubic',
                                  bounds_error=False,
                                  fill_value=(NS_EoS_RHO[-1], NS_EoS_RHO[0]))
    
    # Määritetään x-akselin paineen arvoille uusi tiheys.
    # //
    # Let's define a new density for the x-axis pressure values
    NS_EoS_P_new = np.logspace(np.log10(NS_EoS_P[0]), np.log10(NS_EoS_P[-1]), 1000)
    
    # Piirretään interpoloidut datapisteet.
    # //
    # Let's plot the interpolated data points
    graph(NS_EoS_P_new, NS_EoS_interpolate(NS_EoS_P_new), plt.plot,
          "NS EoS, (rho, P) interpolate", "Paine, P (m^-2)", 
          "Energiatiheys, rho (m^-2)", 'log', "NS:n interpoloitu energiatiheys paineen ftiona")
    
    # Ratkaistaan nyt TOV uudestaan NS:n datalle ja mallinnetaan koko
    # tähden rakenne säteen funktiona.
    # //
    # Let's now solve the TOV again for NS data and model the whole
    # star structure as a function of radius.
    NS_r, NS_m, NS_p, NS_rho = main("CUSTOM", (
        3, 
        R_NS0, 
        0,
        NS_EoS_interpolate(NS_EoS_P_new[2])+0j,
        NS_EoS_P_new[2],
        NS_EoS_interpolate(NS_EoS_P_new[2])+0j,
        NS_EoS_P_new[2],
        2,
        NS_EoS_interpolate,
        1,
        "Neutronitähden"))
    # Vielä avaruuden kaarevuus luonnollisissa yksiköissä
    # neutronitähden sisällä.
    # //
    # Also the curvature of space in natural units
    # inside a neutron star.
    Ricci_scalar(unit_conversion(
        1, "P", unit_conversion(0, "P", NS_p, 1), -1), 
        unit_conversion(
            1, "RHO", unit_conversion(0, "RHO", NS_rho, 1), -1), NS_r)        
            
    return NS_r, NS_m, NS_p, NS_rho
    

