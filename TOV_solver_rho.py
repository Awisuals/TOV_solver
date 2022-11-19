# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5  2022

@author: Antero
"""
from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from functions import *
from structure_equations import *
"""
Write DE-group as:
    EoS -> pressure
    TOV -> Energy density
And solve it.
"""

def set_initial_conditions(rmin, G, K, rho0=0., p0=0., a=0):
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
            a = 3 is for given rho0 and computes p0.

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
    rho_values0 = [rho0, EoS_choiser(0, p=p0, Gamma=G, Kappa=K), rho0, rho0]
    p_values0 = [EoS_choiser(1, rho=rho0, Gamma=G, Kappa=K), p0, p0, EoS_choiser(2, rho=rho0)]
    if a == 0:
        rho = rho_values0[a]
        p = p_values0[a]
    if a == 1:
        p = p_values0[a]
        rho = rho_values0[a]
    if a == 2:
        rho = rho0
        p = p0
    if a == 3:
        rho = rho_values0[a]
        p = p_values0[a]
    m = 4./3.*np.pi*rho0*rmin**3
    return m, p, rho


def TOV_rho(r, y, K, G, interpolation, eos_choise, tov_choise, rho_center):
    
    # HUOM! Tänne alkuarvaukset luonnollisissa yksiköissä GeV^x!
    # NOTE! Here are the initial guesses in natural units GeV^x!

    # Asetetaan muuttujat taulukkoon
    # Paine valitaan valitsin-funktiossa.
    # //
    # Let's set the variables in the table. 
    # The energy density is selected in the selector function.
    m = y[0].real + 0j
    rho = y[1].real + 0j
    p = EoS_choiser(eos_choise, interpolation, G, K, 0, rho).real + 0j
    
    # Ratkaistavat yhtälöt // Equations to be solved
    dy = np.empty_like(y)
    # Massa ja Energiatiheys DY // Mass and energy density DE
    dy[0] = Mass_in_radius(rho, r)                  # dmdr
    dy[1] = TOV_choiser(tov_choise, m ,p, rho, r)   # drhodr

    return dy


def found_radius(t, y, d1, d2, d3, d4, d5, d6):
    """
    Event function: Defined boundary of pressure
    ODE integration stops when this function returns True

    Parameters
    ----------
    t : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.
    d1, d2, d3, d4, d5, d6 : args
        Dummy params.

    Returns
    -------
    None
        Checks when energy density reaches zero.

    """
    d1, d2, d3, d4, d5, d6 = d1, d2, d3, d4, d5, d6
    pressure = EoS_degelgas(y[1].real)
    return pressure >= 0*d6# 4.3e21 # 1e16


# Määritellään funktio TOV-yhtälöiden ratkaisemiseksi ja koodin ajon
# helpottamiseksi. Funktiolle annetaan kasa parametreja ja se ratkaisee
# aijemmin määritellyt yhtälöt.
# //
# Let's define a function to solve the TOV equations and to help run the code
# easier. The function is given a bunch of parameters and it solves
# previously defined equations.
def TOV_solver(ir=[], n=0, R_body=0, kappa_choise=0, rho_K=0, p_K=0, rho_c=0, 
               p_c=0, a=0, eos_choise=0, tov_choise=0, interpolation=0, body=""):
    """
    Appropriate initial values and equations can be chosen. Solves TOV equations
    in this case for the corresponding astrophysical body. As a solution
    the mass, pressure, energy density and radius of an astrophysical body 
    are obtained.
    
    Solver works in natural units (but can be manually changed) so be sure 
    to input values in correct units! Function returns solutions in natural
    units also.

    Parameters
    ----------
    ir : Array
        Integration range, pre defined to [5.067e12, np.inf].
    n : Float
        Polytrope index.
    R_body : Float, optional
        Approximate the radius of the astrophysical body
        to be modeled. The default is 0..
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
        Can be:
            a=0, 1, 2, 3
        Choice for given initial value. Please refer to 
        set_initial_conditions-function documentation for
        additional information. The default is 0..
    EoS_choise : Int, optional
        Choise for what EoS is used to compute initial values.
        Choise:
            0=Polytrope EoS.
            1=Interpolated EoS from data.
            2=Degenerate electron gas
        Please refer to EoS_choiser-function for additional information.
        The default is 0..
    TOV_choise : int, optional
        Choise for tov-equation:
            0=TOV, integrate p
            1=NEWT, integrate p
            2=TOV, integrate rho
            3=NEWT, integrate rho
    interpolation : interpolate, optional
        Has to be given if choise EoS_choise=1. Otherwise can be ignored.
        The default is 0..
    body : String
        Changes title for graphs. Input depending what is modeled.
        ATM isn't necessary.

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
     # Määrätään vakioita.
    # //
    # Determine constants.
    M_sun = 1.9891e30              # kg
    R_earth = 6371                 # km
    
    # Asetetaan integrointiparametrit.
    # Integraattori adaptiivinen, lopettaa integroinnin tähden rajalla.
    # //
    # Let's set the integration parameters.
    # Integrator adaptive, stops the integration at the star boundary.
    if len(ir)!=0: rmin, rmax = ir[0], ir[1]
    else: rmin, rmax = 5.067e12, np.inf
    
    # Ode-ratkaisijan lopettaminen ehdon täyttyessä.
    # //
    # Termination of ode solver when met with condition.
    found_radius.terminal = True # Should be true when works
    found_radius.direction = -1    
    
    # Asetetaan alkuarvot // Set initial values
    if n != 0: Gamma = gamma_from_n(n); Kappa = kappa_choiser(kappa_choise, p_K, rho_K, Gamma, R_body, n) 
    else: Gamma, Kappa = 0.1, 0.1
    
    m0, p0, rho0 = set_initial_conditions(rmin, Gamma, Kappa, rho_c, p_c, a)
    y0 = m0, p0, rho0
    
    # Tulostetaan annetut parametrit // Print given params
    print("\n Model of your choise and semi-realistic params for it:" + 
    "\n Integration range = " + str(ir) +
    "\n Model = "        + body +
    "\n n = "            + str(n) +
    "\n R_body = "       + str(R_body) +
    "\n kappa_choise = " + str(kappa_choise) +
    "\n rho_K = "        + str(rho_K) +
    "\n p_K = "          + str(p_K) +
    "\n rho_c = "        + str(rho_c) +
    "\n p_c = "          + str(p_c) +
    "\n a = "            + str(a) +
    "\n eos_choise = "   + str(eos_choise) +
    "\n tov_choise = "   + str(tov_choise) +
    "\n interpolate = "  + str(interpolation) + "\n \n")
    
    print("Tulostetaan polytrooppivakiot:" 
          + "\n Kappa: " + str(Kappa)
          + "\n Gamma: " + str(Gamma) + "\n \n")
          
    print("Asetetut alkuarvot (m, p ja rho):"
          + "\n m: " + str(y0[0]) 
          + "\n p: " + str(y0[1]) 
          + "\n rho: " + str(y0[2]) + "\n \n")
    
    # Ratkaistaan TOV annetuilla parametreilla 
    # // 
    # Let's solve the TOV with the given parameters
    # NOTE Best performance/resolution: One solution max_step=1e20 
    #                                   Closer to core smaller 1e18
    #                                   MR-relation zoom 5e19.
    soln = solve_ivp(TOV_rho, (rmin, rmax), (m0.real, rho0.real), method='BDF',
    dense_output=False, events=found_radius, max_step = 1e18,
    args=(Kappa, Gamma, interpolation, eos_choise, tov_choise, rho0.real))
    
    print("\n Solverin parametreja:")
    print(soln.nfev, 'evaluations required')
    print(soln.t_events)
    print(soln.y_events)
    print("\n")

    # Energiatiheyden alkuarvaus // Energy denisty initial guess
    # SI-units
    rho0_si = '{:0.2e}'.format(rho0.real * 2.0852e37)
    # TOV ratkaisut // TOV solutions
    # Ratkaisut yksiköissä // Solutions in units:
    # [m] = GeV, [p] = GeV^4 ja [rho] = GeV^4
    r = soln.t * 1.9733e-16 * 1e-3 / R_earth # 2.6544006e-25*1e9
    m = soln.y[0].real * 1.7827e-27 / M_sun # / 1.9891e30 # 1.7827e-36
    rho = soln.y[1].real * 2.0852e37 # 3.16435553043e40
    p = EoS_degelgas(rho) * 2.0852e37 # 3.16435553043e40
    
    print("Saadut TOV ratkaisut ([m] = GeV, [p] = GeV^4 ja [rho] = GeV^4): \n")
    print("Säde: \n \n" + str(r.real) + 
    "\n \n Massa: \n \n" + str(m.real) + 
    "\n \n Energiatiheys: \n \n" + str(rho.real) + 
    "\n \n Paine : \n \n" + str(p.real) + "\n \n")
    
    # # # Piirretään ratkaisun malli kuvaajiin yksiköissä:
    # # # //
    # # # Let's plot the model of the solution on graphs in units:
    # # # [m] = kg, [p] = Pascal ja [rho] = J/m^3 
    gs = gridspec.GridSpec(2, 2)
    plt.figure()
    
    ax1 = plt.subplot(gs[0, :])
    ax2 = plt.subplot(gs[1, 0])
    ax3 = plt.subplot(gs[1, 1])
    ax1.plot(r, m, color='r', label=fr'Massa, ' '\n' fr'$\rho_{"c"}$ = {rho0_si}' r' $\frac{\mathrm{J}}{\mathrm{m}^{3}}$')
    ax2.plot(r, p, color='g', label=fr'Paine, ''\n' fr'$\rho_{"c"}$ = {rho0_si}' r' $\frac{\mathrm{J}}{\mathrm{m}^{3}}$')
    ax3.plot(r, rho, color='b', label=fr'Energiatiheys,' '\n' fr'$\rho_{"c"}$ = {rho0_si}' r' $\frac{\mathrm{J}}{\mathrm{m}^{3}}$')
    
    ax1.set(xlabel=r'Säde, r ($R_{Earth}$)', 
            ylabel= r'Massa, m ($M_{Sun}$)', 
            xscale="linear", yscale="linear")
    ax1.set_title('a)', loc="left")
    ax1.legend(shadow=True, fancybox=True)
    ax1.grid()
    ax2.set(xlabel=r'Säde, r ($R_{Earth}$)', 
            ylabel=r'Paine, p (Pa)', 
            xscale="linear", yscale="linear")
    ax2.set_title('b)', loc="right")
    ax2.legend(shadow=True, fancybox=True)
    ax2.grid()
    ax3.set(xlabel=r'Säde, r ($R_{Earth}$)', 
            ylabel=r'Energiatiheys, $\rho$ $(\frac{\mathrm{J}}{\mathrm{m}^{3}})$', 
            xscale="linear", yscale="linear")
    ax3.set_title('c)', loc="right")
    ax3.legend(shadow=True, fancybox=True)
    ax3.grid()
    
    plt.show()
    
    print("Tähden säde: \n" + str(r[-1]) + 
          "\n Tähden massa: \n" + str(m[-1]) + 
          "\n \n")
    
    return r.real, m.real, p.real, rho.real


def Models(model, args=[]):
    """
    Main function to drive TOV_solver. couple of example
    parameters given and then passes them to solver.

    Parameters
    ----------
    model : string
        Choose between given models in model_choise array.
    args : list, optional
        Insert custom params, by default []
    """    
    model_choise = ["WD_NREL", "WD_REL"]
    model_params = [[0, 0, 0, 0, 0, 2e-14+0j, 0, 3, 2, 3, 0, 
                     "Non-relativistic White Dwarf"], 
                    [0, 0, 0, 0, 0, 2e-11+0j, 0, 3, 2, 2, 0, 
                     "Relativistic White Dwarf"]]
    
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
        p_func          =args[9]
        interpolation   =args[10]
        body            =args[11]
    else:
        for i, m in enumerate(model_choise):
            if m == model:
                print(m)
                print(i)
                n               =model_params[i][0]
                R_body          =model_params[i][1] 
                kappa_choise    =model_params[i][2]
                rho_K           =model_params[i][3]
                p_K             =model_params[i][4]
                rho_c           =model_params[i][5]
                p_c             =model_params[i][6]
                a               =model_params[i][7]
                rho_func        =model_params[i][8]
                p_func          =model_params[i][9]
                interpolation   =model_params[i][10]
                body            =model_params[i][11]
    
    TOV_solver(ir=[], 
               n=n, 
               R_body=R_body, 
               kappa_choise=kappa_choise, 
               rho_K=rho_K, 
               p_K=p_K, 
               rho_c=rho_c, 
               p_c=p_c, 
               a=a, 
               eos_choise=rho_func, 
               tov_choise=p_func, 
               interpolation=interpolation, 
               body=body)

Models("WD_REL")
