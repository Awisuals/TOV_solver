# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5  2022

@author: Antero
"""
from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt

from functions import *
from structure_equations import *
"""
Write DE-group as:
    EoS -> pressure
    TOV -> Energy density
And solve it.
"""
step = 0

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
    # m = 0
    # print("m, p, rho: " + str(m) + str(p) + str(rho))
    return m, p, rho


def TOV_rho(r, y, K, G, interpolation, eos_choise, tov_choise, rho_center):
    
    # HUOM! Tänne alkuarvaukset luonnollisissa yksiköissä

    # Asetetaan muuttujat taulukkoon
    # Paine valitaan valitsin-funktiossa.
    # //
    # Let's set the variables in the table. 
    # The energy density is selected in the selector function.
    m = y[0].real + 0j
    rho = y[1].real + 0j
    p = EoS_choiser(eos_choise, interpolation, G, K, 0, rho).real + 0j
    
    global step    
    
    # Ratkaistavat yhtälöt // Equations to be solved
    dy = np.empty_like(y)
    # Massa ja Energiatiheys DY // Mass and energy density DE
    dy[0] = Mass_in_radius(rho, r)                  # dmdr
    dy[1] = TOV_choiser(tov_choise, m ,p, rho, r)  # drhodr
    
    # print("\n \n DEBUG printing \n" + 
    #       "\n Step: " + str(step) +
    #       "\n Radius: " + str(r) +
    #       "\n Mass: " + str(m) + 
    #       "\n Energy density: " + str(rho) + 
    #       "\n Pressure: " + str(p) + 
    #       "\n Mass derivate: " + str(dy[0]) + 
    #       "\n Energy density derivate: " + str(dy[1]))

    step += 1    
    return dy


def found_radius(t, y, d1, d2, d3, d4, d5, d6):
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
def TOV_solver(ir=[], n=0, R_body=0, kappa_choise=0, rho_K=0, p_K=0, rho_c=0, p_c=0, a=0, eos_choise=0, tov_choise=0, interpolation=0, body=""):
    """
    Appropriate initial values and equations are chosen. Solves TOV equations
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
    EoS_choise : Int, optional
        Choise for what EoS is used to compute energy density.
        Choise:
            0=Polytrope EoS.
            1=Interpolated EoS from data.
        The default is 0..
    TOV_choise : int, optional
        Choise for either newtonian or relativistic pressure.:
            0=TOV
            1=NEWT
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
    
    # ir = np.linspace(rmin, rmax, 500)
    
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
    soln = solve_ivp(TOV_rho, (rmin, rmax), (m0.real, rho0.real), method='BDF',
    dense_output=False, events=found_radius, max_step = 1e18, # for whole solution max_step=1e20 and closer to core smaller
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
    # [m] = kg, [p] = m**-2 ja [rho] = m**-2
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
    graph(r, m,
          plt.plot, fr'Massa, ' '\n' fr'$\rho_{"c"}$ = {rho0_si}' r' $\frac{\mathrm{J}}{\mathrm{m}^{3}}$', 
          r'Säde, r ($R_{Earth}$)', r'Massa, m ($M_{Sun}$)', 'linear', 'linear',
          body + " " + "mass as a function of radius \n", 1)
    graph(r, p,
          plt.plot, fr'Paine, ''\n' fr'$\rho_{"c"}$ = {rho0_si}' r' $\frac{\mathrm{J}}{\mathrm{m}^{3}}$', 
          r'Säde, r ($R_{Earth}$)', r'Paine, p (Pa)', 'linear', 'linear',
          body + " " + "pressure as a function of radius \n", 1)
    graph(r, rho, plt.plot,
          fr'Energiatiheys,' '\n' fr'$\rho_{"c"}$ = {rho0_si}' r' $\frac{\mathrm{J}}{\mathrm{m}^{3}}$',
          r'Säde, r ($R_{Earth}$)', r'Energiatiheys, $\rho$ $(\frac{\mathrm{J}}{\mathrm{m}^{3}})$', 'linear', 'linear', 
          body + " " + "energy density as a function of radius \n", 1)
    
    print("Tähden säde: \n" + str(r[-1]) + 
          "\n Tähden massa: \n" + str(m[-1]) + 
          "\n \n")
    
    return r.real, m.real, p.real, rho.real


def models(model, args=[]):
    
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


def Relativistic_Terms():
    rho_center =  2e-7+0j        # 2e-14+0j       # 2e-7+0j
    skaala = 0.13169421547674232  # 4.284971901021712         # 0.13169421547674232
    kerroin = 0.9 # kuinka lähellä ollaan tähden keskipistettä
    rho_center_si = '{:0.2e}'.format(rho_center.real * 2.0852e37)
    R_koko = '{:0.2e}'.format(skaala)
    
    
    r_tov, m_tov, p_tov, rho_tov = TOV_solver(ir=[5.067e12, kerroin*(skaala/(1.9733e-16 * 1e-3 / 6371))], 
            n=0, 
            R_body=0, 
            kappa_choise=0, 
            rho_K=0, 
            p_K=0, 
            rho_c=rho_center, 
            p_c=0, 
            a=3, 
            eos_choise=2, 
            tov_choise=2, 
            interpolation=0, 
            body="TOV White dwarf")
    
    r_newt, m_newt, p_newt, rho_newt = TOV_solver(ir=[5.067e12, kerroin*(skaala/(1.9733e-16 * 1e-3 / 6371))], 
            n=0, 
            R_body=0, 
            kappa_choise=0, 
            rho_K=0, 
            p_K=0, 
            rho_c=rho_center, 
            p_c=0, 
            a=3, 
            eos_choise=2, 
            tov_choise=3, 
            interpolation=0, 
            body="NEWT White dwarf")
    
    Delta_Rho = rho_newt[:-1] / rho_tov[:-1]
    
    graph(np.flip(r_newt[:-1])/skaala, Delta_Rho, plt.plot, 
          fr'Teorioiden välinen suhde keskipisteen energiatiheydellä' '\n' fr'$\rho_{"c"}$ = {rho_center_si}' r' $\frac{\mathrm{J}}{\mathrm{m}^{3}}$' '\n' r'$R_{WD}$' fr' = {R_koko}' r' $R_{Earth}$', 
          r'Säde, r ($R_{WD}$)', r'$\frac{\rho_{newt}}{\rho_{tov}}$', 'linear', 'log', "", 1, 1)
    
    return

# models("WD_REL")

Relativistic_Terms()
