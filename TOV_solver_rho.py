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
    m = 4./3.*np.pi*rho*rmin**3
    # m = 0
    # print("m, p, rho: " + str(m) + str(p) + str(rho))
    return m, p, rho


def TOV_rho(r, y, K, G, interpolation, eos_choise, tov_choise):
    
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
    
    print("\n \n DEBUG printing \n" + 
          "\n Step: " + str(step) +
          "\n Radius: " + str(r) +
          "\n Mass: " + str(m) + 
          "\n Energy density: " + str(rho) + 
          "\n Pressure: " + str(p) + 
          "\n Mass derivate: " + str(dy[0]) + 
          "\n Energy density derivate: " + str(dy[1]))

    step += 1    
    return dy

# Määritellään funktio TOV-yhtälöiden ratkaisemiseksi ja koodin ajon
# helpottamiseksi. Funktiolle annetaan kasa parametreja ja se ratkaisee
# aijemmin määritellyt yhtälöt.
# //
# Let's define a function to solve the TOV equations and to help run the code
# easier. The function is given a bunch of parameters and it solves
# previously defined equations.
def TOV_solver(ir=[], n=0, R_body=0, kappa_choise=0, rho_K=0, p_K=0,
              rho_c=0, p_c=0, a=0, eos_choise=0, tov_choise=0, interpolation=0, body=""):
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
    rs, rf = ir[0], ir[1]
    
    # Asetetaan alkuarvot // Set initial values
    
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
    "\n eos_choise = "     + str(eos_choise) +
    "\n tov_choise = "       + str(tov_choise) +
    "\n interpolate = "  + str(interpolation) + "\n \n")
    
    Gamma = gamma_from_n(n)
    Kappa = kappa_choiser(kappa_choise, p_K, rho_K, Gamma, R_body, n)
    
    m, p, rho = set_initial_conditions(rs, Gamma, Kappa, rho_c, p_c, a)
    y0 = m, p, rho
    
    print("Tulostetaan polytrooppivakiot:" 
          + "\n Kappa: " + str(Kappa)
          + "\n Gamma: " + str(Gamma) + "\n \n")
          
    print("Asetetut alkuarvot (m, p ja rho):"
          + "\n m: " + str(y0[0]) 
          + "\n p: " + str(y0[1]) 
          + "\n rho: " + str(rho) + "\n \n")
    
    # Ratkaistaan TOV annetuilla parametreilla 
    # // 
    # Let's solve the TOV with the given parameters
    # soln = solve_ivp(TOV, (r0, rf), y0, method='BDF',
    #                  dense_output=True, events=found_radius,
    #                  args=(Kappa, Gamma, interpolation, rho_func, p_func))

    soln = solve_ivp(TOV_rho, (rs, rf), (m.real, rho.real), method='BDF',
    first_step=1e-6, dense_output=False, events=found_radius, 
    args=(Kappa, Gamma, interpolation, eos_choise, tov_choise))
    
    print("\n Solverin parametreja:")
    print(soln.nfev, 'evaluations required')
    print(soln.t_events)
    print(soln.y_events)
    print("\n")

    # TOV ratkaisut // TOV solutions
    # Ratkaisut yksiköissä // Solutions in units:
    # [m] = kg, [p] = m**-2 ja [rho] = m**-2
    r = soln.t
    m = soln.y[0].real
    rho = soln.y[1].real
    p = EoS_choiser(2, rho=rho)

    print("Saadut TOV ratkaisut ([m] = eV, [p] = eV^4 ja [rho] = eV^4): \n")
    print("Säde: \n \n" + str(r.real) + 
    "\n \n Massa: \n \n" + str(m.real) + 
    "\n \n Energiatiheys: \n \n" + str(rho.real) + 
    "\n \n Paine : \n \n" + str(p.real) + "\n \n")

    # rho_c0 = unit_conversion(2, "RHO", rho_c.real, -1)
    
    # # # Piirretään ratkaisun malli kuvaajiin yksiköissä:
    # # # //
    # # # Let's plot the model of the solution on graphs in units:
    # # # [m] = kg, [p] = erg/cm**3 ja [rho] = g/cm**3 
    # graph(r, unit_conversion(1, "M", m, -1),
    #       plt.plot, "Mass", "Radius, r (m)", "Mass, m (kg)", 'linear',
    #       body + " " + "mass as a function of radius \n", 1)
    # graph(r, unit_conversion(2, "P", p, -1),
    #       plt.plot, "Pressure", "Radius, r (m)", "Pressure (erg/cm^3)", 'linear',
    #       body + " " + "pressure as a function of radius \n", 1)
    # graph(r, unit_conversion(2, "RHO", rho, -1), plt.plot,
    #       fr'$\rho_c$ = {rho_c0}' '\n'
    #       fr'$K$ = {Kappa.real}' '\n' 
    #       fr'$\Gamma$ = {Gamma}',
    #       "Radius, r", "Energy density, rho (g/cm^3)", 'linear', 
    #       body + " " + "energy density as a function of radius \n", 1, 1)
    
    graph(r, m,
          plt.plot, "Mass", "Radius, r (m)", "Mass, m (kg)", 'linear',
          body + " " + "mass as a function of radius \n", 1)
    graph(r, p,
          plt.plot, "Pressure", "Radius, r (m)", "Pressure (erg/cm^3)", 'linear',
          body + " " + "pressure as a function of radius \n", 1)
    graph(r, rho, plt.plot,
          fr'$\rho_c$ = {rho_c}' '\n'
          fr'$K$ = {Kappa.real}' '\n' 
          fr'$\Gamma$ = {Gamma}',
          "Radius, r", "Energy density, rho (g/cm^3)", 'linear', 
          body + " " + "energy density as a function of radius \n", 1, 1)
    
    
    print("Tähden säde: \n" + str(r[-1]) + 
          "\n Tähden massa: \n" + str(m[-1]) + 
          "\n \n")
    
    return r.real, m.real, p.real, rho.real


def found_radius(t, y, d1, d2, d3, d4, d5):
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
    d1, d2, d3, d4, d5 = d1, d2, d3, d4, d5
    return y[1].real


def main(model, args=[]):
    
    model_choise = ["WD_NREL", "WD_REL"]
    model_params = [[1.5, 0, 0, 0, 0, 1e14+0j, 0, 3, 2, 1, 0, 
                     "Non-relativistic White Dwarf"], 
                    [3, 0, 0, 0, 0, 1e22+0j, 0, 3, 2, 2, 0, 
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
    
    # Määrätään vakioita.
    # //
    # Determine constants.
    M_sun = 2e30              # kg
    R_WD0 = 7e8               # m
    
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
    rmin, rmax = 1e-3, np.inf # 1e-3m
    N = 500
    rspan = np.linspace(rmin, rmax, N)
    
    # Initiaalirajat säteelle. // Initial limits for the radius.
    r0, rf = rmin, rmax
    
    # Ode-ratkaisijan lopettaminen ehdon täyttyessä.
    # //
    # Termination of ode solver when met with condition.
    found_radius.terminal = True # Should be true when works
    found_radius.direction = -1    
    
    r_sol, m_sol, p_sol, rho_sol = TOV_solver([r0, rf], n, R_body, kappa_choise, rho_K, p_K, rho_c, p_c, 
                             a, rho_func, p_func, interpolation, body)
    
    return r_sol, m_sol, p_sol, rho_sol

main("WD_REL")
