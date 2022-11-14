# -*- coding: utf-8 -*-
"""
Created on Mon Oct  31  2022

@author: Antero
"""
# from re import M
import numpy as np
import scipy.constants as sc
import natpy as nat
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from functions import graph

step = 0
NEWT_M = []
TOV_M = []

# M = []
# RHO = []
# P = []
# R = []

# Natural units
def EoS_degelgas(rho):
    m_e = nat.convert(sc.electron_mass * nat.kg, nat.GeV).value
    m_p = nat.convert(sc.proton_mass * nat.kg, nat.GeV).value
    a = (m_e**4)/(8*np.pi**2)
    b = ((3*np.pi**2)/(2*m_p*m_e**3))**(1/3)
    def x(rho):
        return b*(rho)**(1/3)
    def f(x):
        return (1/3)*x**3*(1+x**2)**(1/2)*(2*x**3-3)+np.log(x+(1+x**2)**(1/2))
    # print(a, b)
    return a*f(x(rho))

# Natural units
def Eos_degelgas_deriv(rho):
    m_e = nat.convert(sc.electron_mass * nat.kg, nat.GeV).value
    m_p = nat.convert(sc.proton_mass * nat.kg, nat.GeV).value
    a = (m_e**4)/(8*np.pi**2)
    b = ((3*np.pi**2)/(2*m_p*m_e**3))**(1/3)
    x = lambda y: b*y**(1/3)
    dpdrho = a*b*(3*rho**(2/3))**(-1)*((14*x(rho)**7+12*x(rho)**5-12*x(rho)**4-9*x(rho)**2+3)/(3*(x(rho)**2+1)**(1/2)))
    return dpdrho

def Mass_in_radius(rho, r):
    dmdr = 4*np.pi*rho*r**2
    # print("\n mass derivate from function Mass_in radius: \n " + str(dmdr))
    return dmdr

def ToV(m=0., p=0., rho=0., r=0.):
    """_summary_

    Parameters
    ----------
    choise : _type_
        _description_
    m : _type_
        _description_
    p : _type_
        _description_
    rho : _type_
        _description_
    r : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    # G = sc.gravitational_constant
    # G = 1.7518e-45
    G = 6.707113-57
    
    # drhodr
    # Geometrizied units (C=G=1)
    # tov = (-m*rho/r**2)*(1+p/rho)*(1+((4*np.pi*r**3*p)/(m)))*(1-2*m/r)**(-1)*(Eos_degelgas_deriv(rho))**(-1)
    
    # Natural units drhodr in modular form!
    # tov = ((-(G*m*rho)/(r**2))*(1+p/rho)*(1+(4*np.pi*r**3*p)/(m))*(1-(2*G*m)/(r))**(-1)*(Eos_degelgas_deriv(rho))**(-1))
    
    # Natural units drhodr in TOV-form!
    tov = -(((rho+p)*(G*m + 4*np.pi*G*r**3*p))/(r*(r-2*G*m)))*(Eos_degelgas_deriv(rho))**(-1)
    
    # print("\n TOV choiser value from function TOV_choiser: \n " + str(tov))
    
    return tov


def TOV_rho(r, y, rho_center):
    
    # HUOM! Tänne alkuarvaukset luonnollisissa yksiköissä

    # Asetetaan muuttujat taulukkoon
    # Paine valitaan valitsin-funktiossa.
    # //
    # Let's set the variables in the table. 
    # The energy density is selected in the selector function.
    # G = 6.6743e-11
    # eV
    # G = 6.707113e-57
    # GeV
    G = 6.707113e-39
    m = y[0].real + 0j
    rho = y[1].real + 0j
    p = EoS_degelgas(rho).real
    global step    
    # global M
    # global RHO
    # global P
    # global R
    global NEWT_M
    global TOV_M
    # Ratkaistavat yhtälöt // Equations to be solved
    dy = np.empty_like(y)
    # Massa ja Energiatiheys DY // Mass and energy density DE
    # dy[0] = 4*np.pi*rho*r**2                  # dmdr
    # dy[1] = ToV(m ,p, rho, r)  # drhodr
    
    dy[0] = 4*np.pi*rho*r**2
    # dy[1] = -(((rho+p)*(G*m + 4*np.pi*G*r**3*p))/(r*(r-2*G*m)))*(Eos_degelgas_deriv(rho))**(-1)
    dy[1] = ((-(G*m*rho)/(r**2))*(1+p/rho)*(1+(4*np.pi*r**3*p)/(m))*(1-(2*G*m)/(r))**(-1)*(Eos_degelgas_deriv(rho))**(-1))
    # dy[1] = (-(G*m*rho)/(r**2))*(1-(2*G*m)/(r))**(-1)*(Eos_degelgas_deriv(rho))**(-1)
    
    newt_p_deriv = (-(G*m*rho)/(r**2))*(Eos_degelgas_deriv(rho))**(-1)
    tov_p_deriv = ((-(G*m*rho)/(r**2))*(1+p/rho)*(1+(4*np.pi*r**3*p)/(m))*(1-(2*G*m)/(r))**(-1)*(Eos_degelgas_deriv(rho))**(-1))
    
    NEWT_M.append(newt_p_deriv)
    TOV_M.append(tov_p_deriv)
    
    print("\n \n DEBUG printing \n" + 
          "\n Step: " + str(step) +
          "\n Radius: " + str(r) +
          "\n Mass: " + str(m) + 
          "\n Energy density: " + str(rho) + 
          "\n Pressure: " + str(p) + 
          "\n Mass derivate: " + str(dy[0]) + 
          "\n Energy density derivate: " + str(dy[1]))
    
    # print("\n \n parts of TOV: \n")
    # print("\n First term:")
    # print((rho+p))
    # print("\n Second term:")
    # print((G*m + 4*np.pi*G*r**3*p))
    # print("\n Third term:")
    # print((r*(r-2*G*m)))
    # print("\n Fourth term:")
    # print((Eos_degelgas_deriv(rho))**(-1))
    
    # M.append(m)
    # RHO.append(rho)
    # P.append(p)
    # R.append(r)
    
    # if step > 500000:
        
    #     graph(R, M,
    #         plt.plot, "Mass", "Radius, r (m)", "Mass, m (kg)", 'linear',
    #         "mass as a function of radius \n", new=1, show=1)

    #     graph(R, P, 
    #         plt.plot, "Pressure", "Radius, r", "Pressure, p (erg/cm^3)", 'linear', 
    #         "pressure as a function of radius \n", new=1, show=1)


    #     graph(R, RHO, 
    #         plt.plot, "Energy density", "Radius, r", "Energy density, rho (g/cm^3)", 'linear', 
    #         "energy density as a function of radius \n", new=1, show=1)

    step += 1    
    return dy.real

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
    "\n eos_choise = "   + str(eos_choise) +
    "\n tov_choise = "   + str(tov_choise) +
    "\n interpolate = "  + str(interpolation) + "\n \n")
    
    m = 4/3*np.pi*rho_c*rs**3  # Mass_in_radius(rho_c, rs)
    rho_center = rho_c
    # print("Tulostetaan polytrooppivakiot:" 
    #       + "\n Kappa: " + str(Kappa)
    #       + "\n Gamma: " + str(Gamma) + "\n \n")
          
    # print("Asetetut alkuarvot (m, p ja rho):"
    #       + "\n m: " + str(y0[0]) 
    #       + "\n p: " + str(y0[1]) 
    #       + "\n rho: " + str(rho) + "\n \n")
    
    # Ratkaistaan TOV annetuilla parametreilla 
    # // 
    # Let's solve the TOV with the given parameters
    soln = solve_ivp(TOV_rho, (rs, rf), (m, rho_c), method='BDF',
    dense_output=False, events=found_radius, max_step = 1e21, args=[rho_center]) # ,max_step = 10 , first_step=1e-10
    
    print("\n Solverin parametreja:")
    print(soln.nfev, 'evaluations required')
    print(soln.t_events)
    print(soln.y_events)
    print("\n")

    # TOV ratkaisut // TOV solutions
    # Ratkaisut yksiköissä // Solutions in units:
    # [m] = kg, [p] = m**-2 ja [rho] = m**-2
    r = soln.t * 1.9733e-16 * 1e-3 # 2.6544006e-25*1e9
    m = soln.y[0].real # * 1.7827e-27 # 1.7827e-36
    rho = soln.y[1].real # * 2.0852e37 # 3.16435553043e40
    p = EoS_degelgas(rho) # * 2.0852e37 # 3.16435553043e40

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
    graph(r, m,
          plt.plot, "Mass", "Radius, r (m)", "Mass, m (kg)", 'linear',
          body + " " + "mass as a function of radius \n")
    graph(r, p,
          plt.plot, "Pressure", "Radius, r (m)", "Pressure (erg/cm^3)", 'linear',
          body + " " + "pressure as a function of radius \n", 1)
    graph(r, rho, plt.plot,
          fr'$\rho_c$ = {rho_c}' '\n',
          "Radius, r", "Energy density, rho (g/cm^3)", 'linear', 
          body + " " + "energy density as a function of radius \n", 1, 1)
    
    
    # print("\n Newtonian energy denisty derivative: \n")
    # print(NEWT_M)
    # print("\n Tolman-Oppenheimer-VOlkoff energy denisty derivative: \n")
    # print(TOV_M)
    
    print("Tähden säde: \n" + str(r[-1]) + 
          "\n Tähden massa: \n" + str(m[-1]) + 
          "\n \n")
    
    return r.real, m.real, p.real, rho.real


def found_radius(t, y, d1):
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
    # d1, d2, d3, d4, d5 = d1, d2, d3, d4, d5
    d1 = d1
    pressure = EoS_degelgas(y[1].real)
    return  y[1].real >= 1e-4*d1# 4.3e21 # 1e16


def main(model, args=[]):
    
    model_choise = ["WD_NREL", "WD_REL"]
    model_params = [[1.5, 0, 0, 0, 0, 1e14+0j, 0, 3, 2, 1, 0, 
                     "Non-relativistic White Dwarf"], 
                    [3, 0, 0, 0, 0, 4.3e-13, 0, 3, 2, 2, 0, 
                     "Relativistic White Dwarf"]]
    # Variation range for REL_WD is 4.3e21 - 4.3e25
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
    
    # Luonnollisia yksiköitä 
    # // 
    # Natural units
    c = 1
    hbar = 1
    
    # Asetetaan integrointiparametrit.
    # Integraattori adaptiivinen, lopettaa integroinnin tähden rajalla.
    # //
    # Let's set the integration parameters.
    # Integrator adaptive, stops the integration at the star boundary.
    rmin, rmax = 5.067e12, np.inf # 
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

"""
Ratkaistaan massa-säde relaatio. Etsitään TOV-yhtälöiden ratkaisuja
jollakin rhospan-alueella. Ratkaistaan yhtälöitä siis tähden keskipisteen eri
energiatiheyksien arvoilla. Etsii ratkaisuja rho-muotoisesta tov-yhtälöstä.

Etsitään tähden raja (find_radius) paineen ratkaisusta ja sitä vastaava
massa massan kuvaajasta. Tallennetaan nämä arvot taulukkoon ja piirretään
kuvaaja.

Mallinnetaan nyt useaa tähteä ja piirretään
Massa-Säde - relaatio.
//
Let's solve the mass-radius relation. We are looking for solutions to the
TOV equations in some rhospan area. So let's solve the equations
from the center of the star with varying values ​​of energy densities. Finds 
solutions to a tov equation in rho form.


Let's find the limit of the star (find_radius) from the pressure solution 
and its equivalent mass from the mass solution. Let's save these values ​​in
an array and plot them.

Now let's model several stars and plot them
Mass-Radius - relation.
"""

# TODO korjaa
def MR_relaatio(rho_min, rho_max, N_MR):
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
    # Build N_MR amount of star models
    
    # rhospan = np.linspace(rho_min, rho_max, N_MR)
    rhospan = np.logspace(np.log10(rho_min), np.log10(rho_max), N_MR)
    print("rhospan: " + str(rhospan))
    R = []
    M = []
    # KAPPA = 6.07706649646 # NREL
    # Ratkaise TOV jokaiselle rho0:lle rhospan alueessa.
    # //
    # Solve the TOV for each rho0 in the range of rhospan.
    for rho0 in rhospan:
        # r, m, p, rho = main("CUSTOM", [1.5, 7e8, KAPPA, rho0+0j, 0, rho0+0j, 0, 0, 0, 0, 0, 
        #                "Not Relativistic White Dwarf"]) 
        # KAPPA += 0.5
        
        r, m, p, rho = main("CUSTOM", [3, 0, 0, 0, 0, rho0, 0, 3, 2, 2, 0, 
                        "Relativistic White Dwarf"]) 

        # r_boundary = find_radius(p, r, raja=0.05)
        r_boundary = r[-1]
        # m_boundary = find_mass_in_radius(m, r, r_boundary)
        m_boundary = m[-1]
        # if m_boundary > 0:
        R.append(r_boundary)
        M.append(m_boundary)
    # Printtaa ja plottaa massa-säde - relaation. 
    # //
    # Print and plot the mass-radius relation.
    print("Tulostetaan ratkaistut massat ja niitä vastaavat säteet: \n")
    print("Säteet: \n " + str(R) + "\n Massat: \n" + str(M))
    R = np.array(R)
    M = np.array(M)

    graph(R, M, plt.scatter, "Massa-säde - relaatio", "Säde",
          "Massa", 'linear', "Massa-säde", 1, 1)
    graph(R, M, plt.plot, "Massa-säde - relaatio", "Säde",
          "Massa", 'linear', "Massa-säde", 1, 1)
    return R, M

# MR_relaatio(5e-15, 5e-2, 200)



main("WD_REL")
