# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5  2022

@author: Antero
"""

def TOV_rho(r, y, K, G, interpolation, eos_choise, tov_choise):
    
    # HUOM! Tänne alkuarvaukset luonnollisissa yksiköissä

    # Asetetaan muuttujat taulukkoon
    # Paine valitaan valitsin-funktiossa.
    # //
    # Let's set the variables in the table. 
    # The energy density is selected in the selector function.
    m = y[0].real + 0j
    rho = y[1].real +0j
    p = EoS_choiser(eos_choise, interpolation, G, K, rho=rho).real + 0j
    
    # Ratkaistavat yhtälöt // Equations to be solved
    dy = np.empty_like(y)
    # Massa ja Energiatiheys DY // Mass and energy density DE
    dy[0] = Mass_in_radius(rho, r)                  # dmdr
    dy[1] = TOV_choiser(tov_choise, m ,p, rho, r)  # drhodr
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
    rho_func : Int, optional
        Choise for what EoS is used to compute energy density.
        Choise:
            0=Polytrope EoS.
            1=Interpolated EoS from data.
        The default is 0..
    p_func : int, optional
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

    soln = solve_ivp(TOV_p, (rs, rf), (m.real, p.real), method='Radau',
    first_step=1e-6, dense_output=True, events=found_radius, 
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
    p = soln.y[1].real
    rho = EoS_p2r(p, Gamma, Kappa)

    print("Saadut TOV ratkaisut ([m] = kg, [p] = m**-2 ja [rho] = m**-2): \n")
    print("Säde: \n \n" + str(r.real) + 
    "\n \n Massa: \n \n" + str(m.real) + 
    "\n \n Paine: \n \n" + str(p.real) + 
    "\n \n Energiatiheys: \n \n" + str(rho.real) + "\n \n")

    rho_c0 = unit_conversion(2, "RHO", rho_c.real, -1)
    
    # # # Piirretään ratkaisun malli kuvaajiin yksiköissä:
    # # # //
    # # # Let's plot the model of the solution on graphs in units:
    # # # [m] = kg, [p] = erg/cm**3 ja [rho] = g/cm**3 
    graph(r, unit_conversion(1, "M", m, -1),
          plt.plot, "Mass", "Radius, r (m)", "Mass, m (kg)", 'linear',
          body + " " + "mass as a function of radius \n", 1)
    graph(r, unit_conversion(2, "P", p, -1),
          plt.plot, "Pressure", "Radius, r (m)", "Pressure (erg/cm^3)", 'linear',
          body + " " + "pressure as a function of radius \n", 1)
    graph(r, unit_conversion(2, "RHO", rho, -1), plt.plot,
          fr'$\rho_c$ = {rho_c0}' '\n'
          fr'$K$ = {Kappa.real}' '\n' 
          fr'$\Gamma$ = {Gamma}',
          "Radius, r", "Energy density, rho (g/cm^3)", 'linear', 
          body + " " + "energy density as a function of radius \n", 1, 1)
    
    print("Tähden säde: \n" + str(r[-1]) + 
          "\n Tähden massa: \n" + str(m[-1]) + 
          "\n \n")
    
    return r.real, m.real, p.real, rho.real
