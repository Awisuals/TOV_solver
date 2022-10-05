# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5  2022

@author: Antero
"""
from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
import scipy.constants as sc
import natpy as nat

from functions import *
from structure_equations import *
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
    KAPPA = 1.94888854486 # REL
    # KAPPA = 6.07706649646 # NREL
    # Ratkaise TOV jokaiselle rho0:lle rhospan alueessa.
    # //
    # Solve the TOV for each rho0 in the range of rhospan.
    for rho0 in rhospan:
        # r, m, p, rho = main("CUSTOM", [1.5, 7e8, KAPPA, rho0+0j, 0, rho0+0j, 0, 0, 0, 0, 0, 
        #                "Not Relativistic White Dwarf"]) 
        # KAPPA += 0.5
        
        r, m, p, rho = main("CUSTOM", [3, 7e8, KAPPA, rho0+0j, 0, rho0+0j, 0, 0, 0, 0, 0, 
                        "Relativistic White Dwarf"]) 

        # r_boundary = find_radius(p, r, raja=0.05)
        r_boundary = r[-1]
        # m_boundary = find_mass_in_radius(m, r, r_boundary)
        m_boundary = m[-1]
        if m_boundary > 0:
            R.append(r_boundary)
            M.append(m_boundary)
    # Printtaa ja plottaa massa-säde - relaation. 
    # //
    # Print and plot the mass-radius relation.
    print("Tulostetaan ratkaistut massat ja niitä vastaavat säteet: \n")
    print("Säteet: \n " + str(R) + "\n Massat: \n" + str(M))
    R = np.array(R)
    M = np.array(M)

    graph(R, unit_conversion(1, "M", M, -1), plt.scatter, "Massa-säde - relaatio", "Säde",
          "Massa", 'linear', "Massa-säde", 1, 1)
    graph(R, unit_conversion(1, "M", M, -1), plt.plot, "Massa-säde - relaatio", "Säde",
          "Massa", 'linear', "Massa-säde", 1, 1)
    return R, M
