# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5  2022

@author: Antero
"""
import numpy as np
import matplotlib.pyplot as plt

from functions import *
from structure_equations import *
from TOV_solver_rho import *
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
from the center of the star with varying values of energy densities. Finds 
solutions to a tov equation in rho form.


Let's find the limit of the star (find_radius) from the pressure solution 
and its equivalent mass from the mass solution. Let's save these values in
an array and plot them.

Now let's model several stars and plot them
Mass-Radius - relation.
"""

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
    rhospan = np.logspace(np.log10(rho_min), np.log10(rho_max), N_MR)
    print("rhospan: " + str(rhospan))
    R_tov = []
    M_tov = []
    
    R_newt = []
    M_newt = []
    # Ratkaise TOV jokaiselle rho0:lle rhospan alueessa.
    # //
    # Solve the TOV for each rho0 in the range of rhospan.
    for rho0 in rhospan:
        r_tov, m_tov, p_tov, rho_tov = TOV_solver(ir=[], 
               n=0, 
               R_body=0, 
               kappa_choise=0, 
               rho_K=0, 
               p_K=0, 
               rho_c=rho0, 
               p_c=0, 
               a=3, 
               eos_choise=2, 
               tov_choise=2, 
               interpolation=0, 
               body="TOV White dwarf")
        r_boundary = r_tov[-1]
        m_boundary = m_tov[-1]
        R_tov.append(r_boundary)
        M_tov.append(m_boundary)
        
        r_newt, m_newt, p_newt, rho_newt = TOV_solver(ir=[], 
               n=0, 
               R_body=0, 
               kappa_choise=0, 
               rho_K=0, 
               p_K=0, 
               rho_c=rho0, 
               p_c=0, 
               a=3, 
               eos_choise=2, 
               tov_choise=3, 
               interpolation=0, 
               body="NEWT White dwarf")
        r_boundary = r_newt[-1]
        m_boundary = m_newt[-1]
        R_newt.append(r_boundary)
        M_newt.append(m_boundary)
        
    # Printtaa ja plottaa massa-säde - relaation. 
    # //
    # Print and plot the mass-radius relation.
    # print("Tulostetaan ratkaistut massat ja niitä vastaavat säteet: \n")
    # print("Säteet: \n " + str(R) + "\n Massat: \n" + str(M))
    # R_tov = np.array(R)
    # M_tov = np.array(M)

    # graph(R, M, plt.scatter, "Massa-säde - relaatio", "Säde",
    #       "Massa", 'linear', "Massa-säde", 1, 1)
    graph(R_tov, M_tov, plt.plot, "Massa-säde - relaatio", "Säde",
          "Massa", 'linear', "Massa-säde")
    graph(R_newt, M_newt, plt.plot, "Massa-säde - relaatio", "Säde",
          "Massa", 'linear', "Massa-säde", 0, 1)
    
    # DELTA_R = R_newt - R_tov
    # DELTA_M = M_newt - M_tov
    
    # graph(DELTA_R, M_newt, plt.plot, "Massa-säde - relaatio", "Säde",
    #       "Massa", 'linear', "Massa-säde", 1, 1)
    return

MR_relaatio(2e-14+0j, 2e-6+0j, 50)

