# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5  2022

@author: Antero
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

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

def MR_relaatio(rho_min1, rho_min2, rho_max, N_MR1, N_MR2):
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
    # Build N_MR1 or N_MR2 amount of star models
    rhospan1 = np.logspace(np.log10(rho_min1), np.log10(rho_max), N_MR1)
    rhospan2 = np.logspace(np.log10(rho_min2), np.log10(rho_max), N_MR2)
    print("\n \n rhospan: " + str(rhospan1))
    
    R_tov = []
    M_tov = []
    
    R_newt = []
    M_newt = []
    
    R_tov_zoom = []
    M_tov_zoom = []
    
    R_newt_zoom = []
    M_newt_zoom = []
    
    # Ratkaise TOV jokaiselle rho0:lle rhospan alueessa.
    # //
    # Solve the TOV for each rho0 in the range of rhospan.
    for rho0 in rhospan1:
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
        
    print("\n \n rhospan2: " + str(rhospan2))
    
    for rho0 in rhospan2:
        r_tov_zoom, m_tov_zoom, p_tov_zoom, rho_tov_zoom = TOV_solver(ir=[], 
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
        r_boundary = r_tov_zoom[-1]
        m_boundary = m_tov_zoom[-1]
        R_tov_zoom.append(r_boundary)
        M_tov_zoom.append(m_boundary)
        
        r_newt_zoom, m_newt_zoom, p_newt_zoom, rho_newt_zoom = TOV_solver(ir=[], 
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
        r_boundary = r_newt_zoom[-1]
        m_boundary = m_newt_zoom[-1]
        R_newt_zoom.append(r_boundary)
        M_newt_zoom.append(m_boundary)
        
    # Plottaa massa-säde - relaation. 
    # //
    # Pplot the mass-radius relation.

    fig, ax1 = plt.subplots() 
    axins1 = inset_axes(ax1, width='30%', height='40%', loc='lower right', 
                        bbox_to_anchor=(-0.12, 0.3, 1.1, 1.2),
                        bbox_transform=ax1.transAxes)
    
    ax1.plot(R_tov, M_tov, color='b', label=fr'TOV')
    ax1.plot(R_newt, M_newt, color='red', label='Newtonilainen' , linestyle='--')

    ax1.set(# title="a)", title_position='left',
            xlabel=r'Säde, r ($R_{Earth}$)', 
            ylabel= r'Massa, m ($M_{Sun}$)', 
            xscale="linear", yscale="linear")
    ax1.set_title('d)', loc="left")
    ax1.legend(shadow=True, fancybox=False)
    ax1.grid()
    
    axins1.plot(R_tov_zoom, M_tov_zoom, color='b')
    axins1.plot(R_newt_zoom, M_newt_zoom, color='red', linestyle='--')
    
    axins1.set(# title="a)", title_position='left',
            xscale="linear", yscale="log")
    axins1.grid()
    
    plt.show()
    
    return

MR_relaatio(2e-14+0j, 8e-10+0j, 8e-6+0j, 50, 75)
