# -*- coding: utf-8 -*-
"""
Created on Fri Nov  18  2022

@author: Antero
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from structure_equations import *
from TOV_solver_rho import *

def Relativistic_Terms():
        """
        Solves TOV-equations and newt-pressure twice for two white dwarves
        with different central energy density and plots the differences between
        these two theories.

        Returns
        -------
        None 
        
        """        

        # Määritellään ratkaistavien WD:en parametreja.
        # //
        # The parameters of the WDs to be solved are defined.
        
        # Pressure at the center.
        rho_center_thin =  2e-11+0j        
        rho_center_dense = 2e-7+0j
        # Radius of WD with previous energy densities.
        skaala_thin = 1.286770048879275 
        skaala_dense = 0.13169421547674232
        # How close to the core we want to solve the equations again
        # to get good resolution for the differenve between theories.
        kerroin_thin = 0.9
        kerroin_dense = 0.9
        # Some formatting for graphs.
        rho_center_d_si = '{:0.2e}'.format(rho_center_dense.real * 2.0852e37)
        rho_thin_d_si = '{:0.2e}'.format(rho_center_thin.real * 2.0852e37)
        R_koko_thin = '{:0.2e}'.format(skaala_thin)
        R_koko_dense = '{:0.2e}'.format(skaala_dense)

        def Tov_Newt_suhde(k, s, rho0):
                """
                Solves the structure of WD with given core density.

                Parameters
                ----------
                k : float
                    How close to the core we want to be.
                s : float
                    Radius of wanted star for integration range.
                rho0 : float
                    Core density.

                Returns
                -------
                array
                    Returns the radius of stars and the ratio 
                    between solutions of these theories.
                """                
                r_tov, m_tov, p_tov, rho_tov = TOV_solver(ir=[5.067e12, k*(s/(1.9733e-16 * 1e-3 / 6371))], 
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

                r_newt, m_newt, p_newt, rho_newt = TOV_solver(ir=[5.067e12, k*(s/(1.9733e-16 * 1e-3 / 6371))], 
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

                Delta_Rho = rho_newt[:-1] / rho_tov[:-1]
                return r_newt, Delta_Rho

        r_thin, rho_thin = Tov_Newt_suhde(kerroin_thin, skaala_thin, rho_center_thin)
        r_dense, rho_dense = Tov_Newt_suhde(kerroin_dense, skaala_dense, rho_center_dense)
        
        # Plotataan kuvat. // Plot all graphs.
        
        gs = gridspec.GridSpec(1, 2)
        plt.figure()

        ax1 = plt.subplot(gs[0, 0])
        ax2 = plt.subplot(gs[0, 1])

        ax1.plot(np.flip(r_dense[:-1])/skaala_dense, rho_dense, color='b', 
                label=fr'Teorioiden välinen suhde keskipisteen parametreilla' '\n' fr'$\rho_{"c"}$ = {rho_center_d_si}' r' $\frac{\mathrm{J}}{\mathrm{m}^{3}}$' '\n' r'$R_{WD}$' fr' = {R_koko_dense}' r' $R_{Earth}$')
        ax2.plot(np.flip(r_thin[:-1])/skaala_thin, rho_thin, color='r', 
                label=fr'Teorioiden välinen suhde keskipisteen parametreilla' '\n' fr'$\rho_{"c"}$ = {rho_thin_d_si}' r' $\frac{\mathrm{J}}{\mathrm{m}^{3}}$' '\n' r'$R_{WD}$' fr' = {R_koko_thin}' r' $R_{Earth}$')

        ax1.set(xlabel=r'Säde, r ($R_{WD}$)', 
                ylabel= r'$\frac{\rho_{newt}}{\rho_{tov}}$', 
                xscale="linear", yscale="log")
        ax1.set_title('e)', loc="left")
        ax1.legend(shadow=True, fancybox=True)
        ax1.grid()

        ax2.set(xlabel=r'Säde, r ($R_{WD}$)', 
                ylabel=r'$\frac{\rho_{newt}}{\rho_{tov}}$', 
                xscale="linear", yscale="log")
        ax2.set_title('f)', loc="left")
        ax2.legend(shadow=True, fancybox=True)
        ax2.grid()

        plt.show()
        return
    
Relativistic_Terms()