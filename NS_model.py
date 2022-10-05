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

from TOV_solver_p import *
from functions import *
from structure_equations import *
"""
Rakennetaan neutronitähden malli paperista "A unified equation
of state of dense matter and neutron star structure" saadulla datalla
sisemmän kuoren tilanyhtälöstä ja ytimen tilanyhtälöstä.
Tilanyhtälöt:
    Ulompi kuori  -> Polytrooppi tilanyhtälö Gamma = 4/3
    Sisempi kuori -> Data paperin taulukosta 3.
    Ydin          -> Data paperin taulukosta 5.
//
Let's build a neutron star model from the paper "A unified equation
of state of dense matter and Neutron star structure" with the obtained data
from the equation of state of the inner shell and the equation of state of 
the core
Equation of states:
    Outer crust     -> Polytrope with Gamma = 4/3
    Inner crust     -> Data from paper array 3.
    Core            -> Data from paper array 5.
"""

# Tilanyhtälöiden muuttujat datasta // Variables of state equations from data:
#     n_b, rho, P, Gamma.
def NS_model(ir=[]):
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
    NS_EoS_oc_r, NS_EoS_oc_m, NS_EoS_oc_P, NS_EoS_oc_RHO = TOV_solver((ir[0], ir[1]), (
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
        0,
        "Neutron star outer core model (polytrope) \n"))
    
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
    graph(unit_conversion(3, "P", NS_EoS_ic_core_P, -1)*1e-9, 
          unit_conversion(3, "RHO", NS_EoS_ic_core_rho, -1)*1e-9, plt.scatter,
          "NS EoS, (P, rho) - ic-core",
          "Pressure, P (eV)", "Energy density, rho (eV)", 'log', # m^-2
          "NS energy density as a function of pressure ic-core")
    
    # Yhdistetään paperin data ja ratkaistu ulomman kuoren malli.
    # //
    # The paper data and the solved model of the outer shell are combined.
    NS_EoS_P = np.flip(np.unique(np.delete(
        np.append(NS_EoS_ic_core_P.real, NS_EoS_oc_P.real), -1)), -1)
    NS_EoS_RHO = np.flip(np.unique(np.delete(
        np.append(NS_EoS_ic_core_rho.real, NS_EoS_oc_RHO.real), -1)), -1)
    
    # Plot
    graph(unit_conversion(3, "P", NS_EoS_P, -1)*1e-9, 
          unit_conversion(3, "P", NS_EoS_RHO, -1)*1e-9, 
          plt.scatter, "NS EoS, (P, rho)",
          "Pressure, P (eV)", "Energy denisty, rho (eV)", 'log', # m^-2
          "NS energy density as a function of pressure")
    
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
    graph(unit_conversion(3, "P", NS_EoS_P_new, -1)*1e-9, 
          unit_conversion(3, "P", NS_EoS_interpolate(NS_EoS_P_new), -1)*1e-9, 
          plt.plot,
          "NS EoS, (rho, P) interpolate", "Pressure, P (eV)", # m^-2
          "eEnergy density, rho (eV)", 'log', "NS interpolated energy density")
    
    # Ratkaistaan nyt TOV uudestaan NS:n datalle ja mallinnetaan koko
    # tähden rakenne säteen funktiona.
    # //
    # Let's now solve the TOV again for NS data and model the whole
    # star structure as a function of radius.
    NS_r, NS_m, NS_p, NS_rho = TOV_solver((ir[0], ir[1]), (
        3, 
        R_NS0, 
        0,
        NS_EoS_interpolate(NS_EoS_P_new[2])+0j,
        NS_EoS_P_new[2],
        NS_EoS_interpolate(NS_EoS_P_new[2])+0j,
        NS_EoS_P_new[2],
        2,
        1,
        0,
        NS_EoS_interpolate,
        "Neutron star"))
    # Vielä avaruuden kaarevuus luonnollisissa yksiköissä
    # neutronitähden sisällä.
    # //
    # Also the curvature of space in natural units
    # inside a neutron star.
    Ricci_scalar(NS_p, NS_rho, NS_r)        
            
    return NS_r, NS_m, NS_p, NS_rho

def main():
    
    
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
    rmin, rmax = 1e-3, np.inf
    N = 500
    rspan = np.linspace(rmin, rmax, N)
    
    # Initiaalirajat säteelle. // Initial limits for the radius.
    r0, rf = rmin, rmax
    
    
    # Ode-ratkaisijan lopettaminen ehdon täyttyessä.
    # //
    # Termination of ode solver when met with condition.
    found_radius.terminal = True
    found_radius.direction = -1
    
    r_sol, m_sol, p_sol, rho_sol = NS_model((r0, rf))
    
    return r_sol, m_sol, p_sol, rho_sol    
