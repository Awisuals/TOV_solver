# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5  2022

@author: Antero
"""
import numpy as np
import scipy.constants as sc
import natpy as nat
"""
Määritetään Tolman-Oppenheimer-Volkoff - yhtälöt (m, p, EoS), 
jotka ratkaisemalla saadaan kuvattua tähden rakenne.
//
Determine the Tolman-Oppenheimer-Volkoff equations (m, p, EoS),
by solving which the structure of the star can be described.
"""

def Mass_in_radius(rho, r):
    dmdr = 4*np.pi*rho*r**2
    return dmdr


def EoS_cd_p2rho(interpolation, p_point):
    """
    Equation of state from custom interpolated (P, RHO)-data.

    Parameters
    ----------
    interpolation : Interpolate function.
        Returns rho when given p.
    p_point : Float, Array
        Pressure of a astrophysical body.

    Returns
    -------
    Float, Array
        Returns RHO as a function of P.

    """
    return interpolation(p_point)


def EoS_r2p(rho, Gamma, Kappa):
    """
    Equation of state, EoS.

    Given the energy density and constants returns the pressure.

    Parameters
    ----------
    rho : Float
        Energy density.
    Gamma : Float
        Polytrope constant.
    Kappa : Float
        Constant of proportionality.

    Returns
    -------
    p : Float
        Pressure.

    """
    p = Kappa*rho**Gamma
    return p


def EoS_p2r(p, Gamma, Kappa):
    """
    Equation of state, EoS.

    Given pressure and constants returns the energy density.

    Parameters
    ----------
    p : Float
        Pressure.
    Gamma : Float
        Polytrope constant.
    Kappa : Float
        Constant of proportionality.

    Returns
    -------
    rho : Float
        Energy density.

    """
    rho = (p/Kappa)**(1/Gamma)
    return rho


def EoS_degelgas(rho):
    """
    Degenerate electron gas equation of state.

    Parameters
    ----------
    rho : float
        Pressure.

    Returns
    -------
    float
        Pressure from given energy density.
    """
    # Tilanyhtälön vakkioita // Equation of state constants
    m_e = nat.convert(sc.electron_mass * nat.kg, nat.GeV).value
    m_p = nat.convert(sc.proton_mass * nat.kg, nat.GeV).value
    a = (m_e**4)/(3*np.pi**2)
    b = ((3*np.pi**2)/(2*m_p*m_e**3))**(1/3)
    # Tilanyhtälö // Equation of state
    x = lambda rho : b*(rho)**(1/3)
    f = lambda x : (1/8)*(x*(1+x**2)**(1/2)*(2*x**2-3)+3*np.log(x+(1+x**2)**(1/2)))
    return a*f(x(rho))


def Eos_degelgas_deriv(rho):
    """
    Degenerate electron gas equation of state derivate.
    dp/drho.
    
    Parameters
    ----------
    rho : float
        Energy density.

    Returns
    -------
    float
        Derivate value.
    """    
    # Vakioita // Constants
    m_e = nat.convert(sc.electron_mass * nat.kg, nat.GeV).value
    m_p = nat.convert(sc.proton_mass * nat.kg, nat.GeV).value
    a = (m_e**4)/(3*np.pi**2)
    b = ((3*np.pi**2)/(2*m_p*m_e**3))**(1/3)
    # Derivaatta // derivate
    x = lambda y: b*y**(1/3)
    dpdrho = (a*b)/(3*rho**(2/3))*((x(rho)**4)/((1+x(rho)**2)**(1/2)))
    return dpdrho


def EoS_choiser(choise, interpolation=0., Gamma=0., Kappa=0., p=0., rho=0.):
    """
    Chooses wanted EoS for DE-group as rho and returns it
    with given parameters.

    Parameters
    ----------
    choise : Int
        Chooses between defined EoS to return.
        Can be 0, 1 or 2.
    interpolation : args
    p : args
    Gamma : argsz
    Kappa : args

    Returns
    -------
    returnable : Float, Array
        Energy density with calculated with wanted EoS and given params.

    """
    if choise == 0:
        returnable = EoS_p2r(p, Gamma, Kappa) # Energy density
    if choise == 1:
        # TODO add interpolate functions and params
        returnable = interpolation # EoS_cd_p2rho(interpolation, p) # Energy density 
    if choise == 2:
        returnable = EoS_degelgas(rho) # Pressure
    return returnable


def TOV_choiser(choise, m=0., p=0., rho=0., r=0.):
    """
    Chooses between different tov functions.

    Parameters
    ----------
    choise : int
        Chooses function:
            choise=0 dpdr tov
            choise=1 dpdr newt
            choise=2 drhodr tov
            choise=3 drhodr newt
    m : float
        Mass.
    p : float
        Pressure.
    rho : float
        Energy density.
    r : float
        Radius.

    Returns
    -------
    float
        Value for chosen hydrostatic equilibrium equation deriv.
    """
    # Gravitational constant in GeV^-2.
    G = 6.707113e-39
    if choise == 0:
        # dpdr tov
        tov = -(rho+p)*(m + 4*np.pi*r**3*p)/(r*(r-2*m))
    if choise == 1:
        # dpdr newt.
        tov = -(m*rho)/(r**2)
    if choise == 2:
        # drhodr tov
        tov = -(((rho+p)*(G*m + 4*np.pi*G*r**3*p))/(r*(r-2*G*m)))*(Eos_degelgas_deriv(rho))**(-1)
    if choise == 3:
        # drhodr newt.
        tov = (-(G*m*rho)/(r**2))*(Eos_degelgas_deriv(rho))**(-1)
    return tov
