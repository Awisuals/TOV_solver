# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5  2022

@author: Antero
"""
import numpy as np
import scipy.constants as sc
import natpy as nat

from functions import graph

"""
Määritetään Tolman-Oppenheimer-Volkoff - yhtälöt (m, p, EoS), 
jotka ratkaisemalla saadaan kuvattua tähden rakenne.
//
Determine the Tolman-Oppenheimer-Volkoff equations (m, p, EoS),
by solving which the structure of the star can be described.
"""

def Mass_in_radius(rho, r):
    dmdr = 4*np.pi*rho*r**2
    # print("\n mass derivate from function Mass_in radius: \n " + str(dmdr))
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

# def Eos_degelgas_deriv(rho):
#     m_e = nat.convert(sc.electron_mass * nat.kg, nat.eV).value
#     m_p = nat.convert(sc.proton_mass * nat.kg, nat.eV).value
#     a = (m_e**4)/(8*np.pi**2)
#     b = ((3*np.pi**2)/(2*m_p*m_e**3))
    
#     dpdrho = ((a*b*(4*b*rho-3)*(((b*rho)**(2/3)+1)**(1/2)))/(3)) + ((2*a*b**2*rho*(b*rho)**(2/3))/(9*((b*rho)**(2/3)+1)**(1/2))) + ((((a*b)/(3*(b*rho)**(2/3)))-((a*b*(b*rho)**(2/3))/(3)))/((b*rho)**(2/3)+1)**(1/2))
#     return dpdrho

def Eos_degelgas_deriv(rho):
    m_e = nat.convert(sc.electron_mass * nat.kg, nat.eV).value
    m_p = nat.convert(sc.proton_mass * nat.kg, nat.eV).value
    a = (m_e**4)/(8*np.pi**2)
    b = ((3*np.pi**2)/(2*m_p*m_e**3))**(1/3)
    x = lambda y: b*y**(1/3)
    dpdrho = a*b*(3*rho**(2/3))**(-1)*((14*x(rho)**7+12*x(rho)**5-12*x(rho)**4-9*x(rho)**2+3)/(3*(x(rho)**2+1)**(1/2)))
    return dpdrho

def EoS_degelgas(rho):
    m_e = nat.convert(sc.electron_mass * nat.kg, nat.eV).value
    m_p = nat.convert(sc.proton_mass * nat.kg, nat.eV).value
    a = (m_e**4)/(8*np.pi**2)
    b = ((3*np.pi**2)/(2*m_p*m_e**3))**(1/3)
    def x(rho):
        return b*(rho)**(1/3)
    def f(x):
        return (1/3)*x**3*(1+x**2)**(1/2)*(2*x**3-3)+np.log(x+(1+x**2)**(1/2))
    # print(a, b)
    return a*f(x(rho))

# def EoS_degelgas(rho):
#     m_e = nat.convert(sc.electron_mass * nat.kg, nat.eV).value
#     mu_e = 2
#     m_mu = nat.convert(1.7762828666e-26 * nat.kg, nat.eV).value
#     a = (m_e**4)/(24*np.pi**2)
#     b = ((3*np.pi**2)/(mu_e*m_mu*m_e**3))**(1/3)
#     y = lambda z : b*z**(1/3)
#     f = lambda x : x*(2*x**2-3)*(x**2+1)**(1/2)+3*np.log(x+(1+x**2)**(1/2))
#     return a*f(y(rho))


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
        returnable = interpolation # EoS_cd_p2rho(interpolation, p) # Energy density # TODO add interpolate functions and params
    if choise == 2:
        returnable = EoS_degelgas(rho) # Pressure
    # print("\n EoS choiser value from function EoS_choiser: \n " + str(returnable))
    return returnable


def TOV_choiser(choise, m=0., p=0., rho=0., r=0.):
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
    G = 4.30091e-3
    # m_e = nat.convert(sc.electron_mass * nat.kg, nat.eV).value
    # mu_e = 2
    # m_mu = nat.convert(1.7762828666e-26 * nat.kg, nat.eV).value
    # a = (m_e**4)/(24*np.pi**2)
    # b = ((3*np.pi**2)/(mu_e*m_mu*m_e**3))**(1/3)
    
    m_e = nat.convert(sc.electron_mass * nat.kg, nat.eV).value
    m_p = nat.convert(sc.proton_mass * nat.kg, nat.eV).value
    a = (m_e**4)/(8*np.pi**2)
    b = ((3*np.pi**2)/(2*m_p*m_e**3))**(1/3)

    if choise == 0:
        # dpdr relativistic
        tov = -(rho+p)*(m + 4*np.pi*r**3*p)/(r*(r-2*m))
    if choise == 1:
        # dpdr newtonian
        tov = -(m*rho)/(r**2)
    if choise == 2:
        # drhodr
        tov = ((-(G*m*rho)/(r**2))*(1+p/rho)*(1+(4*np.pi*r**3*p)/(m))*(1-(2*G*m)/(r))**(-1)*(Eos_degelgas_deriv(rho))**(-1))
    # print("\n TOV choiser value from function TOV_choiser: \n " + str(tov))
    return tov
