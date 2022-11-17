# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5  2022

@author: Antero
"""
import numpy as np
import matplotlib.pyplot as plt

"""

Yleisiä funktioita hyötykäyttöön. // General functions for utility use.

"""

def graph(x, y, style, label, xlabel, ylabel, xscale, yscale, title, new=0, show=0):
    """
    Generates graph with given parameters.

    Parameters
    ----------
    x : Array
        Values for x-axis.
    y : Array
        Values for y-axis.
    style :
        Choose plt.scatter or plt.plot.
    label : String
        Label for legend.
    xlabel : String
        Label for x-axis.
    ylabel : String
        Label for y-axis.
    scale : String
        Scaling for x- and y-axis. Example 'log' or 'linear'.    
    title : String
        Title for graph.

    Returns
    -------
    None.

    """
    if new == 1:
        plt.figure()

    style(x, y, label=label)
    # Piirtää suoran y = 0. // Draws horizontal line y = 0.
    plt.axhline(y=0, color='r', linestyle='--')
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xscale(xscale) 
    plt.yscale(yscale)
    plt.title(title)
    plt.legend()
    if show == 1:
        plt.show()


def unit_conversion(SYS, VAR, VAL, DIR):
    """
    Changes units on desired direction. Handles
    geometrized, natural and SI-units.

    Unit system:
        0 for geometrizied units
        1 for natural units
        2 for [p]=erg/cm**3 and [rho]=g/cm**3 -> GEOM.
        3 for GEOM. -> NATURAL

    Possible conversion apply for:
        M, P and RHO.

    Directions:
        1: from SYS. to SI. MULTIPLY
        -1: from SI to SYS. DIVIDE

    Parameters
    ----------
    SYS : Int
        Choose unit system.
    VAR : String
        Variable that needs unit change.
    VAL : Float
        Value of said variable.
    DIR : Int
        Direction of desired conversion.

    Returns
    -------
    None.

    """
    VAR_TAU = ["M", "P", "RHO"]
    SYS_CON = [[1.3466e27, 1.2102e44, 1.2102e44],
               [1.7827e-27, 2.0852e37, 2.0852e37],
               [1, 8.2627e-46, 7.4261e-25],
               [7.553710663600157e+53, 5803759.831191253, 5803759.831191253]]
    VAR_CON = SYS_CON[SYS]
    for i, var in enumerate(VAR_TAU):
        if VAR == var and DIR == 1:
            VAL = VAL*VAR_CON[i]
        elif VAR == var and DIR == -1:
            VAL = VAL/VAR_CON[i]
    return VAL


def gamma_from_n(n):
    """
    Polytrope power that
    emerges in astrophysical polytrope equation.

    Parameters
    ----------
    n : Float
        polytrope index. (Goes from 0 to inf,
                          depends on the model)

    Returns
    -------
    Float
        Polytrope power

    """
    return (n+1)/n


def kappa_from_r0rho0n(r0, rho0, n):
    """
    Calculates constant of proportionality that
    emergences in polytrope equation.

    Parameters
    ----------
    r0 : Int
        Approximate radius of astrophysical body.
    rho0 : Float
        Center energy density.
    n : Float
        Polytrope index.

    Returns
    -------
    Float
        Constant of proportionality.

    """
    k = (r0**2*4*np.pi*rho0**(1-1/n))/(n+1)
    if k == 0:
        k += 1
    return k


def kappa_from_p0rho0(p0, rho0, G):
    """
    Calculates constant of proportionality that
    emergences in polytrope equation.

    Parameters
    ----------
    p0 : Float
        Pressure at some radius r.
    rho0 : Float
        Energy density at r.
    G : Float
        Polytrope power.

    Returns
    -------
    Float
        Constant of proportionality.

    """
    k=0
    if G != 0 and rho0 != 0:
        k = (p0)/(rho0**G)
    elif k == 0:
        k += 1e-6
    return k


def kappa_choiser(kappa_choise, p_K, rho_K, Gamma, R_body, n):
    if kappa_choise == 0:
        Kappa = kappa_from_p0rho0(p_K, rho_K, Gamma)
    elif kappa_choise == 1:
        Kappa = kappa_from_r0rho0n(R_body, rho_K, n)
    else:
        Kappa = kappa_choise
    return Kappa


def find_radius(p_t, r_t, raja=0):
    """
    Finds the radius corresponding to the boundary of the star.
    The limit can be defined in the function call or
    in the function definition.

    Parameters
    ----------
    p_t : Array
        Pressure values.
    r_t : Array
        Radius values.
    raja : Float 
        Defined boundary of a star. The default is 0.

    Returns
    -------
    R_raja : Float
        A radius corresponding to the star's boundary.

    """
    p = p_t[0]
    R_raja = r_t[0]
    p_raja = raja*p
    i = 0
    while i < len(p_t):
        p = p_t[i]
        R_raja = r_t[i]
        i += 1
        if p < p_raja:
            break
    print(R_raja)
    return R_raja


def find_mass_in_radius(m_t, r_t, r_raja):
    """
    Finds the mass within the radius of the star.

    Parameters
    ----------
    m_t : Array
        Mass values.
    r_t : Array
        Radius values.
    r_raja : Float
        Boundary of the star.

    Returns
    -------
    m : Float
        Mass within the radius of the star.

    """
    r = r_t[0]
    m = m_t[0]
    i = 0
    while i < len(r_t):
        r = r_t[i]
        m = m_t[i]
        i += 1
        if r > r_raja:
            break
    print(m)
    print("==========")
    return m

def Ricci_scalar(p, rho, r):
    """
    Compute the curvature scalar of space - the Ricci scalar
    and plots it. 
    Params must be in geometrizied units:
        [p] = m**-2 ja [rho] = m**-2
    Plots in GeVs.

    Parameters
    ----------
    p : Array
        Pressure solution.
    rho : Array
        Energy density solution.

    Returns
    -------
    None.

    """
    R_scalar = -8*np.pi*(rho - 3*p)
    graph(r, unit_conversion(3, "P", R_scalar, -1)*1e-9, plt.plot,
          "Scalar curvature", "Radius, r", "Ricci scalar, R (eV)", 
          'linear', "Scalar curvature inside neutron star \n")
    
    
def Ricci_scalar(p, rho, r):
    """
    Compute the curvature scalar of space - the Ricci scalar
    and plots it. 
    Params must be in geometrizied units:
        [p] = m**-2 ja [rho] = m**-2
    Plots in GeVs.

    Parameters
    ----------
    p : Array
        Pressure solution.
    rho : Array
        Energy density solution.

    Returns
    -------
    None.

    """
    R_scalar = -8*np.pi*(rho - 3*p)
    graph(r, unit_conversion(3, "P", R_scalar, -1)*1e-9, plt.plot,
          "Scalar curvature", "Radius, r", "Ricci scalar, R (eV)", 
          'linear', "Scalar curvature inside neutron star \n")


