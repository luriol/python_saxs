# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 20:57:07 2019

@author: th0lxl1
"""
from numpy import sin, conj, pi, cos
from numpy import arctan as atan
from numpy import absolute

def vrij_2r(q_mag, volume_fraction, intensity):
    """ vrij_2r(q_mag, volume_fraction, intensity)
    Keyword arguments:
    q_mag -- scattering wavevector
    volume_fraction -- volume fraction of spheres
    intensity -- normalization factor
    radii and sigma for shultz distribution fixed at best fit
    values
    """
    radius1 = 7.8
    radius2 = 6.3
    sigma = .183
    ical1 = vrij_intensity(q_mag, volume_fraction, radius1, sigma)
    ical2 = vrij_intensity(q_mag, volume_fraction/20, radius1, sigma)
    ical3 = vrij_intensity(q_mag, volume_fraction/20, radius2, sigma)
    yfit = ical1*ical3/ical2
    yfit = yfit*intensity
    return yfit

def vrij_intensity(q_mag, volume_fraction, radius, sigma):
    """vrij_intensity(q_mag, arg_in)

    Keyword arguments:
    q_mag -- scattering wavevector
    volume_fraction -- volume fraction of spheres
    sigma -- schults distribution width
    """
    # Translated from the Matlab, Laurence Luri, 1/26/2019
    # VRIJI(x,a)
    #
    # vrij fitting functio, identical to vrij except fits total intensity, not
    # S(Q)
    #
    # Laurence Lurio, May 24, 2016
    # Cleaned up May 26, 2016
    # Removed bug June 6, 2016
    # The vrij S(q) function with analytic
    #integrals from Griffith.
    # Here K is q/a, phi is the
    # volume fraction and sigma is the shultz distribution width paramter
    # A. Vrij, Journal of Chemical Physics, 71, 8 15th October 1979
    # W. L. Griffith, R. Trioolo and A. L. Compere, PRA 33, 2197, 1986
    #
    # For now, K is not an array,
    #but that can probably be fixed
    b_b = sigma**2
    #parameters for Griffith integrals
    c_c = 1/b_b
    xi2 = volume_fraction/(1+2*sigma**2)
    #integrals which don't depend on K
    inorm = volume_fraction/(1+2*sigma**2)/(1+sigma**2)
    #Normalization for integral averages
    yout = q_mag*0
    for i, qval in enumerate(q_mag):
        k_vec = 2*qval*radius
        f_11 = 1-volume_fraction+inorm*24*(gpsi(k_vec, b_b, c_c)/2-\
        k_vec*(1+gchip(k_vec, b_b, c_c))/4)/k_vec**3
        # eqx 38 from vrij.  volume_fraction is the same as xi3
        f_11 = f_11 + 1j*inorm*6*(2-2*gchi(k_vec, b_b, c_c)-\
        k_vec*gpsip(k_vec, b_b, c_c))/k_vec**3
        f_12 = inorm*(6/k_vec**3)*(2*gpsip(k_vec, b_b, c_c)-\
        k_vec*(1+b_b)-k_vec*gchipp(k_vec, b_b, c_c))+ \
        1j*inorm*(6/k_vec**3)*(2-2*gchip(k_vec, b_b, c_c)-\
        k_vec*gpsipp(k_vec, b_b, c_c))
        f_22 = 1-volume_fraction+3*inorm*\
        (gpsipp(k_vec, b_b, c_c)/k_vec+\
        1j*(1+b_b-gchipp(k_vec, b_b, c_c))/k_vec)
        f_21 = .5*(1-volume_fraction)*(1j*k_vec)-3*xi2+ \
        3*inorm*(gpsip(k_vec, b_b, c_c)/k_vec+1j*\
        (1-gchip(k_vec, b_b, c_c))/k_vec)
        t_1 = f_11*f_22-f_12*f_21
        # T function eq. 48
        p_1 = 4*pi*f_12
        # this is only true for a particular choice of fB!
        p_2 = 4*pi*inorm*((6/k_vec**3)*(2*gpsi(k_vec, b_b, c_c)\
        -k_vec-k_vec*gchip(k_vec, b_b, c_c)) + \
        1j*(6/k_vec**3)*(2-2*gchi(k_vec, b_b, c_c)-\
        k_vec*gpsip(k_vec, b_b, c_c)))
        t_2 = f_21*p_1-f_22*p_2
        t_3 = f_12*p_2-f_11*p_1
        i_1 = (inorm*1152*pi**2/k_vec**6)*\
        (4-4*gchi(k_vec, b_b, c_c)-4*k_vec*gpsip(k_vec, b_b, c_c) + \
         k_vec**2*(1+b_b)+k_vec**2*gchipp(k_vec, b_b, c_c))
        i_2 = i_1/16/pi**2
        i_3 = inorm*2*(1+b_b-gchipp(k_vec, b_b, c_c))/k_vec**2
        i_4 = 4*pi*i_2
        i_5 = (inorm*48*pi/k_vec**4)*(2-2*gchip(k_vec, b_b, c_c)-\
        k_vec*gpsipp(k_vec, b_b, c_c))
        i_6 = i_5/4/pi
        d_f = (6/pi)*(i_1*absolute(t_1)**2+i_2*absolute(t_2)**2+\
        9*i_3*absolute(t_3)**2+\
        i_4*(t_1*conj(t_2)+t_2*conj(t_1))+\
        3*i_5*(t_1*conj(t_3)+t_3*conj(t_1))+\
        3*i_6*(t_2*conj(t_3)+t_3*conj(t_2)))
        delta = (absolute(f_11*f_22-f_12*f_21))**2
        # Eq. 39, Leave off the (1-xi3)**4 as it cancels with delta
        yout[i] = absolute(d_f/delta)
        # Don't know where the factor of 1.9156 comes from.
        # This is just a fix to make S(infty)=1
        #yout[i]=yout[i]/i_1/2
        # Comment out this last line to fit I(q)
    return yout


def gpsi(k_k, b_b, c_c):
    """gpsi(k_k, b_b, c_c)"""
    yout = nu_f(1, b_b, k_k)**(c_c/2)*sin(c_c*atan(b_b*k_k))
    return yout


def gpsip(k_k, b_b, c_c):
    """gpsip(k_k, b_b, c_c)"""
    yout = b_b*c_c*nu_f(1, b_b, k_k)**((c_c+1)/2)*\
    sin((c_c+1)*atan(b_b*k_k))
    return yout

def gpsipp(k_k, b_b, c_c):
    """gpsipp(k_k, b_b, c_c)"""
    yout = b_b**2*c_c*(c_c+1)*nu_f(1, b_b, k_k)**((c_c+2)/2)*\
    sin((c_c+2)*atan(b_b*k_k))
    return yout

def gchi(k_k, b_b, c_c):
    """gchi(k_k, b_b, c_c)"""
    yout = nu_f(1, b_b, k_k)**(c_c/2)*cos(c_c*atan(b_b*k_k))
    return yout

def gchip(k_k, b_b, c_c):
    """gchip(k_k, b_b, c_c)"""
    yout = b_b*c_c*nu_f(1, b_b, k_k)**((c_c+1)/2)*\
    cos((c_c+1)*atan(b_b*k_k))
    return yout

def gchipp(k_k, b_b, c_c):
    """gchipp(k_k, b_b, c_c)"""
    yout = b_b**2*c_c*(c_c+1)*nu_f(1, b_b, k_k)**((c_c+2)/2)\
    *cos((c_c+2)*atan(b_b*k_k))
    return yout

def nu_f(m_m, b_b, k_vec):
    """nu(m_m, b_b, c_c)"""
    yout = 1/(m_m**2 + (b_b*k_vec)**2)
    return yout
