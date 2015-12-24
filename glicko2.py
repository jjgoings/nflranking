# just a test to put glicko2 into matrix-vector products 
# following the examples given in the paper
from __future__ import division
import numpy as np
import math

'''
Okay, there are two main parts to this. The first requires taking raw game 
data and making the mu, phi, sig, s, and p arrays. The second then takes 
that data and updates them. 

In practice, we need to also take time data and previous rankings into account.
Thus, this work really only performs a step of the overall ranking process.
Looking WAY into the future, I'd also like to get a plot made of each team's
Gaussian distributed rank and uncertainty. I think it could be the most
informative way to present the data. 

'''
rating = np.array([1500,1400,1550,1700])
RD = np.array([200,30,100,300])
sig = np.array([0.06,0.06,0.06,0.06])

def convert_rating(rating):
    return (rating - 1500)/173.7178

def convert_RD(RD):
    return RD/173.7178

#s = np.array([[0.,1.,0.,0.],[0.,0.,0.,0.],[1.,0.,0.,0.],[1.,0.,0.,0.]])
#p = np.array([[0.,1.,1.,1.],[1.,0.,0.,0.],[1.,0.,0.,0.],[1.,0.,0.,0.]])

#####

def g(phi):
    gphi = 1./(np.sqrt(1. + 3.*phi*phi/(math.pi*math.pi)))
    return gphi

def make_E(gphi,mu,num_teams):
    E = np.zeros((num_teams,num_teams))
    for i in range(num_teams):
        for j in range(num_teams):
            E[i,j] += 1./(1. + np.exp(-gphi[j]*(mu[i] - mu[j]))) 
    return E

def make_v(p,gphi,E,num_teams):
    v = np.zeros(num_teams)
    for i in range(num_teams):
        for j in range(num_teams):
            v[i] += (p[i,j]*gphi[j]*gphi[j]*E[i,j]*(1. - E[i,j]))
    # careful...we could get div by zero. need to change but ok for now since graph is fully connected.
        if abs(v[i]) > 0.000000001: 
            v[i] = 1./v[i]
    return v

def make_delta(p,gphi,E,s,v,num_teams):
    delta = np.zeros(num_teams)
#    for i in range(4):
#        for j in range(4):
    for i in range(num_teams):
        for j in range(num_teams):
            delta[i] += gphi[j]*(s[i,j]-E[i,j])*p[i,j]
        delta[i] *= v[i]
    return delta

def f(x,i,tau,sig,phi,delta,v):
    A = np.exp(x)*(delta[i]**2 - phi[i]**2 - v[i] - np.exp(x))
    B = 2.0*(phi[i]**2 + v[i] + np.exp(x))**2
    C = x - np.log(sig[i]**2)
    D = tau**2  
    return (A/B - C/D)

def update_vol(i,tau,eps,sig,phi,delta,v):
    a = np.log(sig[i]**2)
    A = a
    if delta[i]**2 > phi[i]**2 + v[i]:
        B = np.log(delta[i]**2 - phi[i]**2 -v[i])
    elif delta[i]**2 <= phi[i]**2 + v[i]:
        k = 1
        while f(a - k*tau,i,tau,sig,phi,delta,v) < 0:
            k += 1
        B = a - k*tau
    fA = f(A,i,tau,sig,phi,delta,v)
    fB = f(B,i,tau,sig,phi,delta,v)
    while np.absolute(B - A) > eps:
        C = A + (A-B)*fA/(fB-fA)
        fC = f(C,i,tau,sig,phi,delta,v)
        if fC*fB < 0:
            A = B
            fA = fB
        else:
            fA = 0.5*fA
        B = C
        fB = fC
    return np.exp(A*0.5)

def update_phi(i,phi,sig,v):
    phi_new = np.sqrt(phi[i]**2 + sig[i]**2)
    if v[i] == 0.0:
        v[i] = 0.00000001
    phi_new = 1./(np.sqrt((1./phi_new**2) + 1./v[i]))
    return phi_new

def update_mu(i,mu,phi_new,gphi,s,E,p,num_teams):
    mu_new = mu[i]
    for j in range(num_teams):
        mu_new += (phi_new[i]**2)*gphi[j]*(s[i,j] - E[i,j])*p[i,j]
    return mu_new

def main_update(rating,RD,sig,s,p,tau,eps):
    # Step 2
    mu = convert_rating(rating)
    phi = convert_RD(RD)
    num_teams = len(mu)
    # Step 3
    gphi = g(phi)
    E = make_E(gphi,mu,num_teams)
    v = make_v(p,gphi,E,num_teams)
    # Step 4
    delta = make_delta(p,gphi,E,s,v,num_teams)
    # Step 5
    new_vol = np.zeros(num_teams)
    new_mu = np.zeros(num_teams)
    new_phi  = np.zeros(num_teams)
    for i in range(num_teams):
        new_vol[i] = update_vol(i,tau,eps,sig,phi,delta,v)
        new_phi[i]  = update_phi(i,phi,new_vol,v)
        new_mu[i] = update_mu(i,mu,new_phi,gphi,s,E,p,num_teams)
    return new_mu*173.7178 + 1500, new_phi*173.7178, new_vol, E

#new_rank,new_RD,new_vol = main_update(rating,RD,sig,s,p,0.5,0.00000000001) 
