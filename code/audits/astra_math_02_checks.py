# ---------------------------------------------------------------------------
# VERIFICATION RECORD -- copied unchanged into code/audits/ as a durable audit
# artefact.  Only this header was added.
# origin  : review_packet/astra/math_02_checks.py
# author  : Astra (external reviewer), 2026-09-10
# date    : 2026-09-10
# seed    : numpy default_rng(902)
# command : venv/bin/python3 code/audits/astra_math_02_checks.py
# ---------------------------------------------------------------------------
import numpy as np
from scipy.linalg import eigh
from scipy.optimize import brentq
from scipy.integrate import solve_ivp
from scipy.special import expit
rng=np.random.default_rng(902)
# Finite-horizon attribution, noncommuting PSD matrices, exact spectral integral.
for j in range(200):
 n=5; X=rng.normal(size=(n,n)); Y=rng.normal(size=(n,n))
 U=X@X.T; V=Y@Y.T+0.2*np.eye(n); a=np.linalg.eigvalsh(U)[-1]; k=np.linalg.eigvalsh(V)[0]
 g,Q=eigh(U+V); x=Q.T@rng.normal(size=n); Ut=Q.T@U@Q
 for T in [1e-4,.01,.3,4]:
  integ=-np.expm1(-(g[:,None]+g[None,:])*T)/(g[:,None]+g[None,:])
  E=np.sum(Ut*np.outer(x,x)*integ)
  total=np.sum(x*x*(-np.expm1(-2*g*T)))/2
  assert E/total <= a/(a+k)+1e-12
print('finite-horizon attribution: 800 checks passed')
# General initial encoder, signed context, unequal head initial values with sum zero.
worst=0
for M in [1,7,64,512]:
 for a0,v0 in [(-.7,-.8),(.3,1.2),(1.7,.4)]:
  m=2.9; Delta=m-a0
  u=brentq(lambda u:a0+u+np.sqrt(M)*v0*v0/2*np.sinh(2*np.sqrt(M)*u)-m,0,Delta/(1+M*v0*v0))
  vf=v0*np.cosh(np.sqrt(M)*u); bf=np.sqrt(M)*v0*np.sinh(np.sqrt(M)*u)
  assert u <= Delta/(1+M*v0*v0)+1e-12
  assert abs(vf-v0)<=Delta*Delta/(2*M*abs(v0)**3)+1e-12
  beta=rng.normal(scale=.01,size=M); beta-=beta.mean()
  y0=np.r_[a0,v0,beta]
  def fun(t,y):
   a,v=y[:2]; b=y[2:].sum(); r=expit(-a-b*v)
   return np.r_[r,b*r,np.full(M,v*r)]
  def evt(t,y):return y[0]+y[1]*y[2:].sum()-m
  evt.terminal=True
  sol=solve_ivp(fun,[0,100],y0,events=evt,rtol=2e-10,atol=1e-12)
  last=sol.y[:,-1]; err=np.max(np.abs(last[:2]-[a0+u,vf]));worst=max(err,worst)
  assert len(sol.t_events[0])==1 and err<1e-7
  assert np.max(abs((last[2:]-last[2:].mean())-beta))<1e-10
print('general-initialization toy: 12 full-parameter solves passed; max a/v error',worst)
# BN derivative, inverse-square curvature law, and radial attenuation.
N=8; d=3; H=rng.normal(size=(N,d)); w=rng.normal(size=d); P=np.eye(N)-np.ones((N,N))/N
for eps in [0.,1e-3]:
 def calc(w,ep):
  t=P@H@w; s2=t@t/N; sig=np.sqrt(s2+ep)
  z=t/sig; D=(P-np.outer(t,t)/(N*(s2+ep)))/sig
  jac=D@H; prob=expit(z); G=jac.T@(prob[:,None]*(1-prob[:,None])*jac)/N
  return z,D,G,t,s2
 z,D,G,t,s2=calc(w,eps)
 assert np.linalg.norm(D@np.ones(N))<1e-12
 assert np.allclose(D@t,eps*t/(s2+eps)**1.5,atol=1e-12)
 for c in [.2,2.,5.]:
  zz,DD,GG,tt,ss=calc(c*w,c*c*eps)
  assert np.allclose(zz,z) and np.allclose(GG,G/c**2)
print('BN derivative and exact co-scaled-epsilon curvature law passed')
# ReLU tube inequalities, varying all three parameter groups.
for M in [32,128]:
 N=5; K=2; dx=3; C=2; xx=rng.normal(size=(N,dx)); theta=rng.normal(size=(K,dx)); W=rng.normal(size=(M,K)); b=rng.normal(size=M); V=rng.normal(size=(C,M))/np.sqrt(M)
 X=np.max(np.linalg.norm(xx,axis=1)); Z0=np.sqrt(1+(np.linalg.norm(theta,2)*X)**2); Z=Z0+X
 Rth=.3; Rph=.2
 def outputs(th,ww,bb,vv):
  pre=xx@th.T@ww.T+bb; gate=(pre>0); H=np.maximum(pre,0)/np.sqrt(N)
  Jth=np.stack([np.einsum('ci,ik,l->ckl',vv*gate[n],ww,xx[n]).reshape(C,-1) for n in range(N)]).reshape(N*C,-1)/np.sqrt(N)
  return H,Jth,gate
 H0,J0,g0=outputs(theta,W,b,V); tau=(X*np.linalg.norm(W,axis=1)+Z*Rph)/np.sqrt(M)
 Bmass=np.sum(np.linalg.norm(V,axis=0)*np.linalg.norm(W,axis=1)*np.any(abs(xx@theta.T@W.T+b)<=tau,axis=0))
 for rep in range(100):
  dt=rng.normal(size=theta.shape);dt*=Rth/M/np.linalg.norm(dt)
  arr=rng.normal(size=W.size+b.size+V.size);arr*=Rph/np.sqrt(M)/np.linalg.norm(arr)
  dw=arr[:W.size].reshape(W.shape);db=arr[W.size:W.size+b.size];dv=arr[-V.size:].reshape(V.shape)
  H1,J1,g1=outputs(theta+dt,W+dw,b+db,V+dv)
  assert np.linalg.norm(H1-H0)<=X*np.linalg.norm(W)*np.linalg.norm(dt)+Z*np.sqrt(np.linalg.norm(dw)**2+np.linalg.norm(db)**2)+1e-12
  bound=X*(np.linalg.norm(dv)*np.linalg.norm(W)+np.linalg.norm(V+dv)*np.linalg.norm(dw)+Bmass)
  assert np.linalg.norm(J1-J0,2)<=bound+1e-12
print('ReLU feature/gate tube bounds: 200 random perturbations passed')
