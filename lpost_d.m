function lden = lpost_d(d,y,Dgam,Dmu,sig2,Vtau,taubar,p)
mutau = 0; % prior mean of tau is set to be 0
n = size(y,1);
if d == 0
    Ktau = 1/Vtau;
    tau_hat = mutau;
else
    Ktau = 1/Vtau + d*(Dgam'*Dgam)/sig2;
    tau_hat = Ktau\(d*Dgam'*(y-Dmu)/sig2);    
end
c1 = 1-normcdf((taubar-tau_hat)*Ktau);
c2 = 1-normcdf((taubar-mutau)/Vtau);
lden = -n/2*log(2*pi*sig2) + log(c1/c2) - .5*log(Vtau) - .5*log(Ktau) ...
    - .5/sig2*(y-Dmu)'*(y-Dmu) - .5*mutau^2/Vtau + .5*tau_hat^2*Ktau ...
    + d*log(p) + (1-d)*log(1-p);
end 