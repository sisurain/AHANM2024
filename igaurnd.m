% inverse-Gaussian random number generator
function x = igaurnd(psi,mu,n)
nu0 = randn(n,1).^2;
x1 = mu + mu.^2.*nu0./(2*psi) ...
- mu./(2*psi).*sqrt(4*mu.*psi.*nu0 + mu.^2.*nu0.^2);
x2 = mu.^2./x1;
p = mu./(mu+x1);
U = p>rand(n,1);
x = U.*x1 + (1-U).*x2;
end