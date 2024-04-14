% A High-dimensional Additive Nonparametric Model (AHANM main estimation file)
% DGP equation 56 from 'MARS' by Friedman 1991
clear; close all; clc;
rng('shuffle');  % Seed the random number generator based on the current time

n = 800; % Sample size
%n = 200;

p = 10; % number of components, not including the global intercept

X = cell(p+1,1);
X{1} = zeros(n,1); % global intercept
for xi = 2:p+1
    X{xi} = rand(n,1);
end

e = normrnd(0,1,[n,1]);
X_mat = reshape(cell2mat(X),n,p+1);
Y = X{1};

experiment = 5;
switch experiment
    case 5
        Y = Y+0.1*exp(4*X{2})+4./(1+exp(-20*(X{3}-.5)))+3*X{4}+2*X{5}+1*X{6}+0*X{7}+0*X{8}+0*X{9}+0*X{10}+0*X{11};
end
Y = Y + e;

[b, fitinfo] = lasso(X_mat,Y,'CV',10,'Intercept',true); 
lam = fitinfo.LambdaMinMSE;

% prior
a0 = cell(1, p+1);
Va = cell(1, p+1);
iVa = cell(1, p+1);
for i = 1:p+1
    a0{i} = zeros(1);
    Va{i} = 100*speye(1);
    iVa{i} = Va{i}\speye(1);
end

nusig2 = 3; Ssig2 = .1*(nusig2-1);
p1 = cell(1, p); 
p2 = cell(1, p);
Vtau = cell(1, p);
taubar = cell(1, p);
mutau = cell(1, p);

for p1i = 1:p
    p1{p1i} = .5;
    p2{p1i} = .5;
    Vtau{p1i} = sqrt(5e-6);
    taubar{p1i} = min(Vtau{p1i}/2,.1);
    mutau{p1i} = Vtau{p1i};
end

% construct a few things 
D = cell(1, p); 
A = cell(1, p);
G = cell(1, p);
k = cell(1, p);
DD = cell(1, p);
GG = cell(1, p);
Xa = cell(1, p);

for Di = 1:p
    D{Di} = selectionM(X{Di+1});
    A{Di} = uniquetol(X{Di+1},.001); % sort data
    G{Di} = GM3(X{Di+1});
    [~,k{Di}] = size(D{Di});
    DD{Di} = D{Di}'*D{Di};
    GG{Di} = G{Di}'*G{Di};
    Xa{Di} = G{Di}\[eye(1); sparse(k{Di}-1,1)];
end

% initialize for storage and the Markov chain
d = cell(1, p);
lp_d = cell(1, p);
theta = cell(1, p);
gam = cell(1, p);
irho = cell(1, p);
tau = cell(1, p);
pp = cell(1, p);
for di = 1:p
    d{di} = 0;
    lp_d{di} = zeros(2,1);
    theta{di} = A{di};
    gam{di} = zeros(k{di},1);
    irho{di} = 1/100;
    tau{di} = .05;
    pp{di} = .5;
end

a = cell(1, p+1);
for i = 1:p+1
    a{i} = zeros(1);
end

sig2=var(Y); 

nsim=20000; burnin=2000;

store_para0 = zeros(nsim,2);
store_para = cell(1, p);
store_theta = cell(1, p);

for storei = 1:p
    store_para{storei} = zeros(nsim, 4);
    store_theta{storei} = zeros(nsim, k{storei});
end

store_d = zeros(nsim+burnin, p);

disp('Starting MCMC.... ');
tic;  % Start timer
    
for isim = 1:nsim+burnin

    mu_ahat = cell(1, p);
    mu_a = cell(1, p);
    Kgam = cell(1, p);
    YY = cell(1, p);
    gamhat = cell(1, p);
    prob_d = cell(1, p);
    Ktau = cell(1, p);
    tauhat = cell(1, p);
    Ka = cell(1, p);
    ahat = cell(1, p);


    for j = 1:p
        Dtheta = zeros(n,1);
        for jj = 1:p
            Dtheta = Dtheta + D{jj}*theta{jj};
        end

        % sample theta
        YY{j} = Y - Dtheta + D{j}*theta{j} - a{1};
        mu_ahat{j} = [a{j+1};sparse(k{j}-1,1)];
        mu_a{j} = G{j}\mu_ahat{j};
        Kgam{j} = GG{j}+d{j}*tau{j}^2*DD{j}/sig2;

        gamhat{j} = Kgam{j}\(d{j}*tau{j}*D{j}'*(YY{j}-D{j}*mu_a{j})/sig2);
        gam{j} = gamhat{j}+chol(Kgam{j},'lower')'\randn(k{j},1);
        theta{j} = mu_a{j}+d{j}*tau{j}*gam{j};

        % sample d and tau
        lp_d{j}(1) = lpost_d(0,YY{j},D{j}*gam{j},D{j}*mu_a{j},sig2,Vtau{j},taubar{j},pp{j});
        lp_d{j}(2) = lpost_d(1,YY{j},D{j}*gam{j},D{j}*mu_a{j},sig2,Vtau{j},taubar{j},pp{j});
        prob_d{j} = exp(lp_d{j} - max(lp_d{j}));
        prob_d{j} = prob_d{j}/sum(prob_d{j});
        d{j} = prob_d{j}(2) > rand;

        Ktau{j} = 1/Vtau{j}+d{j}*gam{j}'*DD{j}*gam{j}/sig2;
        tauhat{j} = Ktau{j}\(d{j}*gam{j}'*D{j}'*(YY{j}-D{j}*mu_a{j})/sig2+(1/Vtau{j})*mutau{j});
        tau{j} = tnormrnd(tauhat{j},1/Ktau{j},taubar{j},1e5);

        % sample pp
        pp{j} = betarnd(p1{j}+sum(store_d(:,1)),p2{j}+isim-sum(store_d(:,1)));

        % sample a and irho (Bayesian Lasso)
        Ka{j} = Va{j+1}\speye(1)+(1/sig2)*Xa{j}'*DD{j}*Xa{j};
        ahat{j} = Ka{j}\(iVa{j+1}*a0{j+1}+(1/sig2)*Xa{j}'*D{j}'*(YY{j}-d{j}*tau{j}*D{j}*gam{j}));
        a{j+1} = ahat{j}+chol(Ka{j},'lower')'\randn(1,1); 

        %Ka{j} = (1/sig2)*(irho{j}+Xa{j}'*DD{j}*Xa{j});
        %ahat{j} = (Xa{j}'*D{j}'*(YY{j}-d{j}*tau{j}*D{j}*gam{j}))/(irho{j}+Xa{j}'*DD{j}*Xa{j});
        %a{j+1} = ahat{j}+chol(Ka{j},'lower')'\randn(1,1); 
        %irho{j} = igaurnd(lam^2,(lam^2*sig2/a{j+1}^2)^(1/2),1);
    end

    % sample global intercept a{1}
    Y0 = Y;
    for jjj = 1:p
        Y0 = Y0 - D{jjj}*theta{jjj};
    end
    Ka0 = Va{1}\speye(1)+(1/sig2)*ones(n,1)'*ones(n,1);
    ahat0 = Ka0\(iVa{1}*a0{1}+(1/sig2)*ones(n,1)'*Y0);
    a{1} = ahat0+chol(Ka0,'lower')'\randn(1,1);

    % sample sig2
    e0 = Y0-a{1};
    sig2 = 1/gamrnd(nusig2+n/2,1/(Ssig2+e0'*e0/2));
    
    % store the parameters
    store_d(isim,:) = cell2mat(d);
    
    if isim > burnin
        isave = isim - burnin;
        store_para0(isave,:) = [a{1} sig2];

        for isavei = 1:p
            store_para{isavei}(isave,:) = [a{isavei+1} d{isavei} tau{isavei} pp{isavei}];
            store_theta{isavei}(isave,:) = theta{isavei}';
        end
    end
    
    if (mod(isim, 2000) == 0)
        disp([num2str(isim) ' loops...']);
    end
end

ElapsedTime = toc; % Elapsed time
disp(['MCMC takes ', num2str(ElapsedTime), ' seconds.']);

para_hat0 = mean(store_para0)';
para0_CI = quantile(store_para0,[.05 .95])';

para_hat = cell(1, p);
para_CI = cell(1, p);
theta_hat = cell(1, p);
slope = cell(1, p);
slope_CI = cell(1, p);

for parai = 1:p
    para_hat{parai} = mean(store_para{parai})';
    para_CI{parai} = quantile(store_para{parai},[.05 .95])';
    theta_hat{parai} = mean(store_theta{parai})';
    slope{parai} = full(theta_hat{parai}(1)/A{parai}(1));
    slope_CI{parai} = full(quantile(store_theta{parai}(:,1)/A{parai}(1),[.05 .95])');
end

intercept = para_hat0;
intercept_CI = quantile(store_para0,[.05 .95])';

subplot(3,2,1)
plot(A{1},theta_hat{1}-sum(theta_hat{1}-0.1*exp(4*A{1}))/k{1},'bo','MarkerIndices',1:10:length(theta_hat{1}))
hold on
plot(A{1}, 0.1*exp(4*A{1}),'r+','MarkerIndices',1:10:length(A{1}))
title('1st Component')
legend({'Estimates','True'},'Location','southeast')

subplot(3,2,2)
plot(A{2},theta_hat{2}-sum(theta_hat{2}-4./(1+exp(-20*(A{2}-.5))))/k{2},'bo','MarkerIndices',1:10:length(theta_hat{2}))
hold on
plot(A{2}, 4./(1+exp(-20*(A{2}-.5))),'r+','MarkerIndices',1:10:length(A{2}))
%plot(A{3}, 0*A{3},'r+','MarkerIndices',1:10:length(A{3}))
title('2nd Component')
%axis([0 1 -1 1])

subplot(3,2,3)
plot(A{3},theta_hat{3},'bo','MarkerIndices',1:10:length(theta_hat{3}))
%plot(A{4},theta_hat{4}-sum(theta_hat{4}-7*sin(A{4}.^2))/k{4},'bo','MarkerIndices',1:10:length(theta_hat{4}))
%plot(A{4},theta_hat{4}-sum(theta_hat{4}-7*cos(A{4}.^3+pi/2))/k{4},'bo','MarkerIndices',1:10:length(theta_hat{4}))
%plot(A{4},theta_hat{4}-sum(theta_hat{4}-7*cos(A{4}.^2+pi/2))/k{4},'bo','MarkerIndices',1:10:length(theta_hat{4}))
hold on
plot(A{3}, 3*A{3},'r+','MarkerIndices',1:10:length(A{3}))
%plot(A{4}, 7*sin(A{4}.^2),'r+','MarkerIndices',1:10:length(A{4}))
%plot(A{4}, 7*cos(A{4}.^3+pi/2),'r+','MarkerIndices',1:10:length(A{4}))
%plot(A{4}, 7*cos(A{4}.^2+pi/2),'r+','MarkerIndices',1:10:length(A{4}))
title('3rd Component')
%axis([0 1 -1 1])

subplot(3,2,4)
plot(A{4},theta_hat{4},'bo','MarkerIndices',1:10:length(theta_hat{4}))
%plot(A{5},theta_hat{5}-sum(theta_hat{5}-9*exp(-30*(A{5}-0.5).^2))/k{5},'bo','MarkerIndices',1:10:length(theta_hat{5}))
%plot(A{5},theta_hat{5},'bo','MarkerIndices',1:10:length(theta_hat{5}))
hold on
plot(A{4}, 2*A{4},'r+','MarkerIndices',1:10:length(A{4}))
%plot(A{5}, 9*exp(-30*(A{5}-0.5).^2),'r+','MarkerIndices',1:10:length(A{5}))
%plot(A{5}, 9*log(0*A{5}.^2+1),'r+','MarkerIndices',1:10:length(A{5}))
title('4th Component')
%axis([0 1 -1 1])

subplot(3,2,5)
plot(A{6},theta_hat{6},'bo','MarkerIndices',1:10:length(theta_hat{6}))
hold on
%plot(A{7}, 13*A{7},'r+','MarkerIndices',1:10:length(A{7}))
plot(A{6}, 0*A{6},'r+','MarkerIndices',1:10:length(A{6}))
title('6th Component')
axis([0 1 -1 1])

subplot(3,2,6)
plot(A{8},theta_hat{8},'bo','MarkerIndices',1:10:length(theta_hat{8}))
hold on
%plot(A{9}, 17*A{9},'r+','MarkerIndices',1:10:length(A{9}))
plot(A{8}, 0*A{8},'r+','MarkerIndices',1:10:length(A{8}))
title('8th Component')
axis([0 1 -1 1])

params = cell(2+3*p, 4);

params{1,1} = {'intercept'};
params{2,1} = {'sigma2'};
params{1,2} = intercept(1);
params{1,3} = intercept_CI(1,1);
params{1,4} = intercept_CI(1,2);
params{2,2} = intercept(2);
params{2,3} = intercept_CI(2,1);
params{2,4} = intercept_CI(2,2);


for dsi = 1:p
    dParamName = sprintf('d%d', dsi);  % Create parameter name like 'd1', 'd2', etc.
    slopeParamName = sprintf('slope%d', dsi);  % Create parameter name like 'slope1', 'slope2', etc.
    tauParamName = sprintf('tau%d', dsi);
    params{2+dsi,1} = dParamName;    % Allocate to params array
    params{2+p+dsi,1} = slopeParamName;    % Allocate to params array
    params{2+2*p+dsi,1} = tauParamName; 
    params{2+dsi,2} = para_hat{dsi}(2);
    params{2+dsi,3} = para_CI{dsi}(2,1);
    params{2+dsi,4} = para_CI{dsi}(2,2);
    params{2+p+dsi,2} = slope{dsi};
    params{2+p+dsi,3} = slope_CI{dsi}(1);
    params{2+p+dsi,4} = slope_CI{dsi}(2);
    params{2+2*p+dsi,2} = para_hat{dsi}(3);
    params{2+2*p+dsi,3} = para_CI{dsi}(3,1);
    params{2+2*p+dsi,4} = para_CI{dsi}(3,2);
end
