

function [XSM,xt_res,L,ENL]= MSLM_INT(NE,data,NELIN,NLEVEL,NSMT,ires,inorm,iSYM,iNL,inEQ,indrandperm)
% This function performs long-term simulation by using multi-level stochastic Stuart-Landau modeling
% enabled by DAHD analysis
% Input: data - array of DAHCs from DAHD analysis
%        NE - max time steps to simulate
%        NELIN - model spec: 1 - SL, 0 - linear  
%        NLEVEL - number of levels
%        NSMT - number of stochastic realizations
%        ires - noise specification: 0 - correlated white noise, 1-
%        perumation of last level regression residuals
%        inorm - data normaliztaion, 0 or 1
%        iSYM,iNL,inEQ - parameters for SL model constraints
%        indrandperm - array of random indices for stochastic ensemble
% Output: 
%         array  XSM - simulated data
%         array  xt_res - regression residuals at last model level
%         arrays L and ENL - arrays of model coeffcients
%  
%  Kondrashov D., et al. 2020 Data-adaptive harmonic analysis of oceanic waves and turbulent flows. 
%  Chaos, 30, doi: 10.1063/5.0012077 
%  
%  Kondrashov D., et al. 2026 Accurate and robust real-time prediction of September Arctic sea ice 
%  Chaos: 36, doi: 10.1063/5.0295634
%
%  Chekroun, M. D., and D. Kondrashov, 2017: Data-adaptive harmonic spectra and multilayer Stuart-Landau models. 
%  Chaos, 27 (9), 093 110. doi:10.1063/1.4989400
%
%  Kondrashov, D. et al., 2018: Multiscale Stuart-Landau emulators: Application to wind-driven ocean gyres. 
%  Fluids, 3 (1), 21, doi:10.3390/fluids3010021.
%
%  Written by Dmitri Kondrashov.   Version date 7/26/26
%  Please send comments and suggestions to dkondras@atmos.ucla.edu

DMD = center(data);
stddata = std(DMD);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%NLEVEL=20;
NLN=1;
XT_RES=zeros(size(DMD,1)-1,size(DMD,2),NLEVEL);

XX=zeros(size(DMD,1)-1,size(DMD,2),NLEVEL);

%XX=zeros(NE,size(DMD,2),NLEVEL);

L=zeros(size(DMD,2),size(DMD,2)*NLEVEL,NLEVEL);
ENL=zeros(size(DMD,2),1);
F=zeros(size(DMD,2),NLEVEL);
%RR=zeros(size(DMD,1)-1,2);

nmax = size(DMD,2);
NTT = size(DMD,1);
XX(1:NTT,:,1)= DMD;
stddata = squeeze(std(XX(1:NTT,:,1)));

if inorm == 1 XX(1:NTT,:,1)= XX(1:NTT,:,1)./(ones(NTT,1)*stddata);end

for nl=1:NLEVEL
xt=diff(XX(1:NTT,:,nl));
NTT =size(xt,1);
%%%%%%%%%%%%%%%%%%%%%%do regular MSM %%%%%%%%%%%%%%
if nl==1

NTE = size(xt,1);
%%% total number of coefficients for the pair: (nmax+2)*2;

if NELIN == 1

for n0=1:nmax/2

%%%%%%%%%%%%%%%%%%%%initialization %%%%%%%%%%%%%%%%%%%%%%%%%%
gsvd = zeros(NTE*2,(nmax+2)*2);%% pair with intercept
xg = zeros(NTE*2,1);
bg = zeros((nmax+2)*2,1);


if iSYM ==1 nSYM = 2; else nSYM=0; end;

if iNL ==1 nNL = 1; else nNL=0; end%% equality for nonlinear part

Aeq=zeros(nSYM + nNL,(nmax+2)*2);
beq = zeros(size(Aeq,1),1);

if inEQ ==1 Aneq=zeros(2,(nmax+2)*2);
bneq = zeros(size(Aneq,1),1);
else  Aneq = [];bneq=[];end;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


indk = [2*n0-1 2*n0];

diag1 = indk(1)+1;
off1  = indk(2)+1;

diag2 = nmax+2+indk(2)+1;
off2 =  nmax+2+indk(1)+1;
%%%%%%%%%%diagonal equality condition  %%%%%%%
if iSYM ==1
Aeq(1,diag1)=1;
Aeq(1,diag2)=-1;
%%%%%%%%%%off-diagonal equality condition  %%%%%%
Aeq(2,off1)=1;
Aeq(2,off2)=1;
end
%%%%%%%%%%nonlinear equality condition  %%%%%%
if iNL == 1
Aeq(nSYM+1,nmax+2)=1;
%Aeq(nSYM+1,2*(nmax+2))=1;
Aeq(nSYM+1,2*(nmax+2))=-1;
end
%end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if inEQ == 1 Aneq(1,nmax+2)=1;Aneq(2,2*(nmax+2))=1;
end;

for n_1=1:2

n=indk(n_1);

pred = [ones(NTE,1) XX(1:NTE,:,1) XX(1:NTE,n,1).*(XX(1:NTE,indk(1),1).^2+XX(1:NTE,indk(2),1).^2)];

gsvd(1+(n_1-1)*NTE:n_1*NTE,1+(n_1-1)*(nmax+2):n_1*(nmax+2))=pred;
xg(1+(n_1-1)*NTE:n_1*NTE)=center(xt(:,n));

end %% n_1 cycle

b0 = ones((nmax+2)*2,1);
%options = optimset('Display','off','Algorithm','active-set');
options = optimset('Display','off','Algorithm','interior-point');


[bg,resnorm,residual,exitflag,output,lambda]=lsqlin(gsvd,xg,Aneq,bneq,Aeq,beq,[],[],b0,options);
%display(['DAHC Pair: ' num2str(n0) ', condition # ' num2str(cond(gsvd))]);
if inEQ ==1 tmp=Aneq*bg;
end;

for n_1=1:2
L(indk(n_1),1:nmax,1)=bg(2+(n_1-1)*(nmax+2):n_1*(nmax+2)-1);
val = bg(n_1*(nmax+2));
%if val > 0 val = -val; end;
%ENL(indk(n_1))=val;
ENL(indk(n_1))=bg(n_1*(nmax+2));
F(indk(n_1))=bg(1+(n_1-1)*(nmax+2));
end

residual = xg-gsvd*bg;
XT_RES(1:NTT,indk(1):indk(2),1)= reshape(residual,NTT,2);

end %% n0 cycle

end % end of NELIN ==1

if NELIN == 0
%%%%%%%%%%%simple rgerssion
for n_1=1:nmax
%mxt=mean(xt(:,n_1))
%[beta,tmp,XT_RES(1:NTE,n_1,1)]=regress(xt(:,n_1),center(XX(1:NTE,:,1)));
%L(n_1,1:nmax,1)=beta(1:nmax);
%F(n_1,1)=mxt-beta(1:nmax)'*mean(XX(1:NTE,:,1))';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[beta,tmp,XT_RES(1:NTE,n_1,1)]=regress(xt(1:NTE,n_1),[ones(NTE,1) XX(1:NTE,:,1)]);
L(n_1,1:nmax,1)=beta(2:nmax+1);
F(n_1,1)=beta(1);
%REGR = [ones(NTE,1) XX(1:NTE,:,1)];
%XT_RES(1:NTE,n_1,1)=xt(:,n_1)-REGR*beta;
ENL(n_1)=0;
end

end  %% NELIN ==0

else

NLEV = 1;

%if NLEV  == 1
if NELIN  == 1

for n0=1:nmax/2

indk=[2*n0-1 2*n0];
off = indk;
inds= off; %% pair only
%%inds= 1:nmax; %% all on first levels
for nl_1=1:nl-1
inds = [inds off+nl_1*nmax];
end

pred = [];%pair

%%pred = squeeze(XX(1:NTT,1:nmax,1));
%%for nl_1=2:nl

for nl_1=1:nl
pred = [pred squeeze(XX(1:NTT,indk,nl_1))];
end
%cond(pred)
for n=indk(1):indk(2)
[L(n,inds,nl),BINT,XT_RES(1:NTT,n,nl)] = regress(center(xt(:,n)),pred);
end
end %% end of n0 cycle

else

pred = [];
for nl_1=1:nl
pred = [pred squeeze(XX(1:NTT,:,nl_1))];
end

for n=1:nmax
[beta,BINT,XT_RES(1:NTT,n,nl)] = regress(center(xt(:,n)),[ones(NTT,1) pred]);
L(n,1:nmax*nl,nl)=beta(2:end);
F(n,nl)=beta(1);
end

end

end
%%%%%%%%%%%%%%%%%%%%%%do regular MSM %%%%%%%%%%%%%%
stdr(:,nl)=std(XT_RES(1:NTT,1:nmax,nl));

if nl ~=NLEVEL
XX(1:NTT,1:nmax,nl+1)=center(XT_RES(1:NTT,1:nmax,nl));
if inorm ==1 XX(1:NTT,1:nmax,nl+1)=XX(1:NTT,1:nmax,nl+1)./(ones(NTT,1)*stdr(:,nl)');
%std(XX(1:NTT,1:nmax,nl+1));
end;
end
                                                           
end %% end of NLEVEL


%%%%%%%%%%%%%%INTEGRATION OF THE MODEL
matl = zeros(nmax*NLEVEL);

for nl=1:NLEVEL
for n=1:nmax
matl(n+(nl-1)*nmax,1:nmax*nl) = L(n,1:nmax*nl,nl);
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for nl=1:NLEVEL-1
for n=1:nmax
if inorm == 1 cof = stdr(n,nl); else cof = 1; end;
matl(n+(nl-1)*nmax,n+nl*nmax) = cof;
end
end

[EV,D]=eig(matl);
EL=diag(D);
indel = find(real(EL>0));
if size(indel,1) > 0
fprintf('%f \n',EL(indel));
end;
%ENL
%pause

ind=find(ENL>0);
%%%%%making negative
%ENL(ind)=-ENL(ind);

%ires = 0;
xt_res = XT_RES(1:NTT,:,NLEVEL);
XT_RES_sim = zeros(NTT,size(XT_RES,2));

%covn = cov(XT_RES(1:NTT,:,NLEVEL));
covn = corrcoef(XT_RES(1:NTT,:,NLEVEL))+1.e-10*eye(nmax);
rr = chol(covn)';


%NSMT=4;
if NSMT ==0 XSM = XX; return; end;
XSM=zeros(NE,size(DMD,2),NSMT);
tcount=0;
iter=0;
NSM=0;
NSM1=0;
coff=1;
while NSM~=NSMT

iter = iter+1;
tcount=tcount+1;


for NT=1:NE-1
                                                           


for nl=1:NLEVEL
                                                           
pred1 = [];

for nl_1=1:nl
pred1 = [pred1 squeeze(XX(NT,:,nl_1))];
end
tmp = squeeze(L(:,1:nmax*nl_1,nl))*pred1';

if nl ~= NLEVEL
if inorm ==1 cof = stdr(:,nl); else cof = ones(nmax,1);end;
nr=squeeze(XX(NT,:,nl+1)).*cof';
XX(NT+1,:,nl)=F(:,nl)'+XX(NT,:,nl)+tmp'+nr;
else

if ires == 0 rn = rr*indrandperm(1:nmax,NT,iter).*stdr(:,NLEVEL);
else
timest = indrandperm(NT,iter);
rn=XT_RES(timest,:,end)';
end
XT_RES_sim(NT,:)=rn';


XX(NT+1,:,nl)=F(:,nl)'+XX(NT,:,nl)+tmp'+rn';end;

    if nl ==1 & NELIN ==1

    for n0=1:nmax/2

    indk = [2*n0-1 2*n0];

    for n=indk(1):indk(2)

    pred = XX(NT,n,1).*(XX(NT,indk(1),1).^2+XX(NT,indk(2),1).^2);
                                                           
    %XX(NT+1,n,1)=XX(NT+1,n,1)+ENL(n)*pred;

    if NLN==1 XX(NT+1,n,1)=XX(NT+1,n,1)+F(n)+ENL(n)*pred;
    else XX(NT+1,n,1)=XX(NT+1,n,1)+F(n)+coff*ENL(n)*pred;
    end;

    end

    end

    end

end
end
if size(find(isnan(XX(:,:,1))),1) ==0
if NLN==0 NLN=1; coff=1; end;
NSM1=NSM1+1;
NSM=NSM+1;
if inorm ==1
tmp=ones(NE,1)*stddata;
tmp1=squeeze(XX(1:NE,:,1));
tmp2=tmp1.*tmp;
XSM(1:NE,:,NSM)=tmp2;
else XSM(1:NE,:,NSM)=XX(1:NE,:,1);end;
else %
if NLN==1 NLN=0; end;
coff=coff-0.1;
iter=iter-1;
%%% without linear
%NSM=NSM+1;
end
disp(['TOTAL SIMULATIONs ' num2str(tcount) ' SUCCESSFULL ' num2str(NSM1)]);
end
%size(XT_RES_sim)
return
end




