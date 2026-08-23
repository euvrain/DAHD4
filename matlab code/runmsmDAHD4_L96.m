%clear all;
%close all;

load L96_F6.mat;
K=10;
data = center(data(1:K:end,:));
X=data(:,1:end);
W=160;
D=size(X,2);
NE=size(X,1);
NFE=W;
NP=D;
tic
wt='none';
wt='bartlett';





tic

[fE2,VP,FEP]=DAHD4freq_part_weight(X,W,NFE,NP,wt);



%%
NCMP=0.75*D; 
NCMP=20; 

NFEE=25;%% number of frequencies to emulate
NFEE=30;%% number of frequencies to emulate

ff =[2,10,20,30];
figure
semilogy(fE2,abs(VP),'ro','MarkerSize',4,'MarkerFaceColor','r');
hold on
semilogy(fE2(2:NFEE),abs(VP(1:NCMP,2:NFEE)),'bo','MarkerSize',4,'MarkerFaceColor','b');
semilogy(fE2(ff(1)),abs(VP(1,ff(1))),'ko','MarkerSize',8,'MarkerFaceColor','k');
semilogy(fE2(ff(2)),abs(VP(1,ff(2))),'ko','MarkerSize',8,'MarkerFaceColor','k');
semilogy(fE2(ff(3)),abs(VP(1,ff(3))),'ko','MarkerSize',8,'MarkerFaceColor','k');
semilogy(fE2(ff(4)),abs(VP(1,ff(4))),'ko','MarkerSize',8,'MarkerFaceColor','k');
ylabel('\lambda');
set(gca,'FontSize',16)
%%

NP=D;
EP=zeros((2*W-1)*D,2*NP,NFE);%% 
tic
for iff=1:NFE
ER=DAHM4_ex(FEP(:,:,iff),W,iff,2*NP);
ERR=reshape(ER,(2*W-1)*D,2*NP);
EP(:,:,iff)=ERR;
end
toc
%%

%%%%%%%%%%%%%%
fig=figure('Units','centimeter',...
		   'Position',[0.1 0.1 30 40],...
		   'DefaultAxesFontWeight', 'bold',...
		   'DefaultAxesFontSize', 10,'PaperUnits','centimeter','PaperPositionMode', 'auto','PaperSize',[30 40]);
kk=0;
for pos=1:4
kk=(pos-1)*3;
for K=1:2
kk=kk+1;
subplot(4,3,kk)
set(gca,'FontSize',16);
contourf(reshape(EP(:,K,ff(pos)),(2*W-1),D)','EdgeColor','none');
shading flat
colorbar
if K== 1 title(['f=' num2str(fE2(ff(pos)))]); end;
colormap('jet')
xlabel('Time')
ylabel('Space')
set(gca,'FontSize',16);
end
end

for pos=1:4
ER = reshape(EP(:,1:2,ff(pos)),size(EP,1),2);
A=dahc(X,ER);
subplot(4,3,pos*3)
plot(A,'LineWidth',2)
xlim([1 300])
xlabel('Time')
title(['f=' num2str(fE2(ff(pos)))]);
set(gca,'FontSize',16)
end



%%


%%%%%%%%%%%%%%%
NFS=1;
NFE=NFEE;
%%%%%%%%%%%%%%%
NT0=2;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
NEX = size(X,1);
RXT = zeros(NEX,size(X,2),NFE,NT0);
RRT = zeros(NEX,size(X,2),NFE);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
WW=2*W-1;
NA=size(X,1)-WW+1;
NE = NEX-WW+1;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


XT_RES=zeros(NA,2*size(X,2));
for NF=NFS:NFE
%for NF=NFS:NFE
fprintf('FREQ=%d \n',NF);
%%%%%%%%% for W=20 %%%%%%%%%%%%%%%%%%%%%
if NF==1; %%
NM=size(X,2); %% for NF ==1;
ncmp = NM/2;
%ncmp = NM; 
end
if NF> 1
NM=(size(X,2))*2;
ncmp = NM;
ncmp = NM/2;
ncmp = 3*NM/4;
ncmp=2*NCMP;
end
%%%%%%%% for W=4 or W=10%%%%%%%%%%%%%%%%%%%%%
ER = reshape(EP(:,1:NM,NF),size(EP,1),NM);
A=dahc(X,ER);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
indm=1:ncmp;
RR = hrc(A,ER,size(X,2),indm);
RRT(:,:,NF)=RR;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
RXZ = zeros(size(X,1),size(X,2),NT0);
         NN = size(A,1);
         %NE=4*NA;
         NE=NA;
         DMD=A(:,indm);
         ipls = 0;
         inorm =1;

         % ires=0;
         % L=11;

         ires=0;
         ires=0;
         L=5;
        
         rng('default');
         if ires == 0 indrandperm=randn(size(X,2)*2,NE,NT0);
         else indrandperm=zeros(NE-1,NT0);
         for NT=1:NT0
         indrandperm(:,NT)=randperm(NE-1);
         end
         end
         %NT=NT0;
         %XT_RES=zeros(NA,2*size(X,2));
if NF==1
[xx,xt_res,LL,ENL]= MSLM_INT(NE,DMD(:,:),0,L,NT0,ires,inorm,0,0,0,indrandperm);%%
%XT_RES(1:NA-L,1:ncmp)=xt_res;
else
iNL=1;
iSYM = 1;
inEQ = 1;
[xx,xt_res,LL,ENL]= MSLM_INT(NE,DMD(:,:),1,L,NT0,ires,inorm,iSYM,iNL,inEQ,indrandperm);%%
%display([num2str(ENL','%.5f')])
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%XT_RES(1:NA-L,1:ncmp)=xt_res;
end

for KK=1:NT0
indf = indm;
RXZ(:,:,KK) = hrc(xx(:,indf,KK),ER(:,indf),size(X,2),1:size(A(:,indf),2));
end

%RXT(:,:,NF-NFS+1,:)=RXZ; %no parfor
%RES(:,:,NF-NFS+1)=XT_RES;

RXT(:,:,NF,:)=RXZ; %% parfor
%RES(:,:,NF)=XT_RES;


        end
%%
     
                 NET=2000;
                 figure
                 for KK=1:size(RXT,4)
                 
                 indf = 1:NFE-NFS+1;
              
                 RR=sum(RRT(:,:,:),3);
                 
                 RX=squeeze(sum(RXT(:,:,:,KK),3));
                 display(['Simulation #' num2str(KK) ' Var Recons: ' num2str(sum(var(RR))) ' Var MSLM: '  num2str(sum(var(RX))) ])
                 
                 ND=1;
                indt=NEX-NET:NEX;
                 cmax = 8;
                 cmin = -8;
                 subplot(313)
                contourf(RX(indt,:)',20,'EdgeColor','none')
                         
                          shading flat
                          colorbar
                          colormap('jet')
                          caxis([cmin cmax]);
                          title('MSLM')
                           xlabel('Time')
                           ylabel('Space')
                            set(gca,'FontSize',16)
                          subplot(312)
                         contourf(RR(indt,:)',20,'EdgeColor','none')
                      
                                   shading flat
                                   colorbar
                                   colormap('jet')
                                   caxis([cmin cmax]);
                                   title('DAHD Reconstruction')
                                   xlabel('Time')
                                   ylabel('Space')
                                     set(gca,'FontSize',16)

                                      subplot(311)
                         contourf(X(indt,:)',20,'EdgeColor','none')
                      
                                   shading flat
                                   colorbar
                                   colormap('jet')
                                   caxis([cmin cmax]);
                                   title('Full Data')
                                   xlabel('Time')
                                   ylabel('Space')
                                     set(gca,'FontSize',16)
                                           %pause
                                   end
                                   
                                   
                                   figure
                                 
                                   plot(squeeze(sum(var(RRT))),'r','LineWidth',2)
                                   hold on
                                   plot(squeeze(sum(var(RXT))),'b','LineWidth',2)
                                   legend('Recons','MSLM','Location','NorthWest')
                                   xlabel('Freq #')
                                   ylabel('variance')
                                   set(gca,'FontSize',16)
                                   
                                  
                                   
                                
                                
       statglob(sum(RRT(:,:,:),3),squeeze(sum(RXT(:,:,:,:),3)),300);
                                   
                                   
          
                                  
                                  
                                  
                                  
                                    
                                   
                                  
                           

