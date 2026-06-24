%clear all
generate_data
fpcs=center(x);
W=(size(fpcs,1)+1)/2;% maximum window length
W=65;
WW=2*W-1;
D=size(fpcs,2);
NP=D;% maximum is D
NFE=W;% maximum is W


wt='hann';
wt='blackman';
wt='hanning';
wt='hamming';
wt='bartlett';
%wt='none';
tic

[fE2,VP,FEP]=DAHD4freq_part_weight(fpcs,W,NFE,NP,wt);
toc
%% DAHD spectrum
NP=1;
figure
set(gca,'FontSize',16);
semilogy(fE2(2:end),abs(VP(:,2:end)),'ro','MarkerSize',4,'MarkerFaceColor','r');
hold on
semilogy(fE2(18),abs(VP(1:NP,18)),'bo','MarkerSize',8,'MarkerFaceColor','b');
semilogy(fE2(27),abs(VP(1:NP,27)),'co','MarkerSize',8,'MarkerFaceColor','c');
semilogy(fE2(47),abs(VP(1:NP,47)),'go','MarkerSize',8,'MarkerFaceColor','g');
semilogy(fE2(57),abs(VP(1:NP,57)),'ko','MarkerSize',8,'MarkerFaceColor','k');
ylabel('\lambda');
xlim([0 0.5]);
ylim([1.e-2 Inf]);
xlabel('Freq')
set(gca,'FontSize',16);
%% compute DAHMs 
EP=zeros((2*W-1)*D,2*NP,NFE);%% 
X=reshape(fpcs,size(fpcs,1)*D,1);
tic
for iff=1:NFE
ER=DAHM4_ex(FEP(:,:,iff),W,iff,2*NP);
ERR=reshape(ER,(2*W-1)*D,2*NP);
EP(:,:,iff)=ERR;
end


tt={'(a) f_1','(b) f_2','(c) f_3', '(d) f_4'};
ff =[18,27,47,57];
%%%%%%%%%%%%%%
fig=figure('Units','centimeter',...
		   'Position',[0.1 0.1 20 40],...
		   'DefaultAxesFontWeight', 'bold',...
		   'DefaultAxesFontSize', 10,'PaperUnits','centimeter','PaperPositionMode', 'auto','PaperSize',[20 40]);
kk=0;
for pos=1:4
for K=1:2
kk=kk+1;
subplot(4,2,kk)
set(gca,'FontSize',16);
contourf(reshape(EP(:,K,ff(pos)),WW,6)','EdgeColor','none');
shading flat
colorbar
if K== 1 title(tt(pos)); end;
colormap('jet')
xlabel('Time')
ylabel('Space')
end
end


%% Reconstruction 
ifff=[18 27 47 57];% indices of frequencies of the harmonic modes for maximum window length
NE=size(fpcs,1);
PCF=zeros(NE,D,size(ifff,2));%% harmonic modes reconstruction
AF=zeros(N-(2*W-1)+1,2,size(ifff,2));
NPP=1;% reconstruct only with modes at spectral peak
for i=1:size(ifff,2)
iff=ifff(i);
tmp = squeeze(EP(:,1:2*NPP,iff));%% 
A = dahc(fpcs,tmp);
AF(:,:,i)=A;
PCF(:,:,i) = hrc(A,tmp,size(fpcs,2),1:2*NPP);
end

R1=squeeze(PCF(:,:,1));

R2=squeeze(PCF(:,:,2));

R3=squeeze(PCF(:,:,3));

R4=squeeze(PCF(:,:,4));
%% normalized RMSE of reconstruction
display('normalized RMSE of reconstruction');
display(num2str(sum(sum((xref1-R1).^2,2))/sum(sum(xref1.^2,2))));
display(num2str(sum(sum((xref2-R2).^2,2))/sum(sum(xref2.^2,2))));
display(num2str(sum(sum((xref3-R3).^2,2))/sum(sum(xref3.^2,2))));
display(num2str(sum(sum((xref4-R4).^2,2))/sum(sum(xref4.^2,2))));

plothrcmodes



         
         








