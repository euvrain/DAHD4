function statglob(data,datas,lead);
th=1;
%datas = detrend(datas,'constant');
[N0,L0]=size(data);
[N,K]=size(datas);
datas = reshape(datas,N,L0,K/L0);
figure
subplot(211)
for k=1:K/L0
for i=1:L0
%%%%%%%doing data
tmpd = data(:,i);
cd(:,i) = xcorr(center(tmpd),lead,'coeff');
tmp = datas(:,i,k);
c(:,i) = xcorr(center(tmp),lead,'coeff');
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tmpc = mean(c,2);
tmpcd = mean(cd,2);
plot(1:lead+1,tmpcd(lead+1:end),'r','LineWidth',2)
hold on
plot(1:lead+1,tmpc(lead+1:end),'b','LineWidth',2)
ymin0=min(cd(lead+1:end));
ymin1=min(tmpc(lead+1:end));
ymin=min([ymin0 ymin1]);
ylim([ymin 1]);
xlim([1 lead+1]);
xlabel('Lag')
title('Corr')
legend('Recons','MSLM')
set(gca,'FontSize',16)
end
subplot(212)
L=size(datas,2);
datas = reshape(datas,size(datas,1)*L,size(datas,3));
data = reshape(data,size(data,1)*size(data,2),1);
NN=size(datas,2);
j=0;
tmpd = data;
cmf = 0;
[f,xi] = ksdensity(tmpd);
semilogy(xi,f,'r','LineWidth',2)
%plot(xi,f,'r','LineWidth',2)
hold on
for k=1:NN
tmp = datas(:,k);
[f,xi] = ksdensity(tmp);
%plot(xi,f,'b','LineWidth',1)
semilogy(xi,f,'b','LineWidth',1)
hold on
cm = max(f);
cmf = max(cmf,cm);
end
legend('Recons','MSLM')
cm = max(f);
cmf = max(cmf,cm);
%ylim([0 0.16])
xlim([-8 8]);
title('PDF')
set(gca,'FontSize',16)
end



