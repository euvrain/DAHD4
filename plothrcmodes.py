# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%to plot data as an image repalce 
# %% contourf -> imagesc
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %figure
# fig=figure('Units','centimeter',...
# 					   'Position',[0.1 0.1 20 40],...
# 					   'DefaultAxesFontWeight', 'bold',...
# 					   'DefaultAxesFontSize', 10,'PaperUnits','centimeter','PaperPositionMode', 'auto','PaperSize',[30 20]);
# xmax=0.5;
# xmin=-.5;
# subplot(422)
# set(gca,'FontSize',16);
# contourf(R1',20,'EdgeColor','none');
# shading flat
# colorbar
# colormap('jet')
# title('Recons at f_1');
# xlim([1 129]);
# caxis([xmin xmax]);
# set(gca,'FontSize',16);

# subplot(424)
# set(gca,'FontSize',16);
# contourf(R2',20,'EdgeColor','none');
# colorbar
# colormap('jet')
# shading flat
# title('Recons at f_2');
# xlim([1 129]);
# caxis([xmin xmax]);
# set(gca,'FontSize',16);

# subplot(426)
# set(gca,'FontSize',16);
# contourf(R3',20,'EdgeColor','none');
# shading flat
# colorbar
# colormap('jet')
# title('Recons at f_3');
# xlim([1 129]);
# caxis([xmin xmax]);
# set(gca,'FontSize',16);

# subplot(428)
# set(gca,'FontSize',16);
# contourf(R4',20,'EdgeColor','none');
# shading flat
# title('Recons at f_4');
# colorbar
# colormap('jet')
# xlabel('Time')
# xlim([1 129]);
# caxis([xmin xmax]);
# set(gca,'FontSize',16);
         
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# xmax=0.5;
# xmin=-.5;
# subplot(421)
# set(gca,'FontSize',16);
# contourf(xref1',20,'EdgeColor','none');
# shading flat
# colorbar
# colormap('jet')
# title('Mode at f_1');
# xlim([1 129]);
# caxis([xmin xmax]);
# set(gca,'FontSize',16);

# subplot(423)
# set(gca,'FontSize',16);
# contourf(xref2',20,'EdgeColor','none');
# colorbar
# colormap('jet')
# shading flat
# title('Mode at f_2');
# xlim([1 129]);
# caxis([xmin xmax]);
# set(gca,'FontSize',16);

# subplot(425)
# set(gca,'FontSize',16);
# contourf(xref3',20,'EdgeColor','none');
# shading flat
# colorbar
# colormap('jet')
# title('Mode at f_3');
# xlim([1 129]);
# caxis([xmin xmax]);
# set(gca,'FontSize',16);

# subplot(427)
# set(gca,'FontSize',16);
# contourf(xref4',20,'EdgeColor','none');
# shading flat
# title('Mode at f_4');
# colorbar
# colormap('jet')
# xlabel('Time')
# xlim([1 129]);
# caxis([xmin xmax]);
# set(gca,'FontSize',16);
         

