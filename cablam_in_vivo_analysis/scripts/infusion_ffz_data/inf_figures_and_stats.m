function[] = inf_figures_and_stats(pth)

% load in processed data from infusion experiments, make some figures, do some stats 
load([pth 'cablam_in_vivo_analysis\demo_data\infusion_ffz_data\data_for_plots\res.mat'])

t = ops.win/ops.fs;
t_win = t>0 & t <=2; % window for finding peaks in tactile response, seconds

% for keeping track of N cells response type per animal
% (N)egative, (P)ositive, N(O) Response
N = []; P = []; O = [];

%% bl animals

eps_bl = {};
eps_n_bl = {};

snr_bl = [];
pks_bl = [];
sds_bl = [];

cntr = 1;

for i = 1:4 % the cablam animals are the first 4

    tep = W{i}.tac_eps;
    bts = W{i}.tac_bts;
    resp = W{i}.tac_resp;
    
    gd_ep = nan(size(tep));
    gd_ep_n = nan(size(tep));

    n = 0;
    p = 0;
    o = 0;

    for j = 1:size(tep,1)

        if any(any(resp(j,:,[1 3])))
            ep = squeeze(tep(j,:,:))';
            bt = bts(j,:);
            ep(bt,:) = NaN;
            gd_ep(j,:,:) = ep';
            p = p + 1;
        elseif any(any(resp(j,:,[2 4])))
            n = n + 1;
        else
            o = o + 1;
        end
    end

    N(i) = n; P(i) = p; O(i) = o;

    gd_cell = mean(gd_ep,3,'omitnan');
    gd_cell_idxs = find(~isnan(mean(gd_cell,2)));
    gd_cell = gd_cell(gd_cell_idxs,:,:);
    gd_ep = gd_ep(gd_cell_idxs,:,:);

    sd = std(gd_cell,0,2);
    q = iqr(sd);
    ps = prctile(sd,[25 75]);
    bcs = sd<ps(1)-1.5*q | sd>ps(2)+1.5*q;
    gd_ep = gd_ep(~bcs,:,:);
    gd_cells = gd_cell(~bcs,:,:);

    tmp_pk = squeeze(max(gd_ep(:,t_win,:),[],2));
    tmp_sd = squeeze(std(gd_ep(:,t<0,:),0,2));
    snr_tmp = tmp_pk(:)./tmp_sd(:);

    snr_bl = cat(1,snr_bl,snr_tmp);
    pks_bl = cat(1,pks_bl,tmp_pk(:));
    sds_bl = cat(1,sds_bl,tmp_sd(:));
    
    eps_bl{cntr} = squeeze(gd_cells);

    cntr = cntr + 1;

end

snr_bl(isnan(snr_bl)) = [];
pks_bl(isnan(pks_bl)) = [];
sds_bl(isnan(sds_bl)) = [];

% mean subj responses and all cell stack
subj_mus = [];
cell_stack = [];
cell_stack_ids = [];
for i = 1:size(eps_bl,2)

    tmp = eps_bl{i};

    cell_stack = cat(1,cell_stack,tmp);
    cell_stack_ids = cat(1,cell_stack_ids,i*ones(size(tmp,1),1));

    tmp_mu = mean(tmp,1,'omitnan');
    subj_mus = cat(1,subj_mus,tmp_mu);

end

cell_stack = cell_stack';

jackstat = jackknife(@mean,subj_mus);
err_bl = prctile(jackstat,[2.5 97.5]);
mu_bl = mean(subj_mus);

% all cells, sorted by latency
[mx,ix] = max(cell_stack(t_win,:));
ix = ix + nnz(t<=0);
lix = [];
bc=[];
for i = 1:size(cell_stack,2)
    hlf = mx(i)/2;
    tmp = cell_stack(:,i);
    tmp = tmp(1:ix(i));
    ix_tmp = find(tmp<=hlf,1,'last'); 
    lix(i) = ix_tmp;        
end
[sv,six] = sort(lix);

figure, hold on
imagesc(t,1:numel(six),zscore(cell_stack(:,six))')
for i = 1:numel(six)
    line([t(sv(i)) t(sv(i))],[i-0.5 i+0.5],'Color','w')
end

axis ij
colormap(viridis)
colorbar, clim([-3 3])
ax = gca;
axis tight
ax.XTick = t(1):1:t(end);
ax.YTick = 1:size(cell_stack,2);
ax.XLabel.String = 'Time (Secs)';
ax.YLabel.String = 'ROI';
title('FFZ Infusion: CaBLAM All Cells')

bl_lat = t(sv);

%% gcamp

eps_fl = {};
eps_n_fl = {};

snr_fl = [];
pks_fl = [];
sds_fl = [];

cntr = 1;

for i = 5:7 % the gcamp animals are 5 through 7

    tep = W{i}.tac_eps;
    bts = W{i}.tac_bts;
    resp = W{i}.tac_resp;
    
    gd_ep = nan(size(tep));
    gd_ep_n = nan(size(tep));

    n = 0;
    p = 0;
    o = 0;

    for j = 1:size(tep,1)

        if any(any(resp(j,:,[1 3])))
            ep = squeeze(tep(j,:,:))';
            bt = bts(j,:);
            ep(bt,:) = NaN;
            gd_ep(j,:,:) = ep';
            p = p + 1;
        elseif any(any(resp(j,:,[2 4])))
            n = n + 1;
        else
            o = o + 1;
        end
    end

    N(i) = n;
    P(i) = p;
    O(i) = o;

    gd_cell = mean(gd_ep,3,'omitnan');
    gd_cell_idxs = find(~isnan(mean(gd_cell,2)));
    gd_cell = gd_cell(gd_cell_idxs,:,:);
    gd_ep = gd_ep(gd_cell_idxs,:,:);

    sd = std(gd_cell,0,2);
    q = iqr(sd);
    ps = prctile(sd,[25 75]);
    bcs = sd<ps(1)-1.5*q | sd>ps(2)+1.5*q;
    gd_ep = gd_ep(~bcs,:,:);
    gd_cells = gd_cell(~bcs,:,:);

    tmp_pk = squeeze(max(gd_ep(:,t_win,:),[],2));
    tmp_sd = squeeze(std(gd_ep(:,t<0,:),0,2));
    snr_tmp = tmp_pk(:)./tmp_sd(:);

    snr_fl = cat(1,snr_fl,snr_tmp);
    pks_fl = cat(1,pks_fl,tmp_pk(:));
    sds_fl = cat(1,sds_fl,tmp_sd(:));
    
    eps_fl{cntr} = squeeze(gd_cells);

    cntr = cntr + 1;

end

snr_fl(isnan(snr_fl)) = [];
pks_fl(isnan(pks_fl)) = [];
sds_fl(isnan(sds_fl)) = [];

% mean subj responses and all cell stack
subj_mus = [];
cell_stack = [];
cell_stack_ids = [];
for i = 1:size(eps_fl,2)

    tmp = eps_fl{i};

    cell_stack = cat(1,cell_stack,tmp);
    cell_stack_ids = cat(1,cell_stack_ids,i*ones(size(tmp,1),1));

    tmp_mu = mean(tmp,1,'omitnan');
    subj_mus = cat(1,subj_mus,tmp_mu);

end

cell_stack = cell_stack';

jackstat = jackknife(@mean,subj_mus);
err_fl = prctile(jackstat,[2.5 97.5]);
mu_fl = mean(subj_mus);

% all cells, sorted by latency
[mx,ix] = max(cell_stack(t_win,:));
ix = ix + nnz(t<=0);
lix = [];
bc=[];
for i = 1:size(cell_stack,2)
    hlf = mx(i)/2;
    tmp = cell_stack(:,i);
    tmp = tmp(1:ix(i));
    ix_tmp = find(tmp<=hlf,1,'last'); 
    lix(i) = ix_tmp;        
end
[sv,six] = sort(lix);

figure, hold on
imagesc(t,1:numel(six),zscore(cell_stack(:,six))')
for i = 1:numel(six)
    line([t(sv(i)) t(sv(i))],[i-0.5 i+0.5],'Color','w')
end

axis ij
colormap(viridis)
colorbar, clim([-3 3])
ax = gca;
axis tight
ax.XTick = t(1):1:t(end);
ax.YTick = 1:size(cell_stack,2);
ax.XLabel.String = 'Time (Secs)';
ax.YLabel.String = 'ROI';
title('GCaMP6s All Cells')

fl_lat = t(sv);


%% mean animal tactile response for two groups

figure, hold on

clrs = [0.8 0.4 0.8; ... 
        0.4 0.8 0.4];

colororder(clrs)

yyaxis left
plot(t,mu_bl,'Color',clrs(1,:),'LineWidth',1);
jbfill(t,err_bl(1,:),err_bl(2,:),clrs(1,:),'none',0.5);
ylim([-0.02 0.1])
ylabel('\DeltaL/L_0')

yyaxis right
plot(t,mu_fl,'Color',clrs(2,:),'LineWidth',1);
jbfill(t,err_fl(1,:),err_fl(2,:),clrs(2,:),'none',0.5);
ylim([-0.02 0.1])
xlim([-3 7])
ylabel('\DeltaF/F_0')

xlabel('Time (Sec)')

title('FFZ Infusion: Mean Tactile evoked waveforms across animals')

yyaxis left
line([5 6],[0.095 0.095],'Color',clrs(1,:),'LineStyle','-','Marker','none','LineWidth',4)
line([5 6],[0.09 0.09],'Color',clrs(2,:),'LineStyle','-','Marker','none','LineWidth',4)
text(4,0.095,'CaBLAM')
text(4,0.09,'GCaMP6s')

%% snrs for all trials

% clip out extreme outliers
snr_plt_bl = snr_bl(snr_bl < prctile(snr_bl,75) + 3*iqr(snr_bl)); 
snr_plt_fl = snr_fl(snr_fl < prctile(snr_fl,75) + 3*iqr(snr_fl));

figure, hold on

c = [repmat(clrs(1,:),numel(snr_plt_bl),1); repmat(clrs(2,:),numel(snr_plt_fl),1)];

snrs = [snr_plt_bl;snr_plt_fl];
grp_ids = [ones(numel(snr_plt_bl),1); 2*ones(numel(snr_plt_fl),1)];

swarmchart(grp_ids,snrs,[],c,'filled','MarkerFaceAlpha',0.1);
md1 = median(snr_bl,'omitnan');
md2 = median(snr_fl,'omitnan');
line_len = 0.3;
plot([1-line_len 1+line_len],[md1 md1],'k','LineWidth',1)
plot([2-line_len 2+line_len],[md2 md2],'k','LineWidth',1)
ax = gca;
ax.XTick = [1 2];
ax.XTickLabel = {'CaBLAM','GCaMP6s'};
ylabel('SNR')
title('SNR')
ax.PlotBoxAspectRatio = [1 1 1];
ylim([-5 15])
title('FFZ Infusion: Single tactile trial SNRs')

%% total cells

line_len = 0.25;

CNTS_raw = [N+P+O;P;N;O];

CNTS = 100*CNTS_raw(2:4,:)./sum(CNTS_raw(2:4,:),1); % percentages
figure, hold on
plot(randn(3,4)/20+(1:3)',CNTS(:,1:4),'Color',clrs(1,:),'LineStyle','none','Marker','o','MarkerFaceColor',clrs(1,:),'MarkerEdgeColor','none')
plot([1-line_len 1+line_len],[median(CNTS(1,1:4)) median(CNTS(1,1:4))],'Color',clrs(1,:),'LineWidth',1)
plot([2-line_len 2+line_len],[median(CNTS(2,1:4)) median(CNTS(2,1:4))],'Color',clrs(1,:),'LineWidth',1)
plot([3-line_len 3+line_len],[median(CNTS(3,1:4)) median(CNTS(3,1:4))],'Color',clrs(1,:),'LineWidth',1)
plot(1:3,median(CNTS(:,1:4),2),'Color',clrs(1,:),'LineWidth',1)

plot(randn(3,3)/20+(1:3)',CNTS(:,5:7),'Color',clrs(2,:),'LineStyle','none','Marker','o','MarkerFaceColor',clrs(2,:),'MarkerEdgeColor','none')
plot([1-line_len 1+line_len],[median(CNTS(1,5:7)) median(CNTS(1,5:7))],'Color',clrs(2,:),'LineWidth',1)
plot([2-line_len 2+line_len],[median(CNTS(2,5:7)) median(CNTS(2,5:7))],'Color',clrs(2,:),'LineWidth',1)
plot([3-line_len 3+line_len],[median(CNTS(3,5:7)) median(CNTS(3,5:7))],'Color',clrs(2,:),'LineWidth',1)
plot(1:3,median(CNTS(:,5:7),2),'Color',clrs(2,:),'LineWidth',1)

ax = gca;

ax.XTick = 1:4;
ax.XTickLabel = {'Positive','Negative','Not Responsive'};
xlabel('Cells')
ylabel('Percent of Cells')

ylim([-10 100])
line(xlim,[0 0],'Color',[.7 .7 .7])

line([1.1 1.6],[95 95],'Color',clrs(1,:),'LineStyle','-','Marker','none','LineWidth',4)
line([1.1 1.6],[90 90],'Color',clrs(2,:),'LineStyle','-','Marker','none','LineWidth',4)
text(0.75,95,'CaBLAM')
text(0.75,90,'GCaMP6s')

title('FFZ Infusion: Percentage of cell response types')

%% statistical tests for infusion data

% latency
[lat_p,~,lat_stats] = ranksum(bl_lat,fl_lat);

% snr
[snr_p,~,snr_stats] = ranksum(snr_bl,snr_fl);
[pks_p,~,pks_stats] = ranksum(pks_bl,pks_fl);
[sds_p,~,sds_stats] = ranksum(sds_bl,sds_fl);

%%

fprintf('\n\n----Infusion Results: ')
fprintf('\n--1. Cell Numbers: ')
fprintf(['\nCaBLAM median Cell Ns = ' num2str(median(CNTS_raw(1,1:4)))])
fprintf(['\nCaBLAM iqr Cell Ns = ' num2str(iqr(CNTS_raw(1,1:4)))])
fprintf(['\nGCaMP6s median Cell Ns = ' num2str(median(CNTS_raw(1,5:7)))])
fprintf(['\nGCaMP6s iqr Cell Ns = ' num2str(iqr(CNTS_raw(1,5:7)))])

fprintf('\n--2. Cell Latencies: ')
fprintf(['\nCaBLAM median cell latencies = ' num2str(median(bl_lat))])
fprintf(['\nCaBLAM iqr cell latencies = ' num2str(iqr(bl_lat))])
fprintf(['\nGCaMP6s median cell latencies = ' num2str(median(fl_lat))])
fprintf(['\nGCaMP6s iqr cell latencies = ' num2str(iqr(fl_lat))])
fprintf(['\nLatency Comparison, rank sum, p = ' num2str(lat_p) ', z = ' num2str(lat_stats.zval)])

fprintf('\n--3. Cell SNR: ')
fprintf(['\nCaBLAM median cell SNR = ' num2str(median(snr_bl))])
fprintf(['\nCaBLAM iqr cell SNR = ' num2str(iqr(snr_bl))])
fprintf(['\nGCaMP6s median cell SNR = ' num2str(median(snr_fl))])
fprintf(['\nGCaMP6s iqr cell SNR = ' num2str(iqr(snr_fl))])
fprintf(['\nSNR Comparison, rank sum, p = ' num2str(snr_p) ', z = ' num2str(snr_stats.zval)])

fprintf('\n--4. Cell Peaks: ')
fprintf(['\nCaBLAM median cell peaks = ' num2str(median(pks_bl))])
fprintf(['\nCaBLAM iqr cell peaks = ' num2str(iqr(pks_bl))])
fprintf(['\nGCaMP6s median cell peaks = ' num2str(median(pks_fl))])
fprintf(['\nGCaMP6s iqr cell peaks = ' num2str(iqr(pks_fl))])
fprintf(['\nPeaks Comparison, rank sum, p = ' num2str(pks_p) ', z = ' num2str(pks_stats.zval)])

fprintf('\n--4. Cell SDs: ')
fprintf(['\nCaBLAM median cell SDs = ' num2str(median(sds_bl))])
fprintf(['\nCaBLAM iqr cell SDs = ' num2str(iqr(sds_bl))])
fprintf(['\nGCaMP6s median cell SDs = ' num2str(median(sds_fl))])
fprintf(['\nGCaMP6s iqr cell SDs = ' num2str(iqr(sds_fl))])
fprintf(['\nSDs Comparison, rank sum, p = ' num2str(sds_p) ', z = ' num2str(sds_stats.zval)])








