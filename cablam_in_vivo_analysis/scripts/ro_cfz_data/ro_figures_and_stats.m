function[] = ro_figures_and_stats(pth)

% cablam retro-orbital cfz data figures and stats
data_pth = [pth 'cablam_in_vivo_analysis/demo_data/ro_cfz_data/'];
D2hz = load([data_pth 'ro_2hz.mat']);
D10hz = load([data_pth 'ro_10hz.mat']);

%% mean 2hz responses and example 10 hz response

ep_mus = [];
for i = 1:size(D2hz.EP,2)
    ep_mus = cat(1,ep_mus,mean(D2hz.EP{i},1));
end

jackstat = jackknife(@mean,ep_mus);
err = prctile(jackstat,[2.5 97.5]);

figure, hold on
clr = [0.75 0 0.75; 0.5 0.2 0.2];
plot(D2hz.t,mean(ep_mus),'Color',clr(1,:),'LineWidth',1);
jbfill(D2hz.t,err(1,:),err(2,:),clr(1,:),'none',0.5);
ax = gca;
ax.XTick = -3:1:8;
xlim([-3 8])
ylim([-0.03 0.05])
ylabel('\DeltaL/L_0')
xlabel('Time (Sec)')

t_10hz = -3:1/10:8;
plot(t_10hz,smoothdata(mean(D10hz.EP,1),2,'movmean',5),'Color',clr(2,:));

line([5 6],[0.045 0.045],'Color',clr(1,:),'LineWidth',3)
text(3.4,0.045,'2 Hz (N = 3)')
line([5 6],[0.0425 0.0425],'Color',clr(2,:),'LineWidth',3)
text(3.4,0.0425,'10 Hz (N = 1)')

title('RO CFZ Delivery: Tactile Stimulus Responses')

xlim([-3 7])

%% trial snr and response latency

eps = D2hz.EP;
t = D2hz.t;
bs_idx = t<0;
resp_win = t>=0 & t<2;
snrs = {};
snr_cat = [];
hlf_lats = [];
for i = 1:size(eps,2)

    ep = eps{i};

    bs_sd = std(ep(:,bs_idx),0,2); 
    pk = max(ep(:,resp_win),[],2);
    snr = abs(pk)./bs_sd;
    snrs{i} = snr;
    snr_cat = [snr_cat; snr];

    ep_mu = mean(ep,1);
    [mx,ix] = max(ep_mu(resp_win));
    ix = ix + sum(t<0);
    hlf_lat = t(find(ep_mu(1:ix) < mx/2,1,'last'));

    hlf_lats = [hlf_lats; hlf_lat];

end

snr_2hz_md = median(snr_cat);
snr_2hz_iqr = iqr(snr_cat);

lat_2hz_md = median(hlf_lats);
lat_2hz_iqr = iqr(hlf_lats);

[snr_p,~,snr_stats] = signrank(snr_cat);

%% latencies and snrs for 10 hz run

ep = D10hz.EP;

bs_idx = t_10hz<0;
resp_win = t_10hz>=0 & t_10hz<2;

bs_sd = std(ep(:,bs_idx),0,2);
pk = max(ep(:,resp_win),[],2);
snr = pk./bs_sd;
ep_mu = mean(ep,1);
[mx,ix] = max(ep_mu(resp_win));
ix = ix + sum(t_10hz<0);
hlf_lat = t_10hz(find(ep_mu(1:ix) < mx/2,1,'last'));

snr_10hz_md = median(snr);
snr_10hz_iqr = iqr(snr);

lat_10hz_md = median(hlf_lat);
lat_10hz_iqr = iqr(hlf_lat);

[snr_10hz_p,~,snr_10hz_stats] = ranksum(snr,snrs{1});

%% print stuff

fprintf('\n\n----Retro-Orbital Results: ')
fprintf(['\n2Hz Latency, Median = ' num2str(lat_2hz_md)])
fprintf(['\n2Hz Latency, IQR = ' num2str(lat_2hz_iqr)])
fprintf(['\n2Hz SNR, Median = ' num2str(snr_2hz_md)])
fprintf(['\n2Hz SNR, IQR = ' num2str(snr_2hz_iqr)])
fprintf(['\nSign Rank, H_0: SNR = 0, P = ' num2str(snr_p) ', z = ' num2str(snr_stats.zval)])
fprintf(['\n10Hz Latency, Median = ' num2str(lat_10hz_md)])
fprintf(['\n10Hz Latency, IQR = ' num2str(lat_10hz_iqr)])
fprintf(['\n10Hz SNR, Median = ' num2str(snr_10hz_md)])
fprintf(['\n10Hz SNR, IQR = ' num2str(snr_10hz_iqr)])
fprintf(['\nRanK Sum, H_0: SNR 2Hz = 10Hz, P = ' num2str(snr_10hz_p) ', z = ' num2str(snr_10hz_stats.zval)])













