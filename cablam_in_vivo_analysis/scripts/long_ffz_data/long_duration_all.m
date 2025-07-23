function[] = long_duration_all(pth)

data_pth = [pth 'cablam_in_vivo_analysis\demo_data\long_duration_ffz_data\'];
load([data_pth 'long_duration.mat'],'S');

fs = 10;

% note the loaded demo data is pre-processed because the full data set is
% extremely large:
%  the example data loaded here is a 1 x N vector produced by averaging the 
%  pixel values in a circular ROI (radius = 42 pixels), across the 5 hr 
%  imaging session. The signal is expressed as percent change from baseline, 
%  where baseline is the 60 sec period before ffz application.

%% plot binned timeseries of percent change 

tbl = timetable(S.t_cat',S.psig');
TT = retime(tbl,'regular','mean','TimeStep',minutes(5)); % 5 min bins
T = TT.Time - TT.Time(1);
T = duration(T,'Format','hh:mm');
bgt = T(1) - duration('00:02','Format','hh:mm');

figure,
plot([bgt; T],[0; TT.Var1],'k')
ylim([-20 300])
ax = gca;
ax.XAxis.TickValues = T(1):minutes(15):T(end);
line(xlim,[0 0],'Color',[.5 .5 .5])
xlabel('Time (hours:minutes)')
ylabel('Percent Change')
title('Long Duration CaBLAM Imaging')

%% stim responses

win = -3*fs:7*fs;

EPS = [];
TS = [];

t0 = S.time{1}(1);

for i = 1:size(S.raw_sig,2)

    f = S.piezo_frames{i};
    s = S.raw_sig{i};
    t = S.time{i}-t0;
    s = s-smoothdata(s,2,'movmean',fs*10)+mean(s);
    s = smoothdata(s,2,'movmean',fs*0.5);
    
    eps = [];
    ts = [];
    for j = 1:numel(f)

        ixs = f(j)+win;
        if all(ixs>0) && all(ixs<length(s))

            eps = cat(1,eps,s(ixs));
            ts = cat(1,ts,t(ixs(1)));

        end


    end

    bs_ix = win<0;
    bs = mean(eps(:,bs_ix),2);
    eps_bs = (eps - bs)./bs; 

    EPS = cat(1,EPS,eps_bs); % all trials
    TS = cat(1,TS,ts); % start of every trial relative to start of recording

end

% chop-up responses at 30 min intervals

[h,~] = discretize(TS,minutes(0:30:300));
hs = unique(h(~isnan(h)));

eps_h = nan(numel(hs),size(EPS,2));
np = zeros(numel(hs),1);
snr = [];
pks = [];
for i = 1:numel(hs)

    hix = h==hs(i);
    
    tmp = EPS(hix,:);
    np(i) = size(tmp,1);
    snr = cat(1,snr,[max(tmp(:,win>0&win<=20),[],2)./std(tmp(:,win<0),0,2) i*ones(size(tmp,1),1)]);
    pks = cat(1,pks,[max(tmp(:,win>0&win<=20),[],2) i*ones(size(tmp,1),1)]);

    eps_h(i,:) = mean(tmp,1);

end

% plot tactile responses

c = (jet(numel(hs))./2)+[0.2 0.2 0];

figure
colororder(c)
plot(win./fs,eps_h')

stp = 30;
val = 0;
legnd = {};
for i = 1:numel(hs)
    lgnd{i} = [num2str(val) '-' num2str(val+stp) ' minutes'];
    val = val+stp;
end

legend(lgnd)
ylabel('dl/l')
xlabel('Time (Sec)')
xlim([win(1)./fs win(end)/fs])
title('Long Duration Mean Tactile Responses')

%% test tactile response snr and peak dl/l across time
[p_pks,tbl_pks,stats_pks] = kruskalwallis(pks(:,1),pks(:,2),'off');
[p_snr,tbl_snr,stats_snr] = kruskalwallis(snr(:,1),snr(:,2),'off');

% follow-up on pks
fprintf('\n')
c = multcompare(stats_pks, 'CriticalValueType','tukey-kramer');
ylabel('Time bins')
xlabel('mean ranks')
title('Tukey-Kramer Mulitples Comparison test, Click on each line to see significant differences')

fprintf('\n----Long Duration Results:')
fprintf(['\nKruskal-Wallis of peak DL/L, P = ' num2str(p_pks) ...
    ', Chi-sq = ' num2str(tbl_pks{2,5}) ', df = ' num2str(tbl_pks{2,3})])
fprintf(['\nKruskal-Wallis of SNR, P = ' num2str(p_snr) ...
    ', Chi-sq = ' num2str(tbl_snr{2,5}) ', df = ' num2str(tbl_snr{2,3})])

%% boxplots of time-bin snrs 

figure,hold on
swarmchart(snr(:,2),snr(:,1),'MarkerEdgeColor','none','MarkerFaceColor',[0.6 0.6 0.6],'MarkerFaceAlpha',0.3)
boxplot(snr(:,1),snr(:,2),'whisker',1.5,'symbol','','Color','k','Labels',lgnd);
ylabel('SNR')
xlabel('Time (Minutes)')
xlim([0 11])
title('Long Duration Single-Trial Tactile Responses SNR')

%% boxplots of time-bin peak dl/l

figure,hold on
swarmchart(pks(:,2),pks(:,1),'MarkerEdgeColor','none','MarkerFaceColor',[0.6 0.6 0.6],'MarkerFaceAlpha',0.3)
boxplot(pks(:,1),pks(:,2),'whisker',1.5,'symbol','','Color','k','Labels',lgnd);
ylabel('\DeltaL/L_0')

y_val = 0.035;
y_stp = 0.0025;
for i = 1:size(c,1)

    p = c(i,6);
    if p < 0.05

        grp_cmp = [c(i,1) c(i,2)];
        grp_cmp = sort(grp_cmp,'ascend');

        line(grp_cmp,[y_val y_val],'Color','k','Marker','|')

        if p < 0.01
            text(mean(grp_cmp),y_val+0.0005,'**')
        else
            text(mean(grp_cmp),y_val+0.0005,'*')
        end

        y_val = y_val + y_stp;

    end
end

ylim([-0.015 0.055])
xlabel('Time (Minutes)')

xlim([0 11])

title('Long Duration Single-Trial Tactile Responses Peak \DeltaL/L_0')





















