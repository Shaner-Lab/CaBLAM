function [] = inf_process_all(pth)

%% process data

%----Subject to process----%

id = 'T452';
exp_day = '20250331';

cd([pth 'cablam_in_vivo_analysis\scripts\infusion_ffz_data\'])

%----Processing Parameters----%
ops.fs = 10; % sample rate, Hz
ops.wheel_diam = 15; % diameter of running wheel, CM
ops.run_thresh = 1.5; % zscore threshold for detecting running onset
ops.inter_run_bout = 3; % how much time in seconds between running bouts to label distinct bouts
ops.wrap_around = 200;  % voltage threshold to detect point at which one full wheel revolution occurs and voltage wraps back around to 0 on the wheel rotary encoder
ops.r = 1; % neuropil correction factor
ops.win =  -3*ops.fs:7*ops.fs; % time window around tactile stimulus
ops.trial_rej_thresh = 3; %
ops.h0_t = [-3 0];
ops.h_t = [0 1 1 2];
ops.nboot = 1000;
ops.id = id;
ops.exp_day = exp_day; 
ops.pth2file = [pth 'cablam_in_vivo_analysis\demo_data\infusion_ffz_data\ex_animal\' id '\' exp_day '\'];

%----Begin----%
fprintf(['\nDoing: ' ops.id ', ' ops.exp_day])
inf_process_running(ops);
inf_process_frames(ops);
res = inf_process_main(ops);
fprintf(['\nFinished: ' ops.id ', ' ops.exp_day '\n'])

%% check out some data

figure,
subplot(2,2,[1 2])
df_t = 1/ops.fs:1/ops.fs:size(res.df,2)/ops.fs;
df = res.df';
df0 = smoothdata(df,1,'movmean',ops.fs*120);
df = df-df0;
df = smoothdata(df,1,'movmean',ops.fs*0.5);
plot(df_t,df + linspace(2500,0,size(df,2)),'k')
axis off
title('FFz infusion: Example Animal T452 Output')

subplot(2,2,3)

% pull out positive tactile responders
p_reps = res.tac_resp(:,1) | res.tac_resp(:,3);
eps = res.tac_eps(p_reps,:,:);
bts = res.tac_bts(p_reps,:);

for i = 1:size(eps,1)
    tmp = squeeze(eps(i,:,:));
    bt = bts(i,:);
    tmp(:,bt) = NaN;
    eps(i,:,:) = tmp;
end

mu_eps = squeeze(mean(eps,3,'omitnan'));
mu_t = ops.win/ops.fs;
imagesc(mu_t,1:size(mu_eps,1),mu_eps);
clim([-0.1 0.1])
title('Mean tactile responses for all responsive cells')
xlabel('Time (Sec)')
ylabel('ROI #')
colorbar
f = gcf;
f.Children(1).Label.String = '\DeltaL/L_0';

subplot(2,2,4), hold on
err = bootci(1000,@mean,mu_eps);
jbfill(mu_t,err(1,:),err(2,:),'r','none',0.5);
plot(mu_t,mean(mu_eps,1),'r')
xlabel('Time (Sec)')
xlim([mu_t(1) mu_t(end)])
ylim([-0.05 0.15])
title('Mean across responsive cells')











