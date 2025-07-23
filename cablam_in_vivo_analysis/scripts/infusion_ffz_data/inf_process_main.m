function [res] = inf_process_main(ops)

fprintf('\nGetting df/f, etc.')

pth2file = ops.pth2file;
r = ops.r;

frm = load([pth2file 'Frames.mat']);
piezo_frames = frm.piezo_frames;
run_frames = frm.run_frames;

% pre-labelled indices for start and/or stop of bioluminescent signal
sig_fl = dir([pth2file 'Sig*']);
S = load([sig_fl.folder '\' sig_fl.name]);

% load suite2p output
F = load([pth2file 'Fall.mat']);

% filter ROI signals based on labelled cells
are_cells = logical(F.iscell(:,1));
f = F.F(are_cells,:); % raw roi signals
fneu = F.Fneu(are_cells,:); % roi neuropil signals

% filter for completely flat signals, occasionally suite2p outputs a couple
% of these, not sure why
f_flat = find(var(f,0,2)==0);
f(f_flat,:) = [];
fneu(f_flat,:) = [];

df = f-fneu.*r; % neuropil correction from suite 2p
df = smoothdata(df,2,'movmean',[5 0]); % lowpass filter


% remove data and trial indices that occur within marked
% non-bioluminescent time periods
sig_onoff = S.sig_onoff_idx;
df = df(:,sig_onoff(1):sig_onoff(2));

piezo_frames = piezo_frames-sig_onoff(1)+1;
piezo_frames(piezo_frames<1) = [];
piezo_frames(piezo_frames>size(df,2)) = [];

run_frames = run_frames-sig_onoff(1)+1;
run_frames(run_frames<1) = [];
run_frames(run_frames>size(df,2)) = [];

%% tactile epochs
[eps,bts,nbt,resp_h0] = inf_process_trials(df,piezo_frames,ops);

res.tac_eps = eps;
res.tac_bts = bts;
res.tac_nbts = nbt;
res.tac_resp = resp_h0;

%% process running epochs
[eps,bts,nbt,resp_h0] = inf_process_trials(df,run_frames,ops);

res.run_eps = eps;
res.run_bts = bts;
res.run_nbts = nbt;
res.run_resp = resp_h0;

res.df = df;








