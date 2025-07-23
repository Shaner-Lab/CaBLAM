function [] = inf_process_running(ops)

fprintf('\nDetecting run bouts')

pth2file = ops.pth2file;
csv_file = dir([pth2file '*.csv']); % raw data with wheel data and tactile stimulus timing

wrap_around = ops.wrap_around;
wheel_circ = ops.wheel_diam*pi; % wheel circumference

W = readtable([csv_file.folder '\' csv_file.name],'Delimiter',',');

% using timestamps, in ms
T = cell2mat(W.Var4);
T = string(T(:,12:end-6));
T = milliseconds(duration(T,'InputFormat','hh:mm:ss.SSSSSSS'));
T = T - T(1);

rn_dat = W.Var1; % raw wheel data

% unwrap the wheel data
ixs = find(diff(rn_dat)>wrap_around);
for j = 1:numel(ixs)
    rn_dat(ixs(j)+1:end) = rn_dat(ixs(j)+1:end) - (rn_dat(ixs(j)+1)-rn_dat(ixs(j)));
end
rn_dat = -rn_dat; % flip, the data is recording as descending values with forward running

fs_approx = round(1000/median(diff(T))); % approximate bonsai sample rate -- it's not perfectly fixed
rn_dat = smoothdata(rn_dat,1,'movmedian',fs_approx*2.5); % smooth the running data with a ~2.5 sec moving median

cms = (rn_dat./1024).*wheel_circ; % rescale 10 bit signal, multiply by wheel cicumference to get distance 
secs = T./1000; % signal time in Seconds
dt = diff(secs); 
inst = diff(cms)./dt; % approximate instantaneous speed (ds/dt) 

inst(inst==Inf) = NaN; % this happens because some diff(secs) = 0 due to bonsai wonkiness
inst(inst<0) = 0;
inst = fillmissing(inst,'const',0);

inst = smoothdata(inst,1,'movmean',fs_approx); % smooth again

% detect running onset
ixs = find(zscore(inst)>ops.run_thresh);

% only use bouts that are distinct from one another based on inter_run_bout threshold
x = ixs(1);
prior = ixs(1);
for j = 2:numel(ixs)
    if ixs(j) - prior > ops.inter_run_bout * fs_approx % running bout is more than 5 seconds from the prior one
        x = cat(1,x,ixs(j));
    end
    prior = ixs(j);
end

event_secs = secs(x);
event_ix = x;

save([pth2file 'Running.mat'],'event_secs','event_ix','inst','cms','secs','dt','rn_dat','fs_approx','T','ops')






