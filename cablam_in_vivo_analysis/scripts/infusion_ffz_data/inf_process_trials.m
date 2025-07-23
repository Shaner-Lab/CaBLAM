function [eps,bts,nbt,resp_h0] = inf_process_trials(df,idxs,ops)

win = ops.win;
t = win/ops.fs;
thresh = ops.trial_rej_thresh;
nboot = ops.nboot;

%% gather trials - cells x time x n trials
eps = nan(size(df,1),numel(win),numel(idxs));
missing = [];
for i = 1:size(idxs,1)
    if all(idxs(i)+win>0) && all(idxs(i)+win<size(df,2))        
        tmp = df(:,idxs(i)+win); % cells x time      
        % remove linear trends, add back DC offset
        mu = mean(tmp,2);
        tmp = detrend(tmp')';
        tmp = tmp + mu;
        % baseline to pre stim period
        bs = mean(tmp(:,win<0),2);
        tmp = (tmp-bs)./bs;        
        eps(:,:,i) = tmp;
    else
        missing = [missing; i];
    end
end

eps(:,:,missing) = []; % remove nan trials that were too early or too late relative to total df signal

%% mark excessively noisy trials on a cell-by-cell basis

bts = zeros(size(eps,1),size(eps,3),'logical'); % bad trials
nbt = []; % number of bad trials per cell

for i = 1:size(eps,1) % for each cell

    ep = squeeze(eps(i,:,:));
    
    sds = std(diff(ep)); % standard deviation of approx derivative -- detect big, fast jumps in the data
    qr = iqr(sds);
    ps = prctile(sds,[25 75]);
    bt = sds<ps(1)-qr*thresh | sds>ps(2)+qr*thresh;

    bts(i,bt) = true;
    nbt(i) = nnz(bt);

end

%% test for 'significant' responses

resp_h0 = false(size(eps,1),4); % TF mat, cell x test

h0_t = ops.h0_t;
h_t = ops.h_t;

for i = 1:size(eps,1) % for each cell

    ep = squeeze(eps(i,:,:))';

    % don't test bad trials
    bt = bts(i,:);
    ep(bt,:) = []; 

    x0 = mean(ep(:,t>=h0_t(1)&t<h0_t(2)));

    bci = bootci(nboot,@mean,x0); % confidence interval of baseline 
    x1 = bootci(nboot,@mean,mean(ep(:,t>=h_t(1)&t<h_t(2)),2)); % CI of test window 1
    x2 = bootci(nboot,@mean,mean(ep(:,t>=h_t(3)&t<h_t(4)),2)); % CI of test window 2

    resp_h0(i,1) = x1(1)>bci(2);
    resp_h0(i,2) = x1(2)<bci(1);

    resp_h0(i,3) = x2(1)>bci(2);
    resp_h0(i,4) = x2(2)<bci(1);

end
