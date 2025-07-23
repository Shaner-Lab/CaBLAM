function[] = inf_process_frames(ops)

fprintf('\nGetting frames')

M = matfile([ops.pth2file 'Fall.mat']);
nframes = size(M,'F',2); % get number of imaging frames from suite2p output mat file

fs = ops.fs;
pth2file = ops.pth2file;

csv_file = dir([pth2file '*.csv']);

% load labeled running data
run_fl = dir([pth2file 'Run*']);
R = load([pth2file run_fl.name]);
run_event_secs = R.event_secs;
T = R.T;

% piezo marker signal
W = readtable([csv_file.folder '\' csv_file.name],'Delimiter',',');
piezo_data = rescale(W.Var3);
piezo_idxs = find(diff(piezo_data)>0.9) + 1; % this finds the piezo window onsets
piezo_times = T(piezo_idxs);
piezo_times = piezo_times./1000; % in seconds

imt = 1/fs:1/fs:nframes/fs; % imaging time in seconds

% find nearest frame to occurences of piezo timestamps
piezo_frames = [];
for m = 1:size(piezo_idxs,1)
    tmp = imt - piezo_times(m);
    if any(tmp > 0)
        piezo_frames = cat(1,piezo_frames,find(tmp>0,1,'first'));
    end
end

% run frames
run_frames = [];
for m = 1:size(run_event_secs,1)
    tmp = imt - run_event_secs(m);
    if any(tmp > 0)
        run_frames = cat(1,run_frames,find(tmp>0,1,'first'));
    end
end

%% save
save([pth2file 'Frames.mat'],'piezo_frames','run_frames','imt','T','ops')




