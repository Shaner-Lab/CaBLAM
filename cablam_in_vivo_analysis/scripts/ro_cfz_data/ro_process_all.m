function [] = ro_process_all(pth)

data_pth = [pth 'cablam_in_vivo_analysis\demo_data\ro_cfz_data\'];
fs = 10;

% divide original roi by 2 
% --> tif was binned down (2x2 median) to make it a smaller file for demo_data
roi_c = [348/2 240/2]; 
roi_r = 50/2;

pre = 3;
post = 7;
win = -pre*fs:post*fs;
t = win/fs;

fl = dir([data_pth '*tif']);
imgs = loadtiff([data_pth fl.name]);
nframes = size(imgs,3);

pcsv = dir([data_pth 'Vib*csv']);

pdata = readtable([data_pth pcsv.name],'Delimiter',',');
piezo_data = rescale(pdata.Var1);
T = cell2mat(pdata.Var2);
T = string(T(:,12:end-6));
T = milliseconds(duration(T,'InputFormat','hh:mm:ss.SSSSSSS'));
T = T - T(1);

piezo_idxs = find(diff(piezo_data)>0.9);
piezo_times = T(piezo_idxs);

imt = 1000*(1/fs:1/fs:nframes/fs); % in ms

piezo_frames = [];
for j = 1:size(piezo_idxs,1)
    tmp = imt - piezo_times(j);
    if any(tmp > 0)
        [~,ix] = min(abs(tmp));
        piezo_frames = cat(1,piezo_frames,ix);
    end
end

% get bl signal from roi
f = figure;
im_mu = squeeze(mean(imgs,3));
imagesc(im_mu)
roi = images.roi.Circle(gca,'Center',roi_c,'radius',roi_r);
M = createMask(roi);
close(f)
roi_idxs = M==1;
imgs = reshape(imgs,size(imgs,1)*size(imgs,2),size(imgs,3));
roi_sig = mean(imgs(roi_idxs,:),1);
mu = mean(roi_sig);
roi_sig = roi_sig - smoothdata(roi_sig,2,'movmean',fs*10) + mu;
roi_sig = smoothdata(roi_sig,2,'movmean',fs*0.5);

% make tactile epochs 
[r,~]=find(piezo_frames+win>size(roi_sig,2)|piezo_frames+win<0);
piezo_frames(unique(r)) = [];
eps = roi_sig(piezo_frames+win);
eps = eps';
bs_idx = t<0;
bs = mean(eps(bs_idx,:));
eps_bs = (eps - bs)./bs;

%% plot example stuff
figure, subplot(3,1,1)
imagesc(im_mu-median(im_mu))
clim([-2 10])
images.roi.Circle(gca,'Center',roi_c,'radius',roi_r,'interactionsallowed','none','Color','r');
axis equal
axis off
title('T469 20250617 retro-orbital cfz delivery example data')
subtitle('Mean Image')
subplot(3,1,2)
imagesc(win./fs,1:size(eps_bs,2),eps_bs')
colorbar
axis square
clim([-0.1 0.1])
title('ROI Trials')
ylabel('Trial #')
xlabel('Time (Sec)')

subplot(3,1,3)
plot(win./fs,mean(eps_bs,2),'k')
xlim([-pre post])
ylim([-0.025 0.05])
ylabel('\DeltaL/L_0')
xlabel('Time (Sec)')
axis square
title('Mean ROI tactile response')

f = gcf;
cb = f.Children(2);
cb.Label.String = '\DeltaL/L_0';

