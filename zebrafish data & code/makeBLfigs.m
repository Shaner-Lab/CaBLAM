function [] = makeBLfigs(homedir)

% Code associated with Lambert et. al. 2025 to generate the figures
% describing the zebrafish bioluminescent experiments.
% contains three functions at the end: blSTA to generate the "big movement"
% triggered average and hline/vline for plotting

if nargin<1
homedir = '/Users/schoppik/Desktop/Downloads/';
end

% huC
elavl3(1,:) = '250512/250512.0001/250512 14.47.48.dlm';
elavl3(2,:) = '250512/250512.0002/250512 17.00.03.dlm';
elavl3(3,:) = '250512/250512.0003/250512 17.43.03.dlm';

% psam
psam(1,:) = '250519/250519.0001/250519 11.52.40.dlm';
psam(2,:) = '250519/250519.0002/250519 12.23.04.dlm';
psam(3,:) = '250519/250519.0003/250519 14.09.49.dlm';

% nefma
nefma(1,:) = '250519/250519.0004/250519 14.56.42.dlm';
nefma(2,:) = '250519/250519.0005/250519 16.18.47.dlm';
nefma(3,:) = '250519/250519.0006/250519 16.57.59.dlm';

% nefma controls
nefmac(1,:) = '250520/250520.0003/250520 15.30.15.dlm';
nefmac(2,:) = '250520/250520.0004/250520 16.10.26.dlm';
nefmac(3,:) = '250520/250520.0005/250520 17.14.29.dlm';

% glast
glast(1,:) = '250527/250527.0001/250527 16.17.38.dlm';
glast(2,:) = '250527/250527.0002/250527 16.53.59.dlm';
glast(3,:) = '250527/250527.0003/250527 17.23.46.dlm';

figure
hold on
for i = 1:3
    filename = [homedir elavl3(i,:)];
    sta = blSTA(filename,99.9);
    plot([-500:30*250]./250,sta,'k','LineWidth',1)
end
title('elavl3')
plot([20 30],[.4 .4])
set(gca,'tickdir','out','xlim',[-2 30],'ylim',[-.15 .5],'visible','off')
hline(0),vline(0)

figure
hold on
for i = 1:3
    filename = [homedir psam(i,:)];
    sta = blSTA(filename,99.9);
    plot([-500:30*250]./250,sta,'k','LineWidth',1)
end
title('psam')
plot([20 30],[.4 .4])
set(gca,'tickdir','out','xlim',[-2 30],'ylim',[-.15 .5],'visible','off')
hline(0),vline(0)

figure
hold on
for i = 1:3
    filename = [homedir nefmac(i,:)];
    sta = blSTA(filename);
    plot([-500:30*250]./250,sta,'color',[.6 .6 .6],'LineWidth',1)
end
for i = 1:3
    filename = [homedir nefma(i,:)];
    sta = blSTA(filename);
    plot([-500:30*250]./250,sta,'k','LineWidth',1)
end
title('nefma')
plot([20 30],[.4 .4])
set(gca,'tickdir','out','xlim',[-2 30],'ylim',[-.15 .5],'visible','off')
hline(0),vline(0)

figure
hold on
for i = 1:3
    filename = [homedir glast(i,:)];
    sta = blSTA(filename,99.9975);
    plot([-500:30*250]./250,sta,'k','LineWidth',1)
end
title('glast')
plot([20 30],[.4 .4])
set(gca,'tickdir','out','xlim',[-2 30],'ylim',[-.15 .5],'visible','off')
hline(0),vline(0)

% example figures using 250525 16.17.38.dlm
figure
hold on
filename = [homedir glast(1,:)];
[sta,tmp] = blSTA(filename);
plot([-500:30*250]./250,tmp,'Color',[.6 .6 .6])
plot([-500:30*250]./250,mean(tmp'),'k','LineWidth',1)
plot([20 30],[.4 .4])
set(gca,'tickdir','out','xlim',[-2 30],'ylim',[-.25 1.25],'visible','off')
hline(0),vline(0)

% load up the data
% for spline fitting / sanitization
pmtlow = 0.02;
pmthigh = 0.03;
pmtshift = 100;
buff = 1000; % if you don't have PMT data before the behavior, the spline fitting will fail

data = dlmread(filename,'\t');
lastpmt = find(data(:,1)==0,1)-1;
% sanitize the pmt data. sometimes it misses a timestamp
baddex = (find(diff(data(1:lastpmt,1)) < pmtlow | diff(data(1:lastpmt,1)) > pmthigh));
data(baddex,1) = nan;
st = length(data);
warning off
pmt = spline(data(pmtshift:lastpmt,1),data(pmtshift:lastpmt,2),data(buff:st,3));
warning on
ts = data(buff:st,3)-data(buff,3);
behav = data(buff:st,4);

nback = 50;
v = 278681;
v2 = find(data(:,5)==v);

pmt = smooth(pmt,62);
pmt = detrend(pmt);

%let's plot a long stretch of raw data
figure,
yyaxis left
hold on
plot(pmt(v2-85000:v2+105000).*40,'k','LineWidth',1)
plot([150000 150000],[20000 70000],'k-','LineWidth',1)
plot([150000 157500],[20000 20000],'k-','LineWidth',1)
yyaxis right
hold on
plot(behav(v2-85000:v2+105000),'b','LineWidth',1)
set(gca,'ylim',[0 5000],'visible','off')


nback = 250;
seg = v2-nback-buff:v2+(2500-nback)-buff;

figure,
yyaxis left
hold on
plot(behav(seg),'b','LineWidth',1)
set(gca,'ylim',[0 1000],'visible','off')
yyaxis right
hold on
plot(pmt(seg).*40,'k','LineWidth',1)
plot([2000 2000],[10000 60000],'k-','LineWidth',1)
plot([2000 2250],[10000 10000],'k-','LineWidth',1)


function [sta,tmp] = blSTA(filename,bigBoutPct)

% thresholds
if nargin == 1
    bigBoutPct = 99.9975;
end

framerate = 250;



% for spline fitting / sanitization

pmtlow = 0.02;
pmthigh = 0.03;
pmtshift = 100;
buff = 1000; % if you don't have PMT data before the behavior, the spline fitting will fail

data = dlmread(filename,'\t');
lastpmt = find(data(:,1)==0,1)-1;

% sanitize the pmt data. sometimes it misses a timestamp
baddex = (find(diff(data(1:lastpmt,1)) < pmtlow | diff(data(1:lastpmt,1)) > pmthigh));
%fprintf('You have %d bad elements\n',length(baddex));
data(baddex,1) = nan;
st = length(data);

warning off
pmt = spline(data(pmtshift:lastpmt,1),data(pmtshift:lastpmt,2),data(buff:st,3));
warning on
ts = data(buff:st,3)-data(buff,3);
behav = data(buff:st,4);


bigBoutThreshold = prctile(behav,bigBoutPct);

%figure,yyaxis left,plot(pmt),yyaxis right,plot(behav)

% first, find all big movements
allBigCrossings = find(diff(behav > bigBoutThreshold) == 1)+1;
% since struggles aren't unitary events, exclude all the other big
% movements in the following second
possibleStruggles = allBigCrossings([1;find(diff(allBigCrossings)>framerate)+1]); % give a big window (1 sec)

fprintf('Photons = %d, Struggles = %d, Duration = %d\n',round(mean(pmt)*40),length(possibleStruggles),round(length(pmt)/250/60));

pmt = smooth(pmt,62);


tmp = mexgetwfs(possibleStruggles,pmt,500,framerate.*30);

baseline = repmat(mean(tmp(1:500,:)),[size(tmp,1) 1]);

tmp = (tmp - baseline)./baseline;

sta = mean(tmp');


function vline(hpos,linetype)

ax = axis;

if (nargin == 1)
  plot([hpos hpos],[ax(3) ax(4)],'k:');
elseif (nargin == 2)
  plot([hpos hpos],[ax(3) ax(4)],linetype);
else
  help vline
end