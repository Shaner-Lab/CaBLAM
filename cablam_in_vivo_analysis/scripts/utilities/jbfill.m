function[fillhandle] = jbfill(xpoints,upper,lower,color,edge,transparency)

% USAGE: [fillhandle,msg]=jbfill(xpoints,upper,lower,color,edge,add,transparency)
% This function will fill a region with a color between the two vectors provided
% using the Matlab fill command.

% fillhandle:   handle of the filled region in the plot.
% xpoints:      The horizontal data points (ie frequencies). Note length(Upper)
%               must equal Length(lower)and must equal length(xpoints)!
% upper:        the upper curve values (data can be less than lower)
% lower:        the lower curve values (data can be more than upper)
% color:        the color of the filled area 
% edge:         the color around the edge of the filled area
% transparency: value ranging from 1 for opaque to 0 for invisible for
%               the filled color only.
%
% John A. Bockstege November 2006;
% J Murph 5.31.2012, made better;

if nargin<6;transparency=.5;end %default is to have a transparency of .5
if nargin<5;edge='none';end  %default edge color is none
if nargin<4;color='b';end %default color is blue

if length(upper)==length(lower) && length(lower)==length(xpoints)
    filled=[upper,fliplr(lower)];
    xpoints=[xpoints,fliplr(xpoints)];
    fillhandle=fill(xpoints,filled,color);% plot the data
    set(fillhandle,'EdgeColor',edge,'FaceAlpha',transparency,'EdgeAlpha',transparency);%set edge color
else
    error('Error: Must use the same number of points in each vector');
end
