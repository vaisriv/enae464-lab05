clear all

% finds edge points and saves them to file

global smooth_sigma

% Set read and write directories - PLEASE CHANGE THESE
im_dir = '../../data/mfg_m4_cone/images';
edge_dir = '../../data/mfg_m4_cone/edges';

% the frame numbers that were saved using the PCC software
frames = 0:10;
nframes = length(frames);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Settings for tracing edges %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% if stopstart = 1, choose a stop point as well as a start point; if 0
% just a start point
stopstart = 1;

% First, set initial directions of edge segments
% 'nn' = north; 'nw' = northwest; also 'ww', 'sw', 'ss', 'se', 'ee', 'ne'
directions = {'ee','ee'}; 
no_segs = length(directions);

% Set primary orientation of segments - determines order of directions to
% search for next edge points
% cw = clockwise, ccw = counterclockwise
orient = {'cw','ccw'}; 


% Number of images after which the user resets the input points
% if = 0, only input on first (two) image(s)
input_every = 1;

% Method for pixel-resolution edge detection
% Canny detection most reliable, but other types should be fine too
edge_type = 'canny';

% Sensitivity thresholds for weak and strong edges in the Canny detection 
% and image smoothing factor (standard deviation of Gaussian filter)
threshold = [0.03,0.1]; smooth_sigma = 1.0;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Set up a few more things %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% create figure to plot images on
fig_size = [1200,600]; % for taller images, increase the second number
plot_width = 0.48; % width of each plot in normalized figure units
figure('Position',[50,50,fig_size(1),fig_size(2)])
x_offset = 0.5*(0.5 - plot_width);

% shift in input locations between consecutive images to account for model
% motion - only updated if predict_motion == 1
dx = 0; dy = 0;

% preallocate edge vectors for speed
nmax = 20e3; % maximum number of edge points - can be increased if necessary

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Start edge-tracing procedure %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for kk = 1:nframes
  
  % read in image - change depending on which sequence is desired
    imfile = fullfile(sprintf('%s',im_dir),sprintf('cone_%03d.tif',frames(kk)));
  I = imread(imfile);
  
  % get image size to scale plot sizes
  if kk == 1
    im_size = size(I);
    plot_height = plot_width*fig_size(1)/fig_size(2)*im_size(1)/im_size(2); % get matching plot height
    y_offset = 0.5*(1-plot_height);
  end
  
  % edge detection (without edge-thinning)
  [E,thresh] = edge(I,edge_type,threshold,smooth_sigma);
  
  Einv = -E+1; % take the negative so more easily visible on plot 
  Ia = imadjust(I); % improves image contrast for plotting
  
  % show images
  if kk == 1
    hedge = axes('Position',[x_offset,y_offset,plot_width,plot_height]);
    himage = axes('Position',[0.5+x_offset,y_offset,plot_width,plot_height]);
  end
  % edge image
  axes(hedge)
  imshow(Einv)
  axis off
  % enhanced image for identifying any problematic areas
  axes(himage)
  imshow(Ia)
  axis off

  % input start points (and stopping) points
  axes(hedge)
 
  if stopstart == 0
    title(sprintf('Frame %d\nPlease select all starting points for tracing edge segments',kk))
    [x_inp,y_inp] = ginput(no_segs);
  else
    title(sprintf('Frame %d\nPlease select all starting/stopping point pairs for tracing edge segments',kk))
    [x_inp,y_inp] = ginput(2*no_segs);
  end
  
  % find nearest edge points in E to inputted points
  [x_out, y_out] = find_nearest_edge_points(E,x_inp,y_inp);
  if stopstart == 0
    % only start points
    x_start = x_out;
    y_start = y_out;
  else
    % if stop points included, every second point is a stop point
    x_start = x_out(1:2:end);
    y_start = y_out(1:2:end);
    x_stop = x_out(2:2:end);
    y_stop = y_out(2:2:end);
  end
  
  
  % trace the edges from initial starting points (to stopping points)
  
  for ii=1:no_segs

    if stopstart == 0
      [x_seg,y_seg] = trace_edge(E,x_start(ii),y_start(ii),...
        directions{ii},orient{ii});
    else
      [x_seg,y_seg] = trace_edge_stopstart(E,x_start(ii),y_start(ii),...
        x_stop(ii),y_stop(ii),directions{ii},orient{ii});
    end

    if ii == 1
        xc1 = x_seg;
        yc1 = y_seg;
    else
        xc2 = x_seg; 
        yc2 = y_seg; 
    end

  end

  
  
  % Plot points on images
  axes(hedge)
  hold on
  plot(xc1,yc1,'ro','MarkerFaceColor','r','MarkerEdgeColor','r',...
    'MarkerSize',2)
  plot(xc2,yc2,'bo','MarkerFaceColor','b','MarkerEdgeColor','b',...
    'MarkerSize',2)
  hold off
  axes(himage)
  
  % wait for user to press a key before progressing
  title('Press any key to save edge points and continue')
  pause

  % write edge points to file
  edge_file = fullfile(sprintf('%s',edge_dir),sprintf('frame%d_upper.txt',frames(kk)));
  fid = fopen(edge_file,'wt');
  A = [xc1; yc1];
  fprintf(fid,'x\ty\n');
  fprintf(fid,'%f\t%f\n',A);
  fclose(fid);

  edge_file = fullfile(sprintf('%s',edge_dir),sprintf('frame%d_lower.txt',frames(kk)));
  fid = fopen(edge_file,'wt');
  A = [xc2; yc2];
  fprintf(fid,'x\ty\n');
  fprintf(fid,'%f\t%f\n',A);
  fclose(fid);
    
end

