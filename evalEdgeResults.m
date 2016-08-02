function [ODS, OIS, ODT, AP, R50] = evalEdgeResults(model, subfolder, mode)
% evaluate current edge det model on bsds
% subfolder indexes the current iteration of exp

%% check params
if nargin < 3
  mode = 'fast';
end

% use bsds param
param = globalParam('bsds');

% reset model param (make sure sharpen is open during training)
model.opts.sharpen = 2;
model.opts.multiscale = 1;
model.opts.nms = 1;

%% setup folder structure (bsds is a special case)
imgFolder = fullfile(param.imgPath, 'test');
gtFolder = fullfile(param.edgePath, 'Groundtruth', 'test');
resFolder = fullfile(param.edgePath, subfolder);
if ~exist(resFolder, 'dir')
  mkdir(resFolder);
end

%% run the detector
imgList = dir(fullfile(imgFolder, '*.jpg'));
parfor i=1:length(imgList)
  outfile = fullfile(resFolder, [imgList(i).name(1:end-4) '.png']);
  if exist(outfile, 'file'), continue; end
  img = imread(fullfile(imgFolder, imgList(i).name));
  edgeMap = edgesDetect(img, img, model);
  imwrite(edgeMap, outfile);
end

%% run the benchmark
if strcmp(mode, 'fast')
    tic
    % fast and less accurate evaluation
    [ODS,~,~,ODT,OIS,~,~,AP,R50] = edgesEvalDirFast('resDir', resFolder,...
      'gtDir', gtFolder, 'thrs', 39);
    toc
    fprintf('ODS: %0.3f, OIS: %0.3f, AP: %0.3f\n', ODS, OIS, AP)
    if( 1 ), figure(1); edgesEvalPlot(resFolder, sprintf('%s', strrep(subfolder, '_', ' '))); end
elseif strcmp(mode, 'accurate')
    tic
    % slow and accurate evaluation
    [ODS,~,~,ODT,OIS,~,~,AP,R50] = edgesEvalDir('resDir', resFolder,...
      'gtDir', gtFolder, 'thrs', 99);
    toc
    fprintf('ODS: %0.3f, OIS: %0.3f, AP: %0.3f\n', ODS, OIS, AP)
    if( 1 ), figure(1); edgesEvalPlot(resFolder, sprintf('%s', strrep(subfolder, '_', ' '))); end
else
    fprintf('Mode not recognized\n');
    ODS=0; OIS=0; ODT=0; AP=0; R50=0;
end




