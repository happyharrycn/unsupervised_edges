function [] = detAllEdges(model, param, subfolder)
% run edge detector over all images within current dataset
% subfolder indexes the current iteration of our pipeline

%% params
% reset model param (make sure sharpen is open during training)
% turn off nms, which is not needed for flow
model.opts.sharpen = 2;
model.opts.multiscale = 1;
model.opts.nms = 0;
model.opts.nThreads = 1;

%% setup folder structure
imgFolder = param.imgPath;
resFolder = fullfile(param.edgePath, subfolder);
if ~exist(resFolder, 'dir')
  mkdir(resFolder);
end

%% load the image pair list
fid = fopen(fullfile(param.rootPath, sprintf('%s_pairs.txt', param.dataset)), 'r');
pairList =textscan(fid,'%s %s');
fclose(fid);

%% random sample frames?
if param.sampleFrames
  randIdx = randperm(length(pairList{1}));
  numSamples = min(param.numSamples, length(randIdx));
  randIdx = randIdx(1:numSamples);
else
  randIdx = [1:length(pairList{1})];
end

%% run the edge detector
parfor i=1:length(randIdx)
  [~, curFileName, curFileExt] = fileparts(pairList{1}{randIdx(i)});
  outFile = fullfile(resFolder, [curFileName, '.png']);
  if exist(outFile, 'file'), continue; end
  fprintf('Edge Det for %s\n', outFile)
  img = imread(fullfile(imgFolder, [curFileName, curFileExt]));
  edgeMap = edgesDetect(img, img, model); 
  imwrite(edgeMap, outFile);
end