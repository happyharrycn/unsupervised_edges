function [] = initFlows(param, subfolder)
% run epic flow with sobel edges 
% resutls are stored at subfolder

%% set up folder
% deep match / image folder
dmFolder = param.matchPath;
imgFolder = param.imgPath;
% flow folder (create if not exists)
flowFolder = fullfile(param.flowPath, subfolder);
if ~exist(flowFolder, 'dir')
  mkdir(flowFolder);
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

%% loop over every pair of images (indexed by dmFolder)
parfor i=1:length(randIdx)
  
  % image pairs
  [~, curFileName, ~] = fileparts(pairList{1}{randIdx(i)});
  [~, nextFileName, ~] = fileparts(pairList{2}{randIdx(i)});
  curImgName = fullfile(imgFolder, [curFileName, param.fileExt]);
  nextImgName = fullfile(imgFolder, [nextFileName, param.fileExt]);
  
  % deep matching results
  dmFileName = fullfile(dmFolder, [curFileName '.dm']);
  
  % output file
  outFileName = fullfile(flowFolder, [curFileName '.flo']);
  
  % call epic flow only if necessary
  if ~exist(outFileName, 'file')
    fprintf('EpicFlow for %s\n', outFileName)
    myCmd = sprintf('%s %s %s %s %s -sobel %s', ...
      param.efBin, curImgName, nextImgName, dmFileName, outFileName, param.flowFlag);
    system(myCmd);
  end
  
end