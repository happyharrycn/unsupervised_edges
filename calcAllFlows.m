function [] = calcAllFlows(param, subfolder, vIter, kAlpha)
% run epic flow using specified edge maps

if nargin < 4
  % by default we will smooth the flow map a bit more
  kAlpha = 1.1;
end

if nargin < 3
  vIter = 5; kAlpha = 1.1;
end

%% set up folder
dmFolder = param.matchPath;
imgFolder = param.imgPath;
edgeFolder = fullfile(param.edgePath, subfolder);
flowFolder = fullfile(param.flowPath, subfolder);
if ~exist(flowFolder, 'dir')
  mkdir(flowFolder);
end

%% load the image pair list
fid = fopen(fullfile(param.rootPath, sprintf('%s_pairs.txt', param.dataset)), 'r');
pairList = textscan(fid,'%s %s');
fclose(fid);

%% compute flow for image pairs with edge maps
parfor i=1:length(pairList{1})
  
  % image pairs
  [~, curFileName, ~] = fileparts(pairList{1}{i});
  [~, nextFileName, ~] = fileparts(pairList{2}{i});
  curImgName = fullfile(imgFolder, [curFileName, param.fileExt]);
  nextImgName = fullfile(imgFolder, [nextFileName, param.fileExt]);
  
  % deep matching results
  dmFileName = fullfile(dmFolder, [curFileName '.dm']);
  
  % edge results
  edgeFileName = fullfile(edgeFolder, [curFileName '.png']);
  
  % output file
  outFileName = fullfile(flowFolder, [curFileName '.flo']);
  
  % call epic flow only if edge file exists
  if exist(edgeFileName, 'file') && ~exist(outFileName, 'file')
    fprintf('EpicFlow for %s\n', outFileName)
    % note all flow param should be after flow flag!
    myCmd = sprintf('%s %s %s %s %s -E %s %s -iter %d -alpha %f', ...
      param.efBin, curImgName, nextImgName, dmFileName, outFileName, edgeFileName, param.flowFlag, vIter, kAlpha);
    system(myCmd);
  end
  
end