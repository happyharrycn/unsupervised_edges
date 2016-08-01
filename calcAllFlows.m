function [] = calcAllFlows(param, subfolder, vIter)
% run epic flow using specified edge maps
% subfolder indexes the current iteration of exp
if nargin < 3
  vIter = 5; 
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
    myCmd = sprintf('%s %s %s %s %s -E %s %s -iter %d', ...
      param.efBin, curImgName, nextImgName, dmFileName, outFileName, edgeFileName, param.flowFlag, vIter);
    system(myCmd);
  end
  
end