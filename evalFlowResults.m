function [errList, aep] = evalFlowResults(subfolder)
% valuate current flow results on sintel

%% check params
% use sintel param
param = globalParam('sintel');

%% setup folder structure
gtFolder = fullfile(param.flowPath, 'Groundtruth');
resFolder = fullfile(param.flowPath, subfolder);

%% check all files
gtList = dir(fullfile(gtFolder, '*.flo'));
resList = dir(fullfile(resFolder, '*.flo'));
errList = zeros([1 length(gtList)]); aep = 0;
if length(gtList) ~= length(resList)
  fprintf('Can not match flow results\n');
  return;
end

parfor i=1:length(gtList)
  % read all files
  gtFlow = readFlowFile(fullfile(gtFolder, gtList(i).name));
  resFlow = readFlowFile(fullfile(resFolder, gtList(i).name));
  % end point error
  err = sqrt(sum((gtFlow - resFlow).^2, 3));
  errList(i) = mean(err(:));
end
aep = mean(errList);
fprintf('Optical flow average end point error %0.3f\n', aep)
