% running our pipeline
clear;
clc;

%% this is the top level script for our pipeline
% set up the param struct for all datasets
sintelParam = globalParam('sintel');
videoParam = globalParam('video');
bsdsParam = globalParam('bsds');

% setup iteration numbers
prevFolder = []; currFolder = [];
numIter = 4; startIter = 1;
% we gradually increase our threshold for each iteration (10%)
ratio = 1.1;

% buffer resutls into txt file
fid = fopen('new_result.txt', 'w');

% create tmp folder if it does not exist
if ~isdir(videoParam.tmpFolder)
  mkdir(videoParam.tmpFolder)
end

% re-allocate matlab parpool if any
delete(gcp); pObj = parpool(videoParam.numProc);

%% For each iteration (except the first one), we follow 5 steps
% (1) get motion edges from previous flow
% (2) use motion edges to train an edge detector
% (3) apply the detector to input frames
% (4) re-estimate flow using new edge map
% (5) benchmark edge / flow results
for iter = startIter:numIter
  
  %% set up folders for exps
  prevFolder = sprintf('iter_%03d', iter-1);
  currFolder = sprintf('iter_%03d', iter);

  %% (1) finding motion boundary
  if iter == 1
    % for the first iteration, we need to
    % 1) get init flow results with soble (also randomly sample pairs)
    % 2) get motion boundary using sobel
    initFlows(videoParam, prevFolder);
    initMotEdges(videoParam, prevFolder, currFolder);
    threshBracket = [0.4 0.8];
  else
    oldModel = load(fullfile(videoParam.tmpFolder, 'forest', ...
      [videoParam.dataset '_' prevFolder '.mat']));
    oldModel = oldModel.model;
    detAllMotEdges(videoParam, prevFolder, currFolder, oldModel);
    oldModel = [];
    threshBracket = [0.05 0.4]*ratio;
    % gradually increase the threshold over iterations
    ratio = ratio * ratio;
  end
  
  %% (2) motion boundary -> train edge detector
  if iter == numIter
    % re-allocate matlab parpool (saving us memory)
    delete(gcp); pObj = parpool(2);
    % increase the number of training samples for the last iter (boosting performance a bit)
    model = edgesTrain( videoParam.imgPath, fullfile(videoParam.motEdgePath, currFolder), ...
      'modelFnm', [videoParam.dataset '_' currFolder], 'scale', videoParam.scale, ...
      'modelDir', videoParam.tmpFolder, 'threshBracket', threshBracket, 'nPos', 2e6, 'nNeg', 2e6);
    % recover the matlab pool
    delete(gcp); pObj = parpool(videoParam.numProc);
  else 
    model = edgesTrain( videoParam.imgPath, fullfile(videoParam.motEdgePath, currFolder), ...
      'modelFnm', [videoParam.dataset '_' currFolder], 'scale', videoParam.scale, ...
      'modelDir', videoParam.tmpFolder, 'threshBracket', threshBracket);
  end

  %% (3) run the edge detector using the trained model
  detAllEdges(model, videoParam, currFolder);
  
  %% (4) re-estimate the flow using detected edges
  % (also randomly sample pairs)
  calcAllFlows(videoParam, currFolder);
  
  %% (5) benchmark for edge/flow (kind of slow)
  % benchmark edge detection results on bsds
  if iter == numIter
    [ODS, OIS, ODT, AP, R50] = evalEdgeResults(model, currFolder, 'accurate');
  else
    [ODS, OIS, ODT, AP, R50] = evalEdgeResults(model, currFolder, 'fast');
  end
  
  % bechmarks optical flow results on sintel
  if isdir(sintelParam.imgPath)
    detAllEdges(model, sintelParam, currFolder);
    calcAllFlows(sintelParam, currFolder);
    [errList, aep] = evalFlowResults(currFolder);
  else 
    aep = 0;
  end
  
  %% printing out the results
  fprintf(fid, '%d iteration: ODS %0.3f (ODT %0.3f), OIS %0.3f, AP %0.3f AEP %0.3f\n', ...
    iter, ODS, ODT, OIS, AP, aep);
  
end

%% close file output, all done
fclose(fid);