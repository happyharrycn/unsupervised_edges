function [] = detAllMotEdges(param, prevFolder, currFolder, model)
% detect motion edges on flow field

%% set up folder
flowFolder = fullfile(param.flowPath, prevFolder);
motEdgeFolder = fullfile(param.motEdgePath, currFolder);
imgFolder = param.imgPath;
if ~exist(motEdgeFolder, 'dir')
  mkdir(motEdgeFolder);
end

%% reset model param, note: we need crispy edges for training
model.opts.sharpen = 2;
model.opts.multiscale = 1;
model.opts.nms = 1;
model.opts.nThreads = 1;

%% loop over all flow results
flowList = dir(fullfile(flowFolder, '*.flo'));
parfor i=1:length(flowList)
  
  % check output file
  outFile = fullfile(motEdgeFolder, [flowList(i).name(1:end-4), '.png']);
  imgFile = fullfile(imgFolder, [flowList(i).name(1:end-4), param.fileExt]);
  if exist(outFile, 'file'), continue; end

  % reading flow file
  img = imread(imgFile);
  
  fprintf('Processing flow file...%s\n', outFile)
  flowFile = fullfile(flowFolder, flowList(i).name);
  flow = readFlowFile(flowFile);
  
  % resize the flow (default:by half)
  flow = imresize(flow, param.scale);
  img = imresize(img, param.scale);

  
  % filtering based on flow stats
  flowMag = sqrt(sum(flow.^2, 3)); flowMag = flowMag(:);
  maxFlow = max(flowMag); minFlow = min(flowMag); meanFlow = mean(flowMag); varFlow = var(flowMag);
  if minFlow > 10 || maxFlow<2 || meanFlow < 0.3 || varFlow < 0.5
    continue;
  end

  % run edge detection over the colored flow map
  flowMap = flowToColor(flow);
  motEdge = edgesDetect(flowMap, img, model);

  % refine the edges by align to super pixel boundaries
  seg = spDetect(img, []); mask = (bwdist(seg==0)<3); motEdge = motEdge .* mask;

  % write to png file
  if sum(motEdge(:)>0.4) < 50
    % skip motion edge map without enough samples
    continue;
  end
  imwrite(motEdge, outFile);
  
end