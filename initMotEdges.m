function [] = initMotEdges(param, prevFolder, currFolder)
% initalize motion edges using sobel

%% set up folder
flowFolder = fullfile(param.flowPath, prevFolder);
motEdgeFolder = fullfile(param.motEdgePath, currFolder);
imgFolder = param.imgPath;
if ~exist(motEdgeFolder, 'dir')
  mkdir(motEdgeFolder);
end

%% generate motion edge from flow file
flowList = dir(fullfile(flowFolder, '*.flo'));
for i=1:length(flowList)
  
  % check output file
  outFile = fullfile(motEdgeFolder, [flowList(i).name(1:end-4), param.fileExt]);
  imgFile = fullfile(imgFolder, [flowList(i).name(1:end-4), param.fileExt]);
  if exist(outFile, 'file')
    % skip exisiting files
    continue;
  end

  % reading flow file
  img = imresize(imread(imgFile), param.scale);
  fprintf('Processing flow file...%s\n', outFile)
  flowFile = fullfile(flowFolder, flowList(i).name);
  flow = readFlowFile(flowFile);
  
  % resize the flow (default:by half)
  flow = imresize(flow, param.scale);
  
  % filtering based on flow stats
  flowMag = sqrt(sum(flow.^2, 3)); flowMag = flowMag(:);
  maxFlow = max(flowMag); minFlow = min(flowMag); 
  meanFlow = mean(flowMag); varFlow = var(flowMag);
  if minFlow > 10 || maxFlow<2 || meanFlow < 0.3 || varFlow < 0.5
    continue;
  end

  % run edge detection over the colored flow map (with NMS)
  flowMap = flowToColor(flow); flowMap = single(flowMap)./255;
  flowMap = convTri(flowMap,1); motEdge = sobelDetect(flowMap,1);
  
  % refine the edges by align to super pixel boundaries
  seg = spDetect(img, []); mask = (bwdist(seg==0)<=3); motEdge = motEdge.*mask;
  
  % for debug only
  %if(1)
  %  figure(1), imshow(img); figure(2), imshow(flowMap); 
  %  figure(3), imshow(motEdge); figure(4), imshow(motEdgeMag); pause;
  %end
  
  % write to png file
  if sum(motEdge(:)>0.4) < 50
    % skip motion edge map without enough samples
    continue;
  end
  imwrite(motEdge, outFile);
  
end