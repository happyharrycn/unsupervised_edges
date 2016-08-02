function param = globalParam(dataset)
% setup all the params

%% set up paths
addpath('./flow_utils');
addpath('./structured_edges');
% put piotr's toolbox path here
addpath(genpath('./toolbox-master'));
% put root data folder here
rootPath = '/home/yin/edge_flow_data';

%% set up datasets
% we will mainly use three datasets:
% BSDS for benchmark edge detection
% Sintel for benchmark optical flow
% Video for learning
% each dataset is defined by a set of frame pairs
allDatasets = {'bsds', 'sintel', 'video'};
% if we will sample the frames (or simply keep them all)
allSampleFrames = [0 0 1];
allScales = [1 1 0.5];
allFlowFlags = {'-sintel', '-sintel', '-sintel'};
allFileExt = {'.jpg', '.png', '.png'};

% set up dataset params
index = strcmp(dataset, allDatasets);
if sum(index)==0, param=[]; return; end
param.dataset = allDatasets{index};
param.flowFlag = allFlowFlags{index};
param.sampleFrames = allSampleFrames(index);
param.scale = allScales(index);
param.fileExt = allFileExt{index};

%% setup image / matching / edge / flow paths
param.rootPath = fullfile(rootPath, param.dataset);
param.imgPath = fullfile(rootPath, param.dataset, 'images');
param.matchPath = fullfile(rootPath, param.dataset, 'matches');
param.edgePath = fullfile(rootPath, param.dataset, 'edges');
param.flowPath = fullfile(rootPath, param.dataset, 'flows');
param.motEdgePath = fullfile(rootPath, param.dataset, 'motEdges');

%% number of samples used for training
param.numSamples = 1000;

%% counter and tmp file folder
param.iter = 0;
param.tmpFolder = './tmp';

%% check dataset stats
param.edgeGT = 0;   param.flowGT = 0;
if exist(fullfile(param.edgePath, 'Groundtruth'), 'dir')
    param.edgeGT = 1;
end
if exist(fullfile(param.flowPath, 'Groundtruth'), 'dir')
    param.flowGT = 1;
end

%% binary for deepmatching and epicflow
param.dmBin = './bins/deepmatching';
param.efBin = './bins/epicflow';

%% for parfor
param.numProc = 6;

