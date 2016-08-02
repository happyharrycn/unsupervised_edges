function model = edgesTrain( trnImgDir, trnGtDir, varargin )
% Train structured edge detector.
%
% For an introductory tutorial please see edgesDemo.m.
%
% USAGE
%  opts = edgesTrain()
%  model = edgesTrain( trnImgDir, trnGtDir, opts )
%
% INPUTS
%  trnImgDir  - folder with all training images
%  trnGtDir   - folder with all edge maps
%  opts       - parameters (struct or name/value pairs)
%   (1) model parameters:
%   .imWidth    - [32] width of image patches
%   .gtWidth    - [16] width of ground truth patches
%   (2) tree parameters:
%   .nPos       - [5e5] number of positive patches per tree
%   .nNeg       - [5e5] number of negative patches per tree
%   .nImgs      - [inf] maximum number of images to use for training
%   .nTrees     - [8] number of trees in forest to train
%   .fracFtrs   - [1/4] fraction of features to use to train each tree
%   .minCount   - [1] minimum number of data points to allow split
%   .minChild   - [8] minimum number of data points allowed at child nodes
%   .maxDepth   - [64] maximum depth of tree
%   .discretize - ['pca'] options include 'pca' and 'kmeans'
%   .nSamples   - [256] number of samples for clustering structured labels
%   .nClasses   - [2] number of classes (clusters) for binary splits
%   .split      - ['gini'] options include 'gini', 'entropy' and 'twoing'
%   (3) feature parameters:
%   .nOrients   - [4] number of orientations per gradient scale
%   .grdSmooth  - [0] radius for image gradient smoothing (using convTri)
%   .chnSmooth  - [2] radius for reg channel smoothing (using convTri)
%   .simSmooth  - [8] radius for sim channel smoothing (using convTri)
%   .normRad    - [4] gradient normalization radius (see gradientMag)
%   .shrink     - [2] amount to shrink channels
%   .nCells     - [5] number of self similarity cells
%   .rgbd       - [0] 0:RGB, 1:depth, 2:RBG+depth (for NYU data only)
%   (4) detection parameters (can be altered after training):
%   .stride     - [2] stride at which to compute edges
%   .multiscale - [0] if true run multiscale edge detector
%   .sharpen    - [2] sharpening amount (can only decrease after training)
%   .nTreesEval - [4] number of trees to evaluate per location
%   .nThreads   - [4] number of threads for evaluation of trees
%   .nms        - [0] if true apply non-maximum suppression to edges
%   (5) other parameters:
%   .seed       - [1] seed for random stream (for reproducibility)
%   .useParfor  - [0] if true train trees in parallel (memory intensive)
%   .modelDir   - ['models/'] target directory for storing models
%   .modelFnm   - ['model'] model filename
%   .bsdsDir    - ['BSR/BSDS500/data/'] location of BSDS dataset
%   .scale      = [1] scale factor for gt and images
%
% OUTPUTS
%  model      - trained structured edge detector w the following fields
%   .opts       - input parameters and constants
%   .thrs       - [nNodes x nTrees] threshold corresponding to each fid
%   .fids       - [nNodes x nTrees] feature ids for each node
%   .child      - [nNodes x nTrees] index of child for each node
%   .count      - [nNodes x nTrees] number of data points at each node
%   .depth      - [nNodes x nTrees] depth of each node
%   .eBins      - data structure for storing all node edge maps
%   .eBnds      - data structure for storing all node edge maps
%
% EXAMPLE
%
% See also edgesDemo, edgesChns, edgesDetect, forestTrain
%
% Structured Edge Detection Toolbox      Version 3.01
% Code written by Piotr Dollar, 2014. Modified by Yin
% Licensed under the MSR-LA Full Rights License [see license.txt]

% get default parameters
% I have hard coded some param here
dfs={'imWidth',32, 'gtWidth',16, 'nPos', 5e5, 'nNeg', 5e5, 'nImgs',inf, ...
  'nTrees',8, 'fracFtrs',1/4, 'minCount',1, 'minChild',8, ...
  'maxDepth',64, 'discretize','pca', 'nSamples',256, 'nClasses',2, ...
  'split','gini', 'nOrients',4, 'grdSmooth',0, 'chnSmooth',2, ...
  'simSmooth',8, 'normRad',4, 'shrink',2, 'nCells',5, 'rgbd',0, ...
  'stride',2, 'multiscale',1, 'sharpen',2, 'nTreesEval',4, ...
  'nThreads', 4, 'nms',0, 'seed',1, 'useParfor', 1, 'modelDir','./tmp/', ...
  'modelFnm','model', 'scale', 1, 'nGt', 4, 'threshBracket', [0.1 0.4]};
opts = getPrmDflt(varargin,dfs,1);
if(nargin==0), model=opts; return; end

% if forest exists load it and return
% modelDir -> param.tmpFolder
% modelFnm -> param.dataset + param.iter
cd(fileparts(mfilename('fullpath')));
forestDir = [opts.modelDir '/forest/'];
forestFn = [forestDir opts.modelFnm];
if(exist([forestFn '.mat'], 'file'))
  load([forestFn '.mat']); return; end

% compute constants and store in opts
nTrees=opts.nTrees; nCells=opts.nCells; shrink=opts.shrink;
opts.nPos=round(opts.nPos); opts.nNeg=round(opts.nNeg);
opts.nTreesEval=min(opts.nTreesEval,nTrees);
opts.stride=max(opts.stride,shrink);
imWidth=opts.imWidth; gtWidth=opts.gtWidth;
imWidth=round(max(gtWidth,imWidth)/shrink/2)*shrink*2;
opts.imWidth=imWidth; opts.gtWidth=gtWidth;
nChnsGrad=(opts.nOrients+1)*2; nChnsColor=3;
nChns = nChnsGrad+nChnsColor; opts.nChns = nChns;
opts.nChnFtrs = imWidth*imWidth*nChns/shrink/shrink;
opts.nSimFtrs = (nCells*nCells)*(nCells*nCells-1)/2*nChns;
opts.nTotFtrs = opts.nChnFtrs + opts.nSimFtrs; disp(opts);

% generate stream for reproducibility of model
stream=RandStream('mrg32k3a','Seed',opts.seed);

% train nTrees random trees (can be trained with parfor if enough memory)
if(opts.useParfor),
  parfor i=1:nTrees,
    trainTree(trnImgDir, trnGtDir, opts,stream,i);
  end
else
  for i=1:nTrees,
    trainTree(trnImgDir, trnGtDir, opts,stream,i);
  end
end

% merge trees and save model
model = mergeTrees( opts );
if(~exist(forestDir,'dir')), mkdir(forestDir); end
save([forestFn '.mat'], 'model', '-v7.3');

end

%% merging all trees into forest
function model = mergeTrees( opts )
% accumulate trees and merge into final model
nTrees=opts.nTrees; gtWidth=opts.gtWidth;
treeFn = [opts.modelDir '/tree/' opts.modelFnm '_tree'];
for i=1:nTrees
  t=load([treeFn int2str2(i,3) '.mat'],'tree'); t=t.tree;
  if(i==1), trees=t(ones(1,nTrees)); else trees(i)=t; end
end
nNodes=0; for i=1:nTrees, nNodes=max(nNodes,size(trees(i).fids,1)); end
% merge all fields of all trees
model.opts=opts; Z=zeros(nNodes,nTrees,'uint32');
model.thrs=zeros(nNodes,nTrees,'single');
model.fids=Z; model.child=Z; model.count=Z; model.depth=Z;
model.segs=zeros(gtWidth,gtWidth,nNodes,nTrees,'uint8');
for i=1:nTrees, tree=trees(i); nNodes1=size(tree.fids,1);
  model.fids(1:nNodes1,i) = tree.fids;
  model.thrs(1:nNodes1,i) = tree.thrs;
  model.child(1:nNodes1,i) = tree.child;
  model.count(1:nNodes1,i) = tree.count;
  model.depth(1:nNodes1,i) = tree.depth;
  model.segs(:,:,1:nNodes1,i) = tree.hs-1;
end
% remove very small segments (<=5 pixels)
segs=model.segs; nSegs=squeeze(max(max(segs)))+1;
parfor i=1:nTrees*nNodes, m=nSegs(i);
  if(m==1), continue; end; S=segs(:,:,i); del=0;
  for j=1:m, Sj=(S==j-1); if(nnz(Sj)>5), continue; end
    S(Sj)=median(single(S(convTri(single(Sj),1)>0))); del=1; end
  if(del), [~,~,S]=unique(S); S=reshape(S-1,gtWidth,gtWidth);
    segs(:,:,i)=S; nSegs(i)=max(S(:))+1; end
end
model.segs=segs; model.nSegs=nSegs;
% store compact representations of sparse binary edge patches
nBnds=opts.sharpen+1; eBins=cell(nTrees*nNodes,nBnds);
eBnds=zeros(nNodes*nTrees,nBnds);
parfor i=1:nTrees*nNodes
  if(model.child(i) || model.nSegs(i)==1), continue; end %#ok<PFBNS>
  E=gradientMag(single(model.segs(:,:,i)))>.01; E0=0;
  % eBins stores the sparse edge coordinates for all segments
  % eBnds stores the number of edge pixels for all segments
  for j=1:nBnds, eBins{i,j}=uint16(find(E & ~E0)'-1); E0=E;
    eBnds(i,j)=length(eBins{i,j}); E=convTri(single(E),1)>.01; end
end
eBins=eBins'; model.eBins=[eBins{:}]';
% eBnds now stores the index of each sparse edge structure
eBnds=eBnds'; model.eBnds=uint32([0; cumsum(eBnds(:))]);
end

%% the main function for training
function trainTree( trnImgDir, trnGtDir, opts, stream, treeInd )
% Train a single tree in forest model.

% location of ground truth
% note we will only train on the images with GT results
imgIds=dir(fullfile(trnGtDir, '*.png'));
fileExt = '.png';
if isempty(imgIds);
  imgIds = dir(fullfile(trnGtDir, '*.jpg'));
  fileExt = '.jpg';
end
imgIds=imgIds([imgIds.bytes]>0);
imgIds={imgIds.name};
nImgs=length(imgIds);
for i=1:nImgs,
  imgIds{i}=imgIds{i}(1:end-4);
end

% extract commonly used options
imWidth=opts.imWidth; imRadius=imWidth/2;
gtWidth=opts.gtWidth; gtRadius=gtWidth/2;
nChns=opts.nChns; nTotFtrs=opts.nTotFtrs;
nPos=opts.nPos; nNeg=opts.nNeg; shrink=opts.shrink;

% finalize setup
treeDir = [opts.modelDir '/tree/'];
treeFn = [treeDir opts.modelFnm '_tree'];
if(exist([treeFn int2str2(treeInd,3) '.mat'],'file'))
  fprintf('Reusing tree %d of %d\n',treeInd,opts.nTrees); return; end
fprintf('\n-------------------------------------------\n');
fprintf('Training tree %d of %d\n',treeInd,opts.nTrees); tStart=clock;

% set global stream to stream with given substream (will undo at end)
streamOrig = RandStream.getGlobalStream();
set(stream,'Substream',treeInd);
RandStream.setGlobalStream( stream );

% collect positive and negative patches and compute features
fids=sort(randperm(nTotFtrs,round(nTotFtrs*opts.fracFtrs)));
k = nPos+nNeg; nImgs=min(nImgs,opts.nImgs);
ftrs = zeros(k,length(fids),'single');
labels = zeros(gtWidth,gtWidth,k,'uint8'); k = 0;
tid = ticStatus('Collecting data',30,1);

% modified by YL
for i = 1:nImgs
  % get image and compute channels
  me=imread(fullfile(trnGtDir, [imgIds{i} fileExt]));
  me = im2double(me);
  I=imread(fullfile(trnImgDir, [imgIds{i} fileExt]));
  nGt = opts.nGt;
  
  % resize the image and groundtruth if necessary
  if opts.scale < 1
    %me = imresize(me, opts.scale);
    I = imresize(I, opts.scale);
    assert(size(I,1)==size(me,1) && size(I,2)==size(me,2))
  end
  
  % get image features
  siz=size(I);
  p=zeros(1,4); p([2 4])=mod(4-mod(siz(1:2),4),4);
  if(any(p)), I=imPad(I,p,'symmetric'); end
  [chnsReg,chnsSim] = edgesChns(I,opts);
  
  % sample positive and negative locations
  xy=[]; k1=0; B=false(siz(1),siz(2));
  B(shrink:shrink:end,shrink:shrink:end)=1;
  B([1:imRadius end-imRadius:end],:)=0;
  B(:,[1:imRadius end-imRadius:end])=0;
  
  % generate the threshlist
  thresh = exp(linspace(log(opts.threshBracket(1)), log(opts.threshBracket(2)), nGt));
  gt = cell([1 nGt]);
  
  % all we care about is the pretty positive samples!
  Mneg=~(bwdist(me>0.1)<gtRadius);
  posLabels = [];
  for j=1:nGt
    
    % threshold the motion boundary to get the a boundary map
    Mpos=(me>thresh(j)); gt{j} = Mpos;
    % get the pos sample (if there is a boundary pixel within the patch)
    Mpos(bwdist(Mpos)<gtRadius)=1;
    [y,x]=find(Mpos.*B); k2=min(length(y),ceil(2*nPos/nImgs/nGt));
    rp=randperm(length(y),k2); y=y(rp); x=x(rp);
    xy=[xy; x y ones(k2,1)*j]; k1=k1+k2; %#ok<AGROW>
    posLabels = [posLabels; ones(k2, 1)];
    
    % lower thrshold for neg samples
    [y,x]=find(Mneg.*B); k2=min(length(y),ceil(nNeg/nImgs/nGt));
    rp=randperm(length(y),k2); y=y(rp); x=x(rp);
    xy=[xy; x y ones(k2,1)*j]; k1=k1+k2; %#ok<AGROW>
    posLabels = [posLabels; zeros(k2, 1)];
    
  end
  
  % k1 is the maximum number of samples, but we do not know the exact
  % number until we sample them
  k1 = ceil((nPos + nNeg)/nImgs); kSamples = length(posLabels);
  
  if(k1>size(ftrs,1)-k), k1=size(ftrs,1)-k; xy=xy(1:k1,:); end
  % crop patches and ground truth labels
  psReg=zeros(imWidth/shrink,imWidth/shrink,nChns,k1,'single');
  lbls=zeros(gtWidth,gtWidth,k1,'uint8');
  psSim=psReg; ri=imRadius/shrink; rg=gtRadius;
  
  % check each local patch
  curSampleIdx = 0;
  for j=1:kSamples,
    if curSampleIdx < k1
      % get the sample coordinate
      xy1=xy(j,:); xy2=xy1/shrink;
      
      % crop boundary patch -> find super pixel using boundary map
      % -> remove boundary pixels
      e = gt{xy1(3)}(xy1(2)-rg+1:xy1(2)+rg,xy1(1)-rg+1:xy1(1)+rg);
      s = bwlabel(~e, 4);
      t = spDetectMex('boundaries',uint32(s),single(e),0,1); t = t+ 1;
      if(all(t(:)==t(1)) && posLabels(j)==1),
        continue; % this is not a good edge sample, so skip it
      elseif all(t(:)==t(1))
        curSampleIdx = curSampleIdx + 1;
        lbls(:,:,curSampleIdx)=1; % negtive sample
      else
        curSampleIdx = curSampleIdx + 1;
        [~,~,t]=unique(t);
        lbls(:,:,curSampleIdx)=reshape(t,gtWidth,gtWidth); % positive sample
      end
     
      % crop the feature
      psReg(:,:,:,curSampleIdx)=chnsReg(xy2(2)-ri+1:xy2(2)+ri,xy2(1)-ri+1:xy2(1)+ri,:);
      psSim(:,:,:,curSampleIdx)=chnsSim(xy2(2)-ri+1:xy2(2)+ri,xy2(1)-ri+1:xy2(1)+ri,:);
    end
    
  end
  
  % the actual number of samples
  k1 = curSampleIdx; psReg = psReg(:,:,:,1:k1); psSim = psSim(:,:,:,1:k1);
  lbls = lbls(:,:,1:k1);
  
  % visualization code
%   if(1), figure(1); montage2(squeeze(psReg(:,:,1,:))); drawnow; end
%   if(1), figure(2); montage2(lbls(:,:,:)); drawnow; end
%   if(1), figure(3); imshow(me); drawnow; end
%   pause
  
  % compute features and store
  ftrs1=[reshape(psReg,[],k1)' stComputeSimFtrs(psSim,opts)];
  ftrs(k+1:k+k1,:)=ftrs1(:,fids); labels(:,:,k+1:k+k1)=lbls;
  k=k+k1; if(k==size(ftrs,1)), tocStatus(tid,1); break; end
  tocStatus(tid,i/nImgs);
end
if(k<size(ftrs,1)), ftrs=ftrs(1:k,:); labels=labels(:,:,1:k); end

% train structured edge classifier (random decision tree)
pTree=struct('minCount',opts.minCount, 'minChild',opts.minChild, ...
  'maxDepth',opts.maxDepth, 'H',opts.nClasses, 'split',opts.split);
t=labels; labels=cell(k,1); for i=1:k, labels{i}=t(:,:,i); end
pTree.discretize=@(hs,H) discretize(hs,H,opts.nSamples,opts.discretize);
tree=forestTrain(ftrs,labels,pTree); tree.hs=cell2array(tree.hs);
tree.fids(tree.child>0) = fids(tree.fids(tree.child>0)+1)-1;
if(~exist(treeDir,'dir')), mkdir(treeDir); end
save([treeFn int2str2(treeInd,3) '.mat'],'tree'); e=etime(clock,tStart);
fprintf('Training of tree %d complete (time=%.1fs).\n',treeInd,e);
RandStream.setGlobalStream( streamOrig );

end

%% sim features
function ftrs = stComputeSimFtrs( chns, opts )
% Compute self-similarity features (order must be compatible w mex file).
w=opts.imWidth/opts.shrink; n=opts.nCells; if(n==0), ftrs=[]; return; end
nSimFtrs=opts.nSimFtrs; nChns=opts.nChns; m=size(chns,4);
inds=round(w/n/2); inds=round((1:n)*(w+2*inds-1)/(n+1)-inds+1);
chns=reshape(chns(inds,inds,:,:),n*n,nChns,m);
ftrs=zeros(nSimFtrs/nChns,nChns,m,'single');
k=0; for i=1:n*n-1, k1=n*n-i; i1=ones(1,k1)*i;
  ftrs(k+1:k+k1,:,:)=chns(i1,:,:)-chns((1:k1)+i,:,:); k=k+k1; end
ftrs = reshape(ftrs,nSimFtrs,m)';
end

%% discretize the edge patches
function [hs,segs] = discretize( segs, nClasses, nSamples, type )
% Convert a set of segmentations into a set of labels in [1,nClasses].
persistent cache; w=size(segs{1},1); assert(size(segs{1},2)==w);
if(~isempty(cache) && cache{1}==w), [~,is1,is2]=deal(cache{:}); else
  % compute all possible lookup inds for w x w patches
  is=1:w^4; is1=floor((is-1)/w/w); is2=is-is1*w*w; is1=is1+1;
  kp=is2>is1; is1=is1(kp); is2=is2(kp); cache={w,is1,is2};
end
% compute n binary codes zs of length nSamples
nSamples=min(nSamples,length(is1)); kp=randperm(length(is1),nSamples);
n=length(segs); is1=is1(kp); is2=is2(kp); zs=false(n,nSamples);
for i=1:n, zs(i,:)=segs{i}(is1)==segs{i}(is2); end
zs=bsxfun(@minus,zs,sum(zs,1)/n); zs=zs(:,any(zs,1));
if(isempty(zs)), hs=ones(n,1,'uint32'); segs=segs{1}; return; end
% find most representative segs (closest to mean)
[~,ind]=min(sum(zs.*zs,2)); segs=segs{ind};
% apply PCA to reduce dimensionality of zs
U=pca(zs'); d=min(5,size(U,2)); zs=zs*U(:,1:d);
% discretize zs by clustering or discretizing pca dimensions
d=min(d,floor(log2(nClasses))); hs=zeros(n,1);
for i=1:d, hs=hs+(zs(:,i)<0)*2^(i-1); end
[~,~,hs]=unique(hs); hs=uint32(hs);
if(strcmpi(type,'kmeans'))
  nClasses1=max(hs); C=zs(1:nClasses1,:);
  for i=1:nClasses1, C(i,:)=mean(zs(hs==i,:),1); end
  hs=uint32(kmeans2(zs,nClasses,'C0',C,'nIter',1));
end
% optionally display different types of hs
for i=1:0, figure(i); montage2(cell2array(segs(hs==i))); end
end
