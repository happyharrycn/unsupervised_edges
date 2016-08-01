function [thrs,cntR,sumR,cntP,sumP,V] = edgesEvalImgFast( E, G, varargin )
% Calculate edge precision/recall results for single edge image.
%
% Enhanced replacement for evaluation_bdry_image() from BSDS500 code:
%  http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/
% Uses same format and is fully compatible with evaluation_bdry_image.
% Given default prms results are *identical* to evaluation_bdry_image.
%
% In addition to performing the evaluation, this function can optionally
% create a visualization of the matches and errors for a given edge result.
% The visualization of edge matches V has the following color coding:
%  green=true positive, blue=false positive, red=false negative
% If multiple ground truth labels are given the false negatives have
% varying strength (and true positives can match *any* ground truth).
%
% This function calls the mex file correspondPixels. Pre-compiled binaries
% for some systems are provided in /private, source for correspondPixels is
% available as part of the BSDS500 dataset (see link above). Note:
% correspondPixels is computationally expensive and very slow in practice.
%
% USAGE
%  [thrs,cntR,sumR,cntP,sumP,V] = edgesEvalImg( E, G, [prms] )
%
% INPUTS
%  E          - [h x w] edge probability map (may be a filename)
%  G          - file containing a cell of ground truth boundaries
%  prms       - parameters (struct or name/value pairs)
%   .out        - [''] optional output file for writing results
%   .thrs       - [99] number or vector of thresholds for evaluation
%   .maxDist    - [.0075] maximum tolerance for edge match
%   .thin       - [1] if true thin boundary maps
%   .scale      = [1] resize the gt and image for evalution
%
% OUTPUTS
%  thrs       - [Kx1] vector of threshold values
%  cntR,sumR  - [Kx1] ratios give recall per threshold
%  cntP,sumP  - [Kx1] ratios give precision per threshold
%  V          - [hxwx3xK] visualization of edge matches
%
% EXAMPLE
%
% See also edgesEvalDirFast
%
% Structured Edge Detection Toolbox      Version 3.01
% Code written by Piotr Dollar, 2014. Modified by Yin Li
% Licensed under the MSR-LA Full Rights License [see license.txt]

% get additional parameters
dfs={ 'out','', 'thrs',39, 'maxDist',.0075, 'thin',1 , 'scale', 1.0};
[out,thrs,maxDist,thin, scale] = getPrmDflt(varargin,dfs,1);
if(any(mod(thrs,1)>0)), K=length(thrs); thrs=thrs(:); else
  K=thrs; thrs=linspace(1/(K+1),1-1/(K+1),K)'; end

% load edges (E) and ground truth (G)
if(all(ischar(E))), 
  E=double(imread(E))/255;
  if (scale-1.0) > eps
    E=imresize(E, scale, 'nearest');
  end
end
G=load(G); G=G.groundTruth; n=length(G);
for g=1:n,
  G{g}=double(G{g}.Boundaries);
  if abs(scale-1.0) > eps
    G{g}=imresize(G{g}, scale, 'nearest');
  end
end

% evaluate edge result at each threshold
Z=zeros(K,1); cntR=Z; sumR=Z; cntP=Z; sumP=Z;
if(nargout>=6), V=zeros([size(E) 3 K]); end
for k = 1:K
  % threshhold and thin E
  E1 = double(E>=max(eps,thrs(k)));
  if(thin), E1=double(bwmorph(E1,'thin',inf)); end
  % compare to each ground truth in turn and accumualte
  Z=zeros(size(E)); matchE=Z; matchG=Z; allG=Z;
  for g = 1:n
    [matchE1,matchG1] = correspondPixels(E1,G{g},maxDist);
    matchE = matchE | matchE1>0;
    matchG = matchG + double(matchG1>0);
    allG = allG + G{g};
  end
  % compute recall (summed over each gt image)
  cntR(k) = sum(matchG(:)); sumR(k) = sum(allG(:));
  % compute precision (edges can match any gt image)
  cntP(k) = nnz(matchE); sumP(k) = nnz(E1);
  % optinally create visualization of matches
  if(nargout<6), continue; end; cs=[1 0 0; 0 .7 0; .7 .8 1]; cs=cs-1;
  FP=E1-matchE; TP=matchE; FN=(allG-matchG)/n;
  for g=1:3, V(:,:,g,k)=max(0,1+FN*cs(1,g)+TP*cs(2,g)+FP*cs(3,g)); end
  V(:,2:end,:,k) = min(V(:,2:end,:,k),V(:,1:end-1,:,k));
  V(2:end,:,:,k) = min(V(2:end,:,:,k),V(1:end-1,:,:,k));
end

% if output file specified write results to disk
[thrs2, cntR2, sumR2, cntP2, sumP2] = quadInterpolate(thrs, cntR, sumR, cntP, sumP, 100);
if(isempty(out)), return; end; fid=fopen(out,'w'); assert(fid~=1);
fprintf(fid,'%10g %10g %10g %10g %10g\n',[thrs2 cntR2 sumR2 cntP2 sumP2]');
fclose(fid);

end

% interpolate the PR curve by quadratic function
function [thrs2, cntR2, sumR2, cntP2, sumP2] = quadInterpolate(thrs, cntR, sumR, cntP, sumP,K)
  thrs2 = linspace(0.01,1,K)';thrs2 = [0.005;thrs2];
  sumR2 = interp1(thrs, sumR, thrs2, 'spline'); sumR2 = max(round(sumR2),0);
  sumP2 = interp1(thrs, sumP, thrs2, 'spline');
  [~, ind] = min(sumP2); sumP2(ind+1:end)=0;sumP2 = max(round(sumP2),0);
  R = cntR./(sumR+eps); R2 = interp1(thrs, R, thrs2,'spline');[~, ind] = min(R2);R2(ind+1:end)=0;
  R2 = min(R2,1);R2 = max(R2,0);
  cntR2 = R2.*sumR2;cntR2 = max(round(cntR2),0);

  P = cntP./(sumP+eps);
  try
    P2 = interp1(thrs, P, thrs2, 'spline');
    %P2 = spline(thrs, P, thrs2);
  catch
    print('oops\n');
    P2 = interp1(thrs, P, thrs2, 'linear');
  end
  P2 = min(P2,1);P2 = max(P2,0);
  cntP2 = P2.*sumP2; cntP2 = max(round(cntP2),0);
end
