% Unsupervised Edges
% See also readme.txt
%
% Unsupervised edge learning code is based on the paper:
%  Y. Li, M. Paluri, J. M. Rehg and  P. Dollár
%  "Unsupervised Learning of edges", CVPR 2016. 
% Please cite the above paper if you end up using the edge detector.
%
%
% Unsupervised Edge Learning code:

%   sobelDetect     - simple sobel edge detector
%   initMotEdges    - initalize motion edges using sobel
%   initFlows       - run epic flow with sobel edges 
%   globalParam     - setup all the params
%   calcAllFlows    - run epic flow using specified edge maps
%   detAllEdges     - run edge detector over all images within current dataset
%   detAllMotEdges  - detect motion edges on flow field
%   evalFlowResults - evaluate current flow results on sintel
%   evalEdgeResults - evaluate current edge det model on bsds
%   mainLoop        - running our pipeline
