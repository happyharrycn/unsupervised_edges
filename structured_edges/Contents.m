% EDGES
% See also readme.txt
%
% Fast edge detector code is based on the paper:
%  P. Dollár and C. Zitnick
%  "Structured Forests for Fast Edge Detection", ICCV 2013.
% Please cite the above paper if you end up using the edge detector.
%
% Edge Boxes object proposal generation is based on the paper:
%  C. Zitnick and P. Dollár
%  "Edge Boxes: Locating Object Proposals from Edges", ECCV 2014.
% Please cite the above paper if you end up using the object proposals.
%
% Structured Edge detector code:
%   edgesChns       - Compute features for structured edge detection.
%
% Edge detection evaluation code:
%   edgesEval       - Run and evaluate structured edge detector on BSDS500.
%   edgesEvalDir    - Calculate edge precision/recall results for directory of edge images.
%   edgesEvalImg    - Calculate edge precision/recall results for single edge image.
%   edgesEvalPlot   - Plot edge precision/recall results for directory of edge images.
%
% Sticky Edge Adhesive Superpixel code:
%   spDetect        - Detect Sticky Superpixels in image.
