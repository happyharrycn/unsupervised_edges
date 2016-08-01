function edges = sobelDetect(img, nms)
% simple sobel edge detector

if nargin < 2
  nms = 0;
end

% input params
nC = size(img, 3);
H1 = fspecial('sobel'); H2 = H1';

if isinteger(img)
  img = im2single(img);
end

% get edge responses from max over all color channels
edges = zeros(size(img), 'single');
img = imPad(img, [1 1 1 1],'symmetric');
for i=1:nC
  edges(:,:,i) = sqrt(conv2(img(:,:,i), H1, 'valid').^2 + conv2(img(:,:,i), H2, 'valid').^2);
end

% max over all the channels
edges = max(edges, [], 3);

% run nms if necessary
if nms
  [Ox,Oy]=gradient2(convTri(edges,4));
  [Oxx,~]=gradient2(Ox); [Oxy,Oyy]=gradient2(Oy);
  O=mod(atan(Oyy.*sign(-Oxy)./(Oxx+1e-5)),pi);
  edges=edgesNmsMex(edges,O,3,5,1.01,1);
end