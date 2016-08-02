###################################################################
#                                                                 #
#    Unsupervised Edge Learning V1.0                              #
#    Yin Li (yli440@gatech.edu)                                   #
#    http://cbi.gatech.edu/edges/                                 #
#                                                                 #
###################################################################

1. Introduction.

Pipeline for unsupervised learning of edges from video. Can learn a state-of-art edge detector using motion edges in videos. Code is built on Piotr's Structured Edge Detection Toolbox (see the folder structured_edges and the readme file with the folder). 

If you use the code, we appreciate it if you cite the following paper:

@inproceedings{LiCVPR16edges,
  author    = {Yin Li and Manohar Paluri and James M. Rehg and Piotr Doll\'ar},
  title     = {Unsupervised Learning of Edges},
  booktitle = {CVPR},
  year      = {2016},
}

2. Installation.

a) This code is written for the Matlab interpreter (tested with versions R2014a-2016a) and requires the Matlab Image Processing Toolbox. 

b) Piotr's Matlab Toolbox (version 3.26 or later) is required for Structured Edges. It can be downloaded at:
 https://pdollar.github.io/toolbox/.

c) DeepMatching is needed. It can be downloaded at:
 http://lear.inrialpes.fr/src/deepmatching/
Follow their instructions to compile the code and copy the binary file into ./bins

d) EpicFlow is needed and we included a modified version in folder epicflow. You can build the binary using our provided makefile.
 cd epicflow
 make
Also you need to copy the compiled binary file into ./bins

e) Due to the compilation of (c) and (d), our code ONLY work in linux. Small modifications to makefiles are required for OS X. 

f) Our python script and epicflow code has a dependency on python-opencv and opencv (only 2.4.x is supported for now)  

g) Next, please compile mex code from within Matlab (note: linux64 binaries included):
  mex ./structured_edges/private/edgesDetectMex.cpp -outdir ./structured_edges/private [OMPPARAMS]
  mex ./structured_edges/private/edgesNmsMex.cpp    -outdir ./structured_edges/private [OMPPARAMS]
  mex ./structured_edges/private/spDetectMex.cpp    -outdir ./structured_edges/private [OMPPARAMS]
Here [OMPPARAMS] are parameters for OpenMP and are OS and compiler dependent.
  Windows:  [OMPPARAMS] = '-DUSEOMP' 'OPTIMFLAGS="$OPTIMFLAGS' '/openmp"'
  Linux V1: [OMPPARAMS] = '-DUSEOMP' CFLAGS="\$CFLAGS -fopenmp" LDFLAGS="\$LDFLAGS -fopenmp"
  Linux V2: [OMPPARAMS] = '-DUSEOMP' CXXFLAGS="\$CXXFLAGS -fopenmp" LDFLAGS="\$LDFLAGS -fopenmp"
To compile without OpenMP simply omit [OMPPARAMS]; note that code will be single threaded in this case. We suggest to make the code single threaded to achieve the max degree of parallelism during training.

3. Training 

a) Getting the dataset. We used the following dataset for training and testing

  BSDS500 dataset for edge detection benchmark
  https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html

  Videos from VSB and Youtube Object dataset
  http://lmb.informatik.uni-freiburg.de/resources/datasets/vsb.en.html
  http://calvin.inf.ed.ac.uk/datasets/youtube-objects-dataset/

  [Optional] Sintel dataset for optical flow benchmark
  http://sintel.is.tue.mpg.de/


b) Preparing data. We provide pre-processing scripts in python to select frames within videos (as described in Sec 3.1 of our paper). The scripts are located in the folder py_preprocessing. 

  cd py_preprocessing
  process_videos.py --video_list path/to/video/list \
                    --output_folder path/to/output/folder/dataset_name [--copy]

The script will copy selected frames from a list of videos (see an example in video_list.txt in the same folder) to the output folder. It will also setup folder structure and generate a *_pairs.txt that includes all frame pairs. Use --copy only for sintel dataset (so we dont filter frames)

  match_pairs.py --pair_list path/to/video_pairs.txt \
                 --output_folder path/to/output/folder/dataset_name \
                 --dm_bin ../bins/deepmatching
                 
The script will run deep matching over the selected frame pairs. This is used for training. 

Now you can point rootPath in globalParam.m to your dataset root folder. We assume the following folder hierarchy. 
  root_data_folder
  --dataset_01
    --edges
    --flows
    --images
    --matches
    --motEdges
  ...
  --dataset_02 

For sintel and video datasets, the folder structure is set up automatically by python script. For BSDS, you need to copy 
BSR/BSDS500/data/images --> root_data_folder/bsds/images
BSR/BSDS500/data/groundTruth --> root_data_folder/bsds/edges/GroundTruth

c) Run the pipeline. mainLoop.m includes our training pipeline. Once the datasets are ready, simply call
mainLoop 

Note that we will use ./tmp to cache intermediate results. Do remember to clear the cache (delete all files) if you are re-running the experiments.

4. Trouble Shooting

a) Cannot load any more object with static TLS (when compiled with openmp)
Set environment variable LD_PRELOAD = path/to/libgomp.so before starting Matlab

b) Undefined symbol issues (when using epicflow in matlab)
This is usually caused by the Matlab's own version of libraries. Run 
  ldd path/to/epicflow
in terminal and Matlab to check the lib versions been linked. You can fix it by removing the wrongly linked libs in Matlab folder (e.g. libopencv* in path/to/matlab/bin/glnxa64 or libstdc* in path/to/matlab/sys/os/glnxa64)

For further issues regarding the code, please send me an email at yli440@gatech.edu


###################################################################