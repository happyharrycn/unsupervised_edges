###################################################################
#                                                                 #
#    Unsupervised Edge Learning V1.0                              #
#    Yin Li (yli440@gatech.edu)                                   #
#    http://cbi.gatech.edu/edges/                                 #
#                                                                 #
###################################################################

1. Introduction.

Pipeline for unsupervised learning of edges from video. Can learn a state-of-art edge detector using motion edges in videos. Code is built on Piotr's Structured Edge Detection Toolbox (see the attached readme file from the toolbox). 

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
  mex private/edgesDetectMex.cpp -outdir private [OMPPARAMS]
  mex private/edgesNmsMex.cpp    -outdir private [OMPPARAMS]
  mex private/spDetectMex.cpp    -outdir private [OMPPARAMS]
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





###################################################################
#                                                                 #
#    Structured Edge Detection Toolbox V3.0                       #
#    Piotr Dollar (pdollar-at-gmail.com)                          #
#                                                                 #
###################################################################

1. Introduction.

Very fast edge detector (up to 60 fps depending on parameter settings) that achieves excellent accuracy. Can serve as input to any vision algorithm requiring high quality edge maps. Toolbox also includes the Edge Boxes object proposal generation method and fast superpixel code.

If you use the Structured Edge Detection Toolbox, we appreciate it if you cite an appropriate subset of the following papers:

@inproceedings{DollarICCV13edges,
  author    = {Piotr Doll\'ar and C. Lawrence Zitnick},
  title     = {Structured Forests for Fast Edge Detection},
  booktitle = {ICCV},
  year      = {2013},
}

@article{DollarARXIV14edges,
  author    = {Piotr Doll\'ar and C. Lawrence Zitnick},
  title     = {Fast Edge Detection Using Structured Forests},
  journal   = {ArXiv},
  year      = {2014},
}

@inproceedings{ZitnickECCV14edgeBoxes,
  author    = {C. Lawrence Zitnick and Piotr Doll\'ar},
  title     = {Edge Boxes: Locating Object Proposals from Edges},
  booktitle = {ECCV},
  year      = {2014},
}

###################################################################

2. License.

This code is published under the MSR-LA Full Rights License.
Please read license.txt for more info.

###################################################################

3. Installation.

a) This code is written for the Matlab interpreter (tested with versions R2013a-2013b) and requires the Matlab Image Processing Toolbox. 

b) Additionally, Piotr's Matlab Toolbox (version 3.26 or later) is also required. It can be downloaded at:
 https://pdollar.github.io/toolbox/.

c) Next, please compile mex code from within Matlab (note: win64/linux64 binaries included):
  mex private/edgesDetectMex.cpp -outdir private [OMPPARAMS]
  mex private/edgesNmsMex.cpp    -outdir private [OMPPARAMS]
  mex private/spDetectMex.cpp    -outdir private [OMPPARAMS]
  mex private/edgeBoxesMex.cpp   -outdir private
Here [OMPPARAMS] are parameters for OpenMP and are OS and compiler dependent.
  Windows:  [OMPPARAMS] = '-DUSEOMP' 'OPTIMFLAGS="$OPTIMFLAGS' '/openmp"'
  Linux V1: [OMPPARAMS] = '-DUSEOMP' CFLAGS="\$CFLAGS -fopenmp" LDFLAGS="\$LDFLAGS -fopenmp"
  Linux V2: [OMPPARAMS] = '-DUSEOMP' CXXFLAGS="\$CXXFLAGS -fopenmp" LDFLAGS="\$LDFLAGS -fopenmp"
To compile without OpenMP simply omit [OMPPARAMS]; note that code will be single threaded in this case.

d) Add edge detection code to Matlab path (change to current directory first): 
 >> addpath(pwd); savepath;

e) Finally, optionally download the BSDS500 dataset (necessary for training/evaluation):
 http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/
 After downloading BSR/ should contain BSDS500, bench, and documentation.

f) A fully trained edge model for RGB images is available as part of this release. Additional models are available online, including RGBD/D/RGB models trained on the NYU depth dataset and a larger more accurate BSDS model.

###################################################################

4. Getting Started.

 - Make sure to carefully follow the installation instructions above.
 - Please see "edgesDemo.m", "edgeBoxesDemo" and "spDemo.m" to run demos and get basic usage information.
 - For a detailed list of functionality see "Contents.m".

###################################################################

5. History.

Version NEW
 - now hosting on github (https://github.com/pdollar/edges)
 - suppress Mac warnings, added Mac binaries
 - edgeBoxes: added adaptive nms variant described in arXiv15 paper

Version 3.01 (09/08/2014)
 - spAffinities: minor fix (memory initialization)
 - edgesDetect: minor fix (multiscale / multiple output case)

Version 3.0 (07/23/2014)
 - added Edge Boxes code corresponding to ECCV paper
 - added Sticky Superpixels code
 - edge detection code unchanged

Version 2.0 (06/20/2014)
 - second version corresponding to arXiv paper
 - added sharpening option
 - added evaluation and visualization code
 - added NYUD demo and sweep support
 - various tweaks/improvements/optimizations

Version 1.0 (11/12/2013)
 - initial version corresponding to ICCV paper

###################################################################
