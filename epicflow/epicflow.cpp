#include <stdlib.h>
#include <string.h>
#include <vector>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "epic.h"
#include "image.h"
#include "io.h"
#include "variational.h"


/* show usage information */
void usage(){
    printf("usage:\n");
    printf("    ./epicflow image1 image2 matches outputfile [options]\n");
    printf("Compute EpicFlow between two images using given matches and edges and store it into a .flo file\n");
    printf("Images must be in PPM, JPG or PNG format.\n");
    printf("Edges are read as width*height float32 values in a binary file\n");
    printf("Matches are read from a text file, each match in a different line, each line starting with 4 numbers corresponding to x1 y1 x2 y2\n");
    printf("\n");
    printf("options:\n"); 
    printf("    -h, -help                                                print this message\n");
    printf("  interpolation parameters\n");
    printf("    -nw                                                      use Nadaraya-Watson instead of LA interpolator in the interpolation\n");
    printf("    -p, -prefnn             <int>(25)                        number of neighbors for consisteny checking in the interpolation\n");
    printf("    -n, -nn                 <int>(100)                       number of neighnors for the interpolation\n");
    printf("    -k                      <float>(0.8)                     coefficient of the sigmoid of the Gaussian kernel used in the interpolation\n");
    printf("  energy minimization parameters\n");
    printf("    -i, -iter               <int>(5)                         number of iterations for the energy minimization\n");
    printf("    -a, -alpha              <float>(1.0)                     weight of smoothness term\n");
    printf("    -g, -gamma              <float>(3.0)                     weight of gradient constancy assumption\n");
    printf("    -d, -delta              <float>(2.0)                     weight of color constancy assumption\n");
    printf("    -s, -sigma              <float>(0.8)                     standard deviation of Gaussian presmoothing kernel\n");
    printf("    -E, -edge                                                specify the edge map to use\n");
    printf("    -sobel                                                   using sobel operator for edge detection\n");
    printf("  predefined parameters\n");
    printf("    -sintel                                                  set the parameters to the one optimized on (a subset of) the MPI-Sintel dataset\n");
    printf("    -middlebury                                              set the parameters to the one optimized on the Middlebury dataset\n");
    printf("    -kitti                                                   set the parameters to the one optimized on the KITTI dataset\n");
    printf("\n");
}

/* convert opencv mat to color_image_t */
float_image cvmat_to_float_im(const cv::Mat im_mat, const float scale) {
    // create a new float image
    assert(im_mat.type()==CV_32FC1);
    int nRows = im_mat.rows; 
    int nCols = im_mat.cols;
    float_image im_ci = empty_image(float, nCols, nRows);

    // loop over every pixel 
    int i,j; int idx = 0;
    float* p;
    for( i = 0; i < nRows; ++i) {
        p = const_cast<float*>(im_mat.ptr<float>(i));
        for ( j = 0; j < nCols; ++j, ++idx) {
            im_ci.pixels[idx] = p[j] * scale;
        }
    }

    // note memory is maintained by user
    return im_ci;
}

/* convert color_image_t to opencv mat*/
cv::Mat color_im_to_cvmat(const color_image_t* im_ci, const float scale) {
    // create an empty image
    cv::Mat im_mat(im_ci->height, im_ci->width, CV_32FC3);

    // cont memory
    int nRows = im_mat.rows; 
    int nCols = im_mat.cols;
    
    // loop over every pixel 
    int i,j; int idx = 0;
    float* p;
    for( i = 0; i < nRows; ++i) {
        p = im_mat.ptr<float>(i);
        for ( j = 0; j < nCols; ++j, ++idx) {
            p[3*j]     = im_ci->c3[idx] * scale ;
            p[3*j + 1] = im_ci->c2[idx] * scale;
            p[3*j + 2] = im_ci->c1[idx] * scale;
        }
    }

    return im_mat;
}

/* a wrapper for reading edge file/image from binary or png file*/
float_image my_read_edges(const std::string & filename, const int width, const int height)
{
    float_image edges;
    
    // check file type
    std::size_t found;
    found = filename.find(".png");
    // png file
    if (found!=std::string::npos) {
        cv::Mat img = cv::imread(filename, CV_LOAD_IMAGE_ANYDEPTH);

        if (img.empty()) {
            fprintf(stderr,"Failed to load png file\n");        
        }

        // check the data type of cv::Mat
        // this allows us to load 16bit png file
        float scale = 0.0f;
        if (img.depth() == CV_8U && img.channels()==1) {
            scale = 1.0f/255.0f;
        } else if (img.depth() == CV_16U && img.channels()==1) {
            scale = 1.0f/65535.0f;
        } else {
            fprintf(stderr,"Unknown png format\n");
        }

        img.convertTo(img, CV_32FC1);
        edges = cvmat_to_float_im(img, scale);
    } else
        edges = read_edges(filename.c_str(), width, height);

    return edges;
}


/* this is a quick function for sobel edges using opencv */
float_image sobel_edges(const color_image_t* im) {
    // convert color_image_t to cv::mat
    cv::Mat im_mat = color_im_to_cvmat(im, 1.0f/255.0f);
    cv::Mat edges(im->height, im->width, CV_32FC1, float(0.0));

    // split the color image into channels
    std::vector<cv::Mat> im_chns(3);
    cv::split(im_mat, im_chns);

    // run sobel for each channel
    for(int curChn=0; curChn<3; curChn++) {
        cv::Mat imgX, imgY, imgX2, imgY2;
        cv::Sobel(im_chns[curChn], imgX, CV_32F, 1, 0, 3);
        cv::Sobel(im_chns[curChn], imgY, CV_32F, 0, 1, 3);
        cv::pow(imgX, 2, imgX2); cv::pow(imgY, 2, imgY2);
        cv::Mat sumImg = imgX2 + imgY2;
        edges = cv::max(edges, sumImg);
    }

    // now convert cv::mat back to float_image
    cv::Mat edge_res; sqrt(edges, edge_res); 
    // rescale the edge response to match SE
    edge_res = edge_res / 3.0f;
    float_image edges_im = cvmat_to_float_im(edge_res, 1.0f);
    return edges_im;

}

int main(int argc, char **argv){
    if( argc<5){
        if(argc>1) fprintf(stderr,"Error, not enough arguments\n");
        usage();
        exit(1);
    }

    // read arguments
    color_image_t *im1 = color_image_load(argv[1]);
    color_image_t *im2 = color_image_load(argv[2]);
    float_image edges = empty_edges(im1->width, im1->height);
    float_image matches = read_matches(argv[3]);
    const char *outputfile = argv[4];

    // prepare variables
    epic_params_t epic_params;
    epic_params_default(&epic_params);
    variational_params_t flow_params;
    variational_params_default(&flow_params);
    image_t *wx = image_new(im1->width, im1->height), *wy = image_new(im1->width, im1->height);
    
    // read optional arguments 
    #define isarg(key)  !strcmp(a,key)
    int current_arg = 5;
    while(current_arg < argc ){
        const char* a = argv[current_arg++];
        if( isarg("-h") || isarg("-help") ) 
            usage();
        else if( isarg("-nw") ) 
            strcpy(epic_params.method, "NW");
        else if( isarg("-p") || isarg("-prefnn") ) 
            epic_params.pref_nn = atoi(argv[current_arg++]);
        else if( isarg("-n") || isarg("-nn") ) 
            epic_params.nn = atoi(argv[current_arg++]); 
        else if( isarg("-k") ) 
            epic_params.coef_kernel = atof(argv[current_arg++]);
        else if( isarg("-i") || isarg("-iter") ) 
            flow_params.niter_outer = atoi(argv[current_arg++]); 
        else if( isarg("-a") || isarg("-alpha") ) 
            flow_params.alpha= atof(argv[current_arg++]);  
        else if( isarg("-g") || isarg("-gamma") ) 
            flow_params.gamma= atof(argv[current_arg++]);                                  
        else if( isarg("-d") || isarg("-delta") ) 
            flow_params.delta= atof(argv[current_arg++]);  
        else if( isarg("-s") || isarg("-sigma") ) 
            flow_params.sigma= atof(argv[current_arg++]); 
        else if( isarg("-E") || isarg("-edge") ) {
            free(edges.pixels);           // destroy the empty edges first
            edges = my_read_edges(argv[current_arg++], im1->width, im1->height);
        } 
        else if( isarg("-sobel") ) {
            free(edges.pixels);           // destroy the empty edges first
            edges = sobel_edges(im1);
        }   
        else if( isarg("-sintel") ){ 
            epic_params.pref_nn= 25; 
            epic_params.nn= 160; 
            epic_params.coef_kernel = 1.1f; 
            flow_params.niter_outer = 5;
            flow_params.alpha = 1.0f;
            flow_params.gamma = 0.72f;
            flow_params.delta = 0.0f;
            flow_params.sigma = 1.1f;            
        }  
        else if( isarg("-kitti") ){ 
            epic_params.pref_nn= 25; 
            epic_params.nn= 160; 
            epic_params.coef_kernel = 1.1f;
            flow_params.niter_outer = 2;
            flow_params.alpha = 1.0f;
            flow_params.gamma = 0.77f;
            flow_params.delta = 0.0f;
            flow_params.sigma = 1.7f; 
        }
        else if( isarg("-middlebury") ){ 
            epic_params.pref_nn= 15; 
            epic_params.nn= 65; 
            epic_params.coef_kernel = 0.2f;       
            flow_params.niter_outer = 25;
            flow_params.alpha = 1.0f;
            flow_params.gamma = 0.72f;
            flow_params.delta = 0.0f;
            flow_params.sigma = 1.1f;  
        }
        else{
            fprintf(stderr, "unknown argument %s", a);
            usage();
            exit(1);
        }   
    }
    
    // compute interpolation and energy minimization
    color_image_t *imlab = rgb_to_lab(im1);
    epic(wx, wy, imlab, &matches, &edges, &epic_params, 1);
    // energy minimization
    variational(wx, wy, im1, im2, &flow_params);
    // write output file and free memory
    writeFlowFile(outputfile, wx, wy);
    
    color_image_delete(im1);
    color_image_delete(imlab);
    color_image_delete(im2);
    free(matches.pixels);
    free(edges.pixels);
    image_delete(wx);
    image_delete(wy);

    return 0;
}

