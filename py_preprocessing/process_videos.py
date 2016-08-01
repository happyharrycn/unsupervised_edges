import argparse
import sys
import os
import glob
import time
import cv2
import numpy as np
import shutil


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(
                        description='Pre-process all videos and pick the good frames')
    parser.add_argument('--video_list', dest='video_list_file',
                        help='a file that contains all videos (one video per line)',
                        default=None, type=str)
    parser.add_argument('--output_folder', dest='output_folder',
                        help='output folder to save selected video frames (root of the dataset)',
                        default=None, type=str)
    parser.add_argument('--frames_per_video', dest='frames_per_video', 
                        help='Max number of frames sampled from each video',
                        default=160, type=int)
    parser.add_argument('--copy', dest='copy_frames', help='copy frames without filtering',
                        action='store_true')
 
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

def image_quality(img):
    """
    Implementation of the paper
    "No-Reference Perceptual Quality Assessment of JPEG Compressed Images"
    Used as a quick filter of frames
    """
    # convert bgr image to gray -> float32
    score = 0.0
    if img is None:
        return score

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    x = gray.astype(np.float32)
    h, w = x.shape[0], x.shape[1]

    # horizontal
    d_h = x[:,1:] - x[:,:-1]
    w_bound = int(8*(np.floor(w/8.0)-1)) + 1
    B_h = np.mean(np.abs(d_h[:,7:w_bound:8]))
    A_h = (8.0 * np.mean(np.abs(d_h)) - B_h) / 7.0
    sig_h = np.sign(d_h)
    left_sig, right_sig = sig_h[:,:-2], sig_h[:,1:-1]
    Z_h = np.mean((left_sig * right_sig)<0)

    # vertical
    d_v = x[1:, :] - x[:-1, :]
    h_bound = int(8*(np.floor(h/8.0)-1)) + 1
    B_v = np.mean(np.abs(d_v[7:h_bound:8, :]))
    A_v = (8.0 * np.mean(np.abs(d_v)) - B_v) / 7.0
    sig_v = np.sign(d_v)
    up_sig, down_sig = sig_v[:-2, :], sig_v[1:-1, :]
    Z_v = np.mean((up_sig * down_sig)<0)

    # combine the weights
    B = (B_h + B_v)/2.0
    A = (A_h + A_v)/2.0
    Z = (Z_h + Z_v)/2.0

    # quality prediction
    alpha = -245.8909
    beta = 261.9373
    gamma1 = -239.8886 / 10000.0 
    gamma2 = 160.1664 / 10000.0 
    gamma3 = 64.2859 / 10000.0 

    # corner case of a black / white frame
    if np.abs(A) < 1e-3 or np.abs(B) < 1e-3 or np.abs(Z) < 1e-3:
        score = 0.0
    else:
        score = alpha + beta*(B**gamma1)*(A**gamma2)*(Z**gamma3)

    return score

def vis_matching(src_img, src_pts, dst_pts, M):
    """
    Visualize matching results
    """
    assert dst_pts.shape[0] == 2 and src_pts.shape[0]
    assert dst_pts.shape[1] == src_pts.shape[1]

    # draw the tracked pts
    img = src_img.copy()
    num_matches = src_pts.shape[1]
    for i in range(num_matches):
        cv2.circle(img, (src_pts[0, i], src_pts[1, i]), 3, (0,0,255), 1)
        cv2.line(img,(src_pts[0, i], src_pts[1, i]),
                    (dst_pts[0, i], dst_pts[1, i]),(0,255,0),1)

    return img


def distribute_keypoints(kps, img_size, grid_size, num_features):
    ''' 
    Sample interest points on 2D grid for better homography
    The results is a set of points evenly distributed over 2D
    '''
    # kp index and score
    kp_grids = np.zeros((grid_size, grid_size, num_features), dtype=np.int) - 1
    kp_scores = np.zeros((grid_size, grid_size, num_features), dtype=np.float32) - 1
    grid_width, grid_height = np.ceil(img_size / grid_size)

    # insert kp into grids
    for kp_ind, kp in enumerate(kps):
        x, y = kp.pt
        score = kp.response
        grid_x, grid_y = int(np.floor(x / grid_width)), int(np.floor(y / grid_height))

        # find the lowest scored kp in current grid
        slot = np.argmin(kp_scores[grid_x, grid_y, :])

        # replace slot with current kp if necessary
        if kp_scores[grid_x, grid_y, slot] < score:
            kp_grids[grid_x, grid_y, slot] = kp_ind
            kp_scores[grid_x, grid_y, slot] = score

    # valid list of kp
    kp_list = kp_grids.ravel().tolist()
    kp_list = [ind for ind in kp_list if ind != -1]

    return kp_list


def fit_homography(kp1, kp2, matches, img_size=None, src_img=None):
    ''' 
    Fit homography using matrix using forward/backward matching
    Homography is not going to be accurate anyway
    Tricks here to avoid bias over foreground
    '''
    M = np.zeros((3,3))
    matchesMask = []
    
    if img_size is None:
        # use all matches
        valid_matches = matches
    else:
        # distribute key points across 2D grids
        kp_list = distribute_keypoints(kp1, img_size, 32, 5)
        valid_matches = [m for m in matches if m.queryIdx in kp_list]

    # convert kp to pts
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in valid_matches ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in valid_matches ]).reshape(-1,1,2)

    # call opencv for homography (with a conservative threshold)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 1.2)
    matchesMask = mask.ravel().tolist()

    # visualize matching results
    if src_img is not None:
        src_pts = src_pts.reshape(-1, 2).transpose()
        dst_pts = dst_pts.reshape(-1, 2).transpose()
        valid_ind = np.where(mask==1)[0]

        src_pts = src_pts[:, valid_ind]
        dst_pts = dst_pts[:, valid_ind]

        vis_img = vis_matching(src_img, src_pts, dst_pts, M)
        cv2.imshow('test', vis_img)
        cv2.waitKey(0)

    return M, matchesMask


def match_frames(prev_frame, curr_frame, params):
    """
    Filter out frame pairs based on matching results
    """
    scale = params['fScale']
    num_feats = params['nFeat']
    min_num_kps = params['nMinKp']
    min_num_matches = params['nMinMatch']
    vl_thresh = params['fVlThresh']
    vh_thresh = params['fVhThresh']
    quality_thresh = params['fQualityThresh']
    quality_ratio = params['fQualityRatio']

    # init orb detector/matcher
    orb = cv2.ORB(num_feats)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    # resize frames if necessary
    if np.abs(scale-1) > 1e-3:
        prev_frame = cv2.resize(prev_frame, (0,0), fx=scale, fy=scale)
        curr_frame = cv2.resize(curr_frame, (0,0), fx=scale, fy=scale)
    h, w, c = curr_frame.shape

    # check if prev_frame and curr_frame are blurry
    prev_score = image_quality(prev_frame)
    curr_score = image_quality(curr_frame)
    # print prev_score, curr_score

    if prev_score < quality_thresh or curr_score < quality_thresh:
        return False
    
    if abs(prev_score / (curr_score + 1e-6) - 1) > quality_ratio:
        return False

    # orb detect and descriptors
    kp1, des1 = orb.detectAndCompute(prev_frame, None)
    kp2, des2 = orb.detectAndCompute(curr_frame, None)

    # too few kps: textureless / blurring
    if len(kp1) < min_num_kps or len(kp2) < min_num_kps:
        return False

    # forward / backward matching
    matches = bf.match(des1, des2)

    # not enough matching
    if len(matches) < min_num_matches:
        return False

    # fit homograhpy (with some tricks)
    M, mask = fit_homography( kp1, kp2, matches, np.asarray([w, h]), None)
    
    #  degenerated homography
    if len(mask) == 0:
        return False

    # translational component
    vx, vy = np.abs(M[0,2]), np.abs(M[1,2])

    # motion is too slow or too fast
    if (vx < vl_thresh) and (vy < vl_thresh):
        return False
    if (vx > vh_thresh) or (vy > vh_thresh):
        return False

    # check the off-diagonal of homography (global translational motion)
    # reject a (almost) scaling / translation matrix
    odx, ody = np.abs(M[0,1]), np.abs(M[1, 0])
    if odx <= 1e-3 and ody <= 1e-3:
        return False

    return True


def copy_vid_folder(vid_folder, output_folder, ext=None):
    """
    copy and rename a set of frames into the same folder
    only used to re-organize existing dataset (e.g. sintel)
    Return a set of frame pairs
    """

    if ext is None:
        ext = 'png'
    
    # get video stats
    video_name = os.path.basename(vid_folder)
    video_name.replace(' ', '')
    frame_list = sorted(glob.glob(os.path.join(vid_folder, '*.' + ext)))
    num_frames = len(frame_list)
    frame_pairs = []
    frame_index = 0

    # sanity check
    if len(frame_list) == 0:
        print "Can not open video folder: {:s}".format(vid_folder)
        return frame_pairs
    
    start = time.time()
    for frame_index in xrange(num_frames):

        # copy / rename input frame -> output
        input_frame_file = frame_list[frame_index]
        output_frame_file = os.path.join( output_folder, video_name + "_{:s}".format(
                                             os.path.basename(frame_list[frame_index])
                                        ) )
        shutil.copy(input_frame_file, output_frame_file)

        if frame_index + 1 < num_frames:
            # input frame pair
            paired_frame_file = frame_list[frame_index+1]
            # output frame pair
            output_paired_file = os.path.join( output_folder, video_name + "_{:s}".format(
                                             os.path.basename(frame_list[frame_index+1])
                                        ) )
            # append the list
            frame_pairs.append([output_frame_file, output_paired_file])

    # timing
    end = time.time()
    print "Averge time per frame: {:2f} s. (Total {:d} frames)".format(
        float(end-start)/len(frame_pairs), len(frame_pairs))

    return frame_pairs

def process_vid_folder(vid_folder, output_folder, max_num_samples, 
                       max_frame_range=None, ext=None):
    """
    Process a set of frames within a folder
    Return a set of frame pairs
    """
    # dict that holds params for matching
    params = {}
    params['fScale'] = 0.5
    params['nFeat'] = 2000
    params['nMinKp'] = 50
    params['nMinMatch'] = 30
    params['fVlThresh'] = 0.8
    params['fVhThresh'] = 16.0
    params['fQualityThresh'] = 7.0
    params['fQualityRatio'] = 0.2

    # default params
    if max_frame_range is None:
        max_frame_range = 4
    if ext is None:
        ext = 'png'

    # get video stats
    video_name = os.path.basename(vid_folder)
    video_name.replace(' ', '')
    frame_list = sorted(glob.glob(os.path.join(vid_folder, '*.' + ext)))
    num_frames = len(frame_list)
    frame_pairs = []
    output_frame_list = []
    frame_index = 0

    # sanity check
    if len(frame_list) == 0:
        print "Can not open video folder: {:s}".format(vid_folder)
        return frame_pairs
    
    # fetch the first batch of pairs into buffer
    frame_buffer = []
    for ind in xrange(max_frame_range):
        frame = cv2.imread(frame_list[ind])
        frame_buffer.append(frame.copy())

    start = time.time()
    # FIFO queue 
    # first element: prev frame
    # 2-n element: future frame
    while(len(frame_buffer)>1):
        # fetch frame when possible
        if (frame_index + max_frame_range) < num_frames:
            frame = cv2.imread(frame_list[frame_index + max_frame_range])
            # prevent any corrupted frames
            if frame is not None:
                frame_buffer.append(frame) 

        # de-queue
        prev_frame = frame_buffer.pop(0)

        # run a small trial
        good_pair = -1
        buffer_size = len(frame_buffer)
        pair_ind = np.random.permutation(range(buffer_size))
        pair_ind = pair_ind[:3]
        for ind in pair_ind:
            curr_frame = frame_buffer[ind]
            if match_frames(prev_frame, curr_frame, params):
                good_pair = ind + 1
                break

        # write the images / pairs
        if good_pair > 0:
            output_prev_file = os.path.join( output_folder, video_name + "_{:s}".format(
                                             os.path.basename(frame_list[frame_index])
                                        ) )
            output_curr_file = os.path.join( output_folder, video_name + "_{:s}".format(
                                             os.path.basename(frame_list[frame_index + good_pair])
                                        ) )
            if not os.path.exists(output_prev_file):
                cv2.imwrite(output_prev_file, prev_frame)
                output_frame_list.append(output_prev_file)

            if not os.path.exists(output_curr_file):
                cv2.imwrite(output_curr_file, curr_frame)
                output_frame_list.append(output_curr_file)

            # adding to pairs
            frame_pairs.append([output_prev_file, output_curr_file])

        # de-queue
        frame_index += 1

    # timing
    end = time.time()
    print "Averge time per frame: {:2f} s. Sampled {:d} out of {:d} frames".format(
        float(end-start)/frame_index, len(frame_pairs), frame_index)

    # resample the frame pairs if too many
    if len(frame_pairs) > max_num_samples:
        print "Resample into {:d} frame pairs".format(max_num_samples)
        # resample frame pair index
        rand_ind = np.random.permutation(range(len(frame_pairs)))
        sel_pair_ind = rand_ind[:max_num_samples]
        sel_frame_ind = []

        # get index for frames that we need to keep
        for pair_ind, frame_pair in enumerate(frame_pairs):
            if (pair_ind in sel_pair_ind):
                # add output frame index to selected list
                ind = output_frame_list.index(frame_pair[0])
                if not (ind in sel_frame_ind):
                    sel_frame_ind.append(ind)
                ind = output_frame_list.index(frame_pair[1])
                if not (ind in sel_frame_ind):
                    sel_frame_ind.append(ind)

        # now delete extra frames
        for output_frame_ind, output_frame in enumerate(output_frame_list):
            if not (output_frame_ind in sel_frame_ind):
                os.remove(output_frame)

        # resample the list
        frame_pairs = [frame_pairs[ind] for ind in sel_pair_ind]

    return frame_pairs


def process_vid_file(vid_file, output_folder, max_num_samples, 
                     max_frame_range=None, ext=None):
    """
    Process a single video file 
    Make sure opencv is compiled with ffmpeg
    """
    # dict that holds params for matching
    params = {}
    params['fScale'] = 0.5
    params['nFeat'] = 2000
    params['nMinKp'] = 50
    params['nMinMatch'] = 30
    params['fVlThresh'] = 1.5
    params['fVhThresh'] = 16.0
    params['fQualityThresh'] = 8.0
    params['fQualityRatio'] = 0.2

    # default params
    if max_frame_range is None:
        max_frame_range = 4
    if ext is None:
        ext = 'png'

    # get video stats
    video_name = os.path.basename(vid_file[:-4])
    video_name.replace(' ', '')
    frame_pairs = []
    output_frame_list = []
    frame_index = 0

    # open video file
    cap = cv2.VideoCapture(vid_file)
    if not cap.isOpened():
        print "Can not open video file: {:s}".format(vid_file)
        return frame_pairs
    
    # fetch the first batch of pairs into buffer
    frame_buffer = []
    for ind in xrange(max_frame_range):
        ret, frame = cap.read()
        if ret and (frame is not None):
            frame_buffer.append(frame.copy())

    start = time.time()

    # loop over all frames
    while(len(frame_buffer)>1):

        # read current frame
        ret, frame = cap.read()

        # valid frame?
        if ret and (frame is not None):
            frame_buffer.append(frame)

        # de-queue
        prev_frame = frame_buffer.pop(0)

        # run a small trial
        good_pair = -1
        buffer_size = len(frame_buffer)
        pair_ind = np.random.permutation(range(buffer_size))
        pair_ind = pair_ind[:3]
        for ind in pair_ind:
            curr_frame = frame_buffer[ind]
            if match_frames(prev_frame, curr_frame, params):
                good_pair = ind + 1
                break

        # write the images / pairs
        if good_pair > 0:
            output_prev_file = os.path.join(output_folder, 
                                    video_name + "_{:010d}.{:s}".format(frame_index, ext))
            output_curr_file = os.path.join(output_folder, 
                                    video_name + "_{:010d}.{:s}".format(frame_index + good_pair, ext))

            if not os.path.exists(output_prev_file):
                cv2.imwrite(output_prev_file, prev_frame)
                output_frame_list.append(output_prev_file)

            if not os.path.exists(output_curr_file):
                cv2.imwrite(output_curr_file, curr_frame)
                output_frame_list.append(output_curr_file)

            # adding to pairs
            frame_pairs.append([output_prev_file, output_curr_file])

        # de-queue
        frame_index += 1

    # timing
    end = time.time()
    print "Averge time per frame: {:2f} s. Sampled {:d} out of {:d} frames".format(
        float(end-start)/frame_index, len(frame_pairs), frame_index)

    # resample the frame pairs if too many
    if len(frame_pairs) > max_num_samples:
        print "Resample into {:d} frame pairs".format(max_num_samples)
        # resample frame pair index
        # quick hack: remove first 10% and last 10% frames for video
        rand_ind = np.random.permutation(range(
            int(0.1*len(frame_pairs)), int(0.9*len(frame_pairs))
            ))
        sel_pair_ind = rand_ind[:max_num_samples]
        sel_frame_ind = []

        # get index for frames that we need to keep
        for pair_ind, frame_pair in enumerate(frame_pairs):
            if (pair_ind in sel_pair_ind):
                # add output frame index to selected list
                ind = output_frame_list.index(frame_pair[0])
                if not (ind in sel_frame_ind):
                    sel_frame_ind.append(ind)
                ind = output_frame_list.index(frame_pair[1])
                if not (ind in sel_frame_ind):
                    sel_frame_ind.append(ind)

        # now delete extra frames
        for output_frame_ind, output_frame in enumerate(output_frame_list):
            if not (output_frame_ind in sel_frame_ind):
                os.remove(output_frame)

        # resample the list
        frame_pairs = [frame_pairs[ind] for ind in sel_pair_ind]

    return frame_pairs


if __name__ == '__main__':
    """
    Python scripts for pre-process videos and pick the good frames
    """
    # parse input args
    args = parse_args()
    print('Called with args:')
    print(args)

    video_list_file = args.video_list_file
    output_folder = os.path.join(args.output_folder, 'images')
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    frames_per_video = args.frames_per_video
    copy_frames = args.copy_frames

    # list of all frame pairs
    all_pairs = []

    # read video list file
    video_list = [line.rstrip('\n') for line in open(video_list_file)]
    
    # each video can be either a file or a folder
    for video in video_list:
        curr_pairs = []
        # video file?
        if ('.avi' in os.path.basename(video)) or ('.mp4' in os.path.basename(video)) \
            and os.path.exists(video):
            print "Processing video file: {:s}".format(video)
            curr_pairs = process_vid_file(video, output_folder, frames_per_video)
        # video folder?
        elif os.path.isdir(video):
            if copy_frames:
                # simply copy all frames (used only for prepare sintel dataset)
                print "Coping video folder: {:s}".format(video)
                curr_pairs = copy_vid_folder(video, output_folder)
            else:
                print "Processing video folder: {:s}".format(video)
                curr_pairs = process_vid_folder(video, output_folder, frames_per_video)

        all_pairs = all_pairs + curr_pairs
    
    # now write all pairs to file (in dataset root folder)

    video_pair_file = os.path.join(args.output_folder, 
                            os.path.basename(args.output_folder) + '_pairs.txt')
    fid = open(video_pair_file, 'w')
    for pair in all_pairs:
        fid.write(pair[0] + ' ' + pair[1] + '\n')
    fid.close()

    # finally check the folder stucture
    sub_folders = ['images', 'matches', 'edges', 'flows', 'motEdges']
    for sub_folder in sub_folders:
        curr_sub_folder = os.path.join(args.output_folder, sub_folder)
        if not os.path.exists(curr_sub_folder):
            os.mkdir(curr_sub_folder)
