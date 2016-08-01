import sys
import argparse
import os
from multiprocessing import Pool


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(
                        description='Run deep matching on filtered frame pairs')
    parser.add_argument('--pair_list', dest='pair_list_file',
                        help='a file that contains filtered frame pairs',
                        default=None, type=str)
    parser.add_argument('--output_folder', dest='output_folder',
                        help='output folder to save all matching results (root of the dataset)',
                        default=None, type=str)
    parser.add_argument('--dm_bin', dest='dm_bin',
                        help='deepmatching binary file',
                        default=None, type=str)
    parser.add_argument('--num_threads', dest='num_threads',
                        help='number of processes',
                        default=6, type=int)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


def runCmd(param):
    """
    Wrapper over os.system for multiprocessing
    """
    print param
    os.system(param)


if __name__ == '__main__':
    """
    Python scripts for running matching over filtered frame pairs 
    """
    # parse input args
    args = parse_args()
    print('Called with args:')
    print(args)

    # read video list file
    pair_list_file = args.pair_list_file
    output_folder = os.path.join(args.output_folder, 'matches')
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    dm_bin = args.dm_bin
    all_pairs = [line.rstrip('\n') for line in open(pair_list_file)]

    # for each pair of frames set up the cmd line
    params = []
    for pair in all_pairs:
        frames = pair.split(' ')
        output_file = os.path.basename(frames[0])
        output_file = os.path.splitext(output_file)[0] + '.dm'
        output_file = os.path.join(output_folder, output_file)
        cmd = "{:s} {:s} {:s} -ngh_rad 128 -out {:s}".format(
                dm_bin, frames[0], frames[1], output_file)

        # skip existing file
        if not os.path.exists(output_file):
            params.append(cmd)     

    # multiprocess pool 
    num_threads = args.num_threads
    p = Pool(num_threads)
    p.map(runCmd, params)
    p.close()
    p.join()