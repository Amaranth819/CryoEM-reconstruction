import mrcfile as mf
import numpy as np
import argparse
import os

def syn_2dprojection(input_file, output_file, num):
    if not os.path.exists(input_file):
        raise ValueError('Cannot open the input file!')
        
    cmd = 'relion_project --i %s --o %s --nr_uniform %d' % (input_file, output_file, num)
    os.system(cmd)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type = str, help = 'Input mrc/mrcs file.', default = './data/mask_10288_160.mrc')
    parser.add_argument('--num', type = int, help = 'The total number of generated projections.', default = 4000)
    parser.add_argument('--output', type = str, help = 'Output file path.', default = './data/projs')
    
    args = parser.parse_args()
    input_file = args.input
    num = args.num
    output_file = args.output
    print('Start synthesizing %d projections from %s!' % (num, input_file))
    syn_2dprojection(input_file, output_file, num)
    print('Done writing!')
    
if __name__ == '__main__':
    main()