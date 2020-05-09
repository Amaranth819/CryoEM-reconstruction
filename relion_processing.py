import argparse
import os
import mrcfile as mf
import glob

def resize(orig_structure_path, dest_structure_path, new_box_size):
    if not os.path.exists(orig_structure_path):
        # raise ValueError('Cannot open %s! --resize()' % orig_structure_path)
        return

    with mf.open(orig_structure_path) as f:
        xsize, ysize, zsize = int(f.header['nx']), int(f.header['ny']), int(f.header['nz'])

        if xsize != ysize or xsize != zsize or ysize != zsize:
            return

        xangpix = float(f.voxel_size.x)
        rescale_xangpix = xangpix * xsize / new_box_size
        cmd = 'relion_image_handler --i %s --o %s --new_box %d --angpix %.3f --rescale_angpix %.3f' % (orig_structure_path, dest_structure_path, new_box_size, xangpix, rescale_xangpix)

    os.system(cmd)

    print('Finish creating %s! -- resize()' % dest_structure_path)

def sphere_mask(orig_structure_path, dest_structure_path, box_size):
    if not os.path.exists(orig_structure_path):
        # raise ValueError('Cannot open %s! --sphere_mask()' % orig_structure_path)
        return

    create_mask_cmd = 'relion_mask_create --denovo --box_size %d --o ./data/sphere_mask.mrc --outer_radius %d --width_soft_edge 32' % (box_size, box_size)
    multiply_mask_cmd = 'relion_image_handler --i %s --multiply ./data/sphere_mask.mrc --o %s' % (orig_structure_path, dest_structure_path)

    os.system(create_mask_cmd)
    os.system(multiply_mask_cmd)

    print('Finish creating %s! -- sphere_mask()' % dest_structure_path)

def syn_2dprojection(orig_structure_path, dest_projs_path, num):
    if not os.path.exists(orig_structure_path):
        # raise ValueError('Cannot open %s! --syn_2dprojection()' % orig_structure_path)
        return
        
    cmd = 'relion_project --i %s --o %s --nr_uniform %d' % (orig_structure_path, dest_projs_path, num)
    os.system(cmd)

    print('Finish creating %s! -- syn_2dprojection()' % dest_projs_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_mrc', type = str, default = './data/1081.mrc')
    parser.add_argument('--new_box_size', type = int, default = 160)
    parser.add_argument('--resized_mrc', type = str, default = './data/1081_160.mrc')
    parser.add_argument('--masked_mrc', type = str, default = './data/1081_160_mask.mrc')
    parser.add_argument('--projs', type = str, default = './data/projs.mrcs')
    parser.add_argument('--proj_num', type = int, default = 200)

    # Start processing
    config = parser.parse_args()
    resize(config.input_mrc, config.resized_mrc, config.new_box_size)
    sphere_mask(config.resized_mrc, config.masked_mrc, config.new_box_size)
    syn_2dprojection(config.masked_mrc, config.projs, config.proj_num)


def multi_structures_process(root_path, mrc_data_path, new_box_size, proj_num, structure_postfix = '.mrc', proj_postfix = '.mrcs'):
    structure_paths = os.listdir(root_path + mrc_data_path)
    structure_names = [sp[:-len(structure_postfix)] for sp in structure_paths]

    data_root = root_path + mrc_data_path
    structure_data_paths = [data_root + sn + structure_postfix for sn in structure_names]

    resized_root = root_path + 'resize/'
    if not os.path.exists(resized_root):
        os.makedirs(resized_root)
    resized_structure_names = [resized_root + sn + '_%d' % new_box_size + structure_postfix for sn in structure_names]

    masked_root = root_path + 'mask/'
    if not os.path.exists(masked_root):
        os.makedirs(masked_root)
    masked_structure_names = [masked_root + sn + '_%d_mask' % new_box_size + structure_postfix for sn in structure_names]

    projs_root = root_path + 'projs/'
    if not os.path.exists(projs_root):
        os.makedirs(projs_root)
    proj_names = [projs_root + sn + '_%d_mask_projs' % new_box_size for sn in structure_names]

    # print(structure_data_paths)
    # print(resized_structure_names)
    # print(masked_structure_names)
    # print(proj_names)

    for sdp, rsn, msn, pn in zip(structure_data_paths, resized_structure_names, masked_structure_names, proj_names):
        resize(sdp, rsn, new_box_size)
        sphere_mask(rsn, msn, new_box_size)
        syn_2dprojection(msn, pn, proj_num)


# 3.3
def single_structure_process(mrc_path, new_box_size, proj_num):
    mrc_dir, mrc_filename = os.path.split(mrc_path)
    mrc_name, mrc_postfix = os.path.splitext(mrc_filename)

    resized_mrc_name = mrc_name + '_%d' % new_box_size
    proj_name = resized_mrc_name + '_projs'

    resized_mrc_full_path = mrc_dir + '/' + resized_mrc_name + mrc_postfix
    resize(mrc_path, resized_mrc_full_path, new_box_size)
    # sphere mask
    proj_full_path = mrc_dir + '/' + proj_name
    syn_2dprojection(resized_mrc_full_path, proj_full_path, proj_num)

    return resized_mrc_full_path, proj_full_path


if __name__ == '__main__':
    # multi_structures_process('./data/test/', 'structure/', 160, 200)
    from unzip_pdbe_maps import get_all_mrc_files

    # mrcs = get_all_mrc_files('/home/xulin/Documents/Dataset/PDB/testSet')
    # # print(mrcs)
    # for mrc in mrcs:
    #     single_structure_process(mrc, 64, 24)
    #     # print(mrc)
    # # resize('/home/xulin/Documents/Dataset/PDB/testSet/EMD-1003/emd_1003.mrc', '/home/xulin/Documents/Dataset/PDB/testSet/EMD-1003/emd_1003_64.mrc', 64)

    resized_mrcs = sorted(glob.glob('../data_synthesis/fake/*/*.mrc'))
    for mrc in resized_mrcs:
        proj_path = mrc[:-4] + '_projs'
        syn_2dprojection(mrc, proj_path, 24)
