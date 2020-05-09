import glob
import os


def find_and_unzip_maps(root_dir):
    zip_maps = glob.glob(root_dir + '/*/*.map.gz')
    
    for zip_map in zip_maps:
        cmd = 'gzip -d %s' % zip_map
        print(cmd)
        os.system(cmd)

    print('OK!')

def convert_map_to_mrc(root_dir):
    import mrcfile as mf

    maps = glob.glob(root_dir + '/*/*.map')

    for mp in maps:
        split_name = mp.split('.')
        split_name[-1] = 'mrc'
        mrc_name = '.'.join(n for n in split_name)

        with mf.new(mrc_name) as mrc_file:
            with mf.open(mp) as map_file:
                mrc_file.set_data(map_file.data)
                mrc_file.set_extended_header(map_file.extended_header)
                mrc_file.voxel_size = map_file.voxel_size
                mrc_file.header['nxstart'] = map_file.header['nxstart']
                mrc_file.header['nystart'] = map_file.header['nystart']
                mrc_file.header['nzstart'] = map_file.header['nzstart']
        
        print('Finish writing %s!' % mrc_name)

    print('OK!')

def get_all_mrc_files(root_dir):
    return glob.glob(root_dir + '/*/*.mrc')

def split_file_path(path):
    file_dir, file_name = os.path.split(path)
    file_prefix, file_postfix = os.path.splitext(file_name)
    return file_dir, file_prefix, file_postfix

if __name__ == '__main__':
    find_and_unzip_maps('/home/xulin/Documents/Dataset/PDB/testSet')
    convert_map_to_mrc('/home/xulin/Documents/Dataset/PDB/testSet')

    # dest = '/home/xulin/Documents/Dataset/EMDataResource/'
    # mapgzs = glob.glob(dest + '*.map.gz')

    # for mg in mapgzs:
    #     path, name = os.path.split(mg)
    #     name = name[:-7]
    #     name = name[:3] + '-' + name[4:]
    #     name = str.upper(name)
    #     print(path, name)

    #     newdir = path + '/' + name + '/'
    #     os.makedirs(newdir)

    #     cmd = 'cp %s %s' % (mg, newdir)
    #     os.system(cmd)