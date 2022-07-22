import numpy as np
import shutil
import os
from tqdm import tqdm
import glob

def config_parser():
    import configargparse
    parser = configargparse.ArgumentParser()

    parser.add_argument('--result_path', type=str, required=True,
                        help='PATH_TO_result')
    parser.add_argument('--target_result_path', type=str, required=True,
                        help='path to save result')

    return parser

if __name__ == '__main__':
    parser = config_parser()
    args = parser.parse_args()

    src_root = args.result_path # '/home/yzx/code/jittor-nerf/submit/submit-20220520'
    scenes = ['Car', 'Coffee', 'Easyship', 'Scar', 'Scarf']
    tgt_root = args.target_result_path
    os.makedirs(tgt_root, exist_ok=True)

    for scene in scenes:
        if scene == 'Scar':
            res_lists = sorted(glob.glob(os.path.join(src_root, f'blender_{scene.lower()}/renderonly*')))
        else:
            res_lists = sorted(glob.glob(os.path.join(src_root, f'blender_{scene.lower()}_*/renderonly*')))

        idx = 0
        for sub_path in res_lists:
            imgs_list = sorted(list(filter(lambda x: x.endswith('.png'), os.listdir(sub_path))))
            for img_name in imgs_list:
                src = os.path.join(sub_path, img_name)
                tgt = os.path.join(tgt_root, f'{scene}_r_{idx}.png')
                shutil.copy(src, tgt)
                idx += 1