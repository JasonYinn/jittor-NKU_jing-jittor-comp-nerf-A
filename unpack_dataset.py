import os
import json
import numpy as np
import copy
import shutil

def config_parser():
    import configargparse
    parser = configargparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, required=True,
                        help='PATH_TO_Jrender_dataset')

    return parser

if __name__ == '__main__':
    parser = config_parser()
    args = parser.parse_args()

    scene_list = {'Car'      :[{'train': [0, 107], 'val': [0, 6], 'test': [0, 6]},
                            {'train': [107, 203], 'val': [7, 19], 'test': [7, 17]},
                            {'train': [203, 299], 'val': [20, 29], 'test': [18, 29]}], 
                'Coffee'   : [{'train': [0, 107], 'val': [0, 6], 'test': [0, 6]},
                            {'train': [107, 203], 'val': [7, 18], 'test': [7, 17]},
                            {'train': [203, 299], 'val': [19, 29], 'test': [18, 29]}], 
                'Easyship' : [{'train': [0, 103], 'val': [0, 6], 'test': [0, 9]},
                            {'train': [103, 202], 'val': [7, 15], 'test': [10, 21]},
                            {'train': [202, 299], 'val': [16, 29], 'test': [22, 29]}], 
                'Scarf'    : [{'train': [0, 33], 'val': [0, 2], 'test': [0, 3]},
                            {'train': [30, 65], 'val': [3, 5], 'test': [4, 6]},
                            {'train': [60, 99], 'val': [6, 9], 'test': [7, 9]}]}

    src_root = '/home/yzx/code/jittor-nerf/Jrender_Dataset/Scarf'
    tgt_root = '/home/yzx/code/jittor-nerf/Jrender_Dataset/Scarf_subset2'

    for scene, splits in scene_list.items():
        for idx, infos in enumerate(splits):
            src_root = os.path.join(args.data_path, scene)
            tgt_root = os.path.join(args.data_path, scene + f'_subset{idx + 1}')
            os.makedirs(tgt_root, exist_ok=True)

            for k, val in infos.items():
                with open(os.path.join(src_root, f'normalized_transforms_{k}.json'), 'r') as f:
                    js = json.load(f)
                js_new = copy.deepcopy(js)
                start, end = val
                js_new['frames'] = sorted( list( filter(lambda x: int(x['file_path'].split('_')[-1]) >= start and int(x['file_path'].split('_')[-1]) <= end, 
                                                        js['frames']) ), key=lambda x: int(x['file_path'].split('_')[-1]))
                
                
                img_root = os.path.join(tgt_root, k)
                os.makedirs(img_root, exist_ok=True)
                for frame in js_new['frames']:
                    src = frame['file_path'].replace('.', src_root) + '.png'
                    tgt = frame['file_path'].replace('.', tgt_root) + '.png'
                    if os.path.exists(src):
                        shutil.copy(src, tgt)
                with open(os.path.join(tgt_root, f'normalized_transforms_{k}.json'), 'w') as f:
                    json.dump(js_new, f)