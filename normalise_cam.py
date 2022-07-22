import numpy as np
import json
import copy
import open3d as o3d
import os


def get_tf_cams(cam_dict, target_radius=1.):
    cam_centers = []
    for f in cam_dict:
        C2W = np.array(f['transform_matrix']).reshape(4, 4)
        cam_centers.append(C2W[:3, 3:4])

    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center
    scale = target_radius / radius

    return translate, scale


def normalize_cam_dict(in_cam_dict_file, out_cam_dict_file, translate, scale, target_radius=1., in_geometry_file=None, out_geometry_file=None):
    with open(in_cam_dict_file) as fp:
        in_cam_dict = json.load(fp)

    # translate, scale = get_tf_cams(in_cam_dict['frames'], target_radius=target_radius)

    if in_geometry_file is not None and out_geometry_file is not None:
        # check this page if you encounter issue in file io: http://www.open3d.org/docs/0.9.0/tutorial/Basic/file_io.html
        geometry = o3d.io.read_triangle_mesh(in_geometry_file)
        
        tf_translate = np.eye(4)
        tf_translate[:3, 3:4] = translate
        tf_scale = np.eye(4)
        tf_scale[:3, :3] *= scale
        tf = np.matmul(tf_scale, tf_translate)

        geometry_norm = geometry.transform(tf)
        o3d.io.write_triangle_mesh(out_geometry_file, geometry_norm)
  
    def transform_pose(C2W, translate, scale):
        cam_center = C2W[:3, 3]
        cam_center = (cam_center + translate) * scale
        C2W[:3, 3] = cam_center
        return C2W

    out_cam_dict = copy.deepcopy(in_cam_dict)
    for f_id, f in enumerate(out_cam_dict['frames']):
        C2W = np.array(f['transform_matrix']).reshape(4, 4)
        C2W = transform_pose(C2W, translate, scale)
        assert(np.isclose(np.linalg.det(C2W[:3, :3]), 1.))
        out_cam_dict['frames'][f_id]['transform_matrix'] = C2W.tolist()
    out_cam_dict['T'] = translate.tolist()
    with open(out_cam_dict_file, 'w') as fp:
        json.dump(out_cam_dict, fp, indent=2, sort_keys=True)

def config_parser():
    import configargparse
    parser = configargparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, required=True,
                        help='PATH_TO_Jrender_dataset')

    return parser

if __name__ == '__main__':
    # parser = config_parser()
    # args = parser.parse_args()
    target_radius = 1.0
    
    dataset_root = '/home/yzx/code/jittor-nerf/Jrender_Dataset_b' # args.data_path # '/home/yzx/nerf/nvdiffrec/data/nerf_synthetic'
    scene_list = sorted(list(filter(lambda x: os.path.isdir(os.path.join(dataset_root, x)), os.listdir(dataset_root))))
    scene_list = ['Car', 'Coffee', 'Easyship', 'Scar', 'Scarf']
    for scene in scene_list:
        # json_list = sorted(list(filter(lambda x: x.endswith('.json') and x.startswith('transforms'), os.listdir(os.path.join(dataset_root, scene)))))
        json_list = ['transforms_train.json', 'transforms_test.json', 'transforms_val.json']
        cams = []
        for js in json_list:
            with open(os.path.join(dataset_root, scene, js), 'r') as f:
                cams += json.load(f)['frames']
        translate, scale = get_tf_cams(cams, target_radius=target_radius)
        
        json_list_norm = ['transforms_test_b.json', 'transforms_train.json', 'transforms_test.json', 'transforms_val.json']
        for js in json_list_norm:
            in_cam_dict_file = os.path.join(dataset_root, scene, js)
            out_cam_dict_file = os.path.join(dataset_root, scene, 'normalized_' + js)
            normalize_cam_dict(in_cam_dict_file, out_cam_dict_file, translate, scale, target_radius=target_radius)
    
    # normalize_cam_dict(in_cam_dict_file, out_cam_dict_file, target_radius=1.)