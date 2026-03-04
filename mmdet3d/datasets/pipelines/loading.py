# Copyright (c) OpenMMLab. All rights reserved.
import os

import cv2
import matplotlib.pyplot as plt
import mmcv
import numpy as np
import torch
import torch.fft
import torch.nn.functional as F
from PIL import Image
from pyquaternion import Quaternion
from nuscenes.utils.geometry_utils import transform_matrix

from mmdet3d.core.points import BasePoints, get_points_type
from mmdet.datasets.pipelines import LoadAnnotations, LoadImageFromFile
from ...core.bbox import LiDARInstance3DBoxes
from ..builder import PIPELINES

from torchvision.transforms.functional import rotate
import open3d as o3d

@PIPELINES.register_module()
class LoadOccGTFromFileCVPR2023(object):
    def __init__(self,
                 scale_1_2=False,
                 scale_1_4=False,
                 scale_1_8=False,
                 load_mask=False,
                 load_flow=False,
                 flow_gt_path=None,
                 ignore_invisible=False,
                 group_list=None,
                 ):
        self.scale_1_2 = scale_1_2
        self.scale_1_4 = scale_1_4
        self.scale_1_8 = scale_1_8
        self.ignore_invisible = ignore_invisible
        self.group_list = group_list
        self.load_mask = load_mask
        self.load_flow = load_flow
        self.flow_gt_path = flow_gt_path

    def __call__(self, results):
        occ_gt_path = results['occ_gt_path']
        occ_gt_label = os.path.join(occ_gt_path, "labels.npz")
        occ_gt_label_1_2 = os.path.join(occ_gt_path, "labels_1_2.npz")
        occ_gt_label_1_4 = os.path.join(occ_gt_path, "labels_1_4.npz")
        occ_gt_label_1_8 = os.path.join(occ_gt_path, "labels_1_8.npz")

        occ_labels = np.load(occ_gt_label)

        semantics = occ_labels['semantics']
        if self.load_mask:
            voxel_mask = occ_labels['mask_camera']
            results['voxel_mask_camera'] = voxel_mask.astype(bool)
            if self.ignore_invisible:
                semantics[voxel_mask==0] = 255
        results['voxel_semantics'] = semantics

        if self.scale_1_2:
            occ_labels_1_2 = np.load(occ_gt_label_1_2)
            semantics_1_2 = occ_labels_1_2['semantics']

            if self.load_mask:
                voxel_mask = occ_labels_1_2['mask_camera']
                if self.ignore_invisible:
                    semantics_1_2[voxel_mask==0] = 255
                results['voxel_mask_camera_1_2'] = voxel_mask
            results['voxel_semantics_1_2'] = semantics_1_2

        if self.scale_1_4:
            occ_labels_1_4 = np.load(occ_gt_label_1_4)
            semantics_1_4 = occ_labels_1_4['semantics']

            if self.load_mask:
                voxel_mask = occ_labels_1_4['mask_camera']
                if self.ignore_invisible:
                    semantics_1_4[voxel_mask==0] = 255
                results['voxel_mask_camera_1_4'] = voxel_mask
            results['voxel_semantics_1_4'] = semantics_1_4

        if self.scale_1_8:
            occ_labels_1_8 = np.load(occ_gt_label_1_8)
            semantics_1_8 = occ_labels_1_8['semantics']

            if self.load_mask:
                voxel_mask = occ_labels_1_8['mask_camera']
                if self.ignore_invisible:
                    semantics_1_8[voxel_mask==0] = 255
                results['voxel_mask_camera_1_8'] = voxel_mask
            results['voxel_semantics_1_8'] = semantics_1_8

        return results

@PIPELINES.register_module()
class LoadOccGTFromFileOpenOcc(object):
    def __init__(self, scale_1_2=False, scale_1_4=False, scale_1_8=False, load_ray_mask=False):
        self.scale_1_2 = scale_1_2
        self.scale_1_4 = scale_1_4
        self.scale_1_8 = scale_1_8
        self.load_ray_mask = load_ray_mask

    def __call__(self, results):
        gts_occ_gt_path = results['occ_gt_path']

        occ_ray_mask_path = gts_occ_gt_path.replace('gts', 'openocc_v2_ray_mask')
        occ_ray_mask = os.path.join(occ_ray_mask_path, 'labels.npz')
        occ_ray_mask_1_2 = os.path.join(occ_ray_mask_path, 'labels_1_2.npz')
        occ_ray_mask_1_4 = os.path.join(occ_ray_mask_path, 'labels_1_4.npz')
        occ_ray_mask_1_8 = os.path.join(occ_ray_mask_path, 'labels_1_8.npz')

        occ_gt_path = gts_occ_gt_path.replace('gts', 'openocc_v2')
        occ_gt_label = os.path.join(occ_gt_path, "labels.npz")
        occ_gt_label_1_2 = os.path.join(occ_gt_path, "labels_1_2.npz")
        occ_gt_label_1_4 = os.path.join(occ_gt_path, "labels_1_4.npz")
        occ_gt_label_1_8 = os.path.join(occ_gt_path, "labels_1_8.npz")
        occ_labels = np.load(occ_gt_label)

        semantics = occ_labels['semantics']
        flow = occ_labels['flow']

        if self.scale_1_2:
            occ_labels_1_2 = np.load(occ_gt_label_1_2)
            semantics_1_2 = occ_labels_1_2['semantics']
            flow_1_2 = occ_labels_1_2['flow']
            results['voxel_semantics_1_2'] = semantics_1_2
            results['voxel_flow_1_2'] = flow_1_2
            if self.load_ray_mask:
                ray_mask_1_2 = np.load(occ_ray_mask_1_2)
                ray_mask_1_2 = ray_mask_1_2['ray_mask2']
                results['ray_mask_1_2'] = ray_mask_1_2
        if self.scale_1_4:
            occ_labels_1_4 = np.load(occ_gt_label_1_4)
            semantics_1_4 = occ_labels_1_4['semantics']
            flow_1_4 = occ_labels_1_4['flow']
            results['voxel_semantics_1_4'] = semantics_1_4
            results['voxel_flow_1_4'] = flow_1_4
            if self.load_ray_mask:
                ray_mask_1_4 = np.load(occ_ray_mask_1_4)
                ray_mask_1_4 = ray_mask_1_4['ray_mask2']
                results['ray_mask_1_4'] = ray_mask_1_4
        if self.scale_1_8:
            occ_labels_1_8 = np.load(occ_gt_label_1_8)
            semantics_1_8 = occ_labels_1_8['semantics']
            flow_1_8 = occ_labels_1_8['flow']
            results['voxel_semantics_1_8'] = semantics_1_8
            results['voxel_flow_1_8'] = flow_1_8
            if self.load_ray_mask:
                ray_mask_1_8 = np.load(occ_ray_mask_1_8)
                ray_mask_1_8 = ray_mask_1_8['ray_mask2']
                results['ray_mask_1_8'] = ray_mask_1_8

        if self.load_ray_mask:
            ray_mask = np.load(occ_ray_mask)
            ray_mask = ray_mask['ray_mask2']
            results['ray_mask'] = ray_mask

        results['voxel_semantics'] = semantics
        results['voxel_flows'] = flow

        return results

@PIPELINES.register_module()
class LoadPointsFromFile(object):
    """Load Points From File.

    Load points from file.

    Args:
        coord_type (str): The type of coordinates of points cloud.
            Available options includes:
            - 'LIDAR': Points in LiDAR coordinates.
            - 'DEPTH': Points in depth coordinates, usually for indoor dataset.
            - 'CAMERA': Points in camera coordinates.
        load_dim (int, optional): The dimension of the loaded points.
            Defaults to 6.
        use_dim (list[int], optional): Which dimensions of the points to use.
            Defaults to [0, 1, 2]. For KITTI dataset, set use_dim=4
            or use_dim=[0, 1, 2, 3] to use the intensity dimension.
        shift_height (bool, optional): Whether to use shifted height.
            Defaults to False.
        use_color (bool, optional): Whether to use color features.
            Defaults to False.
        file_client_args (dict, optional): Config dict of file clients,
            refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details. Defaults to dict(backend='disk').
    """

    def __init__(self,
                 coord_type,
                 load_dim=6,
                 use_dim=[0, 1, 2],
                 shift_height=False,
                 use_color=False,
                 file_client_args=dict(backend='disk')):
        self.shift_height = shift_height
        self.use_color = use_color
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        assert max(use_dim) < load_dim, \
            f'Expect all used dimensions < {load_dim}, got {use_dim}'
        assert coord_type in ['CAMERA', 'LIDAR', 'DEPTH']

        self.coord_type = coord_type
        self.load_dim = load_dim
        self.use_dim = use_dim
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def _load_points(self, pts_filename):
        """Private function to load point clouds data.

        Args:
            pts_filename (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            pts_bytes = self.file_client.get(pts_filename)
            points = np.frombuffer(pts_bytes, dtype=np.float32)
        except ConnectionError:
            mmcv.check_file_exist(pts_filename)
            if pts_filename.endswith('.npy'):
                points = np.load(pts_filename)
            else:
                points = np.fromfile(pts_filename, dtype=np.float32)

        return points

    def __call__(self, results):
        """Call function to load points data from file.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the point clouds data.
                Added key and value are described below.

                - points (:obj:`BasePoints`): Point clouds data.
        """
        pts_filename = results['pts_filename']
        points = self._load_points(pts_filename)
        points = points.reshape(-1, self.load_dim)
        points = points[:, self.use_dim]
        attribute_dims = None

        if self.shift_height:
            floor_height = np.percentile(points[:, 2], 0.99)
            height = points[:, 2] - floor_height
            points = np.concatenate(
                [points[:, :3],
                 np.expand_dims(height, 1), points[:, 3:]], 1)
            attribute_dims = dict(height=3)

        if self.use_color:
            assert len(self.use_dim) >= 6
            if attribute_dims is None:
                attribute_dims = dict()
            attribute_dims.update(
                dict(color=[
                    points.shape[1] - 3,
                    points.shape[1] - 2,
                    points.shape[1] - 1,
                ]))

        points_class = get_points_type(self.coord_type)
        points = points_class(
            points, points_dim=points.shape[-1], attribute_dims=attribute_dims)
        results['points'] = points

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__ + '('
        repr_str += f'shift_height={self.shift_height}, '
        repr_str += f'use_color={self.use_color}, '
        repr_str += f'file_client_args={self.file_client_args}, '
        repr_str += f'load_dim={self.load_dim}, '
        repr_str += f'use_dim={self.use_dim})'
        return repr_str


@PIPELINES.register_module()
class PointToMultiViewDepth(object):

    def __init__(self, grid_config, downsample=1):
        self.downsample = downsample
        self.grid_config = grid_config
        self.index = 0
        self.num_cam = 6
        self.std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
        self.mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)

    def vis_depth_img(self, img, depth):
        depth = depth.cpu().numpy()
        img = img.permute(1, 2, 0).cpu().numpy()
        img = img * self.std + self.mean
        img = np.array(img, dtype=np.uint8)
        invalid_y, invalid_x, invalid_c = np.where(img == 0)
        depth[invalid_y, invalid_x] = 0
        y, x = np.where(depth != 0)
        plt.figure()
        plt.imshow(img)
        plt.scatter(x, y, c=depth[y, x], cmap='rainbow_r', alpha=0.5, s=2)
        plt.show()
        self.index = self.index + 1

    def points2depthmap(self, points, height, width):
        height, width = height // self.downsample, width // self.downsample
        depth_map = torch.zeros((height, width), dtype=torch.float32)

        coor = torch.round(points[:, :2] / self.downsample)
        depth = points[:, 2]
        kept1 = (coor[:, 0] >= 0) & (coor[:, 0] < width) & (
            coor[:, 1] >= 0) & (coor[:, 1] < height) & (
                depth < self.grid_config['depth'][1]) & (
                    depth >= self.grid_config['depth'][0])
        coor, depth = coor[kept1], depth[kept1]

        ranks = coor[:, 0] + coor[:, 1] * width
        sort = (ranks + depth / 100.).argsort()
        coor, depth, ranks = coor[sort], depth[sort], ranks[sort]

        kept2 = torch.ones(coor.shape[0], device=coor.device, dtype=torch.bool)
        kept2[1:] = (ranks[1:] != ranks[:-1])
        coor, depth = coor[kept2], depth[kept2]
        coor = coor.to(torch.long)
        depth_map[coor[:, 1], coor[:, 0]] = depth
        return depth_map

    def __call__(self, results):
        # prev process info
        points_lidar = results['points'].tensor
        imgs, sensor2egos, ego2globals, cam2imgs, post_augs, bda = results['img_inputs']
        lidar2imgs = results['lidar2img']
        nt, c, h, w = imgs.shape
        t_frame = nt // 6

        # store list
        depth_maps = []                     # process result

        vis_index = 0
        for cid in range(len(results['cam_names'])):
            lidar2img = lidar2imgs[cid]

            # project lidar point to img plane
            points_img = lidar2img @ torch.cat([points_lidar.T, torch.ones((1, points_lidar.shape[0]))], dim=0)
            points_img = points_img.permute(1, 0)
            points_img = torch.cat([points_img[:, :2] / points_img[:, 2].unsqueeze(1), points_img[:, 2].unsqueeze(1)], dim=1)

            # get corresponding depth value
            depth_map = self.points2depthmap(points_img, h, w)

            # store
            depth_maps.append(depth_map)

            # vis depth img to check the correctness
            # self.vis_depth_img(imgs[cid*t_frame], depth_map)

        results['gt_depth'] = torch.stack(depth_maps)
        return results

def mmlabNormalize(img):
    from mmcv.image.photometric import imnormalize
    to_rgb = True
    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
    img = imnormalize(np.array(img), mean, std, to_rgb)
    img = torch.tensor(img).float().permute(2, 0, 1).contiguous()
    return img


@PIPELINES.register_module()
class PrepareImageInputs(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(
        self,
        data_config,
        is_train=False,
        sequential=False,
        opencv_pp=False,
    ):
        self.is_train = is_train
        self.data_config = data_config
        self.normalize_img = mmlabNormalize
        self.sequential = sequential
        self.opencv_pp = opencv_pp

    def get_rot(self, h):
        return torch.Tensor([
            [np.cos(h), np.sin(h)],
            [-np.sin(h), np.cos(h)],
        ])

    def img_transform(self, img, post_rot, post_tran, resize, resize_dims,
                      crop, flip, rotate):
        # adjust image
        if not self.opencv_pp:
            img = self.img_transform_core(img, resize_dims, crop, flip, rotate)

        # post-homography transformation
        post_rot *= resize
        post_tran -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            post_rot = A.matmul(post_rot)
            post_tran = A.matmul(post_tran) + b
        A = self.get_rot(rotate / 180 * np.pi)
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        post_rot = A.matmul(post_rot)
        post_tran = A.matmul(post_tran) + b
        if self.opencv_pp:
            img = self.img_transform_core_opencv(img, post_rot, post_tran, crop)

        copy_img = img.copy()
        invalid_index = np.where(np.array(copy_img)==0)

        return img, post_rot, post_tran, invalid_index

    def img_transform_core_opencv(self, img, post_rot, post_tran,
                                  crop):
        img = np.array(img).astype(np.float32)
        img = cv2.warpAffine(img,
                             np.concatenate([post_rot,
                                            post_tran.reshape(2,1)],
                                            axis=1),
                             (crop[2]-crop[0], crop[3]-crop[1]),
                             flags=cv2.INTER_LINEAR)
        return img

    def img_transform_core(self, img, resize_dims, crop, flip, rotate):
        # adjust image
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)
        return img

    def sample_augmentation(self, H, W, flip=None, scale=None):
        fH, fW = self.data_config['input_size']
        if self.is_train:
            resize = float(fW) / float(W)
            resize += np.random.uniform(*self.data_config['resize'])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            random_crop_height = self.data_config.get('random_crop_height', False)
            if random_crop_height:
                crop_h = int(np.random.uniform(max(0.3*newH, newH-fH), newH-fH))
            else:
                crop_h = int((1 - np.random.uniform(*self.data_config['crop_h'])) * newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = self.data_config['flip'] and np.random.choice([0, 1])
            rotate = np.random.uniform(*self.data_config['rot'])
            if self.data_config.get('vflip', False) and np.random.choice([0, 1]):
                rotate += 180
        else:
            resize = float(fW) / float(W)
            if scale is not None:
                resize += scale
            else:
                resize += self.data_config.get('resize_test', 0.0)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.data_config['crop_h'])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False if flip is None else flip
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

    def get_sensor_transforms(self, cam_info, cam_name):
        # get sensor2ego
        sensor2ego = transform_matrix(
            translation=cam_info['cams'][cam_name]['sensor2ego_translation'],
            rotation=Quaternion(cam_info['cams'][cam_name]['sensor2ego_rotation'])
        )
        sensor2ego = torch.from_numpy(sensor2ego).to(torch.float32)

        ego2sensor = transform_matrix(
            translation=cam_info['cams'][cam_name]['sensor2ego_translation'],
            rotation=Quaternion(cam_info['cams'][cam_name]['sensor2ego_rotation']),
            inverse=True
        )
        ego2sensor = torch.from_numpy(ego2sensor).to(torch.float32)

        # get sensorego2global
        ego2global = transform_matrix(
            translation=cam_info['cams'][cam_name]['ego2global_translation'],
            rotation=Quaternion(cam_info['cams'][cam_name]['ego2global_rotation'])
        )
        ego2global = torch.from_numpy(ego2global).to(torch.float32)

        global2ego = transform_matrix(
            translation=cam_info['cams'][cam_name]['ego2global_translation'],
            rotation=Quaternion(cam_info['cams'][cam_name]['ego2global_rotation']),
            inverse=True
        )
        global2ego = torch.from_numpy(global2ego).to(torch.float32)

        return sensor2ego, ego2global, ego2sensor, global2ego

    def get_lidar_transformation(self, results):
        # get lidar2ego
        lidar2lidarego = transform_matrix(
            translation=results['curr']['lidar2ego_translation'],
            rotation=Quaternion(results['curr']['lidar2ego_rotation']),
        )
        lidar2lidarego = torch.from_numpy(lidar2lidarego).to(torch.float32)

        # get ego2lidar
        lidarego2lidar = transform_matrix(
            translation=results['curr']['lidar2ego_translation'],
            rotation=Quaternion(results['curr']['lidar2ego_rotation']),
            inverse=True
        )
        lidarego2lidar = torch.from_numpy(lidarego2lidar).to(torch.float32)

        # get ego2global
        lidarego2global = transform_matrix(
            translation=results['curr']['ego2global_translation'],
            rotation=Quaternion(results['curr']['ego2global_rotation']),
        )
        lidarego2global = torch.from_numpy(lidarego2global).to(torch.float32)

        return lidar2lidarego, lidarego2lidar, lidarego2global

    def photo_metric_distortion(self, img, pmd):
        """Call function to perform photometric distortion on images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Result dict with images distorted.
        """
        if np.random.rand()>pmd.get('rate', 1.0):
            return img

        img = np.array(img).astype(np.float32)
        assert img.dtype == np.float32, \
            'PhotoMetricDistortion needs the input image of dtype np.float32,' \
            ' please set "to_float32=True" in "LoadImageFromFile" pipeline'
        # random brightness
        if np.random.randint(2):
            delta = np.random.uniform(-pmd['brightness_delta'],
                                   pmd['brightness_delta'])
            img += delta

        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        mode = np.random.randint(2)
        if mode == 1:
            if np.random.randint(2):
                alpha = np.random.uniform(pmd['contrast_lower'],
                                       pmd['contrast_upper'])
                img *= alpha

        # convert color from BGR to HSV
        img = mmcv.bgr2hsv(img)

        # random saturation
        if np.random.randint(2):
            img[..., 1] *= np.random.uniform(pmd['saturation_lower'],
                                          pmd['saturation_upper'])

        # random hue
        if np.random.randint(2):
            img[..., 0] += np.random.uniform(-pmd['hue_delta'], pmd['hue_delta'])
            img[..., 0][img[..., 0] > 360] -= 360
            img[..., 0][img[..., 0] < 0] += 360

        # convert color from HSV to BGR
        img = mmcv.hsv2bgr(img)

        # random contrast
        if mode == 0:
            if np.random.randint(2):
                alpha = np.random.uniform(pmd['contrast_lower'],
                                       pmd['contrast_upper'])
                img *= alpha

        # randomly swap channels
        if np.random.randint(2):
            img = img[..., np.random.permutation(3)]
        return Image.fromarray(img.astype(np.uint8))

    def get_inputs(self, results, flip=None, scale=None):
        # get cam_names
        cam_names = self.data_config['cams']

        # get store list
        imgs = []
        (sensor2egos, ego2globals, ego2sensors, global2egos, cam2imgs, post_augs,
         lidar2imgs, ego2lidars)  =\
            [], [], [], [], [], [], [], []
        lidar2cams = []
        # get lidar-related transformation
        lidar2lidarego, lidarego2lidar, lidarego2global = self.get_lidar_transformation(results)

        for cam_name in cam_names:
            cam_data = results['curr']['cams'][cam_name]
            filename = cam_data['data_path']
            img = Image.open(filename)

            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)

            # get cam-related transformation
            cam2img = torch.eye(4)
            cam2img[:3, :3] = torch.tensor(cam_data['cam_intrinsic'][:3, :3], dtype=torch.float32)
            sensor2ego, ego2global, ego2sensor, global2ego = self.get_sensor_transforms(results['curr'], cam_name)

            # image view augmentation (resize, crop, horizontal flip, rotate)
            img_augs = self.sample_augmentation(H=img.height, W=img.width, flip=flip, scale=scale)
            resize, resize_dims, crop, flip, rotate = img_augs
            img, post_rot2, post_tran2, invalid_index = \
                self.img_transform(img, post_rot,
                                   post_tran,
                                   resize=resize,
                                   resize_dims=resize_dims,
                                   crop=crop,
                                   flip=flip,
                                   rotate=rotate)

            # for convenience, make augmentation matrices 4x4
            post_aug = torch.eye(4)
            post_aug[:2, :2] = post_rot2
            post_aug[:2, 2] = post_tran2

            # get lidar2img
            lidar2img = cam2img @ ego2sensor @ global2ego @ lidarego2global @ lidar2lidarego
            lidar2img = post_aug @ lidar2img
            lidar2cam = ego2sensor @ global2ego @ lidarego2global @ lidar2lidarego
            if self.is_train and self.data_config.get('pmd', None) is not None:
                img = self.photo_metric_distortion(img, self.data_config['pmd'])

            imgs.append(self.normalize_img(img))

            # adjacent frame use the same aug with current frame
            if self.sequential:
                assert 'adjacent' in results
                for adj_info in results['adjacent']:
                    filename_adj = adj_info['cams'][cam_name]['data_path']
                    img_adjacent = Image.open(filename_adj)
                    if self.opencv_pp:
                        img_adjacent = \
                            self.img_transform_core_opencv(
                                img_adjacent,
                                post_rot[:2, :2],
                                post_tran[:2],
                                crop)
                    else:
                        img_adjacent = self.img_transform_core(
                            img_adjacent,
                            resize_dims=resize_dims,
                            crop=crop,
                            flip=flip,
                            rotate=rotate)
                    imgs.append(self.normalize_img(img_adjacent))
            lidar2cams.append(lidar2cam)
            cam2imgs.append(cam2img)
            sensor2egos.append(sensor2ego)
            ego2globals.append(ego2global)
            ego2sensors.append(ego2sensor)
            global2egos.append(global2ego)
            post_augs.append(post_aug)
            lidar2imgs.append(lidar2img)
        ego2lidars.append(lidarego2lidar)

        if self.sequential:
            for adj_info in results['adjacent']:
                # for convenience
                cam2imgs.extend(cam2imgs[:len(cam_names)])
                post_augs.extend(post_augs[:len(cam_names)])

                # align
                for cam_name in cam_names:
                    sensor2ego, ego2global, ego2sensor, global2ego = \
                        self.get_sensor_transforms(adj_info, cam_name)
                    sensor2egos.append(sensor2ego)
                    ego2globals.append(ego2global)
                    ego2sensors.append(ego2sensor)
                    global2egos.append(global2ego)

        imgs = torch.stack(imgs)
        # sensor2egos and ego2globals containes current and adjacent frame information
        sensor2egos = torch.stack(sensor2egos)
        ego2globals = torch.stack(ego2globals)
        ego2sensors = torch.stack(ego2sensors)
        global2egos = torch.stack(global2egos)
        # cam2imgs and post_augs only contain current frame information
        cam2imgs = torch.stack(cam2imgs)
        post_augs = torch.stack(post_augs)
        # lidar2imgs and ego2lidars only contain current frame information
        lidar2imgs = torch.stack(lidar2imgs)
        ego2lidars = torch.stack(ego2lidars)
        lidar2cams = torch.stack(lidar2cams)
        # store
        results['cam_names'] = cam_names
        results['sensor2sensorego'] = sensor2egos
        results['sensorego2global'] = ego2globals
        results['sensorego2sensor'] = ego2sensors
        results['global2sensorego'] = global2egos
        results['lidar2img'] = lidar2imgs
        results['ego2lidar'] = ego2lidars
        results['lidar2cam'] = lidar2cams
        return (imgs, sensor2egos, ego2globals, cam2imgs, post_augs)

    def __call__(self, results):
        results['img_inputs'] = self.get_inputs(results)
        return results


@PIPELINES.register_module()
class LoadAnnotations(object):

    def __call__(self, results):
        gt_boxes, gt_labels = results['ann_infos']
        gt_boxes = np.array(gt_boxes)
        gt_labels = np.array(gt_labels)
        gt_boxes, gt_labels = torch.Tensor(gt_boxes), torch.tensor(gt_labels)
        if len(gt_boxes) == 0:
            gt_boxes = torch.zeros(0, 9)
        results['gt_bboxes_3d'] = LiDARInstance3DBoxes(gt_boxes, box_dim=gt_boxes.shape[-1], origin=(0.5, 0.5, 0.5))
        results['gt_labels_3d'] = gt_labels
        return results


@PIPELINES.register_module()
class BEVAug(object):

    def __init__(self, bda_aug_conf, classes, is_train=True):
        self.bda_aug_conf = bda_aug_conf
        self.is_train = is_train
        self.classes = classes

    def sample_bda_augmentation(self):
        """Generate bda augmentation values based on bda_config."""
        if self.is_train:
            rotate_bda = np.random.uniform(*self.bda_aug_conf['rot_lim'])
            scale_bda = np.random.uniform(*self.bda_aug_conf['scale_lim'])
            flip_dx = np.random.uniform() < self.bda_aug_conf['flip_dx_ratio']
            flip_dy = np.random.uniform() < self.bda_aug_conf['flip_dy_ratio']
            translation_std = self.bda_aug_conf.get('tran_lim', [0.0, 0.0, 0.0])
            tran_bda = np.random.normal(scale=translation_std, size=3).T
        else:
            rotate_bda = 0
            scale_bda = 1.0
            flip_dx = False
            flip_dy = False
            tran_bda = np.zeros((1, 3), dtype=np.float32)
        return rotate_bda, scale_bda, flip_dx, flip_dy, tran_bda

    def bev_transform(self, rotate_angle, scale_ratio, flip_dx, flip_dy, tran_bda):
        # get rotation matrix
        rotate_angle = torch.tensor(rotate_angle / 180 * np.pi)
        rot_sin = torch.sin(rotate_angle)
        rot_cos = torch.cos(rotate_angle)
        rot_mat = torch.Tensor([
            [rot_cos, -rot_sin, 0],
            [rot_sin, rot_cos, 0],
            [0, 0, 1]])
        scale_mat = torch.Tensor([
            [scale_ratio, 0, 0],
            [0, scale_ratio, 0],
            [0, 0, scale_ratio]])
        flip_mat = torch.Tensor([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]]
        )

        if flip_dx:
            flip_mat = flip_mat @ torch.Tensor([
                [-1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]
            ])
        if flip_dy:
            flip_mat = flip_mat @ torch.Tensor([
                [1, 0, 0],
                [0, -1, 0],
                [0, 0, 1]
            ])

        rot_mat = flip_mat @ (scale_mat @ rot_mat)
        return rot_mat

    def voxel_transform(self, results, flip_dx, flip_dy):
        if flip_dx:
            results['voxel_semantics'] = results['voxel_semantics'][::-1,...].copy()
            if 'voxel_semantics_1_2' in results:
                results['voxel_semantics_1_2'] = results['voxel_semantics_1_2'][::-1, ...].copy()
            if 'voxel_semantics_1_4' in results:
                results['voxel_semantics_1_4'] = results['voxel_semantics_1_4'][::-1, ...].copy()
            if 'voxel_semantics_1_8' in results:
                results['voxel_semantics_1_8'] = results['voxel_semantics_1_8'][::-1, ...].copy()
            if 'voxel_flows' in results:
                results['voxel_flows'] = results['voxel_flows'][::-1, ...].copy()
                results['voxel_flows'][..., 0] = - results['voxel_flows'][..., 0]
                results['voxel_flows'][..., 0][results['voxel_flows'][..., 0] == -255] = 255

        if flip_dy:
            results['voxel_semantics'] = results['voxel_semantics'][:,::-1,...].copy()
            if 'voxel_semantics_1_2' in results:
                results['voxel_semantics_1_2'] = results['voxel_semantics_1_2'][:, ::-1, ...].copy()
            if 'voxel_semantics_1_4' in results:
                results['voxel_semantics_1_4'] = results['voxel_semantics_1_4'][:, ::-1, ...].copy()
            if 'voxel_semantics_1_8' in results:
                results['voxel_semantics_1_8'] = results['voxel_semantics_1_8'][:, ::-1, ...].copy()
            if 'voxel_flows' in results:
                results['voxel_flows'] = results['voxel_flows'][:, ::-1, ...].copy()
                results['voxel_flows'][..., 1] = - results['voxel_flows'][..., 1]
                results['voxel_flows'][..., 1][results['voxel_flows'][..., 1] == -255] = 255

        return results

    def __call__(self, results):
        # sample bda augmentation
        rotate_bda, scale_bda, flip_dx, flip_dy, tran_bda = self.sample_bda_augmentation()
        if 'bda_aug' in results:
            flip_dx, flip_dy = results['bda_aug']['flip_dx'], results['bda_aug']['flip_dy']

        # get bda matrix
        bda_rot = self.bev_transform(rotate_bda, scale_bda, flip_dx, flip_dy, tran_bda)
        bda_mat = torch.zeros(4, 4)
        bda_mat[3, 3] = 1
        bda_mat[:3, :3] = bda_rot
        bda_mat[:3, 3] = torch.from_numpy(tran_bda)

        # do voxel transformation
        results = self.voxel_transform(results, flip_dx=flip_dx, flip_dy=flip_dy)

        # update img_inputs
        imgs, sensor2egos, ego2globals, cam2imgs, post_augs = results['img_inputs']
        results['img_inputs'] = (imgs, sensor2egos, ego2globals, cam2imgs, post_augs, bda_mat)

        return results
@PIPELINES.register_module()
class PointToMultiViewDepth_fix_nomal_triview_fixdirectionByDepthBound_fixattn_fixnormal(object):

    def __init__(self, grid_config, downsample=1,normal_radius=1.2, normal_max_nn=30):
        self.downsample = downsample
        self.grid_config = grid_config
        self.index = 0
        self.num_cam = 6
        self.std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
        self.mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        self.normal_radius = normal_radius  # 法向量邻域搜索半径
        self.normal_max_nn = normal_max_nn
        # 新增：bin划分参数（7.5度/ bin，与需求一致）
        self.wh_bin_size = 22.5  # W/H方向每个bin的角度范围（度）
        self.wh_num_bins = 8   # W/H方向总bin数（180/7.5=24）
        self.depth_bin_size = 18  # 深度方向每个bin的角度范围（度）
        self.depth_num_bins = 5   # 深度方向总bin数（90/7.5=12）
        self.eps_label = 1e-8  # 判断"无label的0值"的阈值（小于此值视为无label）
        # 新增：高度相关配置（可根据LiDAR坐标系调整，默认Z轴为高度）
        self.lidar_height_axis = 2  # LiDAR点云的高度轴（0=X,1=Y,2=Z，默认Z轴）
        self.height_min = grid_config.get('height', [-10.0, 10.0])[0]  # 有效高度最小值（过滤地面以下/异常点）
        self.height_max = grid_config.get('height', [-10.0, 10.0])[1]  # 有效高度最大值（过滤高处/异常点）

    def vis_normal_components(self, img, comp_W, comp_H, comp_depth):
        """可视化法向量在W轴、H轴、深度方向的分量图"""
        # 数据格式转换（Tensor→numpy，无效像素置0）
        img = img.permute(1, 2, 0).cpu().numpy()
        img = img * self.std + self.mean
        img = np.array(img, dtype=np.uint8)
        invalid_mask = (img == 0).any(axis=2)  # 图像黑边标记

        # 三个分量转numpy并处理无效像素
        comp_W = comp_W.cpu().numpy()
        comp_H = comp_H.cpu().numpy()
        comp_depth = comp_depth.cpu().numpy()
        comp_W[invalid_mask] = 0
        comp_H[invalid_mask] = 0
        comp_depth[invalid_mask] = 0

        # 筛选有效像素（非零分量）
        valid_mask = (comp_depth != 0)  # 用深度分量筛选有效点
        y, x = np.where(valid_mask)

        # 创建3子图可视化
        plt.figure(figsize=(18, 6))
        # 子图1：W轴分量（左右倾斜）
        plt.subplot(1, 3, 1)
        plt.imshow(img)
        scatter1 = plt.scatter(x, y, c=comp_W[y, x], cmap='coolwarm', alpha=0.6, s=2)
        plt.title('Normal Component - Image W Axis (Left-Right)')
        plt.colorbar(scatter1, label='Component Value (-1=Left, 1=Right)')

        # 子图2：H轴分量（上下倾斜）
        plt.subplot(1, 3, 2)
        plt.imshow(img)
        scatter2 = plt.scatter(x, y, c=comp_H[y, x], cmap='coolwarm', alpha=0.6, s=2)
        plt.title('Normal Component - Image H Axis (Up-Down)')
        plt.colorbar(scatter2, label='Component Value (-1=Up, 1=Down)')

        # 子图3：深度分量（朝向/背离相机）
        plt.subplot(1, 3, 3)
        plt.imshow(img)
        scatter3 = plt.scatter(x, y, c=comp_depth[y, x], cmap='coolwarm', alpha=0.6, s=2)
        plt.title('Normal Component - Depth Direction (Camera Facing)')
        plt.colorbar(scatter3, label='Component Value (-1=Away, 1=Towards)')

        plt.tight_layout()
        plt.show()
        self.index += 1

    def vis_depth_img_save(self, img, depth, theta_map=None, save_dir=None, show=True):
        """
        可视化深度图和视线角度图，并支持保存到本地
        
        参数:
            img: 原始图像张量 (C, H, W)
            depth: 深度图张量 (H, W)
            theta_map: 视线角度图张量 (H, W)，可选
            save_dir: 保存图像的目录路径，为None时不保存
            show: 是否显示图像，默认为True
        """
        # 1. 数据格式转换（Tensor→numpy）
        depth = depth.cpu().numpy()
        img = img.permute(1, 2, 0).cpu().numpy()  # (H, W, C)
        img = img * self.std + self.mean  # 反归一化
        img = np.array(img, dtype=np.uint8)  # 转为uint8图像格式

        # 2. 处理无效像素（图像黑边）
        invalid_mask = (img == 0).any(axis=2)  # 修正：定义无效区域掩码（黑边）
        depth[invalid_mask] = 0  # 无效区域深度置0
        if theta_map is not None:
            theta_map = theta_map.cpu().numpy()
            theta_map[invalid_mask] = 0  # 无效区域角度置0

        # 3. 筛选有效像素（用于散点图）
        valid_mask = (depth != 0) if theta_map is None else (theta_map != 0)
        y, x = np.where(valid_mask)

        # 4. 创建图像并绘图
        plt.figure(figsize=(14, 6))

        # 子图1：深度图
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        scatter1 = plt.scatter(
            x, y, c=depth[y, x], cmap='rainbow_r', alpha=0.5, s=2
        )
        plt.title('Depth Map (m)')
        plt.colorbar(scatter1, label='Depth')

        # 子图2：视线角度图（若提供）
        if theta_map is not None:
            plt.subplot(1, 2, 2)
            plt.imshow(img)
            scatter2 = plt.scatter(
                x, y, c=theta_map[y, x], cmap='hsv', alpha=0.5, s=2
            )
            plt.title('View Angle Map (°)')
            plt.colorbar(scatter2, label='View Angle (0°=front, 90°=side)')

        plt.tight_layout()

        # 5. 保存图像到本地（若指定目录）
        if save_dir is not None:
            # 确保保存目录存在
            os.makedirs(save_dir, exist_ok=True)
            # 生成带索引的文件名（避免覆盖）
            file_name = f'depth_theta_vis_{self.index}.png'
            save_path = os.path.join(save_dir, file_name)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')  # dpi控制分辨率
            print(f"图像已保存至：{save_path}")

        # 6. 显示图像（可选）
        if show:
            plt.show()
        else:
            plt.close()  # 不显示时关闭画布，释放内存

        self.index += 1  # 更新索引，确保文件名唯一
    def vis_depth_img(self, img, depth,theta_map=None):
        depth = depth.cpu().numpy()
        img = img.permute(1, 2, 0).cpu().numpy()
        img = img * self.std + self.mean
        img = np.array(img, dtype=np.uint8)
        invalid_y, invalid_x, invalid_c = np.where(img == 0)
        depth[invalid_y, invalid_x] = 0
        if theta_map is not None:
            theta_map = theta_map.cpu().numpy()
            theta_map[invalid_mask] = 0

        valid_mask = (depth != 0)  #if theta_map is None else (theta_map != 0)

        y, x = np.where(depth != 0)
        plt.figure(figsize=(14, 6))
        # 子图1：深度图
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        scatter1 = plt.scatter(
            x, y, c=depth[y, x], cmap='rainbow_r', alpha=0.5, s=2
        )
        plt.title('Depth Map (m)')
        plt.colorbar(scatter1, label='Depth')

        # 子图2：视线角度图（若有）
        if theta_map is not None:
            plt.subplot(1, 2, 2)
            plt.imshow(img)
            scatter2 = plt.scatter(
                x, y, c=theta_map[y, x], cmap='hsv', alpha=0.5, s=2
            )
            plt.title('View Angle Map (°)')
            plt.colorbar(scatter2, label='View Angle (0°=front, 90°=side)')

        plt.tight_layout()
        plt.show()
        self.index += 1

    def compute_lidar_normals_fix_direction(self, points_lidar):
        """
        计算LiDAR坐标系下的点云法向量（Open3D加速），并按「法向量与径向向量内积>0」统一朝向
        核心：确保凸包等结构的法向量整体"朝外"（与原点到点的连线方向一致）
        """
        # 1. 张量→numpy（Open3D CPU计算）
        points_np = points_lidar.cpu().numpy()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_np)

        # 2. 估计法向量（适配稀疏点云）
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=self.normal_radius,
                max_nn=self.normal_max_nn
            )
        )
        normals_np = np.asarray(pcd.normals)
        if len(normals_np) == 0:  # 极端情况：无点云时返回空
            return torch.tensor(normals_np, dtype=points_lidar.dtype, device=points_lidar.device)

        # 3. 核心修改：按「法向量与径向向量内积>0」统一朝向（确保凸包法向量朝外）
        # 3.1 计算"径向向量"：原点（LiDAR原点）到每个点的向量（即点的坐标本身）
        radial_vecs = points_np  # shape: (N, 3)，每个元素是[X, Y, Z]（原点→点的向量）
        
        # 3.2 处理径向向量接近原点的异常点（避免长度为0导致的计算问题）
        radial_magnitude = np.linalg.norm(radial_vecs, axis=1, keepdims=True)  # 每个径向向量的长度
        valid_radial_mask = radial_magnitude > 1e-8  # 过滤长度接近0的点（避免无意义方向）
        
        # 3.3 计算法向量与径向向量的点积（逐点判断方向）
        # 点积>0：法向量与径向向量同向（朝外）；点积<0：反向（朝内），需反转
        dot_product = np.sum(normals_np * radial_vecs, axis=1, keepdims=True)  # shape: (N, 1)
        
        # 3.4 反转点积<0的法向量（仅对有效径向向量的点处理）
        reverse_mask = (dot_product > 0) & valid_radial_mask  # 需要反转的点的掩码
        normals_np[reverse_mask.squeeze()] *= -1  # 反转法向量方向

        # （可选）对径向向量无效的点，保留原法向量方向（或按z轴辅助修正，避免孤立点方向混乱）
        invalid_radial_mask = ~valid_radial_mask.squeeze()
        if np.any(invalid_radial_mask):
            mean_z_invalid = np.mean(normals_np[invalid_radial_mask, 2])
            if mean_z_invalid < 0:
                normals_np[invalid_radial_mask] *= -1  # 对无效点按z轴辅助修正

        # 4. numpy→张量（回原设备，与输入点云保持一致）
        normals_lidar = torch.tensor(
            normals_np,
            dtype=points_lidar.dtype,
            device=points_lidar.device
        )
        return normals_lidar

    def points2depthmap(self, points, height, width):
        height, width = height // self.downsample, width // self.downsample
        depth_map = torch.zeros((height, width), dtype=torch.float32)

        coor = torch.round(points[:, :2] / self.downsample)
        depth = points[:, 2]
        kept1 = (coor[:, 0] >= 0) & (coor[:, 0] < width) & (
            coor[:, 1] >= 0) & (coor[:, 1] < height) & (
                depth < self.grid_config['depth'][1]) & (
                    depth >= self.grid_config['depth'][0])
        coor, depth = coor[kept1], depth[kept1]

        ranks = coor[:, 0] + coor[:, 1] * width
        sort = (ranks + depth / 100.).argsort()
        coor, depth, ranks = coor[sort], depth[sort], ranks[sort]

        kept2 = torch.ones(coor.shape[0], device=coor.device, dtype=torch.bool)
        kept2[1:] = (ranks[1:] != ranks[:-1])
        coor, depth = coor[kept2], depth[kept2]
        coor = coor.to(torch.long)
        depth_map[coor[:, 1], coor[:, 0]] = depth
        return depth_map

    def points2view_angle_map(self, points_cam, normals_cam, points_img, height, width):
        """新增：生成视线角度图（法向量与视线向量夹角）"""
        height, width = height // self.downsample, width // self.downsample
        theta_map = torch.zeros((height, width), dtype=torch.float32, device=points_cam.device)

        # 1. 计算视线向量（相机光心→3D点，单位化）
        view_vecs = torch.nn.functional.normalize(points_cam, dim=1)
        # 2. 法向量单位化（确保方向一致性）
        normals_cam = torch.nn.functional.normalize(normals_cam, dim=1)
        # 3. 计算夹角θ（0~180°）：点积→acos→角度转换
        cos_theta = torch.sum(view_vecs * normals_cam, dim=1)
        cos_theta = torch.clamp(cos_theta, -1.0, 1.0)  # 避免数值误差
        theta = torch.acos(cos_theta) * 180 / torch.pi

        # 4. 关联角度到图像像素（复用深度图的有效点筛选逻辑）
        coor = torch.round(points_img[:, :2] / self.downsample)
        depth = points_img[:, 2]
        kept1 = (coor[:, 0] >= 0) & (coor[:, 0] < width) & (
            coor[:, 1] >= 0) & (coor[:, 1] < height) & (
                depth < self.grid_config['depth'][1]) & (
                    depth >= self.grid_config['depth'][0])
        coor, theta = coor[kept1], theta[kept1]

        # 5. 同像素去重（取最近深度对应的角度）
        ranks = coor[:, 0] + coor[:, 1] * width
        sort_idx = (ranks + depth[kept1] / 100.).argsort()
        coor, theta, ranks = coor[sort_idx], theta[sort_idx], ranks[sort_idx]
        kept2 = torch.ones_like(kept1[kept1])
        kept2[1:] = (ranks[1:] != ranks[:-1])
        coor, theta = coor[kept2].long(), theta[kept2]
        theta_map[coor[:, 1], coor[:, 0]] = theta

        return theta_map

    def extract_lidar2cam_extrinsic(self, lidar2img, cam2img):
        # 1. 提取相机内参K并求逆
        K = cam2img[:3, :3]
        K_inv = torch.linalg.inv(K)
        
        # 2. 提取外参矩阵[R | T]（3x4）：K_inv * lidar2img[:3,:4]
        ext_mat = K_inv @ lidar2img[:3, :4]  # ext_mat.shape = (3,4)
        
        # 3. 提取旋转矩阵R并修正正交性
        R_lidar2cam = ext_mat[:3, :3]
        U, _, VT = torch.linalg.svd(R_lidar2cam)
        R_lidar2cam = U @ VT
        if torch.linalg.det(R_lidar2cam) < 0:
            VT[:, -1] *= -1
            R_lidar2cam = U @ VT
        
        # 4. 提取平移向量T（LiDAR原点在相机坐标系下的坐标）
        T_lidar2cam = ext_mat[:3, 3].unsqueeze(1)  # 转为(3,1)，适配矩阵乘法
        
        return R_lidar2cam, T_lidar2cam  # 同时返回R和T
    
    def compute_lidar_normals(self, points_lidar):
        """新增：计算LiDAR坐标系下的点云法向量（Open3D加速）"""
        # 1. 张量→numpy（Open3D CPU计算）
        points_np = points_lidar.cpu().numpy()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_np)

        # 2. 估计法向量（适配稀疏点云）
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=self.normal_radius,
                max_nn=self.normal_max_nn
            )
        )

        # 3. 统一法向量方向（LiDAR坐标系下z轴向上）
        normals_np = np.asarray(pcd.normals)
        mean_z = np.mean(normals_np[:, 2])
        if mean_z < 0:
            normals_np = -normals_np  # 整体反转，确保z分量正向

        # 4. numpy→张量（回原设备）
        normals_lidar = torch.tensor(
            normals_np,
            dtype=points_lidar.dtype,
            device=points_lidar.device
        )
        return normals_lidar

    def points2normal_components_map(self, points_cam, normals_cam, points_img, height, width,post_aug):
        """
        生成法向量在三个方向的分量图：
        - comp_W：图像W轴方向（水平）分量
        - comp_H：图像H轴方向（垂直）分量
        - comp_depth：深度方向（相机视线）分量
        """
        # 下采样后图像尺寸
        height_down, width_down = height // self.downsample, width // self.downsample
        # 初始化三个分量图（与点云同设备）
        comp_W_map = torch.zeros((height_down, width_down), dtype=torch.float32, device=points_cam.device)
        comp_H_map = torch.zeros_like(comp_W_map)
        comp_depth_map = torch.zeros_like(comp_W_map)

        # 步骤1：法向量单位化（确保分量范围在[-1,1]）
        normals_cam = torch.nn.functional.normalize(normals_cam, dim=1)  # (N, 3)，单位向量

        # 步骤2：计算三个方向的分量（直接用点积，因方向向量是单位向量）
        # 深度方向分量：normals_cam · [0,0,1] = normals_cam[:,2]
        comp_depth = normals_cam[:, 2]
        # W轴方向分量：normals_cam · [1,0,0] = normals_cam[:,0]
        comp_W = normals_cam[:, 0]
        # H轴方向分量：normals_cam · [0,-1,0] = -normals_cam[:,1]
        comp_H = -normals_cam[:, 1]

        # 步骤3：从post_aug提取旋转矩阵，修正W/H分量（核心新增）
        R_aug = self.extract_rotation_matrix(post_aug)  # (2,2) numpy矩阵
        R_aug = torch.tensor(R_aug, dtype=comp_W.dtype, device=comp_W.device)  # 转为Tensor

        # 将W/H分量组合为2D向量，应用旋转矩阵
        comp_wh = torch.stack([comp_W, comp_H], dim=1)  # (N, 2)
        comp_wh_rotated = torch.matmul(comp_wh, R_aug.T)  # (N, 2) = (N,2) @ (2,2).T
        comp_W_rot = comp_wh_rotated[:, 0]  # 旋转后的W分量
        comp_H_rot = comp_wh_rotated[:, 1]  # 旋转后的H分量

        # 步骤3：关联分量到图像像素（复用深度图的有效点筛选逻辑）
        # 下采样后的像素坐标
        coor = torch.round(points_img[:, :2] / self.downsample)  # (N, 2)，u/v坐标
        depth = points_img[:, 2]  # 深度值（用于筛选有效点）

        # 筛选有效点：像素在图像内 + 深度在合理范围
        kept1 = (coor[:, 0] >= 0) & (coor[:, 0] < width_down) & (
            coor[:, 1] >= 0) & (coor[:, 1] < height_down) & (
                depth < self.grid_config['depth'][1]) & (
                    depth >= self.grid_config['depth'][0])
        coor, comp_W_rot, comp_H_rot, comp_depth = coor[kept1], comp_W_rot[kept1], comp_H_rot[kept1], comp_depth[kept1]

        # 步骤4：同像素去重（取最近深度对应的分量值，与深度图逻辑一致）
        ranks = coor[:, 0] + coor[:, 1] * width_down  # 2D坐标转1D索引
        # 排序键：像素索引 + 深度/100（同像素时近的在前）
        sort_idx = (ranks + depth[kept1] / 100.).argsort()
        coor, comp_W_rot, comp_H_rot, comp_depth, ranks = coor[sort_idx], comp_W_rot[sort_idx], comp_H_rot[sort_idx], comp_depth[sort_idx], ranks[sort_idx]
        
        # 去重：每个像素只保留第一个点（最近深度）
        kept2 = torch.ones_like(kept1[kept1])
        kept2[1:] = (ranks[1:] != ranks[:-1])
        coor, comp_W_rot, comp_H_rot, comp_depth = coor[kept2].long(), comp_W_rot[kept2], comp_H_rot[kept2], comp_depth[kept2]

        # 步骤6：赋值旋转后的分量到分量图
        comp_W_map[coor[:, 1], coor[:, 0]] = comp_W_rot
        comp_H_map[coor[:, 1], coor[:, 0]] = comp_H_rot
        comp_depth_map[coor[:, 1], coor[:, 0]] = comp_depth

        return comp_W_map, comp_H_map, comp_depth_map

    def points2normal_components_map_fix_direction(self, points_cam, normals_cam, points_img, height, width, post_aug):
        """
        生成法向量在三个方向的分量图：
        - comp_W：图像W轴方向（水平）分量
        - comp_H：图像H轴方向（垂直）分量
        - comp_depth：深度方向（相机视线）分量
        新增：相机坐标系下按「法向量与径向向量内积<0」统一法向量朝向（适配凸包结构）
        """
        # 下采样后图像尺寸
        height_down, width_down = height // self.downsample, width // self.downsample
        # 初始化三个分量图（与点云同设备）
        comp_W_map = torch.zeros((height_down, width_down), dtype=torch.float32, device=points_cam.device)
        comp_H_map = torch.zeros_like(comp_W_map)
        comp_depth_map = torch.zeros_like(comp_W_map)

        # 步骤1：法向量单位化（确保方向计算准确，分量范围在[-1,1]）
        normals_cam = torch.nn.functional.normalize(normals_cam, dim=1)  # (N, 3)，单位向量

        # ---------------------- 新增：相机坐标系下统一法向量朝向 ----------------------
        # 1. 定义"径向向量"：相机原点（光心）→3D点的向量（即points_cam本身）
        radial_vecs_cam = points_cam  # shape: (N, 3)，方向从相机指向3D点
        # 2. 过滤径向向量接近原点的异常点（方向无意义，避免误判）
        radial_mag_cam = torch.norm(radial_vecs_cam, dim=1, keepdim=True)  # 计算径向向量长度
        valid_radial_mask = radial_mag_cam > 1e-6  # 有效点：长度>1e-6（排除近原点噪声）
        # 3. 计算法向量与径向向量的点积（判断朝向是否需要反转）
        # 目标：让有效点的法向量满足「与径向向量内积<0」（凸包整体朝内等一致朝向）
        dot_product = torch.sum(normals_cam * radial_vecs_cam, dim=1, keepdim=True)
        # 4. 反转法向量：点积≥0的点需反转，确保内积<0
        reverse_mask = (dot_product >= 0) & valid_radial_mask  # 需要反转的点掩码
        normals_cam = torch.where(reverse_mask, -normals_cam, normals_cam)  # 逐点精准反转
        # 5. 无效径向向量点的兜底处理（避免孤立点法向量混乱）
        invalid_radial_mask = ~valid_radial_mask.squeeze()
        if invalid_radial_mask.any():
            # 辅助逻辑：让无效点法向量与"相机视线方向（径向向量）"反向，贴合整体趋势
            view_vecs = torch.nn.functional.normalize(radial_vecs_cam[invalid_radial_mask], dim=1)
            dot_view = torch.sum(normals_cam[invalid_radial_mask] * view_vecs, dim=1, keepdim=True)
            normals_cam[invalid_radial_mask] = torch.where(
                dot_view >= 0, 
                -normals_cam[invalid_radial_mask], 
                normals_cam[invalid_radial_mask]
            )
        # --------------------------------------------------------------------------

        # 步骤2：基于统一朝向的法向量，计算三个方向的分量（点积）
        # 深度方向分量：法向量 · 相机Z轴（[0,0,1]）→ 反映朝向/背离相机
        comp_depth = normals_cam[:, 2]
        # W轴方向分量：法向量 · 相机X轴（[1,0,0]）→ 反映左右倾斜
        comp_W = normals_cam[:, 0]
        # H轴方向分量：法向量 · 相机-Y轴（[0,-1,0]）→ 反映上下倾斜（适配图像v轴向下）
        comp_H = -normals_cam[:, 1]

        # 步骤3：从post_aug提取旋转矩阵，修正W/H分量（适配图像增强，原有逻辑保留）
        R_aug = self.extract_rotation_matrix(post_aug)  # (2,2) numpy矩阵
        R_aug = torch.tensor(R_aug, dtype=comp_W.dtype, device=comp_W.device)  # 转为Tensor
        # 将W/H分量组合为2D向量，应用旋转矩阵（对齐增强后的图像W/H轴）
        comp_wh = torch.stack([comp_W, comp_H], dim=1)  # (N, 2)：每行是[W分量, H分量]
        comp_wh_rotated = torch.matmul(comp_wh, R_aug)  # 行向量 × 旋转矩阵（无需转置）
        comp_W_rot = comp_wh_rotated[:, 0]  # 旋转后W轴分量（对齐图像水平方向）
        comp_H_rot = comp_wh_rotated[:, 1]  # 旋转后H轴分量（对齐图像垂直方向）

        # 步骤4：关联分量到图像像素（复用深度图的有效点筛选逻辑）
        coor = torch.round(points_img[:, :2] / self.downsample)  # 下采样后的像素坐标（u/v）
        depth = points_img[:, 2]  # 3D点的深度值（筛选合理深度范围）
        # 筛选有效点：像素在图像内 + 深度在配置区间内
        kept1 = (coor[:, 0] >= 0) & (coor[:, 0] < width_down) & (
            coor[:, 1] >= 0) & (coor[:, 1] < height_down) & (
                depth < self.grid_config['depth'][1]) & (
                    depth >= self.grid_config['depth'][0])
        # 保留有效点的坐标和分量
        coor, comp_W_rot, comp_H_rot, comp_depth = coor[kept1], comp_W_rot[kept1], comp_H_rot[kept1], comp_depth[kept1]

        # 步骤5：同像素去重（取最近深度对应的分量值，避免同像素多值冲突）
        ranks = coor[:, 0] + coor[:, 1] * width_down  # 2D坐标转1D索引（便于排序）
        # 排序键：像素索引 + 深度/100（同像素时，近深度优先保留，避免远遮挡近）
        sort_idx = (ranks + depth[kept1] / 100.).argsort()
        coor, comp_W_rot, comp_H_rot, comp_depth, ranks = coor[sort_idx], comp_W_rot[sort_idx], comp_H_rot[sort_idx], comp_depth[sort_idx], ranks[sort_idx]
        # 去重：仅保留每个像素的第一个点（最近深度对应的分量）
        kept2 = torch.ones_like(kept1[kept1])
        kept2[1:] = (ranks[1:] != ranks[:-1])  # 相邻像素索引不同则保留
        coor, comp_W_rot, comp_H_rot, comp_depth = coor[kept2].long(), comp_W_rot[kept2], comp_H_rot[kept2], comp_depth[kept2]

        # 步骤6：赋值分量到分量图（coor[:,1]是行（图像H轴），coor[:,0]是列（图像W轴））
        comp_W_map[coor[:, 1], coor[:, 0]] = comp_W_rot
        comp_H_map[coor[:, 1], coor[:, 0]] = comp_H_rot
        comp_depth_map[coor[:, 1], coor[:, 0]] = comp_depth

        return comp_W_map, comp_H_map, comp_depth_map
    def extract_rotation_matrix(self, post_aug):
        """从post_aug中提取2D旋转矩阵（假设无剪切，主要是旋转+缩放）"""
        # 提取左上角2x2矩阵（包含旋转和缩放）
        affine_2d = post_aug[:2, :2].cpu().numpy()  # 转为numpy便于处理
        # 计算缩放因子（确保旋转矩阵正交性）
        scale = np.linalg.norm(affine_2d[0, :])  # 取第一行的模长作为缩放因子
        if scale < 1e-6:
            return np.eye(2)  # 避免除零
        # 归一化得到旋转矩阵（去除缩放影响）
        R_aug = affine_2d / scale
        # 确保是旋转矩阵（正交化修正，应对数值误差）
        U, _, VT = np.linalg.svd(R_aug)
        R_aug = U @ VT
        if np.linalg.det(R_aug) < 0:  # 确保行列式为1（右手系）
            VT[-1, :] *= -1
            R_aug = U @ VT
        return R_aug

    def vis_normal_components_save(self, img, comp_W, comp_H, comp_depth, save_dir=None, show=True):
        """
        可视化法向量三个分量图，并支持保存到本地
        
        参数:
            img: 原始图像张量 (C, H, W)
            comp_W: W轴分量图张量 (H, W)
            comp_H: H轴分量图张量 (H, W)
            comp_depth: 深度方向分量图张量 (H, W)
            save_dir: 保存图像的目录路径，为None时不保存
            show: 是否显示图像，默认为True
        """
        # 1. 数据格式转换（Tensor→numpy，反归一化）
        img = img.permute(1, 2, 0).cpu().numpy()  # (H, W, C)
        img = img * self.std + self.mean  # 反归一化到像素值范围
        img = np.array(img, dtype=np.uint8)  # 转为uint8格式

        # 2. 处理无效像素（图像黑边区域）
        invalid_mask = (img == 0).any(axis=2)  # 标记图像黑边（无效区域）
        # 三个分量图的无效区域置0
        comp_W = comp_W.cpu().numpy()
        comp_H = comp_H.cpu().numpy()
        comp_depth = comp_depth.cpu().numpy()
        comp_W[invalid_mask] = 0
        comp_H[invalid_mask] = 0
        comp_depth[invalid_mask] = 0

        # 3. 筛选有效像素（用于散点图绘制）
        valid_mask = (comp_depth != 0)  # 用深度分量筛选有效点
        y, x = np.where(valid_mask)

        # 4. 创建3子图可视化
        plt.figure(figsize=(18, 6))

        # 子图1：W轴分量（左右倾斜）
        plt.subplot(1, 3, 1)
        plt.imshow(img)
        scatter1 = plt.scatter(x, y, c=comp_W[y, x], cmap='coolwarm', alpha=0.6, s=2)
        plt.title('Normal Component - Image W Axis (Left-Right)')
        plt.colorbar(scatter1, label='Component Value (-1=Left, 1=Right)')

        # 子图2：H轴分量（上下倾斜）
        plt.subplot(1, 3, 2)
        plt.imshow(img)
        scatter2 = plt.scatter(x, y, c=comp_H[y, x], cmap='coolwarm', alpha=0.6, s=2)
        plt.title('Normal Component - Image H Axis (Up-Down)')
        plt.colorbar(scatter2, label='Component Value (-1=Up, 1=Down)')

        # 子图3：深度分量（朝向/背离相机）
        plt.subplot(1, 3, 3)
        plt.imshow(img)
        scatter3 = plt.scatter(x, y, c=comp_depth[y, x], cmap='coolwarm', alpha=0.6, s=2)
        plt.title('Normal Component - Depth Direction (Camera Facing)')
        plt.colorbar(scatter3, label='Component Value (-1=Away, 1=Towards)')

        plt.tight_layout()  # 调整子图布局，避免重叠

        # 5. 保存图像到本地（若指定目录）
        if save_dir is not None:
            # 确保保存目录存在（不存在则创建）
            os.makedirs(save_dir, exist_ok=True)
            # 生成带索引的唯一文件名（与depth可视化共用index，确保全局唯一）
            file_name = f'normal_components_vis_{self.index}.png'
            save_path = os.path.join(save_dir, file_name)
            # 高分辨率保存（dpi=300，去除多余空白）
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"法向量分量图已保存至：{save_path}")

        # 6. 显示图像（可选，默认显示）
        if show:
            plt.show()
        else:
            plt.close()  # 不显示时关闭画布，释放内存

        self.index += 1  # 更新索引，确保下次保存文件名唯一

    def points2view_angle_map_fix_direction(self, points_cam, normals_cam, points_img, height, width):
        """
        生成视线角度图（法向量与视线向量夹角）
        新增：相机坐标系下按「法向量与径向向量内积<0」统一法向量朝向（适配凸包等结构）
        """
        height, width = height // self.downsample, width // self.downsample
        theta_map = torch.zeros((height, width), dtype=torch.float32, device=points_cam.device)

        # 1. 基础向量单位化（确保方向计算准确）
        # 视线向量：相机光心→3D点（即points_cam，单位化）
        view_vecs = torch.nn.functional.normalize(points_cam, dim=1)
        # 法向量：先单位化，再统一朝向
        normals_cam = torch.nn.functional.normalize(normals_cam, dim=1)

        # ---------------------- 新增：相机坐标系下统一法向量朝向 ----------------------
        # 2. 定义相机坐标系下的"径向向量"：相机原点→3D点的向量（即points_cam本身）
        radial_vecs_cam = points_cam  # shape: (N, 3)，方向从相机指向3D点
        # 3. 过滤径向向量接近原点的异常点（方向无意义，避免误判）
        radial_mag_cam = torch.norm(radial_vecs_cam, dim=1, keepdim=True)  # 径向向量长度
        valid_radial_mask = radial_mag_cam > 1e-6  # 有效径向向量：长度>1e-6（避免除以0）
        # 4. 计算法向量与径向向量的点积（判断朝向）
        # 点积<0：法向量与径向向量反向；点积≥0：方向同向，需反转以满足"内积<0"的统一要求
        dot_product = torch.sum(normals_cam * radial_vecs_cam, dim=1, keepdim=True)
        # 5. 反转法向量：让有效点的法向量满足"与径向向量内积<0"（统一朝向）
        reverse_mask = (dot_product >= 0) & valid_radial_mask  # 需要反转的点掩码
        normals_cam = torch.where(reverse_mask, -normals_cam, normals_cam)  # 逐点反转
        # （可选）对无效径向向量的点，保留原法向量（或按视线向量辅助修正，避免孤立点混乱）
        invalid_radial_mask = ~valid_radial_mask.squeeze()
        if invalid_radial_mask.any():
            # 辅助逻辑：让无效点法向量与视线向量反向（贴合整体朝向）
            dot_view = torch.sum(normals_cam[invalid_radial_mask] * view_vecs[invalid_radial_mask], dim=1, keepdim=True)
            normals_cam[invalid_radial_mask] = torch.where(dot_view >= 0, -normals_cam[invalid_radial_mask], normals_cam[invalid_radial_mask])
        # --------------------------------------------------------------------------

        # 6. 计算视线夹角θ（0~180°）：基于统一朝向的法向量
        cos_theta = torch.sum(view_vecs * normals_cam, dim=1)
        cos_theta = torch.clamp(cos_theta, -1.0, 1.0)  # 避免数值误差（acos输入需在[-1,1]）
        theta = torch.acos(cos_theta) * 180 / torch.pi  # 弧度转角度

        # 7. 关联角度到图像像素（复用原有效点筛选逻辑）
        coor = torch.round(points_img[:, :2] / self.downsample)  # 下采样后的像素坐标
        depth = points_img[:, 2]  # 3D点的深度值（用于筛选合理深度范围）
        
        # 筛选有效点：像素在图像内 + 深度在配置范围内
        kept1 = (coor[:, 0] >= 0) & (coor[:, 0] < width) & (
            coor[:, 1] >= 0) & (coor[:, 1] < height) & (
                depth < self.grid_config['depth'][1]) & (
                    depth >= self.grid_config['depth'][0])
        coor, theta = coor[kept1], theta[kept1]

        # 8. 同像素去重：取最近深度对应的角度（避免同像素多值冲突）
        ranks = coor[:, 0] + coor[:, 1] * width  # 2D像素坐标转1D索引（便于排序）
        # 排序键：像素索引 + 深度/100（同像素时，近深度优先保留）
        sort_idx = (ranks + depth[kept1] / 100.).argsort()
        coor, theta, ranks = coor[sort_idx], theta[sort_idx], ranks[sort_idx]
        # 去重：仅保留每个像素的第一个点（最近深度对应的角度）
        kept2 = torch.ones_like(kept1[kept1])
        kept2[1:] = (ranks[1:] != ranks[:-1])
        coor, theta = coor[kept2].long(), theta[kept2]

        # 9. 赋值角度到角度图
        theta_map[coor[:, 1], coor[:, 0]] = theta

        return theta_map

    # ---------------------- 新增：1. Bin索引计算核心方法（处理无label与极小值） ----------------------
    def compute_wh_bin_index(self, comp_wh):
        """
        计算W/H方向分量的bin索引：
        - 无label（分量绝对值 < eps_label）→ 索引0
        - 有值（非0）→ 按角度映射到1~24（极小非0值对应1，角度越大索引越大）
        """
        device = comp_wh.device
        bin_index = torch.zeros_like(comp_wh, dtype=torch.long, device=device)  # 初始化无label索引0

        # 1. 筛选"有值的非0分量"（排除无label的0）
        has_label_mask = torch.abs(comp_wh) >= self.eps_label
        if not torch.any(has_label_mask):
            return bin_index  # 全是无label，直接返回0

        # 2. 对有值分量计算角度（度）
        comp_clamped = torch.clamp(comp_wh[has_label_mask], -1.0 + 1e-6, 1.0 - 1e-6)
        theta_rad = torch.acos(comp_clamped)
        theta_deg = theta_rad * 180.0 / torch.pi  # 0~180度

        # 3. 角度→bin索引（1~24）：极小角度（0~7.5度）对应1，依次递增
        bin_idx_0based = torch.floor(theta_deg / self.wh_bin_size).long()
        bin_idx_0based = torch.clamp(bin_idx_0based, 0, self.wh_num_bins - 1)  # 防止越界
        bin_idx_1based = bin_idx_0based + 1  # 转为1-based有效索引

        # 4. 赋值有效索引
        bin_index[has_label_mask] = bin_idx_1based
        return bin_index

    def compute_wh_bin_index_fixRing(self, comp_wh):
        """
        计算W/H方向分量的bin索引（0°与180°附近合并为索引1）：
        - 无label（分量绝对值 < eps_label）→ 索引0
        - 有值时：
        - 0°~3.75° 或 176.25°~180° → 索引1
        - 3.75°~11.25° → 索引2（3.75+7.5）
        - 11.25°~18.75° → 索引3
        - ...（依次递增，每个bin宽7.5°）
        - 168.75°~176.25° → 索引24
        """
        device = comp_wh.device
        bin_index = torch.zeros_like(comp_wh, dtype=torch.long, device=device)  # 无label索引0

        # 1. 筛选"有值的非0分量"（排除无label的0）
        has_label_mask = torch.abs(comp_wh) >= self.eps_label
        if not torch.any(has_label_mask):
            return bin_index

        # 2. 计算原始角度（0°~180°）
        comp_clamped = torch.clamp(comp_wh[has_label_mask], -1.0 + 1e-6, 1.0 - 1e-6)
        theta_rad = torch.acos(comp_clamped)
        theta_deg = theta_rad * 180.0 / torch.pi  # 原始角度：0°~180°

        # 3. 定义每个bin的角度区间（核心：0°附近与180°附近合并为索引1）
        bin_half = self.wh_bin_size / 2  # 3.75°（每个bin的半宽）
        total_deg = 180.0  # 总角度范围

        # 4. 角度→bin索引映射（1~24）
        # 索引1：0°~3.75° 或 176.25°~180°
        mask_bin1 = (theta_deg < bin_half) | (theta_deg >= (total_deg - bin_half))
        
        # 索引2~24：从3.75°开始，每个bin宽7.5°
        # 先将176.25°~180°的角度映射到-3.75°~0°，便于统一计算
        theta_deg_adj = torch.where(
            theta_deg >= (total_deg - bin_half),
            theta_deg - total_deg,  # 176.25°→-3.75°，180°→0°
            theta_deg
        )
        # 对非bin1的角度，计算相对于3.75°的偏移（映射到0~172.5°）
        theta_offset = theta_deg_adj - bin_half  # 3.75°→0°，11.25°→7.5°，...，176.25°→-7.5°（被bin1过滤）
        # 计算索引2~24（过滤掉bin1的角度）
        bin_idx_0based = torch.floor(theta_offset / self.wh_bin_size).long()
        # 确保索引在0~22（对应2~24），并过滤bin1的角度
        bin_idx_0based = torch.clamp(bin_idx_0based, 0, self.wh_num_bins - 2)  # 24-2=22
        bin_idx_2to24 = bin_idx_0based + 2  # 转为2~24

        # 5. 合并索引（1~24）
        bin_idx_1based = torch.where(
            mask_bin1,
            torch.tensor(1, device=device),  # bin1对应索引1
            bin_idx_2to24  # 其他对应2~24
        )

        # 6. 赋值有效索引
        bin_index[has_label_mask] = bin_idx_1based
        return bin_index

    def compute_wh_bin_index_fixRing_divofDotvalue(self, comp_wh):
        """
        计算W/H方向分量的bin索引（0°与180°附近合并为索引1）：
        - 无label（分量绝对值 < eps_label）→ 索引0
        - 有值时：
        - 0°~3.75° 或 176.25°~180° → 索引1
        - 3.75°~11.25° → 索引2（3.75+7.5）
        - 11.25°~18.75° → 索引3
        - ...（依次递增，每个bin宽7.5°）
        - 168.75°~176.25° → 索引24
        """
        device = comp_wh.device
        bin_index = torch.zeros_like(comp_wh, dtype=torch.long, device=device)  # 无label索引0

        # 1. 筛选"有值的非0分量"（排除无label的0）
        has_label_mask = torch.abs(comp_wh) >= self.eps_label
        if not torch.any(has_label_mask):
            return bin_index

        # 2. 计算原始角度（0°~180°）
        comp_clamped = torch.clamp(comp_wh[has_label_mask], -1.0 + 1e-6, 1.0 - 1e-6)
        theta_rad = torch.acos(comp_clamped)
        theta_deg = theta_rad * 180.0 / torch.pi  # 原始角度：0°~180°

        # # 3. 定义每个bin的角度区间（核心：0°附近与180°附近合并为索引1）
        # bin_half = self.wh_bin_size / 2  # 3.75°（每个bin的半宽）
        # total_deg = 180.0  # 总角度范围

        # # 4. 角度→bin索引映射（1~24）
        # # 索引1：0°~3.75° 或 176.25°~180°
        # mask_bin1 = (theta_deg < bin_half) | (theta_deg >= (total_deg - bin_half))
        
        # # 索引2~24：从3.75°开始，每个bin宽7.5°
        # # 先将176.25°~180°的角度映射到-3.75°~0°，便于统一计算
        # theta_deg_adj = torch.where(
        #     theta_deg >= (total_deg - bin_half),
        #     theta_deg - total_deg,  # 176.25°→-3.75°，180°→0°
        #     theta_deg
        # )
        # # 对非bin1的角度，计算相对于3.75°的偏移（映射到0~172.5°）
        # theta_offset = theta_deg_adj - bin_half  # 3.75°→0°，11.25°→7.5°，...，176.25°→-7.5°（被bin1过滤）
        # # 计算索引2~24（过滤掉bin1的角度）
        # bin_idx_0based = torch.floor(theta_offset / self.wh_bin_size).long()
        # # 确保索引在0~22（对应2~24），并过滤bin1的角度
        # bin_idx_0based = torch.clamp(bin_idx_0based, 0, self.wh_num_bins - 2)  # 24-2=22
        # bin_idx_2to24 = bin_idx_0based + 2  # 转为2~24

        # # 5. 合并索引（1~24）
        # bin_idx_1based = torch.where(
        #     mask_bin1,
        #     torch.tensor(1, device=device),  # bin1对应索引1
        #     bin_idx_2to24  # 其他对应2~24
        # )
       
        bin_idx_1based = ((((theta_deg)+(180.0/self.wh_num_bins/2))//(180.0/self.wh_num_bins))%self.wh_num_bins+1)

        # bin_idx_1based = ((((comp_clamped+1)+(2/self.wh_num_bins/2))//(2/self.wh_num_bins))%self.wh_num_bins+1)
        # 6. 赋值有效索引
        bin_index[has_label_mask] = bin_idx_1based.to(torch.long)
        return bin_index

    def compute_depth_bin_index(self, comp_depth):
        """
        计算深度方向分量的bin索引：
        - 无label（分量 < eps_label）→ 索引0
        - 有值（非0）→ 按角度映射到1~12（极小非0值对应1，角度越大索引越大）
        """
        device = comp_depth.device
        bin_index = torch.zeros_like(comp_depth, dtype=torch.long, device=device)  # 初始化无label索引0

        # 1. 筛选"有值的非0分量"（排除无label的0）
        has_label_mask = comp_depth >= self.eps_label  # 深度分量已统一为非负（之前有反转）
        if not torch.any(has_label_mask):
            return bin_index

        # 2. 对有值分量计算角度（度）
        comp_clamped = torch.clamp(comp_depth[has_label_mask], 0.0 + 1e-6, 1.0 - 1e-6)
        theta_rad = torch.acos(comp_clamped)
        theta_deg = theta_rad * 180.0 / torch.pi  # 0~90度

        # 3. 角度→bin索引（1~12）：极小角度（0~7.5度）对应1，依次递增
        bin_idx_0based = torch.floor(theta_deg / self.depth_bin_size).long()
        bin_idx_0based = torch.clamp(bin_idx_0based, 0, self.depth_num_bins - 1)  # 防止越界
        bin_idx_1based = bin_idx_0based + 1  # 转为1-based有效索引

        # 4. 赋值有效索引
        bin_index[has_label_mask] = bin_idx_1based
        return bin_index

    def compute_depth_bin_index_fixRing(self, comp_depth):
        """
        计算深度方向分量的bin索引（精细区间划分）：
        - 无label（分量 < eps_label）→ 索引0
        - 有值时：
        - 0°~3.75° → 索引1
        - 3.75°~11.25° → 索引2（3.75+7.5）
        - 11.25°~18.75° → 索引3
        - ...（依次递增，每个bin宽7.5°）
        - 86.25°~90° → 索引12
        """
        device = comp_depth.device
        bin_index = torch.zeros_like(comp_depth, dtype=torch.long, device=device)  # 无label索引0

        # 1. 筛选"有值的非0分量"（排除无label的0）
        has_label_mask = comp_depth >= self.eps_label  # 深度分量已统一为非负
        if not torch.any(has_label_mask):
            return bin_index

        # 2. 计算原始角度（0°~90°）
        comp_clamped = torch.clamp(comp_depth[has_label_mask], 0.0 + 1e-6, 1.0 - 1e-6)
        theta_rad = torch.acos(comp_clamped)
        theta_deg = theta_rad * 180.0 / torch.pi  # 原始角度：0°~90°

        # 3. 定义每个bin的角度区间（核心：0°附近3.75°为索引1，后续按7.5°递增）
        bin_half = self.depth_bin_size / 2  # 3.75°（每个bin的半宽）
        total_deg = 90.0  # 深度方向总角度范围

        # 4. 角度→bin索引映射（1~12）
        # 索引1：0°~3.75°
        mask_bin1 = theta_deg < bin_half
        
        # 索引2~12：从3.75°开始，每个bin宽7.5°
        # 计算角度相对于3.75°的偏移（映射到0~86.25°）
        theta_offset = theta_deg - bin_half  # 3.75°→0°，11.25°→7.5°，...，90°→86.25°
        # 计算索引2~12（过滤掉bin1的角度）
        bin_idx_0based = torch.floor(theta_offset / self.depth_bin_size).long()
        # 确保索引在0~10（对应2~12），并过滤bin1的角度
        bin_idx_0based = torch.clamp(bin_idx_0based, 0, self.depth_num_bins - 2)  # 12-2=10
        bin_idx_2to12 = bin_idx_0based + 2  # 转为2~12

        # 5. 合并索引（1~12）
        bin_idx_1based = torch.where(
            mask_bin1,
            torch.tensor(1, device=device),  # bin1对应索引1
            bin_idx_2to12  # 其他对应2~12
        )

        # 6. 赋值有效索引
        bin_index[has_label_mask] = bin_idx_1based
        return bin_index


    def compute_depth_bin_index_fixRing_divofDotvalue(self, comp_depth):
        """
        计算深度方向分量的bin索引（精细区间划分）：
        - 无label（分量 < eps_label）→ 索引0
        - 有值时：
        - 0°~3.75° → 索引1
        - 3.75°~11.25° → 索引2（3.75+7.5）
        - 11.25°~18.75° → 索引3
        - ...（依次递增，每个bin宽7.5°）
        - 86.25°~90° → 索引12
        """
        device = comp_depth.device
        bin_index = torch.zeros_like(comp_depth, dtype=torch.long, device=device)  # 无label索引0

        # 1. 筛选"有值的非0分量"（排除无label的0）
        has_label_mask = comp_depth >= self.eps_label  # 深度分量已统一为非负
        if not torch.any(has_label_mask):
            return bin_index

        # 2. 计算原始角度（0°~90°）
        comp_clamped = torch.clamp(comp_depth[has_label_mask], 0.0 + 1e-6, 1.0 - 1e-6)
        theta_rad = torch.acos(comp_clamped)
        theta_deg = theta_rad * 180.0 / torch.pi  # 原始角度：0°~90°

        # # 3. 定义每个bin的角度区间（核心：0°附近3.75°为索引1，后续按7.5°递增）
        # bin_half = self.depth_bin_size / 2  # 3.75°（每个bin的半宽）
        # total_deg = 90.0  # 深度方向总角度范围

        # # 4. 角度→bin索引映射（1~12）
        # # 索引1：0°~3.75°
        # mask_bin1 = theta_deg < bin_half
        
        # # 索引2~12：从3.75°开始，每个bin宽7.5°
        # # 计算角度相对于3.75°的偏移（映射到0~86.25°）
        # theta_offset = theta_deg - bin_half  # 3.75°→0°，11.25°→7.5°，...，90°→86.25°
        # # 计算索引2~12（过滤掉bin1的角度）
        # bin_idx_0based = torch.floor(theta_offset / self.depth_bin_size).long()
        # # 确保索引在0~10（对应2~12），并过滤bin1的角度
        # bin_idx_0based = torch.clamp(bin_idx_0based, 0, self.depth_num_bins - 2)  # 12-2=10
        # bin_idx_2to12 = bin_idx_0based + 2  # 转为2~12

        # # 5. 合并索引（1~12）
        # bin_idx_1based = torch.where(
        #     mask_bin1,
        #     torch.tensor(1, device=device),  # bin1对应索引1
        #     bin_idx_2to12  # 其他对应2~12
        # )
        
        bin_idx_1based = ((((theta_deg))//(90.0/self.depth_num_bins))%self.depth_num_bins+1)

        # bin_idx_1based = ((((comp_clamped))//(1/self.depth_num_bins))%self.depth_num_bins+1)

        # 6. 赋值有效索引
        bin_index[has_label_mask] = bin_idx_1based.to(torch.long)
        return bin_index

    def points2heightmap(self, points_lidar, points_img, height, width):
        """
        生成图像视角的像素级高度图：
        - 输入：points_lidar（LiDAR点云，含高度信息）、points_img（LiDAR点投影到图像的坐标）
        - 输出：height_map（下采样后的像素级高度图，无效像素为0）
        """
        # 1. 下采样后图像尺寸（与深度图保持一致）
        height_down, width_down = height // self.downsample, width // self.downsample
        # 初始化高度图（与LiDAR点云同设备，无效值为0）
        height_map = torch.zeros((height_down, width_down), dtype=torch.float32, device=points_lidar.device)

        # 2. 提取LiDAR点的高度信息（默认Z轴，可通过self.lidar_height_axis调整）
        point_heights = points_lidar[:, 2]  # (N,)，每个LiDAR点的高度值

        # 3. 筛选有效点：像素在图像内 + 深度在合理范围 + 高度在有效范围
        # 3.1 投影坐标下采样（与深度图逻辑一致）
        coor = torch.round(points_img[:, :2] / self.downsample)  # (N,2)，下采样后的像素坐标（u,v）
        depth = points_img[:, 2]  # (N,)，LiDAR点的深度值（复用深度筛选）
        # 3.2 有效点掩码（多条件过滤）
        kept1 = (coor[:, 0] >= 0) & (coor[:, 0] < width_down) & (  # 像素u在范围内
                coor[:, 1] >= 0) & (coor[:, 1] < height_down) & ( # 像素v在范围内
                depth >= self.grid_config['depth'][0]) & (depth <= self.grid_config['depth'][1]) 
                # &  # 深度有效
                # (point_heights >= self.height_min) & (point_heights <= self.height_max)  # 高度有效

        # 4. 保留有效点的坐标、高度、深度（用于同像素去重）
        coor_valid = coor[kept1]
        height_valid = point_heights[kept1]
        depth_valid = depth[kept1]
        if len(coor_valid) == 0:
            return height_map  # 无有效点，返回全0高度图

        # 5. 同像素去重：取「最近深度」对应的高度（避免远景点遮挡近景点）
        # 5.1 2D像素坐标转1D索引（便于排序和去重）
        ranks = coor_valid[:, 0] + coor_valid[:, 1] * width_down  # (M,)，M为有效点数量
        # 5.2 按「像素索引 + 深度/100」排序（同像素内，深度小的在前，即近点优先）
        sort_idx = (ranks + depth_valid / 100.).argsort()  # 深度/100确保同像素内深度优先
        coor_sorted = coor_valid[sort_idx]
        height_sorted = height_valid[sort_idx]
        ranks_sorted = ranks[sort_idx]

        # 5.3 去重：每个像素仅保留第一个点（最近深度对应的高度）
        kept2 = torch.ones_like(ranks_sorted, dtype=torch.bool)
        kept2[1:] = (ranks_sorted[1:] != ranks_sorted[:-1])  # 相邻像素索引不同则保留
        coor_final = coor_sorted[kept2].long()  # 最终有效像素坐标（转为long类型用于索引）
        height_final = height_sorted[kept2]     # 最终有效高度值

        # 6. 赋值高度到高度图（注意：图像坐标v对应高度图的行，u对应列）
        height_map[coor_final[:, 1], coor_final[:, 0]] = height_final

        return height_map

    def vis_height_img(self, img, height_map, depth_map=None):
        """可视化高度图（可选叠加深度图对比）"""
        # 数据格式转换（Tensor→numpy，反归一化）
        height_map_np = height_map.cpu().numpy()
        img_np = img.permute(1, 2, 0).cpu().numpy()
        img_np = img_np * self.std + self.mean
        img_np = np.array(img_np, dtype=np.uint8)

        # 处理无效像素（图像黑边）
        invalid_mask = (img_np == 0).any(axis=2)
        height_map_np[invalid_mask] = 0
        if depth_map is not None:
            depth_map_np = depth_map.cpu().numpy()
            depth_map_np[invalid_mask] = 0

        # 筛选有效高度像素
        valid_height_mask = (height_map_np != 0)
        y, x = np.where(valid_height_mask)

        # 创建可视化子图
        plt.figure(figsize=(14, 6))
        # 子图1：高度图
        plt.subplot(1, 2, 1)
        plt.imshow(img_np)
        scatter1 = plt.scatter(x, y, c=height_map_np[y, x], cmap='viridis', alpha=0.6, s=2)
        plt.title(f'Height Map (m)')
        plt.colorbar(scatter1, label=f'Height (Range: {self.height_min}~{self.height_max}m)')

        # 子图2：深度图（可选，用于对比）
        if depth_map is not None:
            valid_depth_mask = (depth_map_np != 0)
            y_d, x_d = np.where(valid_depth_mask)
            plt.subplot(1, 2, 2)
            plt.imshow(img_np)
            scatter2 = plt.scatter(x_d, y_d, c=depth_map_np[y_d, x_d], cmap='rainbow_r', alpha=0.6, s=2)
            plt.title('Depth Map (m)')
            plt.colorbar(scatter2, label='Depth')

        plt.tight_layout()
        plt.show()
        self.index += 1

    def vis_height_img_save(self, img, height_map, save_dir=None, depth_map=None, show=True):
        """可视化高度图并保存到本地（支持叠加深度图对比）"""
        height_map_np = height_map.cpu().numpy()
        img_np = img.permute(1, 2, 0).cpu().numpy()
        img_np = img_np * self.std + self.mean
        img_np = np.array(img_np, dtype=np.uint8)

        # 处理无效像素
        invalid_mask = (img_np == 0).any(axis=2)
        height_map_np[invalid_mask] = 0
        depth_map_np = None
        if depth_map is not None:
            depth_map_np = depth_map.cpu().numpy()
            depth_map_np[invalid_mask] = 0

        # 筛选有效像素
        valid_height_mask = (height_map_np != 0)
        y, x = np.where(valid_height_mask)

        # 创建画布
        fig_size = (14, 6) if depth_map is not None else (7, 6)
        plt.figure(figsize=fig_size)
        # 高度图子图
        plt.subplot(1, 2, 1) if depth_map is not None else plt.subplot(1, 1, 1)
        plt.imshow(img_np)
        scatter1 = plt.scatter(x, y, c=height_map_np[y, x], cmap='viridis', alpha=0.6, s=2)
        plt.title(f'Height Map (m)')
        plt.colorbar(scatter1, label=f'Height (Range: {self.height_min}~{self.height_max}m)')

        # 深度图子图（可选）
        if depth_map is not None:
            valid_depth_mask = (depth_map_np != 0)
            y_d, x_d = np.where(valid_depth_mask)
            plt.subplot(1, 2, 2)
            plt.imshow(img_np)
            scatter2 = plt.scatter(x_d, y_d, c=depth_map_np[y_d, x_d], cmap='rainbow_r', alpha=0.6, s=2)
            plt.title('Depth Map (m)')
            plt.colorbar(scatter2, label='Depth')

        plt.tight_layout()
        # 保存图像
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            file_name = f'height_vis_{self.index}.png' if depth_map is None else f'height_depth_vis_{self.index}.png'
            save_path = os.path.join(save_dir, file_name)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"高度图已保存至：{save_path}")
        # 显示或关闭
        if show:
            plt.show()
        else:
            plt.close()
        self.index += 1
    def __call__(self, results):
        # 原有：读取输入数据
        points_lidar = results['points'].tensor  # LiDAR点云 (N, 3)
        imgs, sensor2egos, ego2globals, cam2imgs, post_augs, bda = results['img_inputs']
        lidar2imgs = results['lidar2img']  # LiDAR→各相机的投影矩阵 (6, 3, 4)
        nt, c, h, w = imgs.shape
        t_frame = nt // self.num_cam

        # 新增：1. 计算LiDAR坐标系下的法向量
        normals_lidar = self.compute_lidar_normals(points_lidar)

        # 原有：初始化结果存储
        depth_maps = []
        view_angle_maps = []  # 新增：视线角度图存储
        comp_W_maps = []   # W轴分量图列表
        comp_H_maps = []   # H轴分量图列表
        comp_depth_maps = []  # 深度分量图列表

        comp_W_bins = []
        comp_H_bins = []
        comp_depth_bins = []
        height_maps = []
        # 遍历每个相机，生成深度图+视线角度图
        for cid in range(len(results['cam_names'])):
            lidar2img = lidar2imgs[cid]
            cam2img = cam2imgs[cid]  # 相机内参（用于提取外参）
            lidar2cam = results['lidar2cam'][cid]  # 外参（用于转换坐标）
            post_aug = post_augs[cid]  # 当前相机的增强矩阵（3x3）
            # 原有：1. LiDAR点投影到图像平面（u, v, depth）
            points_homo = torch.cat([points_lidar.T, torch.ones((1, points_lidar.shape[0]), device=points_lidar.device)], dim=0)
            points_img_homo = lidar2img @ points_homo  # (3, N)
            points_img = points_img_homo.permute(1, 0)  # (N, 3)
            # 透视除法：(u/z, v/z, z) → (u, v, z)
            points_img = torch.cat([
                points_img[:, :2] / points_img[:, 2].unsqueeze(1),
                points_img[:, 2].unsqueeze(1)
            ], dim=1)

            # 新增：2. 转换法向量和点到相机坐标系
            # 提取LiDAR→相机的旋转矩阵R
            # R_lidar2cam, T_lidar2cam = self.extract_lidar2cam_extrinsic(lidar2img, cam2img)
            R_lidar2cam, T_lidar2cam = lidar2cam[:3, :3],lidar2cam[:3, 3].unsqueeze(1)  # 外参矩阵转换为相机坐标系
            # 转换3D点（相机坐标系：P_cam = R * P_lidar）
            points_cam = (R_lidar2cam @ points_lidar.T + T_lidar2cam).T   # (N, 3)
            # 转换法向量（方向向量仅旋转，无平移）
            normals_cam = (R_lidar2cam @ normals_lidar.T).T  # (N, 3)

            # 原有：3. 生成深度图
            depth_map = self.points2depthmap(points_img, h, w)
            depth_maps.append(depth_map)

            # 新增：4. 生成视线角度图
            theta_map = self.points2view_angle_map(points_cam, normals_cam, points_img, h, w)
            view_angle_maps.append(theta_map)

            comp_W_map, comp_H_map, comp_depth_map = self.points2normal_components_map(
                points_cam, normals_cam, points_img, h, w,post_aug
            )
            my_revers_mask = comp_depth_map<0
            comp_W_map[my_revers_mask] *=-1
            comp_H_map[my_revers_mask] *=-1
            comp_depth_map[my_revers_mask] *=-1
            comp_W_maps.append(comp_W_map)
            comp_H_maps.append(comp_H_map)
            comp_depth_maps.append(comp_depth_map)
            wh_bin_W = self.compute_wh_bin_index_fixRing_divofDotvalue(comp_W_map)
            comp_W_bins.append(wh_bin_W)
            # H方向bin索引（0=无label，1~24=有效方向）
            wh_bin_H = self.compute_wh_bin_index_fixRing_divofDotvalue(comp_H_map)
            comp_H_bins.append(wh_bin_H)
            # 深度方向bin索引（0=无label，1~12=有效方向）
            depth_bin = self.compute_depth_bin_index_fixRing_divofDotvalue(comp_depth_map)
            comp_depth_bins.append(depth_bin)

            height_map = self.points2heightmap(points_lidar, points_img, h, w)
            height_maps.append(height_map)



            #  用于可视化 复原离散值

            #映射表
            # wh_bin_WH2cosvalue = None
            # (torch.arange(-1,1,0.2)+0.1)[bin_idx_1based.to(torch.long)-1]


            #TODO 这里在可视化的时候打开
            # comp_W_map[wh_bin_W!=0] = (torch.arange(-1,1,0.2)+0.1)[wh_bin_W[wh_bin_W!=0]-1]
            # comp_H_map[wh_bin_H!=0] = (torch.arange(-1,1,0.2)+0.1)[wh_bin_H[wh_bin_H!=0]-1]
            # comp_depth_map[depth_bin!=0] = (torch.arange(0,1,0.2)+0.1)[depth_bin[depth_bin!=0]-1]
            #TODO 上面这里在可视化的时候打开
            
            
            
            # comp_H_map[wh_bin_H!=0] = torch.cos((wh_bin_H-1)*self.wh_bin_size)[wh_bin_H!=0]
            # comp_depth_map[depth_bin!=0] = torch.cos((depth_bin-1)*self.depth_bin_size)[depth_bin!=0]
            
            1==1
            #   用于可视化 复原离散值



            # 可选：可视化验证（深度图+视线角度图）
            # if cid == 0:  # 仅可视化第一个相机，避免冗余
            #     self.vis_depth_img(imgs[cid * t_frame], depth_map, theta_map)
            # self.vis_depth_img_save(imgs[cid * t_frame], depth_map, theta_map, save_dir='./vis_results')
            # 可选：可视化验证（每个相机的三个分量图）
            # if cid == 0:
            #     self.vis_normal_components(imgs[cid * t_frame], comp_W_map, comp_H_map, comp_depth_map)
            # self.vis_normal_components_save(
            #     imgs[cid * t_frame], 
            #     comp_W_map, 
            #     comp_H_map, 
            #     comp_depth_map, 
            #     save_dir='./normal_vis', 
            #     show=False
            # )

            # 可选：高度图可视化（与深度图对比）
            # self.vis_height_img_save(imgs[cid * t_frame], height_map, save_dir='./height_vis', depth_map=depth_map, show=False)

        # 原有：保存深度图结果
        results['gt_depth'] = torch.stack(depth_maps)
        # 新增：保存视线角度图结果
        results['view_angle_maps'] = torch.stack(view_angle_maps)
        results['normal_comp_W_maps'] = torch.stack(comp_W_maps)    # (6, H_down, W_down)
        results['normal_comp_H_maps'] = torch.stack(comp_H_maps)    # 6个相机的W轴分量图
        results['normal_comp_depth_maps'] = torch.stack(comp_depth_maps)  # 6个相机的深度分量图
        results['normal_comp_W_bins'] = torch.stack(comp_W_bins)  # (6, H_down, W_down)
        results['normal_comp_H_bins'] = torch.stack(comp_H_bins)
        results['normal_comp_depth_bins'] = torch.stack(comp_depth_bins)
        results['normals_lidar'] = normals_lidar  # (N, 3)
        results['gt_height_maps'] = torch.stack(height_maps)
        return results


@PIPELINES.register_module()
class PointToMultiViewDepth_fix_nomal_triview_fixdirectionByDepthBound_fixattn(object):

    def __init__(self, grid_config, downsample=1,normal_radius=1.2, normal_max_nn=30):
        self.downsample = downsample
        self.grid_config = grid_config
        self.index = 0
        self.num_cam = 6
        self.std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
        self.mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        self.normal_radius = normal_radius  # 法向量邻域搜索半径
        self.normal_max_nn = normal_max_nn
        # 新增：bin划分参数（7.5度/ bin，与需求一致）
        self.wh_bin_size = 22.5  # W/H方向每个bin的角度范围（度）
        self.wh_num_bins = 8   # W/H方向总bin数（180/7.5=24）
        self.depth_bin_size = 18  # 深度方向每个bin的角度范围（度）
        self.depth_num_bins = 5   # 深度方向总bin数（90/7.5=12）
        self.eps_label = 1e-8  # 判断"无label的0值"的阈值（小于此值视为无label）
        # 新增：高度相关配置（可根据LiDAR坐标系调整，默认Z轴为高度）
        self.lidar_height_axis = 2  # LiDAR点云的高度轴（0=X,1=Y,2=Z，默认Z轴）
        self.height_min = grid_config.get('height', [-10.0, 10.0])[0]  # 有效高度最小值（过滤地面以下/异常点）
        self.height_max = grid_config.get('height', [-10.0, 10.0])[1]  # 有效高度最大值（过滤高处/异常点）

    def vis_normal_components(self, img, comp_W, comp_H, comp_depth):
        """可视化法向量在W轴、H轴、深度方向的分量图"""
        # 数据格式转换（Tensor→numpy，无效像素置0）
        img = img.permute(1, 2, 0).cpu().numpy()
        img = img * self.std + self.mean
        img = np.array(img, dtype=np.uint8)
        invalid_mask = (img == 0).any(axis=2)  # 图像黑边标记

        # 三个分量转numpy并处理无效像素
        comp_W = comp_W.cpu().numpy()
        comp_H = comp_H.cpu().numpy()
        comp_depth = comp_depth.cpu().numpy()
        comp_W[invalid_mask] = 0
        comp_H[invalid_mask] = 0
        comp_depth[invalid_mask] = 0

        # 筛选有效像素（非零分量）
        valid_mask = (comp_depth != 0)  # 用深度分量筛选有效点
        y, x = np.where(valid_mask)

        # 创建3子图可视化
        plt.figure(figsize=(18, 6))
        # 子图1：W轴分量（左右倾斜）
        plt.subplot(1, 3, 1)
        plt.imshow(img)
        scatter1 = plt.scatter(x, y, c=comp_W[y, x], cmap='coolwarm', alpha=0.6, s=2)
        plt.title('Normal Component - Image W Axis (Left-Right)')
        plt.colorbar(scatter1, label='Component Value (-1=Left, 1=Right)')

        # 子图2：H轴分量（上下倾斜）
        plt.subplot(1, 3, 2)
        plt.imshow(img)
        scatter2 = plt.scatter(x, y, c=comp_H[y, x], cmap='coolwarm', alpha=0.6, s=2)
        plt.title('Normal Component - Image H Axis (Up-Down)')
        plt.colorbar(scatter2, label='Component Value (-1=Up, 1=Down)')

        # 子图3：深度分量（朝向/背离相机）
        plt.subplot(1, 3, 3)
        plt.imshow(img)
        scatter3 = plt.scatter(x, y, c=comp_depth[y, x], cmap='coolwarm', alpha=0.6, s=2)
        plt.title('Normal Component - Depth Direction (Camera Facing)')
        plt.colorbar(scatter3, label='Component Value (-1=Away, 1=Towards)')

        plt.tight_layout()
        plt.show()
        self.index += 1

    def vis_depth_img_save(self, img, depth, theta_map=None, save_dir=None, show=True):
        """
        可视化深度图和视线角度图，并支持保存到本地
        
        参数:
            img: 原始图像张量 (C, H, W)
            depth: 深度图张量 (H, W)
            theta_map: 视线角度图张量 (H, W)，可选
            save_dir: 保存图像的目录路径，为None时不保存
            show: 是否显示图像，默认为True
        """
        # 1. 数据格式转换（Tensor→numpy）
        depth = depth.cpu().numpy()
        img = img.permute(1, 2, 0).cpu().numpy()  # (H, W, C)
        img = img * self.std + self.mean  # 反归一化
        img = np.array(img, dtype=np.uint8)  # 转为uint8图像格式

        # 2. 处理无效像素（图像黑边）
        invalid_mask = (img == 0).any(axis=2)  # 修正：定义无效区域掩码（黑边）
        depth[invalid_mask] = 0  # 无效区域深度置0
        if theta_map is not None:
            theta_map = theta_map.cpu().numpy()
            theta_map[invalid_mask] = 0  # 无效区域角度置0

        # 3. 筛选有效像素（用于散点图）
        valid_mask = (depth != 0) if theta_map is None else (theta_map != 0)
        y, x = np.where(valid_mask)

        # 4. 创建图像并绘图
        plt.figure(figsize=(14, 6))

        # 子图1：深度图
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        scatter1 = plt.scatter(
            x, y, c=depth[y, x], cmap='rainbow_r', alpha=0.5, s=2
        )
        plt.title('Depth Map (m)')
        plt.colorbar(scatter1, label='Depth')

        # 子图2：视线角度图（若提供）
        if theta_map is not None:
            plt.subplot(1, 2, 2)
            plt.imshow(img)
            scatter2 = plt.scatter(
                x, y, c=theta_map[y, x], cmap='hsv', alpha=0.5, s=2
            )
            plt.title('View Angle Map (°)')
            plt.colorbar(scatter2, label='View Angle (0°=front, 90°=side)')

        plt.tight_layout()

        # 5. 保存图像到本地（若指定目录）
        if save_dir is not None:
            # 确保保存目录存在
            os.makedirs(save_dir, exist_ok=True)
            # 生成带索引的文件名（避免覆盖）
            file_name = f'depth_theta_vis_{self.index}.png'
            save_path = os.path.join(save_dir, file_name)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')  # dpi控制分辨率
            print(f"图像已保存至：{save_path}")

        # 6. 显示图像（可选）
        if show:
            plt.show()
        else:
            plt.close()  # 不显示时关闭画布，释放内存

        self.index += 1  # 更新索引，确保文件名唯一
    def vis_depth_img(self, img, depth,theta_map=None):
        depth = depth.cpu().numpy()
        img = img.permute(1, 2, 0).cpu().numpy()
        img = img * self.std + self.mean
        img = np.array(img, dtype=np.uint8)
        invalid_y, invalid_x, invalid_c = np.where(img == 0)
        depth[invalid_y, invalid_x] = 0
        if theta_map is not None:
            theta_map = theta_map.cpu().numpy()
            theta_map[invalid_mask] = 0

        valid_mask = (depth != 0)  #if theta_map is None else (theta_map != 0)

        y, x = np.where(depth != 0)
        plt.figure(figsize=(14, 6))
        # 子图1：深度图
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        scatter1 = plt.scatter(
            x, y, c=depth[y, x], cmap='rainbow_r', alpha=0.5, s=2
        )
        plt.title('Depth Map (m)')
        plt.colorbar(scatter1, label='Depth')

        # 子图2：视线角度图（若有）
        if theta_map is not None:
            plt.subplot(1, 2, 2)
            plt.imshow(img)
            scatter2 = plt.scatter(
                x, y, c=theta_map[y, x], cmap='hsv', alpha=0.5, s=2
            )
            plt.title('View Angle Map (°)')
            plt.colorbar(scatter2, label='View Angle (0°=front, 90°=side)')

        plt.tight_layout()
        plt.show()
        self.index += 1

    def compute_lidar_normals_fix_direction(self, points_lidar):
        """
        计算LiDAR坐标系下的点云法向量（Open3D加速），并按「法向量与径向向量内积>0」统一朝向
        核心：确保凸包等结构的法向量整体"朝外"（与原点到点的连线方向一致）
        """
        # 1. 张量→numpy（Open3D CPU计算）
        points_np = points_lidar.cpu().numpy()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_np)

        # 2. 估计法向量（适配稀疏点云）
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=self.normal_radius,
                max_nn=self.normal_max_nn
            )
        )
        normals_np = np.asarray(pcd.normals)
        if len(normals_np) == 0:  # 极端情况：无点云时返回空
            return torch.tensor(normals_np, dtype=points_lidar.dtype, device=points_lidar.device)

        # 3. 核心修改：按「法向量与径向向量内积>0」统一朝向（确保凸包法向量朝外）
        # 3.1 计算"径向向量"：原点（LiDAR原点）到每个点的向量（即点的坐标本身）
        radial_vecs = points_np  # shape: (N, 3)，每个元素是[X, Y, Z]（原点→点的向量）
        
        # 3.2 处理径向向量接近原点的异常点（避免长度为0导致的计算问题）
        radial_magnitude = np.linalg.norm(radial_vecs, axis=1, keepdims=True)  # 每个径向向量的长度
        valid_radial_mask = radial_magnitude > 1e-8  # 过滤长度接近0的点（避免无意义方向）
        
        # 3.3 计算法向量与径向向量的点积（逐点判断方向）
        # 点积>0：法向量与径向向量同向（朝外）；点积<0：反向（朝内），需反转
        dot_product = np.sum(normals_np * radial_vecs, axis=1, keepdims=True)  # shape: (N, 1)
        
        # 3.4 反转点积<0的法向量（仅对有效径向向量的点处理）
        reverse_mask = (dot_product > 0) & valid_radial_mask  # 需要反转的点的掩码
        normals_np[reverse_mask.squeeze()] *= -1  # 反转法向量方向

        # （可选）对径向向量无效的点，保留原法向量方向（或按z轴辅助修正，避免孤立点方向混乱）
        invalid_radial_mask = ~valid_radial_mask.squeeze()
        if np.any(invalid_radial_mask):
            mean_z_invalid = np.mean(normals_np[invalid_radial_mask, 2])
            if mean_z_invalid < 0:
                normals_np[invalid_radial_mask] *= -1  # 对无效点按z轴辅助修正

        # 4. numpy→张量（回原设备，与输入点云保持一致）
        normals_lidar = torch.tensor(
            normals_np,
            dtype=points_lidar.dtype,
            device=points_lidar.device
        )
        return normals_lidar

    def points2depthmap(self, points, height, width):
        height, width = height // self.downsample, width // self.downsample
        depth_map = torch.zeros((height, width), dtype=torch.float32)

        coor = torch.round(points[:, :2] / self.downsample)
        depth = points[:, 2]
        kept1 = (coor[:, 0] >= 0) & (coor[:, 0] < width) & (
            coor[:, 1] >= 0) & (coor[:, 1] < height) & (
                depth < self.grid_config['depth'][1]) & (
                    depth >= self.grid_config['depth'][0])
        coor, depth = coor[kept1], depth[kept1]

        ranks = coor[:, 0] + coor[:, 1] * width
        sort = (ranks + depth / 100.).argsort()
        coor, depth, ranks = coor[sort], depth[sort], ranks[sort]

        kept2 = torch.ones(coor.shape[0], device=coor.device, dtype=torch.bool)
        kept2[1:] = (ranks[1:] != ranks[:-1])
        coor, depth = coor[kept2], depth[kept2]
        coor = coor.to(torch.long)
        depth_map[coor[:, 1], coor[:, 0]] = depth
        return depth_map

    def points2view_angle_map(self, points_cam, normals_cam, points_img, height, width):
        """新增：生成视线角度图（法向量与视线向量夹角）"""
        height, width = height // self.downsample, width // self.downsample
        theta_map = torch.zeros((height, width), dtype=torch.float32, device=points_cam.device)

        # 1. 计算视线向量（相机光心→3D点，单位化）
        view_vecs = torch.nn.functional.normalize(points_cam, dim=1)
        # 2. 法向量单位化（确保方向一致性）
        normals_cam = torch.nn.functional.normalize(normals_cam, dim=1)
        # 3. 计算夹角θ（0~180°）：点积→acos→角度转换
        cos_theta = torch.sum(view_vecs * normals_cam, dim=1)
        cos_theta = torch.clamp(cos_theta, -1.0, 1.0)  # 避免数值误差
        theta = torch.acos(cos_theta) * 180 / torch.pi

        # 4. 关联角度到图像像素（复用深度图的有效点筛选逻辑）
        coor = torch.round(points_img[:, :2] / self.downsample)
        depth = points_img[:, 2]
        kept1 = (coor[:, 0] >= 0) & (coor[:, 0] < width) & (
            coor[:, 1] >= 0) & (coor[:, 1] < height) & (
                depth < self.grid_config['depth'][1]) & (
                    depth >= self.grid_config['depth'][0])
        coor, theta = coor[kept1], theta[kept1]

        # 5. 同像素去重（取最近深度对应的角度）
        ranks = coor[:, 0] + coor[:, 1] * width
        sort_idx = (ranks + depth[kept1] / 100.).argsort()
        coor, theta, ranks = coor[sort_idx], theta[sort_idx], ranks[sort_idx]
        kept2 = torch.ones_like(kept1[kept1])
        kept2[1:] = (ranks[1:] != ranks[:-1])
        coor, theta = coor[kept2].long(), theta[kept2]
        theta_map[coor[:, 1], coor[:, 0]] = theta

        return theta_map

    def extract_lidar2cam_extrinsic(self, lidar2img, cam2img):
        # 1. 提取相机内参K并求逆
        K = cam2img[:3, :3]
        K_inv = torch.linalg.inv(K)
        
        # 2. 提取外参矩阵[R | T]（3x4）：K_inv * lidar2img[:3,:4]
        ext_mat = K_inv @ lidar2img[:3, :4]  # ext_mat.shape = (3,4)
        
        # 3. 提取旋转矩阵R并修正正交性
        R_lidar2cam = ext_mat[:3, :3]
        U, _, VT = torch.linalg.svd(R_lidar2cam)
        R_lidar2cam = U @ VT
        if torch.linalg.det(R_lidar2cam) < 0:
            VT[:, -1] *= -1
            R_lidar2cam = U @ VT
        
        # 4. 提取平移向量T（LiDAR原点在相机坐标系下的坐标）
        T_lidar2cam = ext_mat[:3, 3].unsqueeze(1)  # 转为(3,1)，适配矩阵乘法
        
        return R_lidar2cam, T_lidar2cam  # 同时返回R和T
    
    def compute_lidar_normals(self, points_lidar):
        """新增：计算LiDAR坐标系下的点云法向量（Open3D加速）"""
        # 1. 张量→numpy（Open3D CPU计算）
        points_np = points_lidar.cpu().numpy()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_np)

        # 2. 估计法向量（适配稀疏点云）
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=self.normal_radius,
                max_nn=self.normal_max_nn
            )
        )

        # 3. 统一法向量方向（LiDAR坐标系下z轴向上）
        normals_np = np.asarray(pcd.normals)
        mean_z = np.mean(normals_np[:, 2])
        if mean_z < 0:
            normals_np = -normals_np  # 整体反转，确保z分量正向

        # 4. numpy→张量（回原设备）
        normals_lidar = torch.tensor(
            normals_np,
            dtype=points_lidar.dtype,
            device=points_lidar.device
        )
        return normals_lidar

    def points2normal_components_map(self, points_cam, normals_cam, points_img, height, width,post_aug):
        """
        生成法向量在三个方向的分量图：
        - comp_W：图像W轴方向（水平）分量
        - comp_H：图像H轴方向（垂直）分量
        - comp_depth：深度方向（相机视线）分量
        """
        # 下采样后图像尺寸
        height_down, width_down = height // self.downsample, width // self.downsample
        # 初始化三个分量图（与点云同设备）
        comp_W_map = torch.zeros((height_down, width_down), dtype=torch.float32, device=points_cam.device)
        comp_H_map = torch.zeros_like(comp_W_map)
        comp_depth_map = torch.zeros_like(comp_W_map)

        # 步骤1：法向量单位化（确保分量范围在[-1,1]）
        normals_cam = torch.nn.functional.normalize(normals_cam, dim=1)  # (N, 3)，单位向量

        # 步骤2：计算三个方向的分量（直接用点积，因方向向量是单位向量）
        # 深度方向分量：normals_cam · [0,0,1] = normals_cam[:,2]
        comp_depth = normals_cam[:, 2]
        # W轴方向分量：normals_cam · [1,0,0] = normals_cam[:,0]
        comp_W = normals_cam[:, 0]
        # H轴方向分量：normals_cam · [0,-1,0] = -normals_cam[:,1]
        comp_H = -normals_cam[:, 1]

        # 步骤3：从post_aug提取旋转矩阵，修正W/H分量（核心新增）
        R_aug = self.extract_rotation_matrix(post_aug)  # (2,2) numpy矩阵
        R_aug = torch.tensor(R_aug, dtype=comp_W.dtype, device=comp_W.device)  # 转为Tensor

        # 将W/H分量组合为2D向量，应用旋转矩阵
        comp_wh = torch.stack([comp_W, comp_H], dim=1)  # (N, 2)
        comp_wh_rotated = torch.matmul(comp_wh, R_aug.T)  # (N, 2) = (N,2) @ (2,2).T
        comp_W_rot = comp_wh_rotated[:, 0]  # 旋转后的W分量
        comp_H_rot = comp_wh_rotated[:, 1]  # 旋转后的H分量

        # 步骤3：关联分量到图像像素（复用深度图的有效点筛选逻辑）
        # 下采样后的像素坐标
        coor = torch.round(points_img[:, :2] / self.downsample)  # (N, 2)，u/v坐标
        depth = points_img[:, 2]  # 深度值（用于筛选有效点）

        # 筛选有效点：像素在图像内 + 深度在合理范围
        kept1 = (coor[:, 0] >= 0) & (coor[:, 0] < width_down) & (
            coor[:, 1] >= 0) & (coor[:, 1] < height_down) & (
                depth < self.grid_config['depth'][1]) & (
                    depth >= self.grid_config['depth'][0])
        coor, comp_W_rot, comp_H_rot, comp_depth = coor[kept1], comp_W_rot[kept1], comp_H_rot[kept1], comp_depth[kept1]

        # 步骤4：同像素去重（取最近深度对应的分量值，与深度图逻辑一致）
        ranks = coor[:, 0] + coor[:, 1] * width_down  # 2D坐标转1D索引
        # 排序键：像素索引 + 深度/100（同像素时近的在前）
        sort_idx = (ranks + depth[kept1] / 100.).argsort()
        coor, comp_W_rot, comp_H_rot, comp_depth, ranks = coor[sort_idx], comp_W_rot[sort_idx], comp_H_rot[sort_idx], comp_depth[sort_idx], ranks[sort_idx]
        
        # 去重：每个像素只保留第一个点（最近深度）
        kept2 = torch.ones_like(kept1[kept1])
        kept2[1:] = (ranks[1:] != ranks[:-1])
        coor, comp_W_rot, comp_H_rot, comp_depth = coor[kept2].long(), comp_W_rot[kept2], comp_H_rot[kept2], comp_depth[kept2]

        # 步骤6：赋值旋转后的分量到分量图
        comp_W_map[coor[:, 1], coor[:, 0]] = comp_W_rot
        comp_H_map[coor[:, 1], coor[:, 0]] = comp_H_rot
        comp_depth_map[coor[:, 1], coor[:, 0]] = comp_depth

        return comp_W_map, comp_H_map, comp_depth_map

    def points2normal_components_map_fix_direction(self, points_cam, normals_cam, points_img, height, width, post_aug):
        """
        生成法向量在三个方向的分量图：
        - comp_W：图像W轴方向（水平）分量
        - comp_H：图像H轴方向（垂直）分量
        - comp_depth：深度方向（相机视线）分量
        新增：相机坐标系下按「法向量与径向向量内积<0」统一法向量朝向（适配凸包结构）
        """
        # 下采样后图像尺寸
        height_down, width_down = height // self.downsample, width // self.downsample
        # 初始化三个分量图（与点云同设备）
        comp_W_map = torch.zeros((height_down, width_down), dtype=torch.float32, device=points_cam.device)
        comp_H_map = torch.zeros_like(comp_W_map)
        comp_depth_map = torch.zeros_like(comp_W_map)

        # 步骤1：法向量单位化（确保方向计算准确，分量范围在[-1,1]）
        normals_cam = torch.nn.functional.normalize(normals_cam, dim=1)  # (N, 3)，单位向量

        # ---------------------- 新增：相机坐标系下统一法向量朝向 ----------------------
        # 1. 定义"径向向量"：相机原点（光心）→3D点的向量（即points_cam本身）
        radial_vecs_cam = points_cam  # shape: (N, 3)，方向从相机指向3D点
        # 2. 过滤径向向量接近原点的异常点（方向无意义，避免误判）
        radial_mag_cam = torch.norm(radial_vecs_cam, dim=1, keepdim=True)  # 计算径向向量长度
        valid_radial_mask = radial_mag_cam > 1e-6  # 有效点：长度>1e-6（排除近原点噪声）
        # 3. 计算法向量与径向向量的点积（判断朝向是否需要反转）
        # 目标：让有效点的法向量满足「与径向向量内积<0」（凸包整体朝内等一致朝向）
        dot_product = torch.sum(normals_cam * radial_vecs_cam, dim=1, keepdim=True)
        # 4. 反转法向量：点积≥0的点需反转，确保内积<0
        reverse_mask = (dot_product >= 0) & valid_radial_mask  # 需要反转的点掩码
        normals_cam = torch.where(reverse_mask, -normals_cam, normals_cam)  # 逐点精准反转
        # 5. 无效径向向量点的兜底处理（避免孤立点法向量混乱）
        invalid_radial_mask = ~valid_radial_mask.squeeze()
        if invalid_radial_mask.any():
            # 辅助逻辑：让无效点法向量与"相机视线方向（径向向量）"反向，贴合整体趋势
            view_vecs = torch.nn.functional.normalize(radial_vecs_cam[invalid_radial_mask], dim=1)
            dot_view = torch.sum(normals_cam[invalid_radial_mask] * view_vecs, dim=1, keepdim=True)
            normals_cam[invalid_radial_mask] = torch.where(
                dot_view >= 0, 
                -normals_cam[invalid_radial_mask], 
                normals_cam[invalid_radial_mask]
            )
        # --------------------------------------------------------------------------

        # 步骤2：基于统一朝向的法向量，计算三个方向的分量（点积）
        # 深度方向分量：法向量 · 相机Z轴（[0,0,1]）→ 反映朝向/背离相机
        comp_depth = normals_cam[:, 2]
        # W轴方向分量：法向量 · 相机X轴（[1,0,0]）→ 反映左右倾斜
        comp_W = normals_cam[:, 0]
        # H轴方向分量：法向量 · 相机-Y轴（[0,-1,0]）→ 反映上下倾斜（适配图像v轴向下）
        comp_H = -normals_cam[:, 1]

        # 步骤3：从post_aug提取旋转矩阵，修正W/H分量（适配图像增强，原有逻辑保留）
        R_aug = self.extract_rotation_matrix(post_aug)  # (2,2) numpy矩阵
        R_aug = torch.tensor(R_aug, dtype=comp_W.dtype, device=comp_W.device)  # 转为Tensor
        # 将W/H分量组合为2D向量，应用旋转矩阵（对齐增强后的图像W/H轴）
        comp_wh = torch.stack([comp_W, comp_H], dim=1)  # (N, 2)：每行是[W分量, H分量]
        comp_wh_rotated = torch.matmul(comp_wh, R_aug)  # 行向量 × 旋转矩阵（无需转置）
        comp_W_rot = comp_wh_rotated[:, 0]  # 旋转后W轴分量（对齐图像水平方向）
        comp_H_rot = comp_wh_rotated[:, 1]  # 旋转后H轴分量（对齐图像垂直方向）

        # 步骤4：关联分量到图像像素（复用深度图的有效点筛选逻辑）
        coor = torch.round(points_img[:, :2] / self.downsample)  # 下采样后的像素坐标（u/v）
        depth = points_img[:, 2]  # 3D点的深度值（筛选合理深度范围）
        # 筛选有效点：像素在图像内 + 深度在配置区间内
        kept1 = (coor[:, 0] >= 0) & (coor[:, 0] < width_down) & (
            coor[:, 1] >= 0) & (coor[:, 1] < height_down) & (
                depth < self.grid_config['depth'][1]) & (
                    depth >= self.grid_config['depth'][0])
        # 保留有效点的坐标和分量
        coor, comp_W_rot, comp_H_rot, comp_depth = coor[kept1], comp_W_rot[kept1], comp_H_rot[kept1], comp_depth[kept1]

        # 步骤5：同像素去重（取最近深度对应的分量值，避免同像素多值冲突）
        ranks = coor[:, 0] + coor[:, 1] * width_down  # 2D坐标转1D索引（便于排序）
        # 排序键：像素索引 + 深度/100（同像素时，近深度优先保留，避免远遮挡近）
        sort_idx = (ranks + depth[kept1] / 100.).argsort()
        coor, comp_W_rot, comp_H_rot, comp_depth, ranks = coor[sort_idx], comp_W_rot[sort_idx], comp_H_rot[sort_idx], comp_depth[sort_idx], ranks[sort_idx]
        # 去重：仅保留每个像素的第一个点（最近深度对应的分量）
        kept2 = torch.ones_like(kept1[kept1])
        kept2[1:] = (ranks[1:] != ranks[:-1])  # 相邻像素索引不同则保留
        coor, comp_W_rot, comp_H_rot, comp_depth = coor[kept2].long(), comp_W_rot[kept2], comp_H_rot[kept2], comp_depth[kept2]

        # 步骤6：赋值分量到分量图（coor[:,1]是行（图像H轴），coor[:,0]是列（图像W轴））
        comp_W_map[coor[:, 1], coor[:, 0]] = comp_W_rot
        comp_H_map[coor[:, 1], coor[:, 0]] = comp_H_rot
        comp_depth_map[coor[:, 1], coor[:, 0]] = comp_depth

        return comp_W_map, comp_H_map, comp_depth_map
    def extract_rotation_matrix(self, post_aug):
        """从post_aug中提取2D旋转矩阵（假设无剪切，主要是旋转+缩放）"""
        # 提取左上角2x2矩阵（包含旋转和缩放）
        affine_2d = post_aug[:2, :2].cpu().numpy()  # 转为numpy便于处理
        # 计算缩放因子（确保旋转矩阵正交性）
        scale = np.linalg.norm(affine_2d[0, :])  # 取第一行的模长作为缩放因子
        if scale < 1e-6:
            return np.eye(2)  # 避免除零
        # 归一化得到旋转矩阵（去除缩放影响）
        R_aug = affine_2d / scale
        # 确保是旋转矩阵（正交化修正，应对数值误差）
        U, _, VT = np.linalg.svd(R_aug)
        R_aug = U @ VT
        if np.linalg.det(R_aug) < 0:  # 确保行列式为1（右手系）
            VT[-1, :] *= -1
            R_aug = U @ VT
        return R_aug

    def vis_normal_components_save(self, img, comp_W, comp_H, comp_depth, save_dir=None, show=True):
        """
        可视化法向量三个分量图，并支持保存到本地
        
        参数:
            img: 原始图像张量 (C, H, W)
            comp_W: W轴分量图张量 (H, W)
            comp_H: H轴分量图张量 (H, W)
            comp_depth: 深度方向分量图张量 (H, W)
            save_dir: 保存图像的目录路径，为None时不保存
            show: 是否显示图像，默认为True
        """
        # 1. 数据格式转换（Tensor→numpy，反归一化）
        img = img.permute(1, 2, 0).cpu().numpy()  # (H, W, C)
        img = img * self.std + self.mean  # 反归一化到像素值范围
        img = np.array(img, dtype=np.uint8)  # 转为uint8格式

        # 2. 处理无效像素（图像黑边区域）
        invalid_mask = (img == 0).any(axis=2)  # 标记图像黑边（无效区域）
        # 三个分量图的无效区域置0
        comp_W = comp_W.cpu().numpy()
        comp_H = comp_H.cpu().numpy()
        comp_depth = comp_depth.cpu().numpy()
        comp_W[invalid_mask] = 0
        comp_H[invalid_mask] = 0
        comp_depth[invalid_mask] = 0

        # 3. 筛选有效像素（用于散点图绘制）
        valid_mask = (comp_depth != 0)  # 用深度分量筛选有效点
        y, x = np.where(valid_mask)

        # 4. 创建3子图可视化
        plt.figure(figsize=(18, 6))

        # 子图1：W轴分量（左右倾斜）
        plt.subplot(1, 3, 1)
        plt.imshow(img)
        scatter1 = plt.scatter(x, y, c=comp_W[y, x], cmap='coolwarm', alpha=0.6, s=2)
        plt.title('Normal Component - Image W Axis (Left-Right)')
        plt.colorbar(scatter1, label='Component Value (-1=Left, 1=Right)')

        # 子图2：H轴分量（上下倾斜）
        plt.subplot(1, 3, 2)
        plt.imshow(img)
        scatter2 = plt.scatter(x, y, c=comp_H[y, x], cmap='coolwarm', alpha=0.6, s=2)
        plt.title('Normal Component - Image H Axis (Up-Down)')
        plt.colorbar(scatter2, label='Component Value (-1=Up, 1=Down)')

        # 子图3：深度分量（朝向/背离相机）
        plt.subplot(1, 3, 3)
        plt.imshow(img)
        scatter3 = plt.scatter(x, y, c=comp_depth[y, x], cmap='coolwarm', alpha=0.6, s=2)
        plt.title('Normal Component - Depth Direction (Camera Facing)')
        plt.colorbar(scatter3, label='Component Value (-1=Away, 1=Towards)')

        plt.tight_layout()  # 调整子图布局，避免重叠

        # 5. 保存图像到本地（若指定目录）
        if save_dir is not None:
            # 确保保存目录存在（不存在则创建）
            os.makedirs(save_dir, exist_ok=True)
            # 生成带索引的唯一文件名（与depth可视化共用index，确保全局唯一）
            file_name = f'normal_components_vis_{self.index}.png'
            save_path = os.path.join(save_dir, file_name)
            # 高分辨率保存（dpi=300，去除多余空白）
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"法向量分量图已保存至：{save_path}")

        # 6. 显示图像（可选，默认显示）
        if show:
            plt.show()
        else:
            plt.close()  # 不显示时关闭画布，释放内存

        self.index += 1  # 更新索引，确保下次保存文件名唯一

    def points2view_angle_map_fix_direction(self, points_cam, normals_cam, points_img, height, width):
        """
        生成视线角度图（法向量与视线向量夹角）
        新增：相机坐标系下按「法向量与径向向量内积<0」统一法向量朝向（适配凸包等结构）
        """
        height, width = height // self.downsample, width // self.downsample
        theta_map = torch.zeros((height, width), dtype=torch.float32, device=points_cam.device)

        # 1. 基础向量单位化（确保方向计算准确）
        # 视线向量：相机光心→3D点（即points_cam，单位化）
        view_vecs = torch.nn.functional.normalize(points_cam, dim=1)
        # 法向量：先单位化，再统一朝向
        normals_cam = torch.nn.functional.normalize(normals_cam, dim=1)

        # ---------------------- 新增：相机坐标系下统一法向量朝向 ----------------------
        # 2. 定义相机坐标系下的"径向向量"：相机原点→3D点的向量（即points_cam本身）
        radial_vecs_cam = points_cam  # shape: (N, 3)，方向从相机指向3D点
        # 3. 过滤径向向量接近原点的异常点（方向无意义，避免误判）
        radial_mag_cam = torch.norm(radial_vecs_cam, dim=1, keepdim=True)  # 径向向量长度
        valid_radial_mask = radial_mag_cam > 1e-6  # 有效径向向量：长度>1e-6（避免除以0）
        # 4. 计算法向量与径向向量的点积（判断朝向）
        # 点积<0：法向量与径向向量反向；点积≥0：方向同向，需反转以满足"内积<0"的统一要求
        dot_product = torch.sum(normals_cam * radial_vecs_cam, dim=1, keepdim=True)
        # 5. 反转法向量：让有效点的法向量满足"与径向向量内积<0"（统一朝向）
        reverse_mask = (dot_product >= 0) & valid_radial_mask  # 需要反转的点掩码
        normals_cam = torch.where(reverse_mask, -normals_cam, normals_cam)  # 逐点反转
        # （可选）对无效径向向量的点，保留原法向量（或按视线向量辅助修正，避免孤立点混乱）
        invalid_radial_mask = ~valid_radial_mask.squeeze()
        if invalid_radial_mask.any():
            # 辅助逻辑：让无效点法向量与视线向量反向（贴合整体朝向）
            dot_view = torch.sum(normals_cam[invalid_radial_mask] * view_vecs[invalid_radial_mask], dim=1, keepdim=True)
            normals_cam[invalid_radial_mask] = torch.where(dot_view >= 0, -normals_cam[invalid_radial_mask], normals_cam[invalid_radial_mask])
        # --------------------------------------------------------------------------

        # 6. 计算视线夹角θ（0~180°）：基于统一朝向的法向量
        cos_theta = torch.sum(view_vecs * normals_cam, dim=1)
        cos_theta = torch.clamp(cos_theta, -1.0, 1.0)  # 避免数值误差（acos输入需在[-1,1]）
        theta = torch.acos(cos_theta) * 180 / torch.pi  # 弧度转角度

        # 7. 关联角度到图像像素（复用原有效点筛选逻辑）
        coor = torch.round(points_img[:, :2] / self.downsample)  # 下采样后的像素坐标
        depth = points_img[:, 2]  # 3D点的深度值（用于筛选合理深度范围）
        
        # 筛选有效点：像素在图像内 + 深度在配置范围内
        kept1 = (coor[:, 0] >= 0) & (coor[:, 0] < width) & (
            coor[:, 1] >= 0) & (coor[:, 1] < height) & (
                depth < self.grid_config['depth'][1]) & (
                    depth >= self.grid_config['depth'][0])
        coor, theta = coor[kept1], theta[kept1]

        # 8. 同像素去重：取最近深度对应的角度（避免同像素多值冲突）
        ranks = coor[:, 0] + coor[:, 1] * width  # 2D像素坐标转1D索引（便于排序）
        # 排序键：像素索引 + 深度/100（同像素时，近深度优先保留）
        sort_idx = (ranks + depth[kept1] / 100.).argsort()
        coor, theta, ranks = coor[sort_idx], theta[sort_idx], ranks[sort_idx]
        # 去重：仅保留每个像素的第一个点（最近深度对应的角度）
        kept2 = torch.ones_like(kept1[kept1])
        kept2[1:] = (ranks[1:] != ranks[:-1])
        coor, theta = coor[kept2].long(), theta[kept2]

        # 9. 赋值角度到角度图
        theta_map[coor[:, 1], coor[:, 0]] = theta

        return theta_map

    # ---------------------- 新增：1. Bin索引计算核心方法（处理无label与极小值） ----------------------
    def compute_wh_bin_index(self, comp_wh):
        """
        计算W/H方向分量的bin索引：
        - 无label（分量绝对值 < eps_label）→ 索引0
        - 有值（非0）→ 按角度映射到1~24（极小非0值对应1，角度越大索引越大）
        """
        device = comp_wh.device
        bin_index = torch.zeros_like(comp_wh, dtype=torch.long, device=device)  # 初始化无label索引0

        # 1. 筛选"有值的非0分量"（排除无label的0）
        has_label_mask = torch.abs(comp_wh) >= self.eps_label
        if not torch.any(has_label_mask):
            return bin_index  # 全是无label，直接返回0

        # 2. 对有值分量计算角度（度）
        comp_clamped = torch.clamp(comp_wh[has_label_mask], -1.0 + 1e-6, 1.0 - 1e-6)
        theta_rad = torch.acos(comp_clamped)
        theta_deg = theta_rad * 180.0 / torch.pi  # 0~180度

        # 3. 角度→bin索引（1~24）：极小角度（0~7.5度）对应1，依次递增
        bin_idx_0based = torch.floor(theta_deg / self.wh_bin_size).long()
        bin_idx_0based = torch.clamp(bin_idx_0based, 0, self.wh_num_bins - 1)  # 防止越界
        bin_idx_1based = bin_idx_0based + 1  # 转为1-based有效索引

        # 4. 赋值有效索引
        bin_index[has_label_mask] = bin_idx_1based
        return bin_index

    def compute_wh_bin_index_fixRing(self, comp_wh):
        """
        计算W/H方向分量的bin索引（0°与180°附近合并为索引1）：
        - 无label（分量绝对值 < eps_label）→ 索引0
        - 有值时：
        - 0°~3.75° 或 176.25°~180° → 索引1
        - 3.75°~11.25° → 索引2（3.75+7.5）
        - 11.25°~18.75° → 索引3
        - ...（依次递增，每个bin宽7.5°）
        - 168.75°~176.25° → 索引24
        """
        device = comp_wh.device
        bin_index = torch.zeros_like(comp_wh, dtype=torch.long, device=device)  # 无label索引0

        # 1. 筛选"有值的非0分量"（排除无label的0）
        has_label_mask = torch.abs(comp_wh) >= self.eps_label
        if not torch.any(has_label_mask):
            return bin_index

        # 2. 计算原始角度（0°~180°）
        comp_clamped = torch.clamp(comp_wh[has_label_mask], -1.0 + 1e-6, 1.0 - 1e-6)
        theta_rad = torch.acos(comp_clamped)
        theta_deg = theta_rad * 180.0 / torch.pi  # 原始角度：0°~180°

        # 3. 定义每个bin的角度区间（核心：0°附近与180°附近合并为索引1）
        bin_half = self.wh_bin_size / 2  # 3.75°（每个bin的半宽）
        total_deg = 180.0  # 总角度范围

        # 4. 角度→bin索引映射（1~24）
        # 索引1：0°~3.75° 或 176.25°~180°
        mask_bin1 = (theta_deg < bin_half) | (theta_deg >= (total_deg - bin_half))
        
        # 索引2~24：从3.75°开始，每个bin宽7.5°
        # 先将176.25°~180°的角度映射到-3.75°~0°，便于统一计算
        theta_deg_adj = torch.where(
            theta_deg >= (total_deg - bin_half),
            theta_deg - total_deg,  # 176.25°→-3.75°，180°→0°
            theta_deg
        )
        # 对非bin1的角度，计算相对于3.75°的偏移（映射到0~172.5°）
        theta_offset = theta_deg_adj - bin_half  # 3.75°→0°，11.25°→7.5°，...，176.25°→-7.5°（被bin1过滤）
        # 计算索引2~24（过滤掉bin1的角度）
        bin_idx_0based = torch.floor(theta_offset / self.wh_bin_size).long()
        # 确保索引在0~22（对应2~24），并过滤bin1的角度
        bin_idx_0based = torch.clamp(bin_idx_0based, 0, self.wh_num_bins - 2)  # 24-2=22
        bin_idx_2to24 = bin_idx_0based + 2  # 转为2~24

        # 5. 合并索引（1~24）
        bin_idx_1based = torch.where(
            mask_bin1,
            torch.tensor(1, device=device),  # bin1对应索引1
            bin_idx_2to24  # 其他对应2~24
        )

        # 6. 赋值有效索引
        bin_index[has_label_mask] = bin_idx_1based
        return bin_index

    def compute_wh_bin_index_fixRing_divofDotvalue(self, comp_wh):
        """
        计算W/H方向分量的bin索引（0°与180°附近合并为索引1）：
        - 无label（分量绝对值 < eps_label）→ 索引0
        - 有值时：
        - 0°~3.75° 或 176.25°~180° → 索引1
        - 3.75°~11.25° → 索引2（3.75+7.5）
        - 11.25°~18.75° → 索引3
        - ...（依次递增，每个bin宽7.5°）
        - 168.75°~176.25° → 索引24
        """
        device = comp_wh.device
        bin_index = torch.zeros_like(comp_wh, dtype=torch.long, device=device)  # 无label索引0

        # 1. 筛选"有值的非0分量"（排除无label的0）
        has_label_mask = torch.abs(comp_wh) >= self.eps_label
        if not torch.any(has_label_mask):
            return bin_index

        # 2. 计算原始角度（0°~180°）
        comp_clamped = torch.clamp(comp_wh[has_label_mask], -1.0 + 1e-6, 1.0 - 1e-6)
        theta_rad = torch.acos(comp_clamped)
        theta_deg = theta_rad * 180.0 / torch.pi  # 原始角度：0°~180°

        # # 3. 定义每个bin的角度区间（核心：0°附近与180°附近合并为索引1）
        # bin_half = self.wh_bin_size / 2  # 3.75°（每个bin的半宽）
        # total_deg = 180.0  # 总角度范围

        # # 4. 角度→bin索引映射（1~24）
        # # 索引1：0°~3.75° 或 176.25°~180°
        # mask_bin1 = (theta_deg < bin_half) | (theta_deg >= (total_deg - bin_half))
        
        # # 索引2~24：从3.75°开始，每个bin宽7.5°
        # # 先将176.25°~180°的角度映射到-3.75°~0°，便于统一计算
        # theta_deg_adj = torch.where(
        #     theta_deg >= (total_deg - bin_half),
        #     theta_deg - total_deg,  # 176.25°→-3.75°，180°→0°
        #     theta_deg
        # )
        # # 对非bin1的角度，计算相对于3.75°的偏移（映射到0~172.5°）
        # theta_offset = theta_deg_adj - bin_half  # 3.75°→0°，11.25°→7.5°，...，176.25°→-7.5°（被bin1过滤）
        # # 计算索引2~24（过滤掉bin1的角度）
        # bin_idx_0based = torch.floor(theta_offset / self.wh_bin_size).long()
        # # 确保索引在0~22（对应2~24），并过滤bin1的角度
        # bin_idx_0based = torch.clamp(bin_idx_0based, 0, self.wh_num_bins - 2)  # 24-2=22
        # bin_idx_2to24 = bin_idx_0based + 2  # 转为2~24

        # # 5. 合并索引（1~24）
        # bin_idx_1based = torch.where(
        #     mask_bin1,
        #     torch.tensor(1, device=device),  # bin1对应索引1
        #     bin_idx_2to24  # 其他对应2~24
        # )
       
        bin_idx_1based = ((((theta_deg)+(180.0/self.wh_num_bins/2))//(180.0/self.wh_num_bins))%self.wh_num_bins+1)

        # bin_idx_1based = ((((comp_clamped+1)+(2/self.wh_num_bins/2))//(2/self.wh_num_bins))%self.wh_num_bins+1)
        # 6. 赋值有效索引
        bin_index[has_label_mask] = bin_idx_1based.to(torch.long)
        return bin_index

    def compute_depth_bin_index(self, comp_depth):
        """
        计算深度方向分量的bin索引：
        - 无label（分量 < eps_label）→ 索引0
        - 有值（非0）→ 按角度映射到1~12（极小非0值对应1，角度越大索引越大）
        """
        device = comp_depth.device
        bin_index = torch.zeros_like(comp_depth, dtype=torch.long, device=device)  # 初始化无label索引0

        # 1. 筛选"有值的非0分量"（排除无label的0）
        has_label_mask = comp_depth >= self.eps_label  # 深度分量已统一为非负（之前有反转）
        if not torch.any(has_label_mask):
            return bin_index

        # 2. 对有值分量计算角度（度）
        comp_clamped = torch.clamp(comp_depth[has_label_mask], 0.0 + 1e-6, 1.0 - 1e-6)
        theta_rad = torch.acos(comp_clamped)
        theta_deg = theta_rad * 180.0 / torch.pi  # 0~90度

        # 3. 角度→bin索引（1~12）：极小角度（0~7.5度）对应1，依次递增
        bin_idx_0based = torch.floor(theta_deg / self.depth_bin_size).long()
        bin_idx_0based = torch.clamp(bin_idx_0based, 0, self.depth_num_bins - 1)  # 防止越界
        bin_idx_1based = bin_idx_0based + 1  # 转为1-based有效索引

        # 4. 赋值有效索引
        bin_index[has_label_mask] = bin_idx_1based
        return bin_index

    def compute_depth_bin_index_fixRing(self, comp_depth):
        """
        计算深度方向分量的bin索引（精细区间划分）：
        - 无label（分量 < eps_label）→ 索引0
        - 有值时：
        - 0°~3.75° → 索引1
        - 3.75°~11.25° → 索引2（3.75+7.5）
        - 11.25°~18.75° → 索引3
        - ...（依次递增，每个bin宽7.5°）
        - 86.25°~90° → 索引12
        """
        device = comp_depth.device
        bin_index = torch.zeros_like(comp_depth, dtype=torch.long, device=device)  # 无label索引0

        # 1. 筛选"有值的非0分量"（排除无label的0）
        has_label_mask = comp_depth >= self.eps_label  # 深度分量已统一为非负
        if not torch.any(has_label_mask):
            return bin_index

        # 2. 计算原始角度（0°~90°）
        comp_clamped = torch.clamp(comp_depth[has_label_mask], 0.0 + 1e-6, 1.0 - 1e-6)
        theta_rad = torch.acos(comp_clamped)
        theta_deg = theta_rad * 180.0 / torch.pi  # 原始角度：0°~90°

        # 3. 定义每个bin的角度区间（核心：0°附近3.75°为索引1，后续按7.5°递增）
        bin_half = self.depth_bin_size / 2  # 3.75°（每个bin的半宽）
        total_deg = 90.0  # 深度方向总角度范围

        # 4. 角度→bin索引映射（1~12）
        # 索引1：0°~3.75°
        mask_bin1 = theta_deg < bin_half
        
        # 索引2~12：从3.75°开始，每个bin宽7.5°
        # 计算角度相对于3.75°的偏移（映射到0~86.25°）
        theta_offset = theta_deg - bin_half  # 3.75°→0°，11.25°→7.5°，...，90°→86.25°
        # 计算索引2~12（过滤掉bin1的角度）
        bin_idx_0based = torch.floor(theta_offset / self.depth_bin_size).long()
        # 确保索引在0~10（对应2~12），并过滤bin1的角度
        bin_idx_0based = torch.clamp(bin_idx_0based, 0, self.depth_num_bins - 2)  # 12-2=10
        bin_idx_2to12 = bin_idx_0based + 2  # 转为2~12

        # 5. 合并索引（1~12）
        bin_idx_1based = torch.where(
            mask_bin1,
            torch.tensor(1, device=device),  # bin1对应索引1
            bin_idx_2to12  # 其他对应2~12
        )

        # 6. 赋值有效索引
        bin_index[has_label_mask] = bin_idx_1based
        return bin_index


    def compute_depth_bin_index_fixRing_divofDotvalue(self, comp_depth):
        """
        计算深度方向分量的bin索引（精细区间划分）：
        - 无label（分量 < eps_label）→ 索引0
        - 有值时：
        - 0°~3.75° → 索引1
        - 3.75°~11.25° → 索引2（3.75+7.5）
        - 11.25°~18.75° → 索引3
        - ...（依次递增，每个bin宽7.5°）
        - 86.25°~90° → 索引12
        """
        device = comp_depth.device
        bin_index = torch.zeros_like(comp_depth, dtype=torch.long, device=device)  # 无label索引0

        # 1. 筛选"有值的非0分量"（排除无label的0）
        has_label_mask = comp_depth >= self.eps_label  # 深度分量已统一为非负
        if not torch.any(has_label_mask):
            return bin_index

        # 2. 计算原始角度（0°~90°）
        comp_clamped = torch.clamp(comp_depth[has_label_mask], 0.0 + 1e-6, 1.0 - 1e-6)
        theta_rad = torch.acos(comp_clamped)
        theta_deg = theta_rad * 180.0 / torch.pi  # 原始角度：0°~90°

        # # 3. 定义每个bin的角度区间（核心：0°附近3.75°为索引1，后续按7.5°递增）
        # bin_half = self.depth_bin_size / 2  # 3.75°（每个bin的半宽）
        # total_deg = 90.0  # 深度方向总角度范围

        # # 4. 角度→bin索引映射（1~12）
        # # 索引1：0°~3.75°
        # mask_bin1 = theta_deg < bin_half
        
        # # 索引2~12：从3.75°开始，每个bin宽7.5°
        # # 计算角度相对于3.75°的偏移（映射到0~86.25°）
        # theta_offset = theta_deg - bin_half  # 3.75°→0°，11.25°→7.5°，...，90°→86.25°
        # # 计算索引2~12（过滤掉bin1的角度）
        # bin_idx_0based = torch.floor(theta_offset / self.depth_bin_size).long()
        # # 确保索引在0~10（对应2~12），并过滤bin1的角度
        # bin_idx_0based = torch.clamp(bin_idx_0based, 0, self.depth_num_bins - 2)  # 12-2=10
        # bin_idx_2to12 = bin_idx_0based + 2  # 转为2~12

        # # 5. 合并索引（1~12）
        # bin_idx_1based = torch.where(
        #     mask_bin1,
        #     torch.tensor(1, device=device),  # bin1对应索引1
        #     bin_idx_2to12  # 其他对应2~12
        # )
        
        bin_idx_1based = ((((theta_deg))//(90.0/self.depth_num_bins))%self.depth_num_bins+1)

        # bin_idx_1based = ((((comp_clamped))//(1/self.depth_num_bins))%self.depth_num_bins+1)

        # 6. 赋值有效索引
        bin_index[has_label_mask] = bin_idx_1based.to(torch.long)
        return bin_index

    def points2heightmap(self, points_lidar, points_img, height, width):
        """
        生成图像视角的像素级高度图：
        - 输入：points_lidar（LiDAR点云，含高度信息）、points_img（LiDAR点投影到图像的坐标）
        - 输出：height_map（下采样后的像素级高度图，无效像素为0）
        """
        # 1. 下采样后图像尺寸（与深度图保持一致）
        height_down, width_down = height // self.downsample, width // self.downsample
        # 初始化高度图（与LiDAR点云同设备，无效值为0）
        height_map = torch.zeros((height_down, width_down), dtype=torch.float32, device=points_lidar.device)

        # 2. 提取LiDAR点的高度信息（默认Z轴，可通过self.lidar_height_axis调整）
        point_heights = points_lidar[:, 2]  # (N,)，每个LiDAR点的高度值

        # 3. 筛选有效点：像素在图像内 + 深度在合理范围 + 高度在有效范围
        # 3.1 投影坐标下采样（与深度图逻辑一致）
        coor = torch.round(points_img[:, :2] / self.downsample)  # (N,2)，下采样后的像素坐标（u,v）
        depth = points_img[:, 2]  # (N,)，LiDAR点的深度值（复用深度筛选）
        # 3.2 有效点掩码（多条件过滤）
        kept1 = (coor[:, 0] >= 0) & (coor[:, 0] < width_down) & (  # 像素u在范围内
                coor[:, 1] >= 0) & (coor[:, 1] < height_down) & ( # 像素v在范围内
                depth >= self.grid_config['depth'][0]) & (depth <= self.grid_config['depth'][1]) 
                # &  # 深度有效
                # (point_heights >= self.height_min) & (point_heights <= self.height_max)  # 高度有效

        # 4. 保留有效点的坐标、高度、深度（用于同像素去重）
        coor_valid = coor[kept1]
        height_valid = point_heights[kept1]
        depth_valid = depth[kept1]
        if len(coor_valid) == 0:
            return height_map  # 无有效点，返回全0高度图

        # 5. 同像素去重：取「最近深度」对应的高度（避免远景点遮挡近景点）
        # 5.1 2D像素坐标转1D索引（便于排序和去重）
        ranks = coor_valid[:, 0] + coor_valid[:, 1] * width_down  # (M,)，M为有效点数量
        # 5.2 按「像素索引 + 深度/100」排序（同像素内，深度小的在前，即近点优先）
        sort_idx = (ranks + depth_valid / 100.).argsort()  # 深度/100确保同像素内深度优先
        coor_sorted = coor_valid[sort_idx]
        height_sorted = height_valid[sort_idx]
        ranks_sorted = ranks[sort_idx]

        # 5.3 去重：每个像素仅保留第一个点（最近深度对应的高度）
        kept2 = torch.ones_like(ranks_sorted, dtype=torch.bool)
        kept2[1:] = (ranks_sorted[1:] != ranks_sorted[:-1])  # 相邻像素索引不同则保留
        coor_final = coor_sorted[kept2].long()  # 最终有效像素坐标（转为long类型用于索引）
        height_final = height_sorted[kept2]     # 最终有效高度值

        # 6. 赋值高度到高度图（注意：图像坐标v对应高度图的行，u对应列）
        height_map[coor_final[:, 1], coor_final[:, 0]] = height_final

        return height_map

    def vis_height_img(self, img, height_map, depth_map=None):
        """可视化高度图（可选叠加深度图对比）"""
        # 数据格式转换（Tensor→numpy，反归一化）
        height_map_np = height_map.cpu().numpy()
        img_np = img.permute(1, 2, 0).cpu().numpy()
        img_np = img_np * self.std + self.mean
        img_np = np.array(img_np, dtype=np.uint8)

        # 处理无效像素（图像黑边）
        invalid_mask = (img_np == 0).any(axis=2)
        height_map_np[invalid_mask] = 0
        if depth_map is not None:
            depth_map_np = depth_map.cpu().numpy()
            depth_map_np[invalid_mask] = 0

        # 筛选有效高度像素
        valid_height_mask = (height_map_np != 0)
        y, x = np.where(valid_height_mask)

        # 创建可视化子图
        plt.figure(figsize=(14, 6))
        # 子图1：高度图
        plt.subplot(1, 2, 1)
        plt.imshow(img_np)
        scatter1 = plt.scatter(x, y, c=height_map_np[y, x], cmap='viridis', alpha=0.6, s=2)
        plt.title(f'Height Map (m)')
        plt.colorbar(scatter1, label=f'Height (Range: {self.height_min}~{self.height_max}m)')

        # 子图2：深度图（可选，用于对比）
        if depth_map is not None:
            valid_depth_mask = (depth_map_np != 0)
            y_d, x_d = np.where(valid_depth_mask)
            plt.subplot(1, 2, 2)
            plt.imshow(img_np)
            scatter2 = plt.scatter(x_d, y_d, c=depth_map_np[y_d, x_d], cmap='rainbow_r', alpha=0.6, s=2)
            plt.title('Depth Map (m)')
            plt.colorbar(scatter2, label='Depth')

        plt.tight_layout()
        plt.show()
        self.index += 1

    def vis_height_img_save(self, img, height_map, save_dir=None, depth_map=None, show=True):
        """可视化高度图并保存到本地（支持叠加深度图对比）"""
        height_map_np = height_map.cpu().numpy()
        img_np = img.permute(1, 2, 0).cpu().numpy()
        img_np = img_np * self.std + self.mean
        img_np = np.array(img_np, dtype=np.uint8)

        # 处理无效像素
        invalid_mask = (img_np == 0).any(axis=2)
        height_map_np[invalid_mask] = 0
        depth_map_np = None
        if depth_map is not None:
            depth_map_np = depth_map.cpu().numpy()
            depth_map_np[invalid_mask] = 0

        # 筛选有效像素
        valid_height_mask = (height_map_np != 0)
        y, x = np.where(valid_height_mask)

        # 创建画布
        fig_size = (14, 6) if depth_map is not None else (7, 6)
        plt.figure(figsize=fig_size)
        # 高度图子图
        plt.subplot(1, 2, 1) if depth_map is not None else plt.subplot(1, 1, 1)
        plt.imshow(img_np)
        scatter1 = plt.scatter(x, y, c=height_map_np[y, x], cmap='viridis', alpha=0.6, s=2)
        plt.title(f'Height Map (m)')
        plt.colorbar(scatter1, label=f'Height (Range: {self.height_min}~{self.height_max}m)')

        # 深度图子图（可选）
        if depth_map is not None:
            valid_depth_mask = (depth_map_np != 0)
            y_d, x_d = np.where(valid_depth_mask)
            plt.subplot(1, 2, 2)
            plt.imshow(img_np)
            scatter2 = plt.scatter(x_d, y_d, c=depth_map_np[y_d, x_d], cmap='rainbow_r', alpha=0.6, s=2)
            plt.title('Depth Map (m)')
            plt.colorbar(scatter2, label='Depth')

        plt.tight_layout()
        # 保存图像
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            file_name = f'height_vis_{self.index}.png' if depth_map is None else f'height_depth_vis_{self.index}.png'
            save_path = os.path.join(save_dir, file_name)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"高度图已保存至：{save_path}")
        # 显示或关闭
        if show:
            plt.show()
        else:
            plt.close()
        self.index += 1
    def __call__(self, results):
        # 原有：读取输入数据
        points_lidar = results['points'].tensor  # LiDAR点云 (N, 3)
        imgs, sensor2egos, ego2globals, cam2imgs, post_augs, bda = results['img_inputs']
        lidar2imgs = results['lidar2img']  # LiDAR→各相机的投影矩阵 (6, 3, 4)
        nt, c, h, w = imgs.shape
        t_frame = nt // self.num_cam

        # 新增：1. 计算LiDAR坐标系下的法向量
        normals_lidar = self.compute_lidar_normals(points_lidar)

        # 原有：初始化结果存储
        depth_maps = []
        view_angle_maps = []  # 新增：视线角度图存储
        comp_W_maps = []   # W轴分量图列表
        comp_H_maps = []   # H轴分量图列表
        comp_depth_maps = []  # 深度分量图列表

        comp_W_bins = []
        comp_H_bins = []
        comp_depth_bins = []
        height_maps = []
        # 遍历每个相机，生成深度图+视线角度图
        for cid in range(len(results['cam_names'])):
            lidar2img = lidar2imgs[cid]
            cam2img = cam2imgs[cid]  # 相机内参（用于提取外参）
            lidar2cam = results['lidar2cam'][cid]  # 外参（用于转换坐标）
            post_aug = post_augs[cid]  # 当前相机的增强矩阵（3x3）
            # 原有：1. LiDAR点投影到图像平面（u, v, depth）
            points_homo = torch.cat([points_lidar.T, torch.ones((1, points_lidar.shape[0]), device=points_lidar.device)], dim=0)
            points_img_homo = lidar2img @ points_homo  # (3, N)
            points_img = points_img_homo.permute(1, 0)  # (N, 3)
            # 透视除法：(u/z, v/z, z) → (u, v, z)
            points_img = torch.cat([
                points_img[:, :2] / points_img[:, 2].unsqueeze(1),
                points_img[:, 2].unsqueeze(1)
            ], dim=1)

            # 新增：2. 转换法向量和点到相机坐标系
            # 提取LiDAR→相机的旋转矩阵R
            # R_lidar2cam, T_lidar2cam = self.extract_lidar2cam_extrinsic(lidar2img, cam2img)
            R_lidar2cam, T_lidar2cam = lidar2cam[:3, :3],lidar2cam[:3, 3].unsqueeze(1)  # 外参矩阵转换为相机坐标系
            # 转换3D点（相机坐标系：P_cam = R * P_lidar）
            points_cam = (R_lidar2cam @ points_lidar.T + T_lidar2cam).T   # (N, 3)
            # 转换法向量（方向向量仅旋转，无平移）
            normals_cam = (R_lidar2cam @ normals_lidar.T).T  # (N, 3)

            # 原有：3. 生成深度图
            depth_map = self.points2depthmap(points_img, h, w)
            depth_maps.append(depth_map)

            # 新增：4. 生成视线角度图
            theta_map = self.points2view_angle_map(points_cam, normals_cam, points_img, h, w)
            view_angle_maps.append(theta_map)

            comp_W_map, comp_H_map, comp_depth_map = self.points2normal_components_map(
                points_cam, normals_cam, points_img, h, w,post_aug
            )
            my_revers_mask = comp_depth_map<0
            comp_W_map[my_revers_mask] *=-1
            comp_H_map[my_revers_mask] *=-1
            comp_depth_map[my_revers_mask] *=-1
            comp_W_maps.append(comp_W_map)
            comp_H_maps.append(comp_H_map)
            comp_depth_maps.append(comp_depth_map)
            wh_bin_W = self.compute_wh_bin_index_fixRing_divofDotvalue(comp_W_map)
            comp_W_bins.append(wh_bin_W)
            # H方向bin索引（0=无label，1~24=有效方向）
            wh_bin_H = self.compute_wh_bin_index_fixRing_divofDotvalue(comp_H_map)
            comp_H_bins.append(wh_bin_H)
            # 深度方向bin索引（0=无label，1~12=有效方向）
            depth_bin = self.compute_depth_bin_index_fixRing_divofDotvalue(comp_depth_map)
            comp_depth_bins.append(depth_bin)

            height_map = self.points2heightmap(points_lidar, points_img, h, w)
            height_maps.append(height_map)



            #  用于可视化 复原离散值

            #映射表
            # wh_bin_WH2cosvalue = None
            # (torch.arange(-1,1,0.2)+0.1)[bin_idx_1based.to(torch.long)-1]


            #TODO 这里在可视化的时候打开
            # comp_W_map[wh_bin_W!=0] = (torch.arange(-1,1,0.2)+0.1)[wh_bin_W[wh_bin_W!=0]-1]
            # comp_H_map[wh_bin_H!=0] = (torch.arange(-1,1,0.2)+0.1)[wh_bin_H[wh_bin_H!=0]-1]
            # comp_depth_map[depth_bin!=0] = (torch.arange(0,1,0.2)+0.1)[depth_bin[depth_bin!=0]-1]
            #TODO 上面这里在可视化的时候打开
            
            
            
            # comp_H_map[wh_bin_H!=0] = torch.cos((wh_bin_H-1)*self.wh_bin_size)[wh_bin_H!=0]
            # comp_depth_map[depth_bin!=0] = torch.cos((depth_bin-1)*self.depth_bin_size)[depth_bin!=0]
            
            1==1
            #   用于可视化 复原离散值



            # 可选：可视化验证（深度图+视线角度图）
            # if cid == 0:  # 仅可视化第一个相机，避免冗余
            #     self.vis_depth_img(imgs[cid * t_frame], depth_map, theta_map)
            # self.vis_depth_img_save(imgs[cid * t_frame], depth_map, theta_map, save_dir='./vis_results')
            # 可选：可视化验证（每个相机的三个分量图）
            # if cid == 0:
            #     self.vis_normal_components(imgs[cid * t_frame], comp_W_map, comp_H_map, comp_depth_map)
            # self.vis_normal_components_save(
            #     imgs[cid * t_frame], 
            #     comp_W_map, 
            #     comp_H_map, 
            #     comp_depth_map, 
            #     save_dir='./normal_vis', 
            #     show=False
            # )

            # 可选：高度图可视化（与深度图对比）
            # self.vis_height_img_save(imgs[cid * t_frame], height_map, save_dir='./height_vis', depth_map=depth_map, show=False)

        # 原有：保存深度图结果
        results['gt_depth'] = torch.stack(depth_maps)
        # 新增：保存视线角度图结果
        results['view_angle_maps'] = torch.stack(view_angle_maps)
        results['normal_comp_W_maps'] = torch.stack(comp_W_maps)    # (6, H_down, W_down)
        results['normal_comp_H_maps'] = torch.stack(comp_H_maps)    # 6个相机的W轴分量图
        results['normal_comp_depth_maps'] = torch.stack(comp_depth_maps)  # 6个相机的深度分量图
        results['normal_comp_W_bins'] = torch.stack(comp_W_bins)  # (6, H_down, W_down)
        results['normal_comp_H_bins'] = torch.stack(comp_H_bins)
        results['normal_comp_depth_bins'] = torch.stack(comp_depth_bins)
        results['normals_lidar'] = normals_lidar  # (N, 3)
        results['gt_height_maps'] = torch.stack(height_maps)
        return results