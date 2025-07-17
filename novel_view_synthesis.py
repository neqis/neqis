import os
import logging
import argparse
import numpy as np
import cv2 as cv
import trimesh
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from shutil import copyfile
from tqdm import tqdm
from pyhocon import ConfigFactory
from models.dataset import Dataset
from networks.rendering_network import RenderingNetwork
from networks.sdf_network import SDFNetwork
from networks.single_variance_network import SingleVarianceNetwork
from networks.nerf import NeRF
from models.renderer import NeQISRenderer

class Runner:
    def __init__(self, conf_path, mode='train', case='CASE_NAME', is_continue=False, checkpoint = False):
        self.device = torch.device('cuda')

        # Configuration
        self.conf_path = conf_path
        f = open(self.conf_path)
        conf_text = f.read()
        conf_text = conf_text.replace('CASE_NAME', case)
        f.close()

        self.conf = ConfigFactory.parse_string(conf_text)
        self.conf['dataset.data_dir'] = self.conf['dataset.data_dir'].replace('CASE_NAME', case)
        self.base_exp_dir = self.conf['general.base_exp_dir']
        os.makedirs(self.base_exp_dir, exist_ok=True)
        self.dataset = Dataset(self.conf['dataset'])
        self.iter_step = 0

        # Training parameters
        self.end_iter = self.conf.get_int('train.end_iter')
        self.save_freq = self.conf.get_int('train.save_freq')
        self.report_freq = self.conf.get_int('train.report_freq')
        self.val_freq = self.conf.get_int('train.val_freq')
        self.val_mesh_freq = self.conf.get_int('train.val_mesh_freq')
        self.batch_size = self.conf.get_int('train.batch_size')
        self.validate_resolution_level = self.conf.get_int('train.validate_resolution_level')
        self.learning_rate = self.conf.get_float('train.learning_rate')
        self.learning_rate_alpha = self.conf.get_float('train.learning_rate_alpha')
        self.use_white_bkgd = self.conf.get_bool('train.use_white_bkgd')
        self.warm_up_end = self.conf.get_float('train.warm_up_end', default=0.0)
        self.anneal_end = self.conf.get_float('train.anneal_end', default=0.0)

        # Weights
        self.color_weight = self.conf.get_float('train.color_weight')
        self.igr_weight = self.conf.get_float('train.igr_weight')
        self.mask_weight = self.conf.get_float('train.mask_weight')
        self.normal_weight = self.conf.get_float('train.normal_weight')
        
        self.is_continue = is_continue
        self.mode = mode
        self.model_list = []
        self.writer = None

        # Networks
        params_to_train = []
        self.nerf_outside = NeRF(**self.conf['model.nerf']).to(self.device)
        self.sdf_network = SDFNetwork(**self.conf['model.sdf_network']).to(self.device)
        self.deviation_network = SingleVarianceNetwork(**self.conf['model.variance_network']).to(self.device)
        self.color_network = RenderingNetwork(**self.conf['model.rendering_network']).to(self.device)
        
        params_to_train += list(self.nerf_outside.parameters())
        params_to_train += list(self.sdf_network.parameters())
        params_to_train += list(self.deviation_network.parameters())
        params_to_train += list(self.color_network.parameters())
        
        self.optimizer = torch.optim.Adam(params_to_train, lr=self.learning_rate)
        self.renderer = NeQISRenderer(self.nerf_outside,
                                     self.sdf_network,
                                     self.deviation_network,
                                     self.color_network,
                                     **self.conf['model.neus_renderer'])

        # Load checkpoint
        latest_model_name = None
        if checkpoint:
             model_list_raw = os.listdir(os.path.join(self.base_exp_dir, 'checkpoints'))
             model_list = []
             for model_name in model_list_raw:
                 if model_name[-3:] == 'pth' and int(model_name[5:-4]) <= self.end_iter:
                     model_list.append(model_name)
             model_list.sort()
             latest_model_name = model_list[checkpoint-1]

        if is_continue:
            model_list_raw = os.listdir(os.path.join(self.base_exp_dir, 'checkpoints'))
            model_list = []
            for model_name in model_list_raw:
                if model_name[-3:] == 'pth' and int(model_name[5:-4]) <= self.end_iter:
                    model_list.append(model_name)
            model_list.sort()
            latest_model_name = model_list[-1]

        if latest_model_name is not None:
            logging.info('Find checkpoint: {}'.format(latest_model_name))
            self.load_checkpoint(latest_model_name)

        # Backup codes and configs for debug
        if self.mode[:5] == 'train':
            self.file_backup()

    def file_backup(self):
        dir_lis = self.conf['general.recording']
        os.makedirs(os.path.join(self.base_exp_dir, 'recording'), exist_ok=True)
        for dir_name in dir_lis:
            cur_dir = os.path.join(self.base_exp_dir, 'recording', dir_name)
            os.makedirs(cur_dir, exist_ok=True)
            files = os.listdir(dir_name)
            for f_name in files:
                if f_name[-3:] == '.py':
                    copyfile(os.path.join(dir_name, f_name), os.path.join(cur_dir, f_name))

        copyfile(self.conf_path, os.path.join(self.base_exp_dir, 'recording', 'config.conf'))

    def load_checkpoint(self, checkpoint_name):
        checkpoint = torch.load(os.path.join(self.base_exp_dir, 'checkpoints', checkpoint_name), map_location=self.device)
        self.nerf_outside.load_state_dict(checkpoint['nerf'])
        self.sdf_network.load_state_dict(checkpoint['sdf_network_fine'])
        self.deviation_network.load_state_dict(checkpoint['variance_network_fine'])
        self.color_network.load_state_dict(checkpoint['color_network_fine'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.iter_step = checkpoint['iter_step']

        logging.info('End')

    def render_novel_image(self, idx_0, idx_1, ratio, resolution_level):
        """
        Interpolate view between two cameras.
        """
        rays_o, rays_d, rot = self.dataset.gen_rays_between(idx_0, idx_1, ratio, resolution_level=resolution_level)
        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
        rays_d = rays_d.reshape(-1, 3).split(self.batch_size)

        out_rgb_fine, out_normal_fine = [], []
        for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
            near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
            background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None

            render_out = self.renderer.render(rays_o_batch,
                                              rays_d_batch,
                                              near,
                                              far,
                                              cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                              background_rgb=background_rgb)

            out_rgb_fine.append(render_out['color_fine'].detach().cpu().numpy())
            
            # normals = normals.sum(dim=1).detach().cpu().numpy()
            out_normal_fine.append(render_out['normals'].detach().cpu().numpy())

            del render_out

        img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3]) * 256).clip(0, 255).astype(np.uint8)
        normal_img = np.concatenate(out_normal_fine, axis=0)
        rot = (rot.detach().cpu().numpy()).T
        
        normal_img_copy = normal_img.reshape([H, W, 3])
        normal_img_copy = normal_img_copy / np.linalg.norm(normal_img_copy, axis=2, keepdims=True)
        normal_img_copy = (normal_img_copy + 1.0) * 0.5 * 255.0
        normal_img_copy = normal_img_copy.astype(np.uint8)
        
        background_color = normal_img_copy[0, 0 :]
        mask = np.all(normal_img_copy == background_color, axis=-1)

        normal_img = np.matmul(rot[None, :, :], normal_img[:, :, None]).reshape([H, W, 3, -1])
        normal_img = normal_img / np.linalg.norm(normal_img, axis=2, keepdims=True)
        normal_img = (normal_img + 1.0) * 0.5 * 255.0
        normal_img = normal_img.astype(np.uint8)
        
        # Set background pixels to 255 (white)
        normal_img = normal_img[..., 0]
        normal_img[mask] = [255, 255, 255]
        
        return img_fine, normal_img
            
    def novel_view_synthesis(self, idx_0, idx_1, ratio, resolution_level):
        img, normal_img = self.render_novel_image(idx_0, idx_1, ratio, resolution_level)

        os.makedirs(os.path.join(self.base_exp_dir, 'novel_view'), exist_ok=True)
        
        cv.imwrite(os.path.join(self.base_exp_dir, 'novel_view', '{:0>8d}_{}_{}.png'.format(self.iter_step, idx_0, idx_1)), img)
        cv.imwrite(os.path.join(self.base_exp_dir, 'novel_view', '{:0>8d}_{}_{}_normal.png'.format(self.iter_step, idx_0, idx_1)), normal_img)
 

if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=FORMAT)

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/base.conf')
    parser.add_argument('--mode', type=str, default='novel_view')
    parser.add_argument('--mcube_threshold', type=float, default=0.0)
    parser.add_argument('--is_continue', default=False, action="store_true")
    parser.add_argument('--checkpoint', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--case', type=str, default='')

    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    runner = Runner(args.conf, args.mode, args.case, args.is_continue)

    _, _, img_idx_0, img_idx_1, ratio = args.mode.split('_')
    img_idx_0 = int(img_idx_0)
    img_idx_1 = int(img_idx_1)
    ratio = float(ratio)
    runner.novel_view_synthesis(img_idx_0, img_idx_1, ratio, resolution_level=1)
