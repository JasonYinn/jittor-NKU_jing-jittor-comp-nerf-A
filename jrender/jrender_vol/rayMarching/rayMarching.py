import numpy as np
import jittor as jt
from jittor import nn
from ..integrator import *

def render_rays_v2(ray_batch,
                network_fn,
                network_query_fn,
                N_samples,
                retraw=False,
                lindisp=False,
                perturb=0.,
                N_importance=0,
                network_fine=None,
                white_bkgd=False,
                raw_noise_std=0.,
                verbose=False,
                **kwargs):
    """Volumetric rendering.
    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_fn: function. Model for predicting RGB and density at each point
        in space.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      network_fine: "fine" network with same spec as network_fn.
      white_bkgd: bool. If True, assume a white background.
      raw_noise_std: ...
      verbose: bool. If True, print more debugging info.
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] # [N_rays, 3] each
    viewdirs = ray_batch[:,-3:] if ray_batch.shape[-1] > 8 else None
    bounds = jt.reshape(ray_batch[...,6:8], [-1,1,2])
    near, far = bounds[...,0], bounds[...,1] # [-1,1]

    far_factor = kwargs.get('far_factor')
    z_vals = sample(N_rays, N_samples, lindisp, perturb, near, far)
    z_vals2 = sample(N_rays, N_samples, lindisp, perturb, near, far*far_factor) #ship,coffe:0.5 car:1.1
    pts = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * z_vals.unsqueeze(-1) # [N_rays, N_samples, 3]
    pts2 = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * z_vals2.unsqueeze(-1)
    pts = jt.concat([pts, pts2], dim=-1)#embed_fn, input_ch = get_embedder(args.multires, args.i_embed, dim=6) 

    raw = network_query_fn(pts, viewdirs, network_fn)
    rgb_map, disp_map, acc_map, weights, depth_map = integrator(raw, z_vals, rays_d, raw_noise_std, white_bkgd)

    rgb_map_0, disp_map_0, acc_map_0, weights0 = rgb_map, disp_map, acc_map, weights
    #usefulRayIndex = jt.nonzero(acc_map > 0.1)
    if  N_importance > 0:
        # importance sampling
        z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], N_importance, det=(perturb==0.))
        z_samples = z_samples.detach()
        _, z_vals = jt.argsort(jt.concat([z_vals, z_samples], -1), -1)
        pts = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * z_vals.unsqueeze(-1) # [N_rays, N_samples + N_importance, 3]
        
        z_vals_mid2 = .5 * (z_vals2[...,1:] + z_vals2[...,:-1])
        z_samples2 = sample_pdf(z_vals_mid2, weights[...,1:-1], N_importance, det=(perturb==0.))
        z_samples2 = z_samples2.detach()
        _, z_vals2 = jt.argsort(jt.concat([z_vals2, z_samples2], -1), -1)
        pts2 = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * z_vals2.unsqueeze(-1) # [N_rays, N_samples + N_importance, 3]
        pts = jt.concat([pts, pts2], dim=-1)

        run_fn = network_fn if network_fine is None else network_fine
        raw = network_query_fn(pts, viewdirs, run_fn)
        rgb_map, disp_map, acc_map, weights, depth_map = integrator(raw, z_vals, rays_d, raw_noise_std, white_bkgd)
        
        # w_err = n_err.detach()
        # wets = jt.misc.split(w_err,1,dim=1)
        # num = weights0.shape[1]
        # ratio = len(wets)
        # weights0 = weights0 + 1e-5
        # wei0s = jt.misc.split(weights0,int(num//ratio),dim=1)
        # wet = [jt.nn.relu(wei0s[i]-wets[i].expand_as(wei0s[i])) for i in range(len(wets))]
        # wet = jt.concat(wet,dim=1)
        # wet = wet**2
        # n_err0  = jt.sum(wet,-1)

    ret = {'rgb_map' : rgb_map, 'disp_map' : disp_map, 'acc_map' : acc_map}
    if retraw:
        ret['raw'] = raw
    if N_importance > 0:
        ret['rgb0'] = rgb_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0
        # ret['ne0'] = n_err0

    return ret

def render_rays_v1(ray_batch,
                network_fn,
                network_query_fn,
                N_samples,
                retraw=False,
                lindisp=False,
                perturb=0.,
                N_importance=0,
                network_fine=None,
                white_bkgd=False,
                raw_noise_std=0.,
                verbose=False,
                **kwargs):
    """Volumetric rendering.
    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_fn: function. Model for predicting RGB and density at each point
        in space.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      network_fine: "fine" network with same spec as network_fn.
      white_bkgd: bool. If True, assume a white background.
      raw_noise_std: ...
      verbose: bool. If True, print more debugging info.
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] # [N_rays, 3] each
    viewdirs = ray_batch[:,-3:] if ray_batch.shape[-1] > 8 else None
    bounds = jt.reshape(ray_batch[...,6:8], [-1,1,2])
    near, far = bounds[...,0], bounds[...,1] # [-1,1]

    # [N_rays, N_samples]
    z_vals = sample(N_rays, N_samples, lindisp, perturb, near, far)
    pts = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * z_vals.unsqueeze(-1) # [N_rays, N_samples, 3]
    if kwargs.get('embed_depth', False):
      pts = jt.concat([pts, z_vals.unsqueeze(-1)], dim=-1)
    if kwargs.get('embed_radius', False):
      pts = jt.concat([pts, jt.norm(pts[..., :3], dim=-1, keepdim=True)], dim=-1)
    
    if kwargs.get('model', 'NeRF') == 'OffsetNeRF':
      offset_dir = network_query_fn(pts, viewdirs, network_fn)
      offset_dir = offset_dir / (jt.norm(offset_dir, dim=-1, keepdim=True) + 1e-6)
      offset = offset_dir * z_vals.unsqueeze(-1)
      offset_pts = pts[..., :3] + offset
      if kwargs.get('embed_depth', False):
        offset_pts = jt.concat([offset_pts, z_vals.unsqueeze(-1)], dim=-1)
      if kwargs.get('embed_radius', False):
        offset_pts = jt.concat([offset_pts, jt.norm(offset_pts[..., :3], dim=-1, keepdim=True)], dim=-1)
      raw = network_query_fn(offset_pts, viewdirs, network_fine)
      
      rgb_map, disp_map, acc_map, weights, depth_map = integrator(raw, z_vals, rays_d, raw_noise_std, white_bkgd)
      rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map

    else:
      raw = network_query_fn(pts, viewdirs, network_fn)
      # if kwargs.get('model', 'NeRF') == 'OpacityNeRF':
      #   rgb_map, disp_map, acc_map, weights, depth_map = integrator_opacity(raw, z_vals, rays_d, raw_noise_std, white_bkgd)
      # else:
      rgb_map, disp_map, acc_map, weights, depth_map = integrator(raw, z_vals, rays_d, raw_noise_std, white_bkgd)

      rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map
      #usefulRayIndex = jt.nonzero(acc_map > 0.1)
      if  N_importance > 0:
          # importance sampling
          z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
          z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], N_importance, det=(perturb==0.))
          z_samples = z_samples.detach()

          _, z_vals = jt.argsort(jt.concat([z_vals, z_samples], -1), -1)
          pts = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * z_vals.unsqueeze(-1) # [N_rays, N_samples + N_importance, 3]
          
          run_fn = network_fn if network_fine is None else network_fine
          if kwargs.get('embed_depth', False):
            pts = jt.concat([pts, z_vals.unsqueeze(-1)], dim=-1)
          if kwargs.get('embed_radius', False):
            pts = jt.concat([pts, jt.norm(pts[..., :3], dim=-1, keepdim=True)], dim=-1)
          raw = network_query_fn(pts, viewdirs, run_fn)
          rgb_map, disp_map, acc_map, weights, depth_map = integrator(raw, z_vals, rays_d, raw_noise_std, white_bkgd)
        
    ret = {'rgb_map' : rgb_map, 'disp_map' : disp_map, 'acc_map' : acc_map}
    if retraw:
        ret['raw'] = raw
    if N_importance > 0:
        ret['rgb0'] = rgb_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0

    return ret
