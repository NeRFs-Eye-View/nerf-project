import torch
import numpy as np
from pynput import keyboard
from threading import Lock
import os, sys
import imageio
import cv2
#from run_nerf_helpers import *

img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)

class CameraViewer:
    def __init__(self, c2w, hwf, K, poses, bds, **kargs):

        # self.device = torch.device("cuda")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.c2w_3_4 = torch.tensor(c2w, dtype=torch.float32, device=self.device)
        self.cam_R = torch.eye(3, dtype=torch.float32, device=self.device)
        self.cam_t = torch.zeros(3, dtype=torch.float32, device=self.device)
        self.cam_yaw = torch.tensor([0.], dtype=torch.float32, device=self.device)
        self.cam_pitch = torch.tensor([0.], dtype=torch.float32, device=self.device)
        self.K = torch.tensor(K, dtype=torch.float32, device=self.device)
        self.hwf = hwf  # Assuming hwf is not used in tensor operations directly
        self.poses = torch.tensor(poses, dtype=torch.float32, device=self.device)
        self.bds = torch.tensor(bds, dtype=torch.float32, device=self.device)
        self.kargs = kargs  # Ensure all tensors in kargs are also moved to GPU
        self.translation_step = .1
        self.rotation_step = 5
        self.N_imgs = 1
        self.running = True
        self.current_keys = set()
        self.current_keys_lock = Lock()
        self.listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release
        )

        print('This cam\'s c2w matrix: ', self.c2w_3_4)


    def render(self, new_view):
        raise NotImplementedError("NeRF Model must implement render method")

    def update_camera(self):
        print("[ Rotation ] yaw: %f pitch: %f" % (self.cam_yaw, self.cam_pitch))
        print("[ Translation ] move to (%f, %f, %f)" % (self.cam_t[0], self.cam_t[1], self.cam_t[2]))

        one_row = torch.tensor([[0, 0, 0, 1]], dtype=torch.float32, device=self.device)
        c2w_affine = torch.cat((self.c2w_3_4, one_row), dim=0)
        transform_affine = torch.cat((torch.cat((self.cam_R, self.cam_t[:, None]), dim=1), one_row), dim=0)
        new_view = torch.matmul(c2w_affine, transform_affine)
        print('[ New View ]\n', new_view)
        self.c2w_3_4 = new_view[:3, :4]


        with torch.no_grad():
            rgb, disp, acc, _ = self.render(new_view, **self.kargs)
            rgb, disp = rgb.cpu().numpy(), disp.cpu().numpy()
            print('Done rendering')
            rgb8 = to8b(rgb)
            rgb8 = to8b(rgb)
            filename = os.path.join('./', '{:03d}.png'.format(self.N_imgs))
            imageio.imwrite(filename, rgb8)
            self.N_imgs += 1

            # bgr8 = cv2.cvtColor(rgb8, cv2.COLOR_RGB2BGR)
            # cv2.imshow('transformed', bgr8)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()



    def on_press(self, key):
        with self.current_keys_lock:
            self.current_keys.add(key)
        self.handle_key_event()

    def on_release(self, key):
        with self.current_keys_lock:
            if key in self.current_keys:
                self.current_keys.remove(key)
        if key == keyboard.Key.esc or key == keyboard.KeyCode.from_char('q'):
            self.running = False
            self.listener.stop()
            print("=========================[ STOP VIEWER ]=========================")
            return False

    def handle_key_event(self):
        shift_pressed = any([keyboard.Key.shift_l in self.current_keys, keyboard.Key.shift_r in self.current_keys])
        
        forward = torch.tensor([0, 0, -self.translation_step], dtype=torch.float32, device=self.device)  # forward
        right = torch.tensor([self.translation_step, 0, 0], dtype=torch.float32, device=self.device)  # right
        # up = np.array([0, 1, 0])  # top

        # translation or 
        if shift_pressed:
            # (panning/tilting)
            with_other_keys = False
            if keyboard.Key.right in self.current_keys:
                self.cam_yaw -= self.rotation_step
                with_other_keys = True
            elif keyboard.Key.left in self.current_keys:
                self.cam_yaw += self.rotation_step
                with_other_keys = True
            if keyboard.Key.up in self.current_keys:
                self.cam_pitch += self.rotation_step
                with_other_keys = True
            elif keyboard.Key.down in self.current_keys:
                self.cam_pitch -= self.rotation_step
                with_other_keys = True

            if with_other_keys:
                self.apply_rotation(self.cam_yaw, self.cam_pitch)
                self.update_camera()

                # self.apply_rotation(pitch=-self.rotation_step)  # rotate down
        else:
            # translation
            delta = torch.tensor([0, 0, 0], dtype=torch.float32, device=self.device)
            if keyboard.Key.right in self.current_keys:
                delta += right  # right
            elif keyboard.Key.left in self.current_keys:
                delta -= right  # left
            elif keyboard.Key.up in self.current_keys:
                delta += forward  # forward
            elif keyboard.Key.down in self.current_keys:
                delta -= forward  # backward
            self.apply_translation(delta)
            self.update_camera()


    def apply_translation(self, delta):
        self.cam_t += delta


    def apply_rotation(self, yaw=0, pitch=0):
        yaw, pitch = torch.deg2rad(yaw), torch.deg2rad(pitch)
        # R_pitch = torch.tensor([
        #     [1, 0, 0],
        #     [0, torch.cos(pitch), torch.sin(pitch)],
        #     [0, -torch.sin(pitch), torch.cos(pitch)],
        R_pitch = torch.tensor([
            [1, 0, 0],
            [0, torch.cos(pitch), -torch.sin(pitch)],
            [0, torch.sin(pitch), torch.cos(pitch)]
        ], dtype=torch.float32)
        R_yaw = torch.tensor([
            [torch.cos(yaw),   0,  torch.sin(yaw) ],
            [0,             1,  0           ],
            [-torch.sin(yaw),  0,  torch.cos(yaw) ]
        ], dtype=torch.float32)
        self.cam_R = R_yaw @ R_pitch # update camera rotation matrix


    def start(self):
        import sys
        import termios
        import atexit
        from select import select

        def disable_echo():
            # 터미널 속성 가져오기
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            
            # atexit으로 프로그램 종료 시 원래 설정 복구
            atexit.register(lambda: termios.tcsetattr(fd, termios.TCSADRAIN, old_settings))
            
            # 현재 설정 변경
            new_settings = termios.tcgetattr(fd)
            new_settings[3] = new_settings[3] & ~termios.ECHO  # ECHO 비활성화
            
            termios.tcsetattr(fd, termios.TCSADRAIN, new_settings)

        disable_echo()
        self.listener.start()
        print("=========================[ START VIEWER ]=========================")
        print("Camera simulator started. Press 'q' or 'Esc' to exit.")
        while self.running:
            pass
        self.listener.join()


class NeRFViewer(CameraViewer):
    """
    The code related to ray and render are from https://github.com/yenchenlin/nerf-pytorch
    """
    def __get_rays(self, H, W, K, c2w):
        i, j = torch.meshgrid(torch.linspace(0, W-1, W, dtype=torch.float32, device=self.device), 
                              torch.linspace(0, H-1, H, dtype=torch.float32, device=self.device), indexing="ij")
        i = i.T
        j = j.T
        f_x, f_y = K[0, 0], K[1, 1]
        c_x, c_y = K[0, 2], K[1, 2]
        dirs = torch.stack([
            (i - c_x) / f_x,
            -(j - c_y) / f_y,
            -torch.ones_like(i, dtype=torch.float32, device=self.device)  # z axis = -1
        ], dim=-1)
        rays_d = torch.reshape(torch.matmul(c2w[:3, :3], dirs[..., None]), (dirs.shape[0], dirs.shape[1], 3))
        rays_o = c2w[:3, -1].expand_as(rays_d)
        return rays_o, rays_d
    

    def __get_closest_pose(self, view, poses, bds):
        """
        calculate closest pose with current updated view using cosine similarity method
        """
        def normalize(v):
            norm = torch.linalg.norm(v, axis=-1, keepdims=True)
            return v / norm

        # poses 배열과 view 벡터의 z-방향
        z_vecs = poses[:, :3, 2]
        z_view = view[:3, 2]

        # 벡터들을 정규화
        z_vecs_normalized = normalize(z_vecs)
        z_view_normalized = normalize(z_view.reshape(1, 3))  # z_view를 2D 배열로 변경
        z_cosine_similarities = (z_vecs_normalized @ z_view_normalized.T).flatten()

        closest_idx = torch.argmax(z_cosine_similarities)
        print('current view: ', view[:3, 2], ', V.S. closest view: ', poses[closest_idx, :, 2])
        return closest_idx

    # def __get_new_bds(self, new_view, pose, near, far):
    #     get_distance = lambda vec1, vec2: torch.sqrt(torch.sum(torch.square(vec1 - vec2)))
    #     dist = get_distance(new_view[:3, 3], pose[:3, 3]) # calculate distance with translation vector
    #     new_view_t = new_view[:, 3]
    #     new_view_z = new_view[:, 2]
    #     pose_t = pose[:, 3]
    #     pose_z = pose[:, 2]

    #     is_inner_new_view = (torch.dot(new_view_t, new_view_z) < 0)
    #     is_inner_pose = (torch.dot(pose_t, pose_z) < 0)

    #     origin = torch.tensor([0, 0, 0], dtype=torch.float32, device=self.device)
    #     dist_from_origin_new_view = get_distance(new_view[:3, 3], origin)
    #     dist_from_origin_pose = get_distance(pose[:3, 3], origin)

    #     """
    #     When both vectors are facing inward and the new_view vector is closer to the origin,
    #     then the distance from new_view to the near point is closer.
    #     두 벡터 모두 안쪽을 보고 있다면, 원점과 가까운 벡터가 near point와의 거리가 더 가깝다. 
    #     두 벡터 모두 바깥 쪽을 보고 있다면, 원점과의 거리가 먼 벡터가 near point와의 거리가 더 가깝다. 
    #     """
    #     adjustment = dist
    #     if is_inner_new_view and is_inner_pose:  # Both vectors are either facing inwards or outwards
    #         adjustment = -dist if (dist_from_origin_new_view < dist_from_origin_pose) else dist
    #     elif not is_inner_new_view and not is_inner_pose:
    #         adjustment = -dist if (dist_from_origin_new_view > dist_from_origin_pose) else dist

    #     near += adjustment
    #     far += adjustment
    #     print('adjustment: ', adjustment)
    #     return near, far
    
    
    def batchify_rays(self, rays_flat, chunk, render_rays, near, far, ndc, **kargs):
        """
        Render rays in smaller minibatches to avoid OOM.
        """
        all_ret = {}
        # rays_flat = torch.tensor(rays_flat, dtype=torch.float32)
        for i in range(0, rays_flat.shape[0], chunk):
            ret = render_rays(rays_flat[i:i+chunk], **kargs)
            for k in ret:
                if k not in all_ret:
                    all_ret[k] = []
                all_ret[k].append(ret[k])
        all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
        return all_ret
    

    def render(self, new_view, use_viewdirs, chunk,**kargs):
        print("nerf viewer!")
        print(self.hwf[0])
        print(self.K)
        # print(super.)
        rays_o, rays_d = self.__get_rays(self.hwf[0], self.hwf[1], self.K, self.c2w_3_4)

        # use viewdirs
        viewdirs = rays_d
        if use_viewdirs:
            viewdirs = viewdirs.reshape((-1, 3))
        ray_shape = rays_d.shape
        rays_o = rays_o.reshape((-1, 3)) # make matrix to list of vectors
        rays_d = rays_d.reshape((-1, 3)) # make matrix to list of vectors
        # print('rays_d after reshape: ', rays_d.shape)

        # print('poses shape: ', self.poses.shape)

        closest_idx = self.__get_closest_pose(new_view, self.poses, self.bds)
        closest_pose = self.poses[closest_idx]
        near, far = self.bds[closest_idx]
        print(new_view.dtype)
        print(closest_pose.dtype)
        print('previous bounds: (%lf, %lf)' % (near, far))
        # near, far = self.__get_new_bds(new_view, closest_pose, near, far)
        # print('current bounds: (%lf, %lf)' % (near, far))
        # self.bds[0] = near
        # self.bds[1] = far

        near = near * torch.ones_like(rays_d[..., :1], dtype=torch.float32, device=self.device) # pixel 개수만큼의 near point
        far = far * torch.ones_like(rays_d[..., :1], dtype=torch.float32, device=self.device) # pixel 개수만큼의 far point

        #! pixel 개수만큼의 rays_o, rays_d, near, far, viewdir
        #? rays dim: (num_pixels, 3+3+1+1+3) = (num_pixels, 11)
        rays = torch.cat([rays_o, rays_d, near, far, viewdirs], -1)
        all_ret = self.batchify_rays(rays, chunk, **kargs)

        print("finished!")

        for k in all_ret:
            k_sh = list(ray_shape[:-1]) + list(all_ret[k].shape[1:])
            all_ret[k] = torch.reshape(all_ret[k], k_sh)

        k_extract = ['rgb_map', 'disp_map', 'acc_map']
        ret_list = [all_ret[k] for k in k_extract]
        ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
        return ret_list + [ret_dict]

        # Render and reshape

    
