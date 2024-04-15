import torch
import numpy as np
from pynput import keyboard
from threading import Lock
from run_nerf_helpers import *
from run_nerf import render as nerf_render


class CameraViewer:
    def __init__(self, c2w, hwf, K):
        self.c2w_3_4 = c2w
        self.cam_R = torch.transpose(c2w[:3, :3], 0, 1) # R^T (otrhogonal -> R^{-1} = R^T)
        self.cam_t = -self.cam_R @ c2w[:3, -1] # -R^Tt
        self.delta
        self.K = K
        self.hwf = hwf
        self.translation_step = 3.0  # translation step
        self.rotation_step = np.deg2rad(30)  # rotation step (rad)
        self.running = True
        self.current_keys = set()
        self.current_keys_lock = Lock()
        self.listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release
        )
        print('C2W matrix: ', c2w)
        print('Rotation matrix: ', self.cam_R)
        print('translation matrix: ', self.cam_t)

    def render(self):
        raise NotImplementedError("NeRF Model must implement render method")

    def update_camera(self):
        # print("Updated C2W Matrix:\n", self.c2w.cpu().numpy())
        self.c2w_3_4[:, :3] = torch.transpose(self.cam_R, 0, 1)
        self.c2w_3_4[:, -1] = -torch.transpose(self.cam_R, 0, 1) @ self.cam_t
        print('C2W matrix: ', self.c2w_3_4)
        print('Rotation matrix: ', self.cam_R)
        print('translation matrix: ', self.cam_t)
        print('updated C2W Matrix\n', self.c2w_3_4.cpu().numpy())
        self.render()

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
            return False

    def handle_key_event(self):
        shift_pressed = any([keyboard.Key.shift_l in self.current_keys, keyboard.Key.shift_r in self.current_keys])
        
        forward = torch.tensor([0, 0, 1], utype=torch.float32).cuda()  # forward
        left = torch.tensor([1, 0, 0], dtype=torch.float32).cuda()  # right
        up = torch.tensor([0, 1, 0], dtype=torch.float32).cuda()  # top

        # translation or 
        if shift_pressed:
            # (panning/tilting)
            if keyboard.Key.right in self.current_keys:
                self.apply_rotation(yaw=self.rotation_step)  # rotate right
            elif keyboard.Key.left in self.current_keys:
                self.apply_rotation(yaw=self.rotation_step)  # rotate left
            if keyboard.Key.up in self.current_keys:
                self.apply_rotation(pitch=self.rotation_step)  # rotate up
            elif keyboard.Key.down in self.current_keys:
                self.apply_rotation(pitch=self.rotation_step)  # rotate down
        else:
            # translation
            delta = torch.tensor([0, 0, 0], dtype=torch.float32).cuda()
            if keyboard.Key.right in self.current_keys:
                self.delta -= left  # right
            elif keyboard.Key.left in self.current_keys:
                self.delta += left  # left
            elif keyboard.Key.up in self.current_keys:
                self.delta += forward  # forward
            elif keyboard.Key.down in self.current_keys:
                self.delta -= forward  # backward
            self.apply_translation(delta)

        self.update_camera()

    def apply_translation(self, delta):
        self.cam_t += delta # update camera translation vector


    def apply_rotation(self, yaw=0, pitch=0):
        R_pitch = np.array([
            [np.cos(pitch), -np.sin(pitch), 0],
            [np.sin(pitch), np.cos(pitch), 0],
            [0, 0, 1]
        ])
        R_yaw = np.array([
            [np.cos(yaw), 0, np.sin(yaw)],
            [0, 1, 0],
            [-np.sin(yaw), 0, np.cos(yaw)]
        ])
        self.cam_R = self.cam_R @ R_yaw @ R_pitch # update camera rotation matrix


    def start(self):
        self.listener.start()
        print("Camera simulator started. Press 'q' or 'Esc' to exit.")
        while self.running:
            pass
        self.listener.join()


class NeRFViewer(CameraViewer):
    def render(self):
        nerf_render()