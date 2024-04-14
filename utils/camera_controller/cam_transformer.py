import torch
import numpy as np
from pynput import keyboard
from threading import Lock
from run_nerf_helpers import *

class CameraTransformer:
    def __init__(self, c2w_matrix, K):
        self.c2w_matrix = c2w_matrix.cuda()  # C2W 변환 행렬을 GPU로 이동
        self.yaw = 0
        self.pitch = 0
        self.delta
        self.K = K
        self.translation_step = 1.0  # translation step
        self.rotation_step = np.deg2rad(30)  # rotation step (rad)
        self.running = True
        self.current_keys = set()
        self.current_keys_lock = Lock()
        self.listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release
        )

    def update_camera(self):
        print("Updated C2W Matrix:\n", self.c2w_matrix.cpu().numpy())
        hwf = c2w[:, 4]
        H, W, f = hwf[0], hwf[1], hwf[2]
        rays_o, rays_d = get_rays(self.c2w)
        # call render()


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
        
        forward = torch.tensor([0, 0, 1], dtype=torch.float32).cuda()  # forward
        right = torch.tensor([1, 0, 0], dtype=torch.float32).cuda()  # right
        up = torch.tensor([0, 1, 0], dtype=torch.float32).cuda()  # top

        # translation or 
        if shift_pressed:
            # (panning/tilting)
            if keyboard.Key.right in self.current_keys:
                self.apply_rotation(yaw=self.yaw - self.rotation_step)  # rotate right
            elif keyboard.Key.left in self.current_keys:
                self.apply_rotation(yaw=self.yaw + self.rotation_step)  # rotate left
            if keyboard.Key.up in self.current_keys:
                self.apply_rotation(pitch=self.pitch + self.rotation_step)  # rotate up
            elif keyboard.Key.down in self.current_keys:
                self.apply_rotation(pitch=self.pitch -self.rotation_step)  # rotate down
        else:
            # translation
            delta = torch.tensor([0, 0, 0], dtype=torch.float32).cuda()
            if keyboard.Key.right in self.current_keys:
                self.delta += right  # right
            elif keyboard.Key.left in self.current_keys:
                self.delta -= right  # left
            if keyboard.Key.up in self.current_keys:
                self.delta += forward  # forward
            elif keyboard.Key.down in self.current_keys:
                self.delta -= forward  # backward

            self.cam_t += delta
            self.apply_translation(delta)

        self.update_camera()

    def apply_translation(self, direction):
        self.c2w[:, 3] += direction


    def apply_rotation(self, yaw=0, pitch=0):
        cam_R = np.array([
            [np.cos(yaw), np.sin(yaw), 0], 
            [-np.sin(yaw)*np.sin(pitch), np.cos(yaw) * np.sin(pitch), -np.cos(pitch)],
            [-np.sin(yaw)*np.cos(pitch), np.cos(yaw) * np.cos(pitch), np.sin(pitch)]
        ])
        self.c2w[:, :3] = self.c2w[:, :3] @ self.cam_R



    def start(self):
        self.listener.start()
        print("Camera simulator started. Press 'q' or 'Esc' to exit.")
        while self.running:
            pass
        self.listener.join()