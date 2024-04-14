import cv2
import numpy as np
from pynput import keyboard
from threading import Lock

class CameraTransformationController:
    def __init__(self, extrinsic_params, intrinsic_params):
        self.extrinsic_params = extrinsic_params
        self.intrinsic_params = intrinsic_params
        self.rotation_step = 10  # 회전 단계(도)
        self.translation_step = 10  # 이동 단계(픽셀)
        self.running = True
        self.current_keys = set()
        self.current_keys_lock = Lock()
        self.listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release
        )
        # 가상 카메라 이미지 생성
        self.image = np.zeros((400, 400, 3), dtype=np.uint8)

    def update_camera(self):
        # 카메라 변환 시뮬레이션 및 이미지 업데이트 로직
        print(f"Updated extrinsic parameters: {self.extrinsic_params}")

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
            cv2.destroyAllWindows()
            self.listener.stop()
            return False

    def handle_key_event(self):
        shift_pressed = any([keyboard.Key.shift_l in self.current_keys, keyboard.Key.shift_r in self.current_keys])
        
        if keyboard.Key.left in self.current_keys:
            if shift_pressed:
                # 왼쪽 회전
                self.extrinsic_params['rotation'][1] -= self.rotation_step
            else:
                # 왼쪽 이동
                self.extrinsic_params['translation'][0] -= self.translation_step
        elif keyboard.Key.right in self.current_keys:
            if shift_pressed:
                # 오른쪽 회전
                self.extrinsic_params['rotation'][1] += self.rotation_step
            else:
                # 오른쪽 이동
                self.extrinsic_params['translation'][0] += self.translation_step
        elif keyboard.Key.up in self.current_keys:
            if shift_pressed:
                # 위로 회전
                self.extrinsic_params['rotation'][0] -= self.rotation_step
            else:
                # 위로 이동
                self.extrinsic_params['translation'][1] -= self.translation_step
        elif keyboard.Key.down in self.current_keys:
            if shift_pressed:
                # 아래로 회전
                self.extrinsic_params['rotation'][0] += self.rotation_step
            else:
                # 아래로 이동
                self.extrinsic_params['translation'][1] += self.translation_step
                
        self.update_camera()

    def start(self):
        self.listener.start()
        cv2.imshow('Camera View', self.image)
        while self.running:
            if cv2.waitKey(100) & 0xFF in [ord('q'), 27]:
                self.running = False
        cv2.destroyAllWindows()
        self.listener.join()

# 사용 예시
if __name__ == "__main__":
    extrinsic_params = {
        'translation': [0, 0, 0],  # 카메라 위치
        'rotation': [0, 0, 0]  # 카메라 방향
    }
    intrinsic_params = {
        # 예시적인 intrinsic 파라미터; 실제 사용 시에는 적절한 값으로 설정해야 함
    }
    simulator = CameraTransformationController(extrinsic_params, intrinsic_params)
    simulator.start()

