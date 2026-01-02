import json

# camera_registry is at config/camera_registry.json
# cameras_runtime is at config/cameras_runtime.json

# create a camera_controller class
class CameraController:
    def __init__(self):
        # check if the camera files exist
        try:
            with open('config/camera_registry.json', 'r') as file:
                self.camera_registry = json.load(file)
            with open('config/cameras_runtime.json', 'r') as file:
                self.cameras_runtime = json.load(file)
        except FileNotFoundError:
            raise FileNotFoundError("Camera registry/runtime files not found")
        
        # create a dictionary to store the camera objects
        self.cameras = {}
        for camera_id in self.cameras_runtime:
            self.cameras[camera_id] = Camera(camera_id, self.cameras_runtime[camera_id])


    def get_camera_info(self, camera_id):
        if camera_id in self.camera_registry:
            return self.camera_registry[camera_id]
        else:
            return None
        
# camera class
class Camera:
    def __init__(self, camera_id, camera_info):
        self.id = camera_id
        self.mac = camera_info['mac']
        self.ip = camera_info['ip']
        self.resolution = camera_info['resolution']
        self.rtsp = camera_info['rtsp']

