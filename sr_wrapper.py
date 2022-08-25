import yaml
import numpy as np
from model.edsr import edsr
from model import resolve_single

""" Implementation of APIs for super-resolution models

    Example usage (given a video file)
    =================================
    video = np.array(imageio.mimread(video_name))
    video_array = img_as_float32(video)
    video_array = video_array.transpose((0, 3, 1, 2))
    source = video_array[:1, :, :, :]
    target = video_array[1:2, :, :, :]
    
    model = SuperResolutionModel("temp.yaml")
    prediction = predict_with_lr_video(target_lr)
"""
class SuperResolutionModel():
    def __init__(self, config_path, checkpoint='None'):
        super(SuperResolutionModel, self).__init__()
        
        with open(config_path) as f:
            config = yaml.safe_load(f)

        # config parameters
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        generator_params = config['model_params']['generator_params']
        self.shape = config['dataset_params']['frame_shape']
        self.use_lr_video = generator_params.get('use_lr_video', True)
        self.lr_size = generator_params.get('lr_size', 256)
        self.generator_type = generator_params.get('generator_type', 'edsr')
        self.depth = generator_params.get('depth', 16)
        self.scale = int(self.shape[1] / self.lr_size)

        # configure modules
        self.generator = edsr(scale=self.scale, num_res_blocks=self.depth)
        # initialize weights
        if checkpoint == 'None':
            checkpoint = config['checkpoint_params']['checkpoint_path']
        self.generator.load_weights(checkpoint)

        # set to test mode
        self.generator.eval()

        timing_enabled = True
        self.times = []
        self.start = torch.cuda.Event(enable_timing=timing_enabled)
        self.end = torch.cuda.Event(enable_timing=timing_enabled)


    def get_shape(self):
        return tuple(self.shape)


    def get_lr_video_info(self):
        return self.use_lr_video, self.lr_size


    def predict_with_lr_video(self, target_lr):
        """ predict and return the target RGB frame 
            from a low-res version of it. 
        """

        return resolve_single(self.generator, target_lr)

