import cv2
import math
import numpy as np
import torch
from torch import nn
from torchvision import transforms as T
from copy import deepcopy
import torch
from itertools import cycle
from einops import rearrange

class Keypoints2Image:
    def __init__(self, mode='openpose_body_25', image_shape=(256, 256), background_white=False):

        self.background_white = background_white
        self.height = image_shape[0]
        self.width = image_shape[1]
        
        bgr_colors =cycle([(255,0,0), 
            (255,165,0),
            (218,165,32),
            (255,255,0),
            (0,255,0),
            (144,238,133),
            (144,238,133),
            (255,0,0),
            (124,252,0),
            (144,238,144),
            (135,206,235),
            (30,144,255),
            (128,0,128),
            (128,0,128),
            (255,0,255),
            (255,0,255),
            (75,0,130),
            (75,0,130),])        
        
        if mode == 'openpose_body_25':
            edges = [(0,1), (1,2), (2,3), (3,4), (1,5), (5,6), \
                      (6,7), (1,8), (8,9,), (9,10), (10,11), (8,12), (12,13), \
                      (13,14), (0, 15), (15, 17), (0,16), (16,18)]
            self.segments = [(edge, next(bgr_colors)) for edge in edges]
            
        elif mode == 'mediapipe':
            edges = [(8,6), (6,5), (5,4), (4,0), (0,1), (1,2), (2,3), (3,7), (9,10), \
                     (18,20), (16,18), (16,20), (16,22), (14,22), (12,14), (11,12), \
                     (11,13), (13,15), (15,21), (15,17), (17,19), (15,19), (12,24), \
                     (23,24), (11,23), (23,25), (25,27), (27,29), (27,31), (29,31), \
                     (24,26), (26,28), (28,32), (28,30), \
                     (30,32),  ]
            self.segments = []
            count = 0
            for edge in edges:
                if count % 2 == 0:
                    color = next(bgr_colors)
                count += 1
                self.segments.append((edge, color))
        else:
            raise ValueError(f"Invalid mode f{mode}")
            
            
    def _get_coords(self, keypoint):
        x = int(keypoint[0] * self.width)
        y = int(keypoint[1] * self.height)
        return tuple((x, y))
    
    def __call__(self, keypoints, threshold = 0.0):
        if self.background_white:
            img = 255 * np.ones((self.height, self.width, 3), np.uint8)
        else:
            img = np.zeros((self.height, self.width, 3), np.uint8)
        for person in keypoints:
            for points, color in self.segments:
                kp1 = person[points[0]]
                kp2 = person[points[1]]
                if kp1[-1] > threshold and kp2[-1] > threshold:
                    cv2.line(img, self._get_coords(kp1), 
                             self._get_coords(kp2), color, 2)

        
        #if type(keypoints) == torch.Tensor:
        img = img/255.
        img = T.ToTensor()(img.astype(np.float32))#.to(keypoints.device)
        #else:
        #    img = img.astype(np.uint8)
        return img

def keypoints_to_heatmap(keypoints,
                         threshold = 0.2, 
                         fraction=False, 
                         image_shape=(256, 256),
                         sigma=4.):
    
    height, width = image_shape[:2]
    heatmap = np.zeros((len(keypoints), height, width), np.float32)
    
    for i, kp in enumerate(keypoints):
        if kp[-1] <= threshold:
            continue
        center_x, center_y = kp[0] * height, kp[1] * width
        if fraction:
            center_x = int(center_x * width)
            center_y = int(center_y * height)

        th = 1.6052
        delta = math.sqrt(th * 2)
        x0 = int(max(0, center_x - delta * sigma))
        y0 = int(max(0, center_y - delta * sigma))

        x1 = int(min(width, center_x + delta * sigma))
        y1 = int(min(height, center_y + delta * sigma))

        # gaussian filter
        for y in range(y0, y1):
            for x in range(x0, x1):
                d = (x - center_x) ** 2 + (y - center_y) ** 2
                exp = d / 2.0 / sigma / sigma
                if exp > th:
                    continue
                heatmap[i][y][x] = max(heatmap[i][y][x], math.exp(-exp))
                heatmap[i][y][x] = min(heatmap[i][y][x], 1.0)
    return heatmap

def heatmap_to_image(heatmaps):
    x = heatmaps.sum(axis=0)
    x/=x.max()
    return torch.unsqueeze(x, axis=0).repeat(3,1,1)

def heatmap_to_skeleton(heatmaps):

    keypoints = []
    for heatmap in heatmaps:
        coords = list(np.squeeze((heatmap==torch.max(heatmap)).nonzero().detach().cpu().numpy()))[::-1]
        if len(coords)==2:
            coords.append(1.0)
            keypoints.append(coords)
        else:
            keypoints.append([0 ,0, 0.])
    skeleton_img = keypoints_to_image(keypoints, fraction=False)

    #heatmap_img = heatmap_to_image(heatmaps)
    #mix_img = 0.3*heatmap_img + 0.7*skeleton_img
    #return mix_img
    return skeleton_img.to(heatmaps.device)

class PoseVisualizer:
    def __init__(self, pose_format, image_shape=(256, 256), background_white=False):
        self.pose_format = pose_format
        if self.pose_format == 'image':
            self.fn = lambda x: x
        elif self.pose_format == 'heatmap':
            self.fn = lambda x: heatmap_to_skeleton(x[0])
        elif self.pose_format == 'keypoint':
            kp2im = Keypoints2Image('openpose_body_25', image_shape, background_white)
            self.fn = lambda x : kp2im(x)
        else:
            raise(ValueError)

    def convert(self, x):
        return self.fn(x)

    
class RandomRotateScale(object):
    def __init__(self, angle_degree=(0., 0.), scale=(1,1)):
        self.angle_degree = angle_degree
        self.scale = scale
    
    def __call__(self, sample):
        image, keypoints = sample['image'], sample['keypoints']
        # get random degree and scale
        angle = np.random.uniform(self.angle_degree[0], self.angle_degree[1])
        scale = np.random.uniform(self.scale[0], self.scale[1])
        # rotate image
        height, width = image.shape[:2]
        center = (width/2, height/2)
        rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=angle, scale=scale)
        rotated_image = cv2.warpAffine(src=image, M=rotate_matrix, dsize=(width, height))
        # rotate keypoint
        
        kp_ = deepcopy(keypoints)
        kp_[:,2] = 1.
        center = (0.5, 0.5)
        rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=angle, scale=scale)
        
        new_kp = np.dot(kp_, rotate_matrix.transpose())
        new_kp = np.concatenate((new_kp, np.expand_dims(keypoints[:,2].transpose(), 1)), axis=1)
        
        return {'image':rotated_image, 'keypoints':new_kp.astype(np.float32)}

class RandomCrop(object):
    
    def __init__(self, margins=(0.05, 1.)):
        self.margin = margins
        
    def __call__(self, sample):
        image, keypoints = sample['image'], sample['keypoints']
        kps = keypoints.copy()
        height, width = image.shape[:2]
        
        left_x, top_y, right_x = np.random.uniform(self.margin[0], self.margin[1], size=3)
        right_x = 1 - right_x
        
        crop_h = crop_w = right_x - left_x
        if top_y + crop_h > 1:
            crop_h = crop_w = 1 - top_y

        right_x = left_x + crop_w
        bottom_y = top_y + crop_h

        # crop keypoints
        kps[:,0] = (kps[:,0] - left_x)/crop_w
        kps[:,1] = (kps[:,1] - top_y)/crop_h

        x_indices = np.where(np.logical_and(kps[:,0]<0, kps[:,0]>1.))[0]
        y_indices = np.where(np.logical_and(kps[:,1]<0, kps[:,1]>1.))[0]
        kps[list(set(y_indices) | set(x_indices))] = [0., 0., 0.]
        
        # crop images
        left_x = int(left_x * width)
        top_y = int(top_y * height)
        right_x = left_x + int(width*crop_w)
        bottom_y = top_y + int(height*crop_h)
        crop_image = image[top_y:bottom_y, 
                           left_x:right_x, :]
        crop_image = cv2.resize(crop_image, (width, height), interpolation=cv2.INTER_AREA)

        return {'image':crop_image, 'keypoints':kps}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        image, keypoints = sample['image'], sample['keypoints']
        image = image.transpose((2, 0, 1))
        image = image.astype(np.float32)/255.
        return {'image':torch.from_numpy(image), \
                'keypoints': torch.from_numpy(keypoints)}

class ConcatSamples(object):    
    def __call__(self, sample):
        images, keypoints = sample['image'], sample['keypoints']
        kps = keypoints.copy()
        h, w, _ = images[0].shape
        left_half = images[0][:,int(0.25*h):int(0.75*h),:]
        right_half = images[1][:,int(0.25*h):int(0.75*h),:]
        combined_image = np.hstack((left_half, right_half))

        kps[0] = [[max(x-0.25, 0), y, c] for x, y, c in kps[0]]
        kps[1] = [[min(x+0.25, 1), y, c] for x, y, c in kps[1]]

        return {'image':combined_image, 'keypoints':kps}

class CenterCropResize(object):
    
    def __init__(self, image_shape=(256, 256)):
        self.image_shape = image_shape
        
    def __call__(self, sample):
        image, keypoints = sample['image'], sample['keypoints']
        kps = np.array(keypoints.copy())
        
        height, width = image.shape[:2]
        new_height, new_width = height, width
        if width > height:
            left_margin = (width - height)/2/width
            right_margin = 1 - left_margin
            top_margin = 0.
            bottom_margin = 1.
            new_width = height
        elif height > width:
            left_margin = 0.
            right_margin = 1.
            top_margin = (height - width)/2/height
            bottom_margin = 1 - top_margin
            new_height = width
            
        # crop keypoints
        crop_w = new_width/width
        crop_h = new_height/height
        kps[:,:,0] = (kps[:,:,0] - left_margin)/crop_w
        kps[:,:,1] = (kps[:,:,1] - top_margin)/crop_h
        x_indices = np.where(np.logical_and(kps[:,:,0]<0, kps[:,:,0]>1.))[0]
        y_indices = np.where(np.logical_and(kps[:,:,1]<0, kps[:,:,1]>1.))[0]
        kps[:,list(set(y_indices) | set(x_indices))] = [0., 0., 0.]
        
        # crop images
        left_x = int(left_margin*width)
        top_y = int(top_margin*height)
        right_x = int(right_margin*width)
        bottom_y = int(bottom_margin*height)
        crop_image = image[top_y:bottom_y, 
                           left_x:right_x, :]
        crop_image = cv2.resize(crop_image, self.image_shape, interpolation=cv2.INTER_AREA)

        return {'image':crop_image, 'keypoints':kps}

def pad_keypoints(keypoints, max_num, num_keypoints=25):
    num_person = keypoints.shape[0]
    return np.insert(keypoints, tuple((max_num - num_person)*[num_person]), 0, 0)
'''
def pad_keypoints(keypoints, max_num, num_keypoints=25):
    padded = np.zeros((max_num, num_keypoints, 3), dtype=np.float32)
    for i, keypoint in enumerate(keypoints):
        if i >= max_num:
            break
        padded[i] = keypoint.astype(np.float32)
    return padded
'''
'''
class KPE(nn.Module):
    def __init__(self, max_num_people, d_model, projection=None):
        self.max_num_people = max_num_people
        super().__init__()
        if projection == 'linear':
            self.layer = nn.Linear(max_num_people, d_model)
        elif projection == None:
            assert 3*max_num_people <= d_model
            self.layer = lambda x: x
        else:
            raise ValueError(f"Invalid projection of ({projection})")
    
    def forward(self, x):
        return self.layer(x)

    def preprocess(self, keypoints:np.array):    # keyoints of single person
        num_person = keypoints.shape[0]
        repeats = tuple((self.max_num_people - num_person)*[num_person])
        return np.insert(keypoints, repeats, 0, 0)
'''

class KPE:
    def __init__(self, max_num_people):
        super().__init__()
        self.max_num_people = max_num_people
        self.num_keypoints = 25

    '''
        assume single sample, no batch dimension
    '''
    def __call__(self, keypoints:np.array):    # keyoints of single person

        num_person = keypoints.shape[0]
        repeats = tuple((self.max_num_people - num_person)*[num_person])
        padded_keypoints = np.insert(keypoints, repeats, 0, 0)
        return rearrange(padded_keypoints, 'a b c -> b (a c)')

    def decode_single(self, torch_tokens):
        tokens = torch_tokens.cpu().detach()
        num_people = tokens.shape[1]//3
        num_kp = tokens.shape[0]
        result = np.zeros((num_people, num_kp, 3))
        for p in range(num_people):
            for i in range(self.num_keypoints):
                result[p,i,:] = tokens[i][3*p:3*(p+1)]
                
        valid = np.mean(result, axis=(1,2))!=0
        return result[valid]

    def decode(self, tokens): # batch
        return [self.decode_single(t) for t in tokens]