import os
from data.base_dataset import BaseDataset, get_transform, transform_normalize, transform_img_and_label, \
    get_img_transform, transform_gamma, transform_tensor, transform_tensor_for_img_label
from data.image_folder import make_dataset
from PIL import Image
import random
import numpy as np
from collections import namedtuple
import torchvision.transforms.functional as trans_f


class UnalignedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """
    ACDCClass = namedtuple('ACDCClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                         'has_instances', 'ignore_in_eval', 'color'])
    classes = [
        ACDCClass('unlabeled', 0, 255, 'void', 0, False, True, (0, 0, 0)),
        ACDCClass('ego vehicle', 1, 255, 'void', 0, False, True, (0, 0, 0)),
        ACDCClass('rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0)),
        ACDCClass('out of roi', 3, 255, 'void', 0, False, True, (0, 0, 0)),
        ACDCClass('static', 4, 255, 'void', 0, False, True, (0, 0, 0)),
        ACDCClass('dynamic', 5, 255, 'void', 0, False, True, (111, 74, 0)),
        ACDCClass('ground', 6, 255, 'void', 0, False, True, (81, 0, 81)),
        ACDCClass('road', 7, 0, 'flat', 1, False, False, (128, 64, 128)),
        ACDCClass('sidewalk', 8, 1, 'flat', 1, False, False, (244, 35, 232)),
        ACDCClass('parking', 9, 255, 'flat', 1, False, True, (250, 170, 160)),
        ACDCClass('rail track', 10, 255, 'flat', 1, False, True, (230, 150, 140)),
        ACDCClass('building', 11, 2, 'construction', 2, False, False, (70, 70, 70)),
        ACDCClass('wall', 12, 3, 'construction', 2, False, False, (102, 102, 156)),
        ACDCClass('fence', 13, 4, 'construction', 2, False, False, (190, 153, 153)),
        ACDCClass('guard rail', 14, 255, 'construction', 2, False, True, (180, 165, 180)),
        ACDCClass('bridge', 15, 255, 'construction', 2, False, True, (150, 100, 100)),
        ACDCClass('tunnel', 16, 255, 'construction', 2, False, True, (150, 120, 90)),
        ACDCClass('pole', 17, 5, 'object', 3, False, False, (153, 153, 153)),
        ACDCClass('polegroup', 18, 255, 'object', 3, False, True, (153, 153, 153)),
        ACDCClass('traffic light', 19, 6, 'object', 3, False, False, (250, 170, 30)),
        ACDCClass('traffic sign', 20, 7, 'object', 3, False, False, (220, 220, 0)),
        ACDCClass('vegetation', 21, 8, 'nature', 4, False, False, (107, 142, 35)),
        ACDCClass('terrain', 22, 9, 'nature', 4, False, False, (152, 251, 152)),
        ACDCClass('sky', 23, 10, 'sky', 5, False, False, (70, 130, 180)),
        ACDCClass('person', 24, 11, 'human', 6, True, False, (220, 20, 60)),
        ACDCClass('rider', 25, 12, 'human', 6, True, False, (255, 0, 0)),
        ACDCClass('car', 26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
        ACDCClass('truck', 27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
        ACDCClass('bus', 28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
        ACDCClass('caravan', 29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
        ACDCClass('trailer', 30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
        ACDCClass('train', 31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
        ACDCClass('motorcycle', 32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
        ACDCClass('bicycle', 33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
        ACDCClass('license plate', -1, 255, 'vehicle', 7, False, True, (0, 0, 142)),
    ]
    train_id_to_color = [c.color for c in classes if (c.train_id != -1 and c.train_id != 255)]
    train_id_to_color.append([0, 0, 0])
    train_id_to_color = np.array(train_id_to_color)
    id_to_train_id = np.array([c.train_id for c in classes])

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image
        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))

        self.transform_Normalize = transform_normalize()
        self.transform_img_label = transform_img_and_label(self.opt)
        self.transform_img = get_img_transform(self.opt)

        self.transform_A_gamma = transform_gamma(self.opt)
        self.transform_A_gamma_tensor = transform_tensor_for_img_label(self.opt)
        self.transform_A_tensor_only = transform_tensor(self.opt)

    @classmethod
    def encode_target(cls, target):
        return cls.id_to_train_id[np.array(target)]

    @classmethod
    def decode_target(cls, target):
        target[target == 255] = 19
        return cls.train_id_to_color[target]

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')

        label_path_prefix = '/home/datasets/ACDC/gt_trainval/gt/night/trainval'
        label_suffix = 'gt_labelIds.png'
        img_name = A_path.split('/')[-1]
        file_name = img_name.split('_')[0]
        img_prefixs = img_name.split('_', 3)
        label_prefix = ''
        for i in range(3):
            label_prefix = label_prefix + img_prefixs[i] + '_'
        label_prefix = label_prefix + label_suffix
        label_path = os.path.join(label_path_prefix, file_name, label_prefix)
        A_label = Image.open(label_path)

        B = self.transform_img(B_img)  # unpaired

        # gamma correction
        A_img, A_label = self.transform_A_gamma(A_img, A_label)  # unpaired
        A = trans_f.adjust_gamma(A_img, gamma=0.5)
        A, A_target = self.transform_A_gamma_tensor(A, A_label)
        A_old = self.transform_A_tensor_only(A_img)

        # A, A_target = self.transform_img_label(A_img, A_label)
        real_A_target = self.encode_target(A_target)
        A = self.transform_Normalize(A)
        A_old = self.transform_Normalize(A_old)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path, 'real_A_target': real_A_target, 'A_old': A_old}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
