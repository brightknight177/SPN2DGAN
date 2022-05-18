from data.base_dataset import BaseDataset, transform_resize
from data.image_folder import make_dataset
from PIL import Image
from data.base_dataset import transform_normalize, transform_img_and_label, \
    get_img_transform, transform_tensor, transform_resize_only, transform_toTenandNormal
import torchvision.transforms.functional as trans_f


class SingleDataset(BaseDataset):
    """This dataset class can load a set of images specified by the path --dataroot /path/to/data.

    It can be used for generating CycleGAN results only for one side with the model option '-model test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.A_paths = sorted(make_dataset(opt.dataroot, opt.max_dataset_size))
        input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        # self.transform = get_transform(opt, grayscale=(input_nc == 1))
        self.transform = transform_resize(self.opt)

        self.transform_Normalize = transform_normalize()
        self.transform_img_label = transform_img_and_label(self.opt)

        self.transform_resize_only = transform_resize_only(self.opt)
        self.transform_tensor_only = transform_tensor(self.opt)
        self.transform_TenAndNor = transform_toTenandNormal(self.opt)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A and A_paths
            A(tensor) - - an image in one domain
            A_paths(str) - - the path of the image
        """
        A_path = self.A_paths[index]
        A_img = Image.open(A_path).convert('RGB')
        A_img = self.transform_resize_only(A_img)
        A = trans_f.adjust_gamma(A_img, gamma=0.5)
        A = self.transform_TenAndNor(A)
        A_old = self.transform_TenAndNor(A_img)

        return {'A': A, 'A_paths': A_path, 'A_old': A_old}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.A_paths)
