import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from data.unaligned_dataset import UnalignedDataset
from models.segment_base import deeplabv3plus_resnet50
import kornia


class CycleGANModel(BaseModel):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
        Dropout is not used in the original CycleGAN paper.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')

        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B', 'S_A',
                           'sobel_A', 'sobel_B', 'P_A', 'P_B']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_B', 'rec_A', 'real_A_target', 'fake_B_pred', 'pred_real_A_mask', 'static_mask_A', 'real_A_old']
        visual_names_B = ['real_B', 'fake_A', 'rec_B', 'real_B_pred']
        if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, gen='dual')
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, gen='normal')

        if self.isTrain:  # define discriminators
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netS_A = deeplabv3plus_resnet50(num_classes=19, output_stride=16).to(self.device)
            self.netM_A = deeplabv3plus_resnet50(num_classes=19, output_stride=16).to(self.device)
            # load maskSegNet ckpt
            pretrained_maskNet = torch.load("./pre_model/best_deeplabv3plus_resnet50_acdc_os16.pth")
            self.netM_A.load_state_dict(pretrained_maskNet["model_state"])
            for p in self.netM_A.parameters():
                p.requires_grad = False
            self.netM_A.eval()
            # load FakeBSegNet ckpt
            pretrained_net = torch.load("./pre_model/best_deeplabv3plus_resnet50_cityscapes_os16.pth")
            self.netS_A.load_state_dict(pretrained_net["model_state"])
            for q in self.netS_A.parameters():
                q.requires_grad = False
            self.netS_A.eval()

        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            self.criterionCE = torch.nn.CrossEntropyLoss(ignore_index=255, reduction='mean')  # add the ce loss
            self.criterionBoundary = torch.nn.L1Loss()  # add the boundary loss
            self.criterionVGG = networks.VGGLoss_for_trans(gpu_id=self.device)
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(
                itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

        self.real_A_target = input['real_A_target'].to(device=self.device, dtype=torch.int64)
        self.real_A_old = input['A_old'].to(self.device)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        upsample_256 = torch.nn.Upsample(size=[256, 256], mode='bilinear')

        pred_A_mask = upsample_256(self.netM_A(self.real_A_old))
        pred_A_mask = pred_A_mask.detach().max(dim=1)[1]

        self.pred_real_A_mask = UnalignedDataset.decode_target(pred_A_mask.cpu())
        self.pred_real_A_mask = torch.Tensor(self.pred_real_A_mask.transpose(0, 3, 1, 2))
        pred_A_mask[pred_A_mask < 2] = 1
        pred_A_mask[pred_A_mask >= 2] = 0
        self.static_mask = pred_A_mask

        self.fake_B = self.netG_A(self.real_A, self.static_mask)  # G_A(A)
        self.rec_A = self.netG_B(self.fake_B)   # G_B(G_A(A))
        self.fake_A = self.netG_B(self.real_B)  # G_B(B)

        pred = upsample_256(self.netS_A(self.real_B))
        pred_B = pred.detach().max(dim=1)[1]
        self.real_B_pred = UnalignedDataset.decode_target(pred_B.cpu())
        self.real_B_pred = torch.Tensor(self.real_B_pred.transpose(0, 3, 1, 2))
        pred_B[pred_B < 2] = 1
        pred_B[pred_B >= 2] = 0
        self.static_mask_B = pred_B
        self.rec_B = self.netG_A(self.fake_A, self.static_mask_B)   # G_A(G_B(B))

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity  # ??????????????????
        lambda_A = self.opt.lambda_A  # ??????????????????
        lambda_B = self.opt.lambda_B  # ??????????????????
        lambda_sobel = 1.0
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A = self.netG_A(self.real_B, self.static_mask_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # segmentation loss
        upsample_256 = torch.nn.Upsample(size=[256, 256], mode='bilinear')
        S_A_pred = upsample_256(self.netS_A(self.fake_B))
        self.loss_S_A = self.criterionCE(S_A_pred, self.real_A_target)
        self.real_A_target = UnalignedDataset.decode_target(self.real_A_target.cpu())
        self.real_A_target = torch.Tensor(self.real_A_target.transpose(0, 3, 1, 2))
        self.fake_B_pred = UnalignedDataset.decode_target(S_A_pred.detach().max(dim=1)[1].cpu())
        self.fake_B_pred = torch.Tensor(self.fake_B_pred.transpose(0, 3, 1, 2))
        self.static_mask_A = UnalignedDataset.decode_target(self.static_mask.detach().cpu())
        self.static_mask_A = torch.Tensor(self.static_mask_A.transpose(0, 3, 1, 2))

        # edge loss
        edge_real_A = kornia.spatial_gradient(self.real_A)
        edge_real_B = kornia.spatial_gradient(self.real_B)
        edge_fake_A = kornia.spatial_gradient(self.fake_A)
        edge_fake_B = kornia.spatial_gradient(self.fake_B)
        self.loss_sobel_A = self.criterionBoundary(edge_fake_B, edge_real_A) * lambda_sobel
        self.loss_sobel_B = self.criterionBoundary(edge_fake_A, edge_real_B) * lambda_sobel

        # VGG perceptal loss for content and style
        self.loss_P_A = self.criterionVGG(self.fake_B, self.real_A, self.real_B, weights=[0, 0, 0, 1.0 / 4, 1.0])
        self.loss_P_B = self.criterionVGG(self.fake_A, self.real_B, self.real_A, weights=[0, 0, 0, 1.0 / 4, 1.0])

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A \
                      + self.loss_idt_B + self.loss_S_A + self.loss_sobel_A + self.loss_sobel_B + \
                      self.loss_P_A + self.loss_P_B
        self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_A()      # calculate gradients for D_A
        self.backward_D_B()      # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights
