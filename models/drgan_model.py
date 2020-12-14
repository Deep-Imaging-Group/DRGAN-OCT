import torch
import itertools
from .base_model import BaseModel
from lib import GANLoss
from lib import init_net, psnr, ImagePool, get_scheduler
from .drgan_nets import ContentEncoder, StyleEncoder, Decoder, NoisyGenerator, NLayerDiscriminator


class DRGANModel(BaseModel):
    def __init__(self, args):
        BaseModel.__init__(self, args)

        if self.mode == 'train':
            self.fake_C_pool = ImagePool(args.pool_size)  # create image buffer to store previously generated images
            self.fake_N_pool = ImagePool(args.pool_size)  # create image buffer to store previously generated images
            self.fake_PNX_pool = ImagePool(args.pool_size)  # create image buffer to store previously generated images
            self.fake_PNY_pool = ImagePool(args.pool_size)  # create image buffer to store previously generated images

            visual_images_X = ['imageX', 'clearX', 'cycleX', 'reconX', 'labelX']
            visual_images_Y = ['imageY', 'noisyY', 'cycleY', 'reconY', 'imageN']
            self.visual_losses = ['loss_D_C', 'loss_D_N', 'loss_G', 'loss_cycle', 'loss_recon', 'loss_noise']
            self.nrow = max(len(visual_images_X), len(visual_images_Y))
            self.visual_images = visual_images_X + visual_images_Y
            self.visual_losses.append('train_psnr')
        else:
            self.visual_images = ['imageX', 'clearX', 'labelX']
            self.visual_losses.append('valid_psnr')

    def setup(self):
        if self.mode == 'train':
            self.set_random_seed(self.args.seed)
            self.networks = {
                'netE_C': ContentEncoder(self.args.n_downsample, self.args.n_res, self.args.input_dim, self.args.dim,
                                         self.args.norm, self.args.activ, self.args.pad_type),
                'netE_N': StyleEncoder(self.args.n_downsample, self.args.input_dim, self.args.dim, self.args.style_dim,
                                       self.args.norm, self.args.activ, self.args.pad_type),
                'netG_C': Decoder(self.args.n_upsample, self.args.n_res, self.args.dim, self.args.output_dim,
                                  'in', self.args.activ, self.args.pad_type, self.args.skip_connect),
                'netG_N': NoisyGenerator(self.args.n_downsample, self.args.dim, self.args.style_dim,
                                         self.args.input_dim, self.args.output_dim,
                                         self.args.mlp_dim, self.args.n_res, self.args.activ, self.args.pad_type,
                                         self.args.skip_connect),
                'netD_C': NLayerDiscriminator(in_nc=self.args.input_dim),
                'netD_N': NLayerDiscriminator(in_nc=self.args.input_dim),
                'netD_PN': NLayerDiscriminator(in_nc=self.args.input_dim)}
            self.networks = init_net(self.networks, self.args.init_type, self.args.gpu_ids)
            self.optimizers = {'optim_G': torch.optim.Adam(itertools.chain(self.networks['netE_C'].parameters(),
                                                                           self.networks['netE_N'].parameters(),
                                                                           self.networks['netG_C'].parameters(),
                                                                           self.networks['netG_N'].parameters()),
                                                           lr=self.args.lr, betas=(0.5, 0.999), weight_decay=1e-4),
                               'optim_D': torch.optim.Adam(itertools.chain(self.networks['netD_C'].parameters(),
                                                                           self.networks['netD_N'].parameters(),
                                                                           self.networks['netD_N'].parameters(),
                                                                           self.networks['netD_PN'].parameters()),
                                                           lr=self.args.lr, betas=(0.5, 0.999), weight_decay=1e-4)}
            self.schedulers = [get_scheduler(optimizer, self.args) for optimizer in list(self.optimizers.values())]
            self.objectives = {'GANLoss': GANLoss(self.args.gan_type).to(self.device),
                               'L1Loss': torch.nn.L1Loss().to(self.device),
                               }
        else:
            self.networks = {
                'netE_C': ContentEncoder(self.args.n_downsample, self.args.n_res, self.args.input_dim, self.args.dim,
                                         self.args.norm, self.args.activ, self.args.pad_type),
                'netG_C': Decoder(self.args.n_upsample, self.args.n_res, self.args.dim, self.args.output_dim,
                                  'in', self.args.activ, self.args.pad_type, skip_connect=self.args.skip_connect)}
            self.load_networks(self.networks, self.args.load_epoch)

    def set_input(self, input):
        self.imageX = input['imageX'].to(self.device)
        if self.mode == 'train':
            self.labelX = input['labelX'].to(self.device)
            self.imageY = input['imageY'].to(self.device)
            self.imageN = input['imageN'].to(self.device)
        elif self.mode == 'valid':
            self.labelX = input['labelX'].to(self.device)
            self.image_name = input['image_name']
        else:  # test
            self.image_name = input['image_name']

    def forward(self):
        self.contentX1 = self.networks['netE_C'](self.imageX)
        self.clearX = self.networks['netG_C'](self.contentX1)
        if self.args.mode == 'train':
            self.metric = psnr(self.clearX, self.labelX)
            self.train_psnr = self.metric

            self.noiseX1 = self.networks['netE_N'](self.imageX)
            self.noiseY1 = self.networks['netE_N'](self.imageY)

            self.contentY1 = self.networks['netE_C'](self.imageY)
            self.noisyY = self.networks['netG_N'](self.contentY1, self.noiseX1)

            self.reconX = self.networks['netG_N'](self.contentX1, self.noiseX1)
            self.reconY = self.networks['netG_C'](self.contentY1)

            self.contentX2 = self.networks['netE_C'](self.clearX)
            self.contentY2 = self.networks['netE_C'](self.noisyY)
            self.noiseX2 = self.networks['netE_N'](self.clearX)
            self.noiseY2 = self.networks['netE_N'](self.noisyY)

            self.cycleX = self.networks['netG_N'](self.contentX2, self.noiseY2)
            self.cycleY = self.networks['netG_C'](self.contentY2)

        if self.args.mode == 'valid':
            self.valid_psnr = psnr(self.clearX, self.labelX)

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
        loss_D_real = self.objectives['GANLoss'](pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.objectives['GANLoss'](pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_C(self):
        """Calculate GAN loss for discriminator D_C"""
        fake_C = self.fake_C_pool.query(self.clearX)
        self.loss_D_C = self.backward_D_basic(self.networks['netD_C'], self.imageY, fake_C)

    def backward_D_N(self):
        """Calculate GAN loss for discriminator D_N"""
        fake_N = self.fake_N_pool.query(self.noisyY)
        self.loss_D_N = self.backward_D_basic(self.networks['netD_N'], self.imageX, fake_N)

    def backward_D_PN(self):
        """Calculate GAN loss for discriminator D_N"""
        fake_PNX = self.fake_PNX_pool.query(self.imageX - self.clearX)
        fake_PNY = self.fake_PNY_pool.query(self.noisyY - self.imageY)
        self.loss_D_PNX = self.backward_D_basic(self.networks['netD_PN'], self.imageN, fake_PNX)
        self.loss_D_PNY = self.backward_D_basic(self.networks['netD_PN'], self.imageN, fake_PNY)

    def backward_D(self):
        self.set_requires_grad([self.networks['netD_C'], self.networks['netD_N'], self.networks['netD_PN']], True)
        self.optimizers['optim_D'].zero_grad()  # set Ds's gradients to zero
        self.backward_D_C()  # calculate gradients for D_C
        self.backward_D_N()  # calculate graidents for D_N
        self.backward_D_PN()  # calculate graidents for D_PN
        self.optimizers['optim_D'].step()  # update Ds's weights

    def backward_G(self):
        self.set_requires_grad([self.networks['netD_C'], self.networks['netD_N'], self.networks['netD_PN']], False)
        self.optimizers['optim_G'].zero_grad()

        lambda_recon = self.args.lambda_recon
        lambda_cycle = self.args.lambda_cycle
        lambda_noise = self.args.lambda_noise

        # GAN loss
        self.loss_GAN_C = self.objectives['GANLoss'](self.networks['netD_C'](self.clearX), True)
        self.loss_GAN_N = self.objectives['GANLoss'](self.networks['netD_N'](self.noisyY), True)

        # cycle loss
        self.loss_cycle = lambda_cycle * self.objectives['L1Loss'](self.cycleX, self.imageX) + \
                          lambda_cycle * self.objectives['L1Loss'](self.cycleY, self.imageY)

        # recon loss
        self.loss_recon = lambda_recon * self.objectives['L1Loss'](self.reconX, self.imageX) + \
                          lambda_recon * self.objectives['L1Loss'](self.reconY, self.imageY)

        # noise loss
        self.loss_noise = lambda_noise * self.objectives['GANLoss'](self.networks['netD_PN'](self.imageX - self.clearX), True) + \
                          lambda_noise * self.objectives['GANLoss'](self.networks['netD_PN'](self.noisyY - self.imageY), True)

        # total loss
        self.loss_G = self.loss_GAN_C + self.loss_GAN_N + self.loss_cycle + self.loss_recon + self.loss_noise
        self.loss_G.backward()
        self.optimizers['optim_G'].step()

    def optimize_parameters(self, idx):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()
        self.backward_G()
        if idx % self.args.num_critics == 0:
            self.backward_D()
