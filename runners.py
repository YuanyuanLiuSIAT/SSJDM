import time
from utils.dataset import get_dataset
from utils.dataset import get_dataset1
from models.ema import EMAHelper
from models import get_optimizer,get_sigmas
from models.POCS import BPOCS3
from models.ncsnv2 import NCSNv2
from utils.utils import *
from losses.dsm import anneal_dsm_score_estimation
import mat73
import scipy.io as scio


def get_model(config):
    configG = config.model.model_POCS
    configG.device = config.device
    netG = BPOCS3(configG).to(config.device)
    score  =  NCSNv2(config).to(config.device)
    return netG, score

def normalize(gen_img, estimated_mvue):
    '''
        Estimate mvue from coils and normalize with 99% percentile.
    '''
    scaling = torch.quantile(estimated_mvue.abs(), 0.99)
    return gen_img * scaling


def unnormalize(gen_img, estimated_mvue):
    '''
        Estimate mvue from coils and normalize with 99% percentile.
    '''
    scaling = torch.quantile(estimated_mvue.abs(), 0.99)
    return gen_img / scaling


class Runner():
    def __init__(self, args, config):
        self.args = args
        self.config = config

    def pre_train(self):
        # config model saving dir
        model_dir = "./weights/pretrain"
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        self.config.testing.weights = model_dir + '/epoch-' + str(self.config.training.n_epochs) + '.pkl'

        # load dataset
        dataloader = get_dataset1(self.config, 'training')

        netG, _ = get_model(self.config)

        optimizer = torch.optim.Adam(netG.parameters(), lr=1e-4)

        if self.config.model.model_POCS.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(netG)

        mask_dir = "/data0/shucong/code/SSJDM_contrast_code/self_diffusion_t1rho/mask/paired_random2d_72x176_%dx.mat" % (self.config.pretraining.selfacc)
        I = mat73.loadmat(mask_dir)
        mask = I['mask']
        mask = mask.astype(np.complex64)
        mask = np.expand_dims(mask, axis=0)
        mask = np.expand_dims(mask, axis=0)
        mask = np.expand_dims(mask, axis=0)
        mask = torch.from_numpy(mask)
        sub_mask = I['sub_mask']
        sub_mask = sub_mask.astype(np.complex64)
        sub_mask = np.expand_dims(sub_mask, axis=0)
        sub_mask = np.expand_dims(sub_mask, axis=0)
        sub_mask = np.expand_dims(sub_mask, axis=0)
        sub_mask = torch.from_numpy(sub_mask)

        I = mat73.loadmat('/data0/shucong/code/SSJDM_contrast_code/self_diffusion_t1rho_clean/mask/filter_72x176.mat')
        filt = I['weight']
        filt = filt.astype(np.complex64)
        filt = np.expand_dims(filt, axis=0)
        filt = np.expand_dims(filt, axis=0)
        filt = np.expand_dims(filt, axis=0)
        filt = torch.from_numpy(filt)

        sigmaN = self.config.pretraining.sigmaN
        sigmaK = self.config.pretraining.sigmaK

        for epoch in range(self.config.training.n_epochs):
            step = 0
            loss_sum = 0
            for index, point in enumerate(dataloader):
                t0 = time.time()
                netG.train()
                step += 1
                org, csm = point
                org = c2r(org).type(torch.FloatTensor).to(self.config.device)
                mask = mask.to(self.config.device)
                sub_mask = sub_mask.to(self.config.device)
                filt = filt.to(self.config.device)
                atb = c2r(r2c(org)*sub_mask)
                if self.config.data.dataset_name == "fastmri":
                    out, mu, sigma = netG(atb,sub_mask)
                else:
                    csm = c2r(csm).type(torch.FloatTensor).to(self.config.device)
                    out, mu, sigma = netG(atb,csm,sub_mask)
                    out = Emat_xyt(out, False, csm, mask)
                    out = Emat_xyt(out, True, csm, 1)

                label = Emat_xyt(org, True, csm, 1)
                loss_data = torch.sum(torch.sum(torch.pow(torch.abs(c2r(r2c(out-label))), 2), 0))
                loss_mu = 0
                loss_sigma1 = 0
                loss_sigma2 = 0
                for j in range(len(sigma)):
                    loss_mu += torch.sum(torch.pow(mu[j],2))
                    loss_sigma1 += torch.sum(torch.pow(sigma[j],2))
                    loss_sigma2 += torch.sum(torch.log(sigma[j]+1e-10))

                loss = loss_data + (sigmaN/sigmaK)**2*(loss_mu+loss_sigma1)- 2*sigmaN**2*loss_sigma2
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_sum += loss

                param_num = sum(param.numel() for param in netG.parameters())
                # if step % 10 == 0:
                print('Epoch', epoch + 1, '/', self.config.training.n_epochs, 'Step', step, 'loss = ', loss.cpu().data.numpy(),
                        'loss mean =', loss_sum.cpu().data.numpy() / (step + 1),
                        'time', time.time() - t0, 'param_num', param_num)

            if (epoch + 1) % 10 == 0:
                torch.save(netG.state_dict(), "%s/G_epoch-%d.pkl" % (model_dir, epoch + 1))
    

    def train(self):
        # config model saving dir
        model_dir = "./weights/score/"
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        # load dataset
        dataloader = get_dataset(self.config, 'training')

        netG, score = get_model(self.config)
        netG.load_state_dict(torch.load(self.config.training.weights), strict=True)
        set_requires_grad(netG, requires_grad=False)

        optimizer = get_optimizer(self.config, score.parameters())

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(score)

        sigmas = get_sigmas(self.config)
        mask_dir = "/data0/shucong/code/SSJDM_contrast_code/self_diffusion_t1rho_clean/mask/paired_random2d_72x176_PDR%dx.mat" % (self.config.pretraining.selfacc)
        I = mat73.loadmat(mask_dir)
        mask = I['mask']
        mask = mask.astype(np.complex64)
        mask = np.expand_dims(mask, axis=0)
        mask = np.expand_dims(mask, axis=0)
        mask = np.expand_dims(mask, axis=0)
        mask = torch.from_numpy(mask)

        for epoch in range(self.config.training.n_epochs):
            step = 0
            loss_sum = 0
            for index, point in enumerate(dataloader):
                t0 = time.time()
                score.train()
                step += 1
                org, csm = point
                org = c2r(org).type(torch.FloatTensor).to(self.config.device)
                mask = mask.to(self.config.device)

                csm = c2r(csm).type(torch.FloatTensor).to(self.config.device)
                label, _, _ = netG(org,csm,mask)

                nb, nc, nt, nx, ny = label.size()
                label = label.view(nb,nc*nt,nx,ny)
                loss = anneal_dsm_score_estimation(score, label, sigmas, None, self.config.training.anneal_power)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_sum += loss

                if self.config.model.ema:
                    ema_helper.update(score)

                param_num = sum(param.numel() for param in score.parameters())
                if step % 10 == 0:
                    print('Epoch', epoch + 1, '/', self.config.training.n_epochs, 'Step', step, 'loss = ', loss.cpu().data.numpy(),
                            'loss mean =', loss_sum.cpu().data.numpy() / (step + 1),
                            'time', time.time() - t0, 'param_num', param_num)

            if (epoch + 1) % 10 == 0:
                torch.save(score.state_dict(), "%s/S_epoch-%d.pkl" % (model_dir, epoch + 1))


    def test(self):
        print('load weights:', self.config.testing.weights)
        result_dir = './results/recon'
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        _, score = get_model(self.config)
        score.load_state_dict(torch.load(self.config.testing.weights), strict=True)

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(score)
            ema_helper.load_state_dict(torch.load(self.config.testing.weights))
            ema_helper.ema(score)
        
        sigmas = get_sigmas(self.config)

        # load dataset
        dataloader = get_dataset(self.config, 'test')

        score.eval()

        print('testDataLen:',len(dataloader))
        for index, point in enumerate(dataloader):
            print('---------------------------------------------')
            print('---------------- point:', index, '------------------')
            print('---------------------------------------------')
            t0=time.time()
            k0, csm = point
            csm = c2r(csm).type(torch.FloatTensor).to(self.config.device)
            k0 = k0.to(self.config.device)

            mask_dir = "/data0/shucong/code/SSJDM_contrast_code/self_diffusion_t1rho_clean/mask/paired_random2d_72x176_PDR6x.mat"

            I = mat73.loadmat(mask_dir)
            mask = I['mask']
            mask = mask.astype(np.complex64)
            mask = np.expand_dims(mask, axis=0)
            mask = np.expand_dims(mask, axis=0)
            mask = np.expand_dims(mask, axis=0)
            mask = torch.from_numpy(mask)
            mask = mask.to(self.config.device)

            atb = k0 * mask
            atb_to_image = Emat_xyt(c2r(atb), True, csm, 1)
            atb_to_image = r2c(atb_to_image)

            samples = torch.rand(self.config.testing.batch_size, self.config.data.channels,
                    self.config.data.image_size[0], self.config.data.image_size[1],
                    device=self.config.device)

            with torch.no_grad():
                for c, sigma in enumerate(sigmas):
                    if c % 10 == 0:
                        print('sigma:', c)
                    sigma_index = torch.ones(samples.shape[0], device=samples.device) * c
                    sigma_index = sigma_index.long()
                    step_size = self.config.testing.step_lr * (sigma / sigmas[-1]) ** 2
                    for s in range(self.config.testing.n_steps_each):
                        noise = torch.randn_like(samples) * torch.sqrt(step_size * 2)
                        grad = score(samples, sigma_index)
                        nb, nc, nx, ny = samples.size()
                        meas_grad = Emat_xyt(samples.view(nb,2,11,nx,ny), False, csm, mask) - c2r(atb)
                        meas_grad = Emat_xyt(meas_grad, True, csm, mask)
                        meas_grad = meas_grad.view(nb,nc,nx,ny)
                        meas_grad = meas_grad.type(torch.cuda.FloatTensor)
                        meas_grad /= torch.norm(meas_grad)
                        meas_grad *= torch.norm(grad)
                        meas_grad *= self.config.mse

                        # combine measurement gradient, prior gradient and noise Langevin MCMC
                        samples = samples + step_size * (grad - meas_grad) + noise

                if self.config.testing.denoise:
                    last_noise = (len(sigmas) - 1) * torch.ones(samples.shape[0], device=samples.device)
                    last_noise = last_noise.long()
                    samples = samples + sigmas[-1] ** 2 * score(samples, last_noise)

                recon = normalize(samples.view(nb,2,11,nx,ny), atb_to_image)
                recon = r2c(recon)
                print('time:',time.time()-t0)

                recon=recon.cpu().numpy()
                if index == 0:
                   recon_stack = recon
                else:
                   recon_stack = np.concatenate((recon_stack, recon), axis=0)

        recon_stack = np.squeeze(recon_stack)
        recon_stack = np.transpose(recon_stack, [0, 3, 2, 1])
        scio.savemat(os.path.join(result_dir, 'recon_SSJDM.mat'), {'recon_SSJDM': recon_stack})
