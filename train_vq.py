import os
import options.option_vq as option_vq
args = option_vq.get_args_parser()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import json

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import models.vqvae as vqvae

import utils.utils_model as utils_model
from dataset import dataset_VQ
import utils.eval_mlm as eval_trans

import warnings
warnings.filterwarnings('ignore')
from torch.autograd import Variable
CUDA_LAUNCH_BLOCKING=1


def update_lr_warm_up(optimizer, nb_iter, warm_up_iter, lr):

    current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
    for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr

    return optimizer, current_lr

def loss_quantile(mu: Variable, labels: Variable, quantile: Variable):
    loss = 0
    for i in range(mu.shape[1]):
        mu_e = mu[:, i]
        labels_e = labels[:, i]

        I = (labels_e >= mu_e).float()
        each_loss = 2 * (torch.sum(quantile * ((labels_e - mu_e) * I) + (1 - quantile) * (mu_e - labels_e) * (1 - I)))
        loss += each_loss

    denom = torch.sum(torch.abs(labels)).item()
    loss = loss / denom

    return loss


##### ---- Exp dirs ---- #####
# args = option_vq.get_args_parser()
torch.manual_seed(args.seed)

args.out_dir = os.path.join(args.out_dir, f'{args.exp_name}')
os.makedirs(args.out_dir, exist_ok = True)

##### ---- Logger ---- #####
logger = utils_model.get_logger(args.out_dir)
writer = SummaryWriter(args.out_dir)
logger.info(json.dumps(vars(args), indent=4, sort_keys=True))




logger.info(f'Training on {args.dataname}')




##### ---- Dataloader ---- #####
train_loader = dataset_VQ.DATALoader(args.dataname,
                                        args.batch_size,
                                        window_size=args.window_size,
                                        unit_length=2**args.down_t, dataset_type='train')

train_loader_iter = dataset_VQ.cycle(train_loader)

val_loader = dataset_VQ.DATALoader(args.dataname,
                                        args.batch_size,
                                        window_size=args.window_size,
                                        unit_length=2**args.down_t, dataset_type='train')#val
test_loader = dataset_VQ.DATALoader(args.dataname,
                                        args.batch_size,
                                        window_size=args.window_size,
                                        unit_length=2**args.down_t, dataset_type='train')
MSELoss = torch.nn.MSELoss()
##### ---- Network ---- #####
net = vqvae.HumanVQVAE(args, ## use args to define different parameters in different quantizers
                       args.nb_code,
                       args.code_dim,
                       args.down_t,
                       args.stride_t,
                       args.width,
                       args.depth,
                       args.dilation_growth_rate,
                       args.vq_act,
                       args.vq_norm)


if args.resume_pth : 
    logger.info('loading checkpoint from {}'.format(args.resume_pth))
    ckpt = torch.load(args.resume_pth, map_location='cpu')
    net.load_state_dict(ckpt['net'], strict=True)
net.train()
net.cuda()

##### ---- Optimizer & Scheduler ---- #####
optimizer = optim.AdamW(net.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_scheduler, gamma=args.gamma)
  


##### ------ warm-up ------- #####
avg_recons, avg_perplexity, avg_commit = 0., 0., 0.

for nb_iter in range(1, args.warm_up_iter):

    optimizer, current_lr = update_lr_warm_up(optimizer, nb_iter, args.warm_up_iter, args.lr)

    input = next(train_loader_iter)
    input = input.cuda().float() # (bs, 64, dim)

    # if nb_iter%2==0:
    #     input = input[:,:96,:]

    output, loss_commit, perplexity = net(input)
    loss_recons = MSELoss(output, input)
    # fft1 = torch.fft.fft(output,dim=1, norm='forward')
    # fft2 = torch.fft.fft(input,dim=1, norm='forward')
    # loss_fft = MSELoss(torch.real(fft1), torch.real(fft2)) + MSELoss(torch.imag(fft1), torch.imag(fft2))

    # loss_recons = MSELoss(torch.real(fft1), torch.real(fft2)) + MSELoss(torch.imag(fft1), torch.imag(fft2))

    loss = args.recon*loss_recons + args.commit * loss_commit# + args.fft*loss_fft


    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    avg_recons += loss_recons.item()
    avg_perplexity += perplexity.item()
    avg_commit += loss_commit.item()

    if nb_iter % args.print_iter ==  0 :
        avg_recons /= args.print_iter
        avg_perplexity /= args.print_iter
        avg_commit /= args.print_iter

        logger.info(f"Warmup. Iter {nb_iter} :  lr {current_lr:.5f} \t Commit. {avg_commit:.5f} \t PPL. {avg_perplexity:.2f} \t Recons.  {avg_recons:.8f}")

        avg_recons, avg_perplexity, avg_commit = 0., 0., 0.

####---- Training ---- #####
avg_recons, avg_perplexity, avg_commit = 0., 0., 0.

best_iter, best_ds, best_mse, writer, logger = eval_trans.evaluation_vqvae(args,args.out_dir, val_loader, net, logger, writer, 0, best_iter=0, best_ds=99999, best_mse = 99999)

for nb_iter in range(1, args.total_iter + 1):
    
    input = next(train_loader_iter)
    input = input.cuda().float() # bs, nb_joints, joints_dim, seq_len
    
    output, loss_commit, perplexity = net(input)

    loss_recons = MSELoss(output, input)
    loss = args.recon*loss_recons + args.commit * loss_commit# + args.fft*loss_fft


    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    
    avg_recons += loss_recons.item()
    avg_perplexity += perplexity.item()
    avg_commit += loss_commit.item()
    
    if nb_iter % args.print_iter == 0:
        avg_recons /= args.print_iter
        avg_perplexity /= args.print_iter
        avg_commit /= args.print_iter
        
        writer.add_scalar('./Train/Recons', avg_recons, nb_iter)
        writer.add_scalar('./Train/PPL', avg_perplexity, nb_iter)
        writer.add_scalar('./Train/Commit', avg_commit, nb_iter)
        
        logger.info(f"Train. Iter {nb_iter} : \t Commit. {avg_commit:.8f} \t PPL. {avg_perplexity:.2f} \t Recons.  {avg_recons:.8f}")
        
        avg_recons, avg_perplexity, avg_commit = 0., 0., 0.,

    if nb_iter % args.eval_iter==0:
        best_iter, best_ds, best_mse, writer, logger = eval_trans.evaluation_vqvae(args,args.out_dir, val_loader, net, logger, writer, nb_iter, best_iter, best_ds, best_mse)
msg_final = f"Train. Iter {best_iter} : , MAE. {best_ds:.4f}, MSE. {best_mse:.6f}"
logger.info(msg_final)