import torch
import torch.optim as optim
import torch_mimicry as mmc
from torch_mimicry.nets import sngan, infomax_gan
import argparse

if __name__ == "__main__":
    def parse_args():
         parser = argparse.ArgumentParser(description="")
         parser.add_argument('--name', type=str, default=None)
         parser.add_argument('--use_nfl', type=bool, default=False)
         parser.add_argument('--key_size', type=int, default=32)
         parser.add_argument('--val_size', type=int, default=32)
         parser.add_argument('--n_heads', type=int, default=4)
         parser.add_argument('--topk', type=int, default=4)

         args = parser.parse_args()
         return args

    args = parse_args()
    print(args)
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    print(device)
    dataset = mmc.datasets.load_dataset(root='/home/anirudh/lsun/', name='lsun_bedroom_128')
    print(device)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2)
    print(device)

    # Define models and optimizers

    #netG = sngan.SNGANGenerator32().to(device)
    #netD = sngan.SNGANDiscriminator32().to(device)
    print(args)
    netG = infomax_gan.InfoMaxGANGenerator128(use_nfl=args.use_nfl, key_size=args.key_size, val_size=args.val_size, n_heads = args.n_heads, topk=args.topk).to(device)
    netD = infomax_gan.InfoMaxGANDiscriminator128().to(device)


    optD = optim.Adam(netD.parameters(), 2e-4, betas=(0.0, 0.9))
    optG = optim.Adam(netG.parameters(), 2e-4, betas=(0.0, 0.9))

    # Start training
    trainer = mmc.training.Trainer(
        netD=netD,
        netG=netG,
        optD=optD,
        optG=optG,
        n_dis=2,
        num_steps=100000,
        lr_decay='linear',
        dataloader=dataloader,
        log_dir='/home/anirudh/nips2020/mimicry/log/' + args.name,
        device=device)
    trainer.train()

    # Evaluate fid
    mmc.metrics.evaluate(
        metric='fid',
        log_dir='/home/anirudh/nips2020/mimicry/log/' + args.name,
        netG=netG,
        dataset_name='lsun_bedroom_128',
        num_real_samples=50000,
        num_fake_samples=50000,
        evaluate_step=100000,
        device=device)

    '''
    # Evaluate kid
    mmc.metrics.evaluate(
        metric='kid',
        log_dir='/scratch/anirudhg/nips2020/mimicry/log/' + args.name,
        netG=netG,
        dataset_name='cifar10',
        num_subsets=50,
        subset_size=1000,
        evaluate_step=100000,
        device=device)
    '''

    # Evaluate inception score
    mmc.metrics.evaluate(
        metric='inception_score',
        log_dir='/home/anirudh/nips2020/mimicry/log/' + args.name,
        netG=netG,
        num_samples=50000,
        evaluate_step=100000,
        device=device)

