import torch
import torch.optim as optim
import torch_mimicry as mmc
from torch_mimicry.nets import sngan, infomax_gan
import argparse


if __name__ == "__main__":
    # Data handling objects

    def parse_args():
        parser = argparse.ArgumentParser(description="")
        parser.add_argument('--name', type=str, default=None)
        parser.add_argument('--use_nfl', type=bool, default=True)
        parser.add_argument('--key_size', type=int, default=32)
        parser.add_argument('--val_size', type=int, default=32)
        parser.add_argument('--n_heads', type=int, default=4)
        parser.add_argument('--topk', type=int, default=4)
        args = parser.parse_args()
        return args

    args = parse_args()
    print(args)



    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    dataset = mmc.datasets.load_dataset(root='/home/anirudh/nips2020/mimicry/examples/datasets', name='cifar10')
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=64, shuffle=True, num_workers=4)

    # Define models and optimizers

    #netG = sngan.SNGANGenerator32().to(device)
    #netD = sngan.SNGANDiscriminator32().to(device)

    netG = infomax_gan.InfoMaxGANGenerator32(use_nfl=args.use_nfl, key_size=args.key_size, val_size=args.val_size, n_heads = args.n_heads, topk=args.topk).to(device)
    netD = infomax_gan.InfoMaxGANDiscriminator32().to(device)

    #netG = infomax_gan.InfoMaxGANGenerator32().to(device)
    #netD = infomax_gan.InfoMaxGANDiscriminator32().to(device)


    optD = optim.Adam(netD.parameters(), 2e-4, betas=(0.0, 0.9))
    optG = optim.Adam(netG.parameters(), 2e-4, betas=(0.0, 0.9))

    # Start training
    trainer = mmc.training.Trainer(
        netD=netD,
        netG=netG,
        optD=optD,
        optG=optG,
        n_dis=5,
        num_steps=100000,
        lr_decay='linear',
        dataloader=dataloader,
        log_dir='/home/anirudh/nips2020/mimicry/examples/log/' + args.name,
        device=device)
    trainer.train()

    # Evaluate fid
    mmc.metrics.evaluate(
        metric='fid',
        log_dir='/home/anirudh/nips2020/mimicry/examples/log/' + args.name,
        netG=netG,
        dataset_name='cifar10',
        num_real_samples=50000,
        num_fake_samples=50000,
        evaluate_step=100000,
        device=device)


    # Evaluate inception score
    mmc.metrics.evaluate(
        metric='inception_score',
        log_dir='/home/anirudh/nips2020/mimicry/examples/log/' + args.name,
        netG=netG,
        num_samples=50000,
        evaluate_step=100000,
        device=device)
