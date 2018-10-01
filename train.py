#-*-coding:utf-8-*—
# 把data augmentation 融合进入网络训练的过程
# 可选：不在选用max 而是 区间距离

import torch 
from torch import nn
import matplotlib.pyplot as plt 
import torchvision
from torchvision import transforms
import argparse

from models import Generator,Discriminator

def visulize(tensor):
    imgs = torchvision.utils.make_grid(tensor*0.5+0.5).cpu()
    plt.imshow(imgs.permute(1,2,0).detach().numpy())
    plt.show()


def train(args):
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3,[0.5]*3)
    ])

    dataset = torchvision.datasets.CIFAR10(root = './data/',
                        transform = transform, download = True)

    dataloader = torch.utils.data.DataLoader(dataset,args.batch_size,
                 shuffle = True, num_workers=args.workers)
    #network
    d = Discriminator(args)
    g = Generator(args)
    #optimizer
    optim_D = torch.optim.Adam(d.parameters(),lr = args.lr,
                                betas = (args.beta1, 0.999))
    optim_G = torch.optim.Adam(g.parameters(), lr = args.lr, 
                                betas = (args.beta1, 0.999))
    #loss criterion
    criterion = torch.nn.BCELoss()

    fix_noise = torch.randn(args.batch_size, args.nz, 1, 1)
    if args.gpu:
        fix_noise = fix_noise.cuda()
        d.cuda()
        g.cuda()
        criterion.cuda()

    print('start training!')
    for epoch in range(args.epoch):
        for i, data in enumerate(dataloader):
            input, _ = data
            label = torch.ones(input.size(0))
            noise = torch.randn(input.size(0),args.nz, 1, 1)
            
            if args.gpu:
                noise = noise.cuda()
                input = input.cuda()
                label = label.cuda()
                
            #=====train discriminator=======
            d.zero_grad()
            #train net d with real image
            output = d(input)
            error_real = criterion(output.squeeze(),label)
            error_real.backward()
            D_x = output.mean()
            #train net d with fake image
            fake_pic = g(noise)
            output2 = d(fake_pic)
            label = torch.zeros(input.size(0))
            if args.gpu:
                label = label.cuda()
            error_fake = criterion(output2.squeeze(), label)
            error_fake.backward()
            D_x2 = error_fake.mean()
            error_D = error_real + error_fake
            optim_D.step()

            #======train generator======
            g.zero_grad()
            label = torch.ones(input.size(0))
            if args.gpu:
                label = label.cuda()
            noise = torch.normal(noise)
            fake_pic = g(noise)
            output = d(fake_pic)
            error_G = criterion(output.squeeze(), label)
            error_G.backward()
            optim_G.step()
            D_G_z2 = output.mean()
            if (i+1) % 500 == 0:
                print('epoch%d LossD: %f, LossG: %f' %(epoch, error_D[0], error_G[0]))
        
        if epoch % 2 == 0:
            fake_u = g(fix_noise)
            visulize(fake_u)

    torch.save(d.state_dict(), 'dcgan_d.pth')
    torch.save(d.state_dict(), 'dcgan_g.pth')
    

    
def main():
    parser = argparse.ArgumentParser(description='easy!')
    parser.add_argument('--lr', default = 0.0002, type = float,
                        help = 'Learning Rate')
    parser.add_argument('--nz', default = 100, type = int,
                        help = 'Noise dimension')
    parser.add_argument('--beta1', default = 0.5, type = float,
                        help = 'Beta 1 for adam')
    parser.add_argument('--epoch', default = 10, type = int,
                        help = 'the number of epoch')
    parser.add_argument('--batch_size', default = 64, type = int,
                        help = 'the num of batch size')
    parser.add_argument('--workers', default = 2, type = int,
                        help = 'the num of workers')
    parser.add_argument('-gpu', default = True, type = bool,
                        help = 'whether to use gpu')
    args = parser.parse_args()
    print(args)
    
    train(args)


if __name__ == '__main__':
    main()