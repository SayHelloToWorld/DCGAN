from torch import nn



class Generator(nn.Module):
    
    def __init__(self, args):
        super(Generator,self).__init__()
        self.nz = args.nz
        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(self.nz,64*8,4,1,0,bias=False),
            nn.BatchNorm2d(64*8),
            nn.ReLU(True))
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(64*8, 64*4, 4, 2, 1, bias= False),
            nn.BatchNorm2d(64*4),
            nn.ReLU(True)
        )
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(64*4, 64*2,4,2,1,bias = False),
            nn.BatchNorm2d(64*2),
            nn.ReLU(True)
        )
        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(64*2, 64, 4,2,1, bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )
        self.layer5 = nn.Sequential(
            nn.ConvTranspose2d(64, 3,4,2,1, bias = False),
            nn.Tanh()
        )
             

    def forward(self, input):
        out = self.layer1(input)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        
        return out

class Discriminator(nn.Module):
    def __init__(self,args):
        super(Discriminator, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias = False),
            nn.LeakyReLU(0.2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(64,64*2,4,2,1,bias = False),
            nn.BatchNorm2d(64*2),
            nn.LeakyReLU(0.2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64*2,64*4,4,2,1,bias = False),
            nn.BatchNorm2d(64*4),
            nn.LeakyReLU(0.2)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(64*4,64*8, 4,2,1, bias = False),
            nn.BatchNorm2d(64*8),
            nn.LeakyReLU(0.2)
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(64*8,1,4,1,0,bias = False),
            nn.Sigmoid()
        )
    
    def forward(self, input):
        out = self.layer1(input)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        return out
    