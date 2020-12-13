import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd.variable import Variable
from torchvision import transforms, datasets
from torchvision.utils import save_image

seed = 123
torch.manual_seed(seed)
#standardizing to range (-1, 1)
compose = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize([0.5], [0.5])
        ])
data = datasets.MNIST(root='./data', train=True, transform=compose, download=True)
data_loader = torch.utils.data.DataLoader(data, batch_size=100, shuffle=True)

class discriminator(torch.nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        
        self.transform1 =  nn.LeakyReLU(0.2)
        self.transform2 =  nn.Linear(784, 512)
        self.transform3 =  nn.Linear(512, 256)
        self.transform4 =  nn.Linear(256, 1)
        self.transform5 =  nn.Sigmoid()
        
    def forward(self, x):
        x = self.transform2(x)
        x = self.transform1(x)
        x = self.transform3(x)
        x = self.transform1(x)
        x = self.transform4(x)
        x = self.transform5(x)
        return x

class generator(torch.nn.Module):
    def __init__(self):
        super(generator, self).__init__()
        
        self.transform1 =  nn.LeakyReLU(0.2)
        self.transform2 =  nn.Linear(100,256)
        self.transform3 = nn.Linear(256,512)
        self.transform4 =  nn.Linear(512, 784)
        self.transform5 =  nn.Tanh() #matches standardization
        
    def forward(self, x):
        x = self.transform2(x)
        x = self.transform1(x)
        x = self.transform3(x)
        x = self.transform1(x)
        x = self.transform4(x)
        x = self.transform5(x)
        return x
    
dis = discriminator()
gen = generator()
print(dis);print(gen)                      


def images_to_vectors(images):
    return images.view(images.size(0), 784)

def vectors_to_images(vectors):
    return vectors.view(vectors.size(0), 1, 28, 28)

g_optim = optim.Adam(gen.parameters(), lr=1e-4)
d_optim = optim.Adam(dis.parameters(), lr=1e-4)
loss = nn.BCELoss()

for epochs in range(200):
    for batch_num, (batch_dat,_) in enumerate(data_loader):

        # train discriminator
        d_optim.zero_grad()
        real_dat = Variable(images_to_vectors(batch_dat))
        latent_normal = Variable(torch.randn(batch_dat.size(0), 100))
        fake_dat = gen(latent_normal).detach()
        
        dout_real = dis(real_dat)
        dout_fake = dis(fake_dat)
        derror_real = loss(dout_real, Variable(torch.ones(real_dat.size(0), 1)))
        derror_fake = loss(dout_fake, Variable(torch.zeros(real_dat.size(0), 1)))
        d_error = derror_real + derror_fake
        d_error.backward()
        d_optim.step()
        
        # train generator
        g_optim.zero_grad()
        fake_dat2 = gen(latent_normal)
        dout_fake2 = dis(fake_dat2)
        g_error = loss(dout_fake2, Variable(torch.ones(fake_dat2.size(0), 1)))
        g_error.backward()
        g_optim.step()
        print('epoch = {}, batch number = {}'.format(epochs,batch_num))



test_num = 16
test_latent_normal = Variable(torch.randn(test_num, 100))     
test_images = vectors_to_images(gen(test_latent_normal))
save_image(test_images,"im2.png")