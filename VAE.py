import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd.variable import Variable
from torchvision import transforms, datasets
from torchvision.utils import save_image

seed = 123
torch.manual_seed(seed)
#without standardizing to range (-1, 1)
compose = transforms.Compose(
        [transforms.ToTensor()
        ])
data = datasets.MNIST(root='./data', train=True, transform=compose, download=True)
data_loader = torch.utils.data.DataLoader(data, batch_size=100, shuffle=True)

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.transform1 = nn.Linear(784, 512)
        self.transform2 = nn.Linear(512, 256)
        self.transform31 = nn.Linear(256, 20)
        self.transform32 = nn.Linear(256, 20)
        
        self.transform3 = nn.Linear(20, 256)
        self.transform4 = nn.Linear(256, 512)
        self.transform5 = nn.Linear(512, 784)
        self.transform6 =  nn.LeakyReLU(0.2)
        self.transform7 =  nn.Sigmoid() #matches no standardization

    def encoder(self, x):
        x = self.transform1(x)
        x = self.transform6(x)
        x = self.transform2(x)
        x = self.transform6(x)
        mu = self.transform31(x)
        var = torch.exp(self.transform32(x))
        return mu, var

    def decoder(self, z):
        z = self.transform3(z)
        z = self.transform6(z)
        z = self.transform4(z)
        z = self.transform6(z)
        z = self.transform5(z)
        z = self.transform7(z)
        return z

    def forward(self, x):
        mu, var = self.encoder(x)
        sd = torch.sqrt(var)
        standard_normal = Variable(torch.randn_like(sd))
        z = mu + sd * standard_normal
        # torch.normal not propogating gradients
        # z = torch.normal(mean=mu,std=sd)
        xnew = self.decoder(z)
        return xnew, mu, var


vae_net = VAE()
vae_optim = optim.Adam(vae_net.parameters(), lr=1e-4)
loss_bce = nn.BCELoss(reduction='sum')


def images_to_vectors(images):
    return images.view(images.size(0), 784)

def vectors_to_images(vectors):
    return vectors.view(vectors.size(0), 1, 28, 28)

for epochs in range(200):
    for batch_num, (batch_dat, _) in enumerate(data_loader):
        vae_optim.zero_grad()
        dat = Variable(images_to_vectors(batch_dat))
        xnew, mu, var = vae_net(dat)
        BCE = loss_bce(xnew, dat)
        KLD = -0.5 * torch.sum(1 + torch.log(var) - torch.pow(mu,2) - var)
        error = BCE + KLD
        error.backward()
        vae_optim.step()
        print('epoch = {}, batch number = {}'.format(epochs,batch_num))

        
            
test_num = 16
test_latent_normal = Variable(torch.randn(test_num, 20))
test_images = vectors_to_images(vae_net.decoder(test_latent_normal))
save_image(test_images,'im2.png')
