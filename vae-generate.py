import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import argparse


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Hout = ((Hin-Kernal+2*Padding)/Stride)+1
        # Input shape will be B,3,128,128 -> B,C,H,W
        self.c1 = nn.Conv2d(3,64,3,2,padding=1) # After this layer input will be B,64,64,64
        self.bn1 = nn.BatchNorm2d(64)
        self.act1 = nn.LeakyReLU()
        self.c2 = nn.Conv2d(64,128,3,2,padding=1) # After this later input will be B,128,32,32
        self.bn2 = nn.BatchNorm2d(128)
        self.act2 = nn.LeakyReLU()
        self.c3 = nn.Conv2d(128,128,3,2,padding=1) # After this layer input will be B,128,16,16
        self.bn3 = nn.BatchNorm2d(128)
        self.act3 = nn.LeakyReLU()
        self.c4 = nn.Conv2d(128,128,3,2,padding=1) # After this layer input will be B,128,8,8
        self.bn4 = nn.BatchNorm2d(128)
        self.act4 = nn.LeakyReLU()
        self.c5 = nn.Conv2d(128,128,3,2,padding=1) # After this layer input will be B,128,4,4
        self.bn5 = nn.BatchNorm2d(128)
        self.act5 = nn.LeakyReLU()
        self.c6 = nn.Conv2d(128,128,3,2,padding=1) # After this layer input will be B,128,2,2
        self.bn6 = nn.BatchNorm2d(128)
        self.act6 = nn.LeakyReLU()
        self.flatten_layer = nn.Flatten(start_dim=1,end_dim=-1)
        self.z_mu = nn.Linear(128*2*2,200)
        self.z_logvar = nn.Linear(128*2*2,200)

    def forward(self,x):
        x = self.act1(self.bn1(self.c1(x)))
        x = self.act2(self.bn2(self.c2(x)))
        x = self.act3(self.bn3(self.c3(x)))
        x = self.act4(self.bn4(self.c4(x)))
        x = self.act5(self.bn5(self.c5(x)))
        x = self.act6(self.bn6(self.c6(x)))
        x = self.flatten_layer(x)
        z_mean = self.z_mu(x)
        z_logvar = self.z_logvar(x)
        return z_mean,z_logvar
    
   
class Sampling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,z_mean,z_logvar):
        batch_size,dim = z_mean.shape
        epsilon = torch.randn(batch_size,dim,device=z_mean.device)
        return z_mean + torch.exp(0.5*z_logvar)*epsilon


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense1 = nn.Linear(200,128*2*2)
        self.dbn1 = nn.BatchNorm1d(128*2*2)
        self.dact1 = nn.LeakyReLU()
        # Hout = (Hin-1)*stride -2*padding+kernel+output_padding
        self.ct1 = nn.ConvTranspose2d(128,128,kernel_size=3,stride=2,padding=1,output_padding=1) # After this layer B,128,4,4
        self.bn1 = nn.BatchNorm2d(128)
        self.act1 = nn.LeakyReLU()
        self.ct2 = nn.ConvTranspose2d(128,128,kernel_size=3,stride=2,padding=1,output_padding=1) # After this layer B,128,8,8
        self.bn2 = nn.BatchNorm2d(128)
        self.act2 = nn.LeakyReLU()
        self.ct3 = nn.ConvTranspose2d(128,128,kernel_size=3,stride=2,padding=1,output_padding=1) # B,128,16,16
        self.bn3 = nn.BatchNorm2d(128)
        self.act3 = nn.LeakyReLU()
        self.ct4 = nn.ConvTranspose2d(128,128,kernel_size=3,stride=2,padding=1,output_padding=1) # B,128,32,32
        self.bn4 = nn.BatchNorm2d(128)
        self.act4 = nn.LeakyReLU()
        self.ct5 = nn.ConvTranspose2d(128,64,kernel_size=3,stride=2,padding=1,output_padding=1) # B,64,64,64
        self.bn5 = nn.BatchNorm2d(64)
        self.act5 = nn.LeakyReLU()
        self.ct6 = nn.ConvTranspose2d(64,3,kernel_size=3,stride=2,padding=1,output_padding=1) # B,3,128,128
        #self.bn6 = nn.BatchNorm2d(3)
        # self.act6 = nn.LeakyReLU()
        self.act6 = nn.Sigmoid()

    def forward(self,x):
        x = self.dact1(self.dbn1(self.dense1(x)))
        x = x.view(-1,128,2,2)
        x = self.act1(self.bn1(self.ct1(x)))
        x = self.act2(self.bn2(self.ct2(x)))
        x = self.act3(self.bn3(self.ct3(x)))
        x = self.act4(self.bn4(self.ct4(x)))
        x = self.act5(self.bn5(self.ct5(x)))
        x = self.act6(self.ct6(x))
        return x
    
def KL_Divergence_Loss(z_mean,z_logvar):
    kl_loss = -0.5 * torch.sum(1 + z_logvar - z_mean**2 - torch.exp(z_logvar), dim=1)
    # we dont want to keepdim=true because we just want B not B,1
    kl_loss = kl_loss.mean()
    return kl_loss


class AutoEncoder(nn.Module):
    def __init__(self,encoder,decoder,sampling_layer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.sampling_layer = sampling_layer

    def forward(self,x):
        z_mean,z_logvar = self.encoder(x)
        loss = KL_Divergence_Loss(z_mean,z_logvar)
        z_sample = self.sampling_layer(z_mean,z_logvar)
        out = self.decoder(z_sample)
        return loss,out
    

def generate_samples(num_samples=4):
    zsamples = torch.randn(num_samples, 200).to(device)
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 5))

    with torch.no_grad():
        for i in range(num_samples):
            gen_img = decoder(zsamples[i].unsqueeze(0))  # Unsqueeze to add batch dimension
            gen_img = gen_img.cpu().detach().numpy().squeeze(0)  # Convert to numpy and remove batch dimension
            gen_img = gen_img.transpose(1, 2, 0)  # H,W,C format
            
            axes[i].imshow(gen_img)
            axes[i].axis('off')

        plt.show()


if __name__ == "__main__":
    epochs = 50
    beta = 0.1
    batch_size = 128

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = Encoder().to(device)
    decoder = Decoder().to(device)
    sampling_layer = Sampling().to(device)
    model = AutoEncoder(encoder,decoder,sampling_layer).to(device)
    state_dict = torch.load('final_model_weights.pth', map_location=device)
    model.load_state_dict(state_dict)
    
    model.eval()
    parser = argparse.ArgumentParser(description="VAE Generate Samples")
    parser.add_argument('--samples', type=int, default=6, help='Number of samples to generate')
    args = parser.parse_args()

    generate_samples(args.samples)