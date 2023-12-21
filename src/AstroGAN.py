import torch
import torch.nn as nn
from torch.autograd import Variable
import time as t
import os
from src.utils.metrics import Logger, get_inception_score
from itertools import chain
from torchvision import utils
import numpy as np


class Generator(torch.nn.Module):
    def __init__(self, channels, latent_dim=100, h_channels=[1024, 512, 256], act=nn.ReLU, kernel_size=4):
        super().__init__()
        self.latent_dim = latent_dim
        self.channels = channels
        cnn_modules = []

        in_channels = latent_dim

        for out_channels in h_channels:
            pad = 0 if in_channels == self.latent_dim else 1
            stride = 1 if in_channels == self.latent_dim else 2    
            cnn_modules.append(nn.Sequential(
                nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=pad),
                nn.BatchNorm2d(num_features=out_channels),
                act()
                ))
            
            in_channels = out_channels

        cnn_modules.append(nn.ConvTranspose2d(in_channels=h_channels[-1], out_channels=channels, kernel_size=kernel_size, stride=stride, padding=pad))

        self.cnn_modules = nn.Sequential(*cnn_modules)

        self.act = nn.Tanh()

    def forward(self, x):
        x = self.cnn_modules(x)
        x = self.act(x)
        return x

    def features_extraction(self, x):
        x = self.cnn_modules(x)
        return x

class Discriminator(torch.nn.Module):
    def __init__(self, channels, h_channels=[256, 512, 1024], kernel_size=4):
        super().__init__()
        self.channels = channels
        cnn_modules = []
        in_channels = channels
        for out_channels in h_channels:  
            cnn_modules.append(nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=2, padding=1),
                nn.BatchNorm2d(num_features=out_channels),
                nn.LeakyReLU(0.2, inplace=True)
                ))
            in_channels = out_channels

        self.cnn_module = nn.Sequential(*cnn_modules)

        self.out_module = (nn.Sequential(nn.Conv2d(in_channels=h_channels[-1], out_channels=1, kernel_size=4, stride=1, padding=0),
                                        nn.Sigmoid(),
                                        nn.Flatten(0)))
        
        

    def forward(self, x):
        x = self.cnn_module(x)
        x = self.out_module(x)
        return x
    
    def feature_extraction(self, x):
        x = self.cnn_module(x)
        return x.view(x.size(0), -1)


class ConditionalGenerator(torch.nn.Module):

    def __init__(self, channels, generator_ref, latent_dim_ref=100, latent_dim_sci=100, latent_dim_emb_ref=50, 
                    h_channels=[1024, 512, 256], act=nn.ReLU, kernel_size=4):
        
        super().__init__()
        # Attributes of Conditional generator model
        self.latent_dim_ref = latent_dim_ref
        self.latent_dim_sci = latent_dim_sci
        self.latent_dim_emb_ref = latent_dim_emb_ref
        self.channels = channels
        cnn_modules = []
        
        # Generator module of reference Channel
        self.generator_ref = generator_ref

        in_channels = latent_dim_sci

        for out_channels in h_channels:
            pad = 0 if in_channels == self.latent_dim_sci else 1
            stride = 1 if in_channels == self.latent_dim_sci else 2    
            cnn_modules.append(nn.Sequential(
                nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=pad),
                nn.BatchNorm2d(num_features=out_channels),
                act()
                ))
            
            in_channels = out_channels

        # Linear transformation of embedding.
        self.W = nn.Linear(latent_dim_ref, latent_dim_emb_ref)

        # Generator of Science Channel
        cnn_modules.append(nn.ConvTranspose2d(in_channels=h_channels[-1], out_channels=channels, kernel_size=kernel_size, stride=stride, padding=pad))

        self.cnn_modules = nn.Sequential(*cnn_modules)

        self.act = nn.Tanh()

    def forward(self, x):
        # x dim [batch, (ref_dim+(sci_dim-emb_dim)]

        z_ref = x[:, :self.latent_dim_ref]
        
        z_emb = self.W(z_ref.view(-1, self.latent_dim_ref)).view(-1, self.latent_dim_emb_ref, 1, 1)
        z_sci = x[:, self.latent_dim_ref:]

        # z dim = [batch, latent_sci_dim]
        z = torch.hstack((z_emb, z_sci))
        # Calculation of the science features (not activated)
        x_sci = self.cnn_modules(z)

        # Calcualtion of generated image with detach gradient.
        x_ref = self.generator_ref(z_ref).detach()

        # Embedding
        x_sci[:,0,:,:] = x_sci[:,0,:,:] + x_ref[:,0,:,:]

        # Activation of linear combination of features
        x = self.act(x_sci)
        return x
    
    def generate_sci_ref(self, x):
        z_ref = x[:, :self.latent_dim_ref]
        
        z_emb = self.W(z_ref.view(-1, self.latent_dim_ref)).view(-1, self.latent_dim_emb_ref, 1, 1)
        z_sci = x[:, self.latent_dim_ref:]

        # z dim = [batch, latent_sci_dim]
        z = torch.hstack((z_emb, z_sci))
        # Calculation of the science features (not activated)
        x_sci = self.forward(x)

        # Calcualtion of generated image with detach gradient.
        x_ref = self.generator_ref(z_ref).detach()

        x_combined = torch.hstack((x_sci, x_ref))

        return x_combined



    def features_extraction(self, x):
        x = self.cnn_modules(x)
        return x


class AstroGAN(object):
    def __init__(self, device = "cuda", epochs=50, batch_size=128):
        print("AstroGAN model initalization.")
        self.G_ref= Generator(1).to(device)
        self.G_sci = None
        self.D_ref = Discriminator(1).to(device)
        self.D_sci = Discriminator(2).to(device)
        self.C = 1

        self.G_ref_trained = False
        self.G_sci_trained = False

        # binary cross entropy loss and optimizer
        self.loss = nn.BCELoss()

        self.device = device

        # Using lower learning rate than suggested by (ADAM authors) lr=0.0002  and Beta_1 = 0.5 instead od 0.9 works better [Radford2015]
        self.d_optimizer_ref = torch.optim.Adam(self.D_ref.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.g_optimizer_ref = torch.optim.Adam(self.G_ref.parameters(), lr=0.0002, betas=(0.5, 0.999))

        self.d_optimizer_sci = None #torch.optim.Adam(self.D_sci.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.g_optimizer_sci = None #torch.optim.Adam(self.G_sci.parameters(), lr=0.0002, betas=(0.5, 0.999))

        self.epochs = epochs
        self.batch_size = batch_size

        # Set the logger
        self.logger_ref = Logger('./logs_ref')
        self.logger_sci = Logger('./logs_sci')
        self.number_of_images = 64


    def train_ref(self, train_loader):
        self.t_begin = t.time()
        generator_iter = 0

        for epoch in range(self.epochs):
            self.epoch_start_time = t.time()

            for i, (images, _) in enumerate(train_loader):
                # Check if round number of batches
                if i == train_loader.dataset.__len__() // self.batch_size:
                    break

                z = torch.rand((self.batch_size, 100, 1, 1))
                real_labels = torch.ones(self.batch_size)
                fake_labels = torch.zeros(self.batch_size)

                images, z = Variable(images).to(self.device), Variable(z).to(self.device)
                real_labels, fake_labels = Variable(real_labels).to(self.device), Variable(fake_labels).to(self.device)

                # Train discriminator
                # Compute BCE_Loss using real images
                outputs = self.D_ref(images)
                d_loss_real = self.loss(outputs, real_labels)
                real_score = outputs

                # Compute BCE Loss using fake images
                z = Variable(torch.randn(self.batch_size, 100, 1, 1)).to(self.device)

                fake_images = self.G_ref(z)
                outputs = self.D_ref(fake_images)
                d_loss_fake = self.loss(outputs.flatten(), fake_labels)
                fake_score = outputs

                # Optimize discriminator
                d_loss = d_loss_real + d_loss_fake
                self.D_ref.zero_grad()
                d_loss.backward()
                self.d_optimizer_ref.step()

                # Train generator
                # Compute loss with fake images
                z = Variable(torch.randn(self.batch_size, 100, 1, 1)).to(self.device)

                fake_images = self.G_ref(z)
                outputs = self.D_ref(fake_images)
                g_loss = self.loss(outputs.flatten(), real_labels)

                # Optimize generator
                self.D_ref.zero_grad()
                self.G_ref.zero_grad()
                g_loss.backward()
                self.g_optimizer_ref.step()
                generator_iter += 1


                if generator_iter % 1000 == 0:
                    print('Epoch-Ref-{}'.format(epoch + 1))
                    self.save_model()

                    if not os.path.exists('training_ref_result_images/'):
                        os.makedirs('training_ref_result_images/')

                    # Denormalize images and save them in grid 8x8
                    z = Variable(torch.randn(800, 100, 1, 1)).to(self.device)
                    samples = self.G_ref(z)
                    samples = samples.mul(0.5).add(0.5)
                    samples = samples.data.cpu()[:64]
                    grid = utils.make_grid(samples)
                    utils.save_image(samples, 'training_result_images/img_generatori_iter_{}.png'.format(str(generator_iter).zfill(3)))

                    time = t.time() - self.t_begin
                    #print("Inception score: {}".format(inception_score))
                    print("Generator iter: {}".format(generator_iter))
                    print("Time {}".format(time))


                if ((i + 1) % 19) == 0:
                    print("Epoch: [%2d] [%4d/%4d] D_loss: %.8f, G_loss: %.8f" %
                          ((epoch + 1), (i + 1), train_loader.dataset.__len__() // self.batch_size, d_loss.data, g_loss.data))

                    z = Variable(torch.randn(self.batch_size, 100, 1, 1).to("cuda"))

                    # TensorBoard logging
                    # Log the scalar values
                    info = {
                        'd_loss': d_loss.data,
                        'g_loss': g_loss.data
                    }

                    for tag, value in info.items():
                        self.logger_ref.scalar_summary(tag, value, generator_iter)

                    # Log values and gradients of the parameters
                    for tag, value in self.D_ref.named_parameters():
                        tag = tag.replace('.', '/')
                        self.logger_ref.histo_summary(tag, self.to_np(value), generator_iter)
                        self.logger_ref.histo_summary(tag + '/grad', self.to_np(value.grad), generator_iter)
                    # Log the images while training
                    info = {
                        'real_images': self.real_images(images, self.number_of_images),
                        'generated_images': self.generate_img_ref(z, self.number_of_images)
                    }

                    for key in info:
                        self.logger_ref.image_summary(key, info[key], generator_iter)


        self.t_end = t.time()
        print('Time of training-{}'.format((self.t_end - self.t_begin)))
        #self.file.close()

        # Save the trained parameters
        self.save_model()

    def train_sci(self, train_loader, load_ref=True, d_ref_dir='./discriminator_ref.pkl', g_ref_dir='./generator_ref.pkl', epochs=1000):
        
        self.epochs = epochs
        if load_ref:
            self.load_model(d_ref_dir, g_ref_dir, ref=True)

        self.G_ref = self.G_ref.to(self.device)
        self.G_sci = ConditionalGenerator(2, self.G_ref).to(self.device)

        self.d_optimizer_sci = torch.optim.Adam(self.D_sci.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.g_optimizer_sci = torch.optim.Adam(self.G_sci.parameters(), lr=0.0002, betas=(0.5, 0.999))

        self.t_begin = t.time()
        generator_iter = 0

        for epoch in range(self.epochs):
            self.epoch_start_time = t.time()

            for i, (images, _) in enumerate(train_loader):
                # Check if round number of batches
                if i == train_loader.dataset.__len__() // self.batch_size:
                    break

                z = torch.rand((self.batch_size, 150, 1, 1))
                real_labels = torch.ones(self.batch_size)
                fake_labels = torch.zeros(self.batch_size)

                images, z = Variable(images).to(self.device), Variable(z).to(self.device)
                real_labels, fake_labels = Variable(real_labels).to(self.device), Variable(fake_labels).to(self.device)

                # Train discriminator
                # Compute BCE_Loss using real images
                outputs = self.D_sci(images)
                d_loss_real = self.loss(outputs, real_labels)
                real_score = outputs

                # Compute BCE Loss using fake images
                z = Variable(torch.randn(self.batch_size, 150, 1, 1)).to(self.device)

                fake_images = self.G_sci(z)
                outputs = self.D_sci(fake_images)
                d_loss_fake = self.loss(outputs.flatten(), fake_labels)
                fake_score = outputs

                # Optimize discriminator
                d_loss = d_loss_real + d_loss_fake
                self.D_sci.zero_grad()
                d_loss.backward()
                self.d_optimizer_sci.step()

                # Train generator
                # Compute loss with fake images
                z = Variable(torch.randn(self.batch_size, 150, 1, 1)).to(self.device)

                fake_images = self.G_sci(z)
                outputs = self.D_sci(fake_images)
                g_loss = self.loss(outputs.flatten(), real_labels)

                # Optimize generator
                self.D_sci.zero_grad()
                self.G_sci.zero_grad()
                g_loss.backward()
                self.g_optimizer_sci.step()
                generator_iter += 1


                if generator_iter % 1000 == 0:
                    print('Epoch-{}'.format(epoch + 1))
                    self.save_model()

                    if not os.path.exists('training_result_images_ag/'):
                        os.makedirs('training_result_images_ag/')

                    # Denormalize images and save them in grid 8x8
                    z = Variable(torch.randn(800, 150, 1, 1)).to(self.device)
                    samples = self.G_sci(z)
                    samples = samples.mul(0.5).add(0.5)
                    samples = samples.data.cpu()[:64]
                    grid = utils.make_grid(samples)
                    utils.save_image(samples, 'training_result_images_ag/img_generatori_iter_{}.png'.format(str(generator_iter).zfill(3)))

                    time = t.time() - self.t_begin
                    #print("Inception score: {}".format(inception_score))
                    print("Generator iter: {}".format(generator_iter))
                    print("Time {}".format(time))


                if ((i + 1) % 19) == 0:
                    print("Epoch Sci: [%2d] [%4d/%4d] D_loss: %.8f, G_loss: %.8f" %
                          ((epoch + 1), (i + 1), train_loader.dataset.__len__() // self.batch_size, d_loss.data, g_loss.data))

                    z = Variable(torch.randn(self.batch_size, 150, 1, 1)).to(self.device)

                    # TensorBoard logging
                    # Log the scalar values
                    info = {
                        'd_loss': d_loss.data,
                        'g_loss': g_loss.data
                    }

                    for tag, value in info.items():
                        self.logger_sci.scalar_summary(tag, value, generator_iter)

                    # Log values and gradients of the parameters
                    for tag, value in self.D_sci.named_parameters():
                        tag = tag.replace('.', '/')
                        self.logger_sci.histo_summary(tag, self.to_np(value), generator_iter)
                        self.logger_sci.histo_summary(tag + '/grad', self.to_np(value.grad), generator_iter)
                    # Log the images while training



        self.t_end = t.time()
        print('Time of training-{}'.format((self.t_end - self.t_begin)))
        #self.file.close()

        # Save the trained parameters
        self.save_model(ref=False)



    #TODO
    def evaluate(self, test_loader, D_model_path, G_model_path):
        self.load_model(D_model_path, G_model_path)
        z = Variable(torch.randn(self.batch_size, 100, 1, 1)).to(self.device)
        samples = self.G(z)
        samples = samples.mul(0.5).add(0.5)
        samples = samples.data.cpu()
        grid = utils.make_grid(samples)
        print("Grid of 8x8 images saved to 'dgan_model_image.png'.")
        utils.save_image(grid, 'dgan_model_image.png')

    #TODO
    def real_images(self, images, number_of_images, c=None):
        if c is None:
            c = self.C
        if (c >1):
            return self.to_np(images.view(-1, self.C, 32, 32)[:self.number_of_images])
        else:
            return self.to_np(images.view(-1, 32, 32)[:self.number_of_images])


    def generate_img_ref(self, z, number_of_images):
        samples = self.G_ref(z).data.cpu().numpy()[:number_of_images]
        generated_images = []
        for sample in samples:
            if self.C == 3:
                generated_images.append(sample.reshape(self.C, 32, 32))
            else:
                generated_images.append(sample.reshape(32, 32))
        return generated_images
    
    def generate_img_sci(self, z, number_of_images):
        samples = self.G_sci(z).data.cpu().numpy()[:number_of_images]
        generated_images = []
        for sample in samples:
            generated_images.append(sample.reshape(2, 32, 32))
        return generated_images
    
    def generate_img(self, z, number_of_images):
        samples = self.G_sci.generate_sci_ref(z).data.cpu().numpy()[:number_of_images]
        generated_images = []
        for sample in samples:
            generated_images.append(sample.reshape(32, 32))
        return generated_images

    def to_np(self, x):
        return x.data.cpu().numpy()

    def save_model(self, ref=True):
        if ref:
            torch.save(self.G_ref.state_dict(), './generator_ref.pkl')
            torch.save(self.D_ref.state_dict(), './discriminator_ref.pkl')
            print('Models save to ./generator_ref.pkl & ./discriminator.pkl ')
        else:
            torch.save(self.G_sci.state_dict(), './generator_sci.pkl')
            torch.save(self.D_sci.state_dict(), './discriminator_sci.pkl')
            print('Models save to ./generator_sci.pkl & ./discriminator_sci.pkl ')

    def load_model(self, D_model_filename, G_model_filename, ref=True):
        D_model_path = os.path.join(os.getcwd(), D_model_filename)
        G_model_path = os.path.join(os.getcwd(), G_model_filename)
        if ref:
            self.D_ref.load_state_dict(torch.load(D_model_path))
            self.G_ref.load_state_dict(torch.load(G_model_path))
        else:
            self.D_sci.load_state_dict(torch.load(D_model_path))
            self.G_sci.load_state_dict(torch.load(G_model_path))

        print('Generator model loaded from {}.'.format(G_model_path))
        print('Discriminator model loaded from {}-'.format(D_model_path))

    def load_full_model(self, D_model_filename_ref='./discriminator_ref.pkl', G_model_filename_ref='./generator_ref.pkl',
                         D_model_filename_sci='./discriminator_sci.pkl', G_model_filename_sci='./generator_sci.pkl'):
        

        self.load_model(D_model_filename_ref, G_model_filename_ref, ref=True)


        self.G_sci = ConditionalGenerator(2, self.G_ref).to(self.device)

        self.load_model(D_model_filename_sci, G_model_filename_sci, ref=False)

        print("AstroGAN Loaded Succesfully")




    def generate_latent_walk(self, number):
        if not os.path.exists('interpolated_images/'):
            os.makedirs('interpolated_images/')

        # Interpolate between twe noise(z1, z2) with number_int steps between
        number_int = 10
        z_intp = torch.FloatTensor(1, 100, 1, 1)
        z1 = torch.randn(1, 100, 1, 1)
        z2 = torch.randn(1, 100, 1, 1)
        if self.cuda:
            z_intp = z_intp.cuda()
            z1 = z1.cuda()
            z2 = z2.cuda()

        z_intp = Variable(z_intp)
        images = []
        alpha = 1.0 / float(number_int + 1)
        print(alpha)
        for i in range(1, number_int + 1):
            z_intp.data = z1*alpha + z2*(1.0 - alpha)
            alpha += alpha
            fake_im = self.G(z_intp)
            fake_im = fake_im.mul(0.5).add(0.5) #denormalize
            images.append(fake_im.view(self.C,32,32).data.cpu())

        grid = utils.make_grid(images, nrow=number_int )
        utils.save_image(grid, 'interpolated_images/interpolated_{}.png'.format(str(number).zfill(3)))
        print("Saved interpolated images to interpolated_images/interpolated_{}.".format(str(number).zfill(3)))
