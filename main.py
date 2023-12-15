import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms
from torch.autograd import Function
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr

from hybrid_conv import ConvDenoiseNet
from datasets import train_dataset, test_dataset

from constant import *
from add_noise import add_gaussian_noise
import argparse

from qiskit_ibm_provider import IBMProvider
from qiskit import Aer, IBMQ

from skimage.metrics import structural_similarity as ssim

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', 
					action='store_true', 
					help="test the model")
    args = parser.parse_args()
    

    train_loader = train_dataset(n_samples=100)
    test_loader = test_dataset(n_samples=10)

    if not args.test:
        # Create model
        model = ConvDenoiseNet()
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # number of epochs to train the model
        n_epochs = 50

        losses = []
        psnrs = []

        for epoch in tqdm(range(1, n_epochs + 1)):
            train_loss = 0.0
            for data in train_loader:
                images, _ = data
                noisy_imgs = add_gaussian_noise(images)
                optimizer.zero_grad()
                outputs = model(noisy_imgs)
                loss = criterion(outputs, images)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * images.size(0)

                outputs = outputs.detach().view(len(images), 1, 28, 28)
                batch_avg_psnr = 0
                for i in range(len(images)):
                    org = np.transpose(images[i], (1, 2, 0)).detach().numpy()
                    denoise = np.transpose(outputs[i], (1, 2, 0)).detach().numpy()
                    batch_avg_psnr += psnr(org, denoise)

            psnrs.append(batch_avg_psnr)

            train_loss = train_loss/len(train_loader)
            losses.append(train_loss)

        print('PSNRs: ', psnrs)

        # Plot losses
        plt.figure()
        plt.plot(losses)
        plt.savefig('losses.png')
        np.savetxt("losses.csv", np.array(losses), delimiter=",")

        # Plot PSNRs
        plt.figure()
        plt.plot(psnrs)
        plt.savefig('psnrs.png')
        np.savetxt("psnrs.csv", np.array(psnrs), delimiter=",")

        # Save the model
        torch.save(model.state_dict(), "model.pt")
    else:
        # Load model
        model = ConvDenoiseNet()
        model.load_state_dict(torch.load("model.pt"))

    # plot the first ten input images and then reconstructed images
    ncols = 10
    fig, axes = plt.subplots(nrows=3, ncols=ncols, sharex=True, sharey=True, figsize=(25, 7))
    # fig.tight_layout()

    model.eval()
    with torch.no_grad():
        for k in range(ncols):
            dataiter = iter(test_loader)
            images, labels = next(dataiter)
            noisy_imgs = add_gaussian_noise(images, 0.25)
            output = model(noisy_imgs)
            noisy_imgs = noisy_imgs.numpy()
            output = output.view(1, 1, 28, 28)
            output = output.cpu().detach().numpy()
            col_axes = axes[:, k]
            col_axes[0].imshow(np.squeeze(images), cmap='gist_gray')
            col_axes[1].imshow(np.squeeze(noisy_imgs), cmap='gist_gray')
            col_axes[2].imshow(np.squeeze(output), cmap='gist_gray')

        plt.savefig('denoised_inputs.png', dpi=400)

    # avg_psnr = 0
    # test_size = 0

    # for data in test_loader:
    #     images = data[0]
    #     noisy_imgs = add_gaussian_noise(images, 0.5)
    #     outputs = model(noisy_imgs)
    #     outputs = outputs.detach().view(len(images), 1, 28, 28)
    #     batch_avg_psnr = 0
    #     for i in range(len(images)):
    #         org = np.transpose(images[i], (1, 2, 0)).numpy()
    #         denoise = np.transpose(outputs[i], (1, 2, 0)).numpy()
    #         batch_avg_psnr += psnr(org, denoise)
    #     avg_psnr += batch_avg_psnr
    #     test_size += len(images)
    # print(
    #     "On Test data of {} examples:\nAverage PSNR: {:.3f}".format(
    #         test_size, avg_psnr / test_size
    #     )
    # )

    model.eval()
    with torch.no_grad():
        for data in test_loader:
            images = data[0]
            noisy_imgs = add_gaussian_noise(images, sigma=1)
            output = model(noisy_imgs)
            output = output.view(len(images), 1, 28, 28)
            output = output.detach().cpu()
            # for i in range(len(images)):
            #     org = images[i].numpy().squeeze()
            #     denoise = output[i].numpy().squeeze()
            #     ssim_val = ssim(org, denoise, full=True, data_range=data_range)
            images = images.numpy().squeeze()
            output = output.numpy().squeeze()
            ssim_val = ssim(images, output, data_range=1.0)

    print("SSIM: ", ssim_val)


if __name__ == "__main__":
    main()