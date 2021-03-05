import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision
from tensorboardX import SummaryWriter
from os import path, makedirs
from options import TrainOptions, printInfo
from glob import glob
from tqdm import tqdm
from dataset import Test_Dataset, Train_Dataset, AVA_Dataset 
from torchvision.utils import save_image
from math import sqrt, ceil
from network import ImageDiscriminator, ImageCropper
import numpy as np

image_generator = ImageCropper()
image_discriminator = ImageDiscriminator()
criterion = nn.CrossEntropyLoss()

CRITIC_ITER = 4

if __name__ == "__main__":
    opts = TrainOptions().parse()
    negative_dataset = Train_Dataset()
    negative_loader = torch.utils.data.DataLoader(negative_dataset, batch_size=opts.batch_size, shuffle=True, num_workers=4, drop_last=True)
    positive_dataset = AVA_Dataset()
    positive_loader = torch.utils.data.DataLoader(positive_dataset, batch_size=opts.batch_size, shuffle=True, num_workers=4, drop_last=True)
    test_dataset = Test_Dataset()
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=opts.batch_size, shuffle=True, num_workers=4, drop_last=True)
    start_epoch = 0

    if opts.resume: #TODO
        pass
    else:
        printInfo("Train from the beginning!")
    
    writer = SummaryWriter(log_dir=opts.log_dir)
    writer.add_graph(image_discriminator, torch.randn(1, 3, 224, 224))
    writer.add_graph(image_generator, torch.randn(1, 3, 640, 480))
    # Multi-GPU support
    if torch.cuda.device_count() > 1 and opts.multi_gpu > 1:
        printInfo("Multiple GPU: %d GPUs are available!" % torch.cuda.device_count())
        image_generator = nn.DataParallel(image_generator, device_ids=[0,1,2,3])
        image_discriminator = nn.DataParallel(image_discriminator, device_ids=[0,1,2,3])
    else:
        image_generator = nn.DataParallel(image_generator, device_ids=[opts.cuda])
        image_discriminator = nn.DataParallel(image_discriminator, device_ids=[opts.cuda])

    
    image_generator.to(opts.device)
    image_discriminator.to(opts.device)

    G_optimizer = optim.Adam(image_generator.parameters(), lr=opts.lr, betas=(0.9, 0.999))
    D_optimizer = optim.Adam(image_discriminator.parameters(), lr=opts.lr, betas=(0.9, 0.999))


    #Training
    Loss_G_list = [0.0]
    Loss_D_list = [0.0]
    Loss_T_list = [0.0]

    for epoch in range(start_epoch, opts.epoches):
        printInfo("Epoch: %d  @lr: %f" % (epoch, G_optimizer.param_groups[0]['lr']))
        bar = tqdm(positive_loader)
        loss_G_list = []
        loss_D_list = []

        good_correct = 0
        good_total = 0
        common_correct = 0
        common_total = 0
        common_image_iter = negative_loader.__iter__()

        for i, good_img in enumerate(bar):
            image_discriminator.zero_grad()
            good_img = good_img.to(opts.device)
            good_scores = image_discriminator(good_img)
            good_labels = torch.ones(good_img.size(0)).long().to(opts.device) #好图片
            good_loss = criterion(good_scores, good_labels)

            good_loss.backward()
            D_optimizer.step()

            loss_D_list.append(good_loss.item())

            _, predicted = torch.max(good_scores, dim=1)
            good_total += good_img.size(0)
            good_correct +=  (predicted == good_labels).sum().item()
            

            if i % CRITIC_ITER == 0:
                image_generator.zero_grad()
                inputs = common_image_iter.__next__()
                inputs = inputs.to(opts.device)
                cropped_pictures = image_generator(inputs)
                common_scores = image_discriminator(cropped_pictures)
                common_labels = torch.zeros(inputs.size(0)).long().to(opts.device) #假图片
                common_loss = criterion(common_scores, common_labels)
                loss_G_list.append(common_loss.item())                
                save_image(inputs, path.join(opts.img_dir, 'original_img%d.png' % (epoch + 1)), nrow=ceil(sqrt(inputs.size(0))), normalize=True)
                save_image(cropped_pictures, path.join(opts.img_dir, 'test_img%d.png' % (epoch + 1)), nrow=ceil(sqrt(cropped_pictures.size(0))), normalize=True)
                common_loss.backward()
                G_optimizer.step()

                _, predicted = torch.max(common_scores.data, dim=1)
                common_correct += (common_labels == predicted).sum().item()
                common_total += inputs.size(0)
            
            bar.set_description("Epoch {} [{}, {}] [G]: {} [D]: {}".format(epoch, i+1, len(positive_loader), loss_G_list[-1], loss_D_list[-1]))
        
        Loss_G_list.append(np.mean(loss_G_list))
        Loss_D_list.append(np.mean(loss_D_list))
        

        with torch.no_grad():
            loss_Test_list = []
            test_correct = 0
            test_total = 0
            for i, test_img in enumerate(test_loader):
                test_img = test_img.to(opts.device)
                test_cropped_picture = image_generator(test_img)

                test_scores = image_discriminator(test_cropped_pictures)
                test_labels = torch.zeros(test_img.size(0)).long() #假图片
                test_loss = criterion(test_scores, test_labels)
                loss_Test_list.append(test_loss.item())

                _, predicted = torch.max(test_scores.data, dim=1)
                test_correct += (test_labels == predicted).sum().item()
                test_total += test_img.size(0)
            
            Loss_T_list.append(np.mean(loss_Test_list))
            writer.add_scalars("Loss", {'good_images': np.mean(loss_D_list), 'common_images': np.mean(loss_G_list), 'test_images': np.mean(loss_Test_list)}, epoch + 1)
            writer.add_scalars("Accuracy", {'good_images': good_correct / good_total, "common_images": common_correct / common_total, "test_images": test_correct / test_total}, epoch + 1)

            save_image(test_img, path.join(opts.img_dir, 'original_img%d.png' % (epoch + 1)), nrow=ceil(sqrt(test_img.size(0))), normalize=True)
            save_image(test_cropped_picture, path.join(opts.img_dir, 'test_img%d.png' % (epoch + 1)), nrow=ceil(sqrt(test_cropped_img.size(0))), normalize=True)


        state = {
            'image_discriminator': image_discriminator.state_dict(),
            'image_generator': image_generator.state_dict(),
            'Loss_G': Loss_G_list,
            'Loss_D': Loss_D_list,
            'Loss_T': Loss_T_list,
            "start_epoch": epoch + 1
        }


        torch.save(state, path.join(opts.model_dir, 'latest.pth'))
        if (epoch + 1) % opts.save_checkpoints == 0:
            torch.save(state, path.join(opts.model_dir, 'model%d.pth' % (epoch + 1)))

    writer.close()
    printInfo("Training complete!")