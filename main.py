import wandb
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
from src.datasets import get_dataset

from ..utils import get_loops, sample_class_data, gradient_loss, TensorDataset, epoch_train
import torchvision.utils as vutils

torch.autograd.set_detect_anomaly(True)
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description="Distillation Training Pipeline")
    parser.add_argument('--dataset', type=str, default='AudioMNIST', help='dataset')
    parser.add_argument('--method', type=str, default='DM', help='DM/GM/TM')
    parser.add_argument("-spc", "--spc", type=float, default=10, help="Number of synthetic samples per class")
    parser.add_argument("-e", "--epochs", type=int, default=800, help="Number of epochs for training")
    parser.add_argument('--num_exp', type=int, default=1, help='the number of experiments')
    parser.add_argument('--num_eval', type=int, default=3, help='the number of evaluating randomly initialized models')
    parser.add_argument('--epoch_eval_train', type=int, default=100, help='epochs to train a model with synthetic data')
    parser.add_argument('--Iteration', type=int, default=2000, help='training iterations')
    parser.add_argument('--lr_img', type=float, default=2.0, help='learning rate for updating synthetic images')
    parser.add_argument('--lr_net', type=float, default=0.1, help='learning rate for updating network parameters')
    parser.add_argument('--batch_real', type=int, default=64, help='batch size for real data')
    parser.add_argument('--batch_train', type=int, default=64, help='batch size for training networks')
    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--save_path', type=str, default='result', help='path to save results')
    parser.add_argument('--use_wandb', type=bool, default=True, help='Use wandb for logging')

    args = parser.parse_args()
    USE_WANDB = False
    NUM_EPOCHS = args.epochs
    PLOT_TSNE = args.plot_tsne
    FEATURE_TYPE = args.feature_type
    LR = 0.1
    LR_NET = 0.01
    DATA_SIZE = (128, 128)
    BATCH_REAL = 256
    METHOD = args.method

    num_classes = 10
    channel = 1
    outer_loops, inner_loops = get_loops(args.spc)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")



    # Initialize synthetic dataset with random data
    synth_data = torch.randn(
        size=(num_classes * args.spc, channel, DATA_SIZE[0], DATA_SIZE[1]),
        dtype=torch.float, requires_grad=True, device=device
    )
    synth_labels = torch.tensor(
        [[i] * args.spc for i in range(num_classes)],
        dtype=torch.long, requires_grad=False, device=device
    ).flatten()

    # Initialize real data
    all_real_data, all_real_labels = [], []
    class_indices = [[] for _ in range(num_classes)]
    all_real_data = [torch.unsqueeze(dataset[i][0], dim=0) for i in range(len(dataset))]
    all_real_labels = [dataset[i][1] for i in range(len(dataset))]
    for i, lab in enumerate(all_real_labels):
        class_indices[lab].append(i)
    all_real_data = torch.cat(all_real_data, dim=0).to(device)
    all_real_labels = torch.tensor(all_real_labels, dtype=torch.long, device=device)

    for class_id in range(num_classes):
        print(f"Class {class_id} ({dataset.genres[class_id]}): {len(class_indices[class_id])} {FEATURE_TYPE}s available")

    # Initialize optimizer and loss function for distillation
    param_groups = [
        {'params': synth_data, 'lr': LR}
    ]
    optimizer_dist = optim.SGD(param_groups, momentum=0.5)
    optimizer_dist.zero_grad()
    criterion = nn.CrossEntropyLoss().to(device)

    # Initialize wandb
    if USE_WANDB:
        wandb.init(project="distillation-training-GTZAN-DM", config={
            "method": args.method,
            "ipc": args.spc,
            "epochs": NUM_EPOCHS,
            "plot_tsne": PLOT_TSNE,
            "feature_type": FEATURE_TYPE,
            "batch_size": BATCH_REAL

        })
    data_save = []

    # Crear carpeta de salida si no existe
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # ---- Training loop ----
    for epoch in range(NUM_EPOCHS):
        # if PLOT_TSNE:
        #     plot_tsne_audio(real_audio_embeddings, synthetic_audio, epoch)
        #     plot_tsne_text(real_text_embeddings, synthetic_text, epoch)

        # Save visualization of synthetic data
        # save_name = os.path.join(args.save_path, 'vis_%s_%s_%s_%dipc_exp%d_iter%d.png'%(args.method, args.dataset, args.model, args.ipc, exp, it))
        # image_syn_vis = copy.deepcopy(image_syn.detach().cpu())
        # for ch in range(channel):
        #     image_syn_vis[:, ch] = image_syn_vis[:, ch]  * std[ch] + mean[ch]
        # image_syn_vis[image_syn_vis<0] = 0.0
        # image_syn_vis[image_syn_vis>1] = 1.0
        # save_image(image_syn_vis, save_name, nrow=args.ipc) # Trying normalize = True/False may get better visual effects.

        net = MusicGenreCNN().to(device)
        # load the pre-trained model
        net.load_state_dict(torch.load('music_genre_cnn.pth'))
        net.train()
        net_parameters = list(net.parameters())
        optimizer_net = torch.optim.SGD(net.parameters(), lr=LR_NET) 
        optimizer_net.zero_grad()
        loss_avg = 0
        if METHOD == 'GM':
            for iter_class in range(outer_loops):
                BN_flag = False
                BNSizePC = 16  # for batch normalization
                for module in net.modules():
                    if 'BatchNorm' in module._get_name(): #BatchNorm
                        BN_flag = True
                if BN_flag:
                    real_sample = torch.cat([sample_class_data(c, BNSizePC, class_indices, all_real_data) for c in range(num_classes)], dim=0)
                    net.train() 
                    output_real = net(real_sample)
                    for module in net.modules():
                        if 'BatchNorm' in module._get_name():
                            module.eval() 

                # update the synthetic data
                loss = torch.tensor(0.0).to(device)
                for c in range(num_classes):
                    real_sample =  sample_class_data(c, BATCH_REAL, class_indices, all_real_data)
                    batch_size, num_chunks, channels, height, width = real_sample.shape
                    real_sample = real_sample.view(batch_size * num_chunks, channels, height, width)
                    lab_real = torch.ones((real_sample.shape[0],), device=device, dtype=torch.long) * c
                    synth_sample = synth_data[c*args.spc:(c+1)*args.spc].reshape((args.spc, channel, DATA_SIZE[0], DATA_SIZE[1]))
                    lab_synth = torch.ones((args.spc,), device=device, dtype=torch.long) * c

                    output_real = net(real_sample)
                    loss_real = criterion(output_real, lab_real)
                    gw_real = torch.autograd.grad(loss_real, net_parameters)
                    gw_real = list((_.detach().clone() for _ in gw_real))

                    output_syn = net(synth_sample)
                    loss_syn = criterion(output_syn, lab_synth)
                    gw_syn = torch.autograd.grad(loss_syn, net_parameters, create_graph=True)
                    loss += gradient_loss(gw_syn, gw_real, device)

                optimizer_dist.zero_grad()
                loss.backward(retain_graph=True)
                optimizer_dist.step()
                loss_avg += loss.item()

                if iter_class == outer_loops- 1:
                    break

                # Update network
                synth_data_train, label_syn_train = copy.deepcopy(synth_data.detach()), copy.deepcopy(synth_labels.detach())  # avoid any unaware modification
                dst_syn_train = TensorDataset(synth_data_train, label_syn_train)
                trainloader = torch.utils.data.DataLoader(dst_syn_train, batch_size=BATCH_REAL, shuffle=True, num_workers=0)
                for il in range(inner_loops):
                    loss_avg_cnn, acc_avg_cnn = epoch_train('train', trainloader, net, optimizer_net, criterion, device)

                optimizer_net.zero_grad()
                loss.backward()
                optimizer_net.step()
                loss_avg += loss.item()

                
                loss_avg /= (num_classes*outer_loops)
        else:
            print(f'Method: {METHOD}')
            for param in list(net.parameters()):
                param.requires_grad = False

            embed = net.embed  # for MPS or single GPU

            loss_avg = 0
            loss = torch.tensor(0.0).to(device)
            for c in range(num_classes):
                real_sample =  sample_class_data(c, BATCH_REAL, class_indices, all_real_data)
                batch_size, num_chunks, channels, height, width = real_sample.shape
                real_sample = real_sample.view(batch_size * num_chunks, channels, height, width)

                synth_sample = synth_data[c*args.spc:(c+1)*args.spc].reshape((args.spc, channel, DATA_SIZE[0], DATA_SIZE[1]))

                output_real = embed(real_sample).detach()
                output_syn = embed(synth_sample)

                loss += torch.sum((torch.mean(output_real, dim=0) - torch.mean(output_syn, dim=0))**2)

                optimizer_dist.zero_grad()
                loss.backward(retain_graph=True)
                optimizer_dist.step()
                loss_avg += loss.item()

            

            loss_avg /= (num_classes)


        if epoch%10 == 0:
            print(f"iter = {epoch:04d}, loss = {loss_avg:.4f}")

        # Guardar visualización de datos sintéticos
        if epoch % 25 == 0:
            synth_data_vis = synth_data.detach().cpu()
            save_path = os.path.join(output_dir, f'synth_data_epoch_{epoch}.png')
            vutils.save_image(synth_data_vis, save_path, nrow=args.spc, normalize=True)

        # if epoch == args.Iteration: # only record the final results
        #     data_save.append([copy.deepcopy(synth_data.detach().cpu()), copy.deepcopy(synth_labels.detach().cpu())])
        #     torch.save({'data': data_save}, os.path.join(args.save_path, 'res_%s_%s_%s_%dipc.pt'%(args.method, args.dataset, args.model, args.ipc)))

        # Log loss to wandb
        if USE_WANDB:
            wandb.log({"loss_distillation": loss_avg})
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {loss}")


    # Finish wandb run
    if USE_WANDB:
        wandb.finish()