import os
import time
import copy
import gc
import argparse
import numpy as np
import wandb
import torch
import torch.nn as nn
from torchvision.utils import save_image
from src.datasets import get_dataset
from src.utils import  get_network, evaluate_synset, get_time, info_nce_loss, sample_class_data, sample_negative_samples

def main():

    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--dataset', type=str, default='AUDIO_MNIST', help='dataset')
    parser.add_argument('--model', type=str, default='SimplifiedConvNet', help='model')
    parser.add_argument('--ipc', type=int, default=10, help='image(s) per class')
    parser.add_argument('--num_exp', type=int, default=1, help='the number of experiments')
    parser.add_argument('--num_eval', type=int, default=10, help='the number of evaluating randomly initialized models')
    parser.add_argument('--epoch_eval_train', type=int, default=100, help='epochs to train a model with synthetic data') # it can be small for speeding up with little performance drop
    parser.add_argument('--Iteration', type=int, default=1000, help='training iterations')
    parser.add_argument('--lr_img', type=float, default=2.0, help='learning rate for updating synthetic images')
    parser.add_argument('--lr_net', type=float, default=0.1, help='learning rate for updating network parameters')
    parser.add_argument('--batch_real', type=int, default=64, help='batch size for real data')
    parser.add_argument('--batch_train', type=int, default=64, help='batch size for training networks')
    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--save_path', type=str, default='result', help='path to save results')
    parser.add_argument('--use_wandb', type=bool, default=True, help='Use wandb for logging')
    parser.add_argument('--use_contrastive', type=bool, default=False, help='Use contrastive loss')
    parser.add_argument('--contrastive_weight', type=float, default=0.2, help='Weight for contrastive loss')

    args = parser.parse_args()
    args.method = 'DM'
    args.device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    USE_WANDB = args.use_wandb

    if not os.path.exists(args.data_path):
        os.mkdir(args.data_path)

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    if USE_WANDB:
        wandb.init(project=f"DisTrackted-{args.dataset}-{args.method}", config={
            "ipc": args.ipc,
            "batch_size": args.batch_real,
            "batch_size_train": args.batch_train,
            "lr_img": args.lr_img,
            "lr_net": args.lr_net,
            "model": args.model,
            "epoch_eval_train": args.epoch_eval_train,
            "num_eval": args.num_eval,
            "Iteration": args.Iteration,
            "use_contrastive": args.use_contrastive,
            "contrastive_weight": args.contrastive_weight
        })

    eval_it_pool = np.arange(0, args.Iteration+1, 500).tolist()
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = get_dataset(args.dataset, args.data_path)
    model_eval_pool = [args.model]

    accs_all_exps = dict() # record performances of all experiments
    for key in model_eval_pool:
        accs_all_exps[key] = []

    data_save = []

    for exp in range(args.num_exp):
        print('\n================== Exp %d ==================\n '%exp)
        print('Hyper-parameters: \n', args.__dict__)
        print('Evaluation model pool: ', model_eval_pool)

        ''' organize the real dataset '''
        images_all = []
        labels_all = []
        indices_class = [[] for c in range(num_classes)]
        images_all = [torch.unsqueeze(dst_train[i][0], dim=0) for i in range(len(dst_train))]
        labels_all = [dst_train[i][1] for i in range(len(dst_train))]
        for i, lab in enumerate(labels_all):
            indices_class[lab].append(i)
        images_all = torch.cat(images_all, dim=0).to(args.device)
        labels_all = torch.tensor(labels_all, dtype=torch.long, device=args.device)


        for c in range(num_classes):
            print('class c = %d: %d real images'%(c, len(indices_class[c])))

        for ch in range(channel):
            print('real images channel %d, mean = %.4f, std = %.4f'%(ch, torch.mean(images_all[:, ch]), torch.std(images_all[:, ch])))


        ''' initialize the synthetic data '''
        image_syn = torch.randn(size=(num_classes*args.ipc, channel, im_size[0], im_size[1]), dtype=torch.float, requires_grad=True, device=args.device)
        label_syn = torch.tensor([np.ones(args.ipc)*i for i in range(num_classes)], dtype=torch.long, requires_grad=False, device=args.device).view(-1) # [0,0,0, 1,1,1, ..., 9,9,9]

        ''' training '''
        optimizer_img = torch.optim.SGD([image_syn, ], lr=args.lr_img, momentum=0.5) # optimizer_img for synthetic data
        optimizer_img.zero_grad()
        print('%s training begins'%get_time())

        for it in range(args.Iteration+1):

            ''' Evaluate synthetic data '''
            if it in eval_it_pool:
                for model_eval in model_eval_pool:
                    print('-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s, iteration = %d'%(args.model, model_eval, it))

                    accs = []
                    accs_train = []
                    for it_eval in range(args.num_eval):
                        net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device) # get a random model
                        image_syn_eval, label_syn_eval = copy.deepcopy(image_syn.detach()), copy.deepcopy(label_syn.detach()) # avoid any unaware modification
                        _, acc_train, acc_test = evaluate_synset(it_eval, net_eval, image_syn_eval, label_syn_eval, testloader, args)
                        accs.append(acc_test)
                        accs_train.append(acc_train)
                        
                        # Memory cleanup
                        del net_eval
                        del image_syn_eval
                        del label_syn_eval
                        
                        if args.device == 'mps':
                            torch.mps.empty_cache()
                        elif args.device == 'cuda':
                            torch.cuda.empty_cache()
                        
                        gc.collect()
                        # End of memory cleanup
                        
                    print('Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------'%(len(accs), model_eval, np.mean(accs), np.std(accs)))
                    if USE_WANDB:
                        wandb.log({f"exp_{exp}": {
                            'iteration': it,
                            'acc_train': np.mean(accs_train),
                            'acc_test': np.mean(accs)
                        }})
                    if it == args.Iteration: # record the final results
                        accs_all_exps[model_eval] += accs

                ''' visualize and save '''
                save_name = os.path.join(args.save_path, 'vis_%s_%s_%s_%dipc_exp%d_iter%d.png'%(args.method, args.dataset, args.model, args.ipc, exp, it))
                image_syn_vis = copy.deepcopy(image_syn.detach().cpu())
                for ch in range(channel):
                    image_syn_vis[:, ch] = image_syn_vis[:, ch]  * std[ch] + mean[ch]
                image_syn_vis[image_syn_vis<0] = 0.0
                image_syn_vis[image_syn_vis>1] = 1.0
                save_image(image_syn_vis, save_name, nrow=args.ipc) # Trying normalize = True/False may get better visual effects.



            ''' Train synthetic data '''
            net = get_network(args.model, channel, num_classes, im_size).to(args.device) # get a random model
            net.train()
            for param in list(net.parameters()):
                param.requires_grad = False
            
            if args.model == 'ConvNet':
                embed = net.module.embed if torch.cuda.device_count() > 1 else net.embed # for GPU parallel
            else:
                embed = net

            loss_avg = 0

            ''' update synthetic data '''
            images_real_all = []
            images_syn_all = []
            loss = torch.tensor(0.0).to(args.device)
            for c in range(num_classes):
                img_real = sample_class_data(c, args.batch_real, indices_class, images_all)
                img_syn = image_syn[c*args.ipc:(c+1)*args.ipc].reshape((args.ipc, channel, im_size[0], im_size[1]))



                images_real_all.append(img_real)
                images_syn_all.append(img_syn)

            images_real_all = torch.cat(images_real_all, dim=0)
            images_syn_all = torch.cat(images_syn_all, dim=0)
            
            output_real = embed(images_real_all).detach()
            output_syn = embed(images_syn_all)

            loss += torch.sum((torch.mean(output_real.reshape(num_classes, args.batch_real, -1), dim=1) - torch.mean(output_syn.reshape(num_classes, args.ipc, -1), dim=1))**2)

            optimizer_img.zero_grad()
            loss.backward()
            optimizer_img.step()
            loss_avg += loss.item()
            
            

            loss_avg /= (num_classes)
            
            if USE_WANDB:
                wandb.log({f"exp_{exp}": {
                    'Synthetic data loss': loss_avg
                }})

            if it%10 == 0:
                print('%s iter = %05d, loss = %.4f' % (get_time(), it, loss_avg))

            if it == args.Iteration: # only record the final results
                data_save.append([copy.deepcopy(image_syn.detach().cpu()), copy.deepcopy(label_syn.detach().cpu())])
                torch.save({'data': data_save, 'accs_all_exps': accs_all_exps, }, os.path.join(args.save_path, 'res_%s_%s_%s_%dipc.pt'%(args.method, args.dataset, args.model, args.ipc)))

            # Memory cleanup
            del net
            del embed
            
            if args.device == 'mps':
                torch.mps.empty_cache()
            elif args.device == 'cuda':
                torch.cuda.empty_cache()

            gc.collect()
            # End of memory cleanup

    # Finalizar wandb de forma segura al terminar
    if USE_WANDB:
        wandb.finish()

    print('\n==================== Final Results ====================\n')
    for key in model_eval_pool:
        accs = accs_all_exps[key]
        print('Run %d experiments, train on %s, evaluate %d random %s, mean  = %.2f%%  std = %.2f%%'%(args.num_exp, args.model, len(accs), key, np.mean(accs)*100, np.std(accs)*100))



if __name__ == '__main__':
    main()


