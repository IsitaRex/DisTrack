import os
import time
import copy
import gc
import argparse
import numpy as np
import wandb
import torch
import torchaudio
import torch.nn as nn
from torchvision.utils import save_image
from src.datasets import get_dataset, MNIST_MEL_SPEC, MNIST_MFCC
from src.distillation_losses import DistillationLosses
from src.utils import get_loops, get_network, evaluate_synset, match_loss, get_time, info_nce_loss, sample_class_data, sample_negative_samples, load_AST_feature_extractor

# Intentar importar wandb de manera segura
#clear mps cache
torch.mps.empty_cache()

def main():

    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--dataset', type=str, default='AUDIO_MNIST', help='dataset')
    parser.add_argument('--ipc', type=int, default=10, help='image(s) per class')
    parser.add_argument('--num_exp', type=int, default=1, help='the number of experiments')
    parser.add_argument('--num_eval', type=int, default=3, help='the number of evaluating randomly initialized models')
    parser.add_argument('--epoch_eval_train', type=int, default=100, help='epochs to train a model with synthetic data') # it can be small for speeding up with little performance drop
    parser.add_argument('--Iteration', type=int, default=1000, help='training iterations')
    parser.add_argument('--lr_img', type=float, default=2.0, help='learning rate for updating synthetic images')
    parser.add_argument('--lr_net', type=float, default=0.1, help='learning rate for updating network parameters')
    parser.add_argument('--batch_real', type=int, default=64, help='batch size for real data')
    parser.add_argument('--batch_train', type=int, default=64, help='batch size for training networks')
    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--save_path', type=str, default='result', help='path to save results')
    parser.add_argument('--use_wandb', type=bool, default=True, help='Use wandb for logging')
    parser.add_argument('--vanilla_weight', type=float, default=0.7, help='Weight for Vanilla Loss')
    parser.add_argument('--joint_weight', type=float, default=0.2, help='Weight for Joint Matching Loss')

    args = parser.parse_args()
    args.outer_loop, args.inner_loop = get_loops(args.ipc)
    # args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # if args.vanilla_weight + args.joint_weight > 1.0:
    #     raise ValueError("The sum of vanilla_weight and joint_weight must be less than or equal to 1.0")
    # if args.joint_weight < 0.0:
    #     raise ValueError("joint_weight must be greater than or equal to 0.0")
    
    args.modality_gap_weight = 1.0 - args.vanilla_weight - args.joint_weight
    args.device = 'mps'
    args.feature = 'combined'
    args.model = 'SimplifiedConvNet'  
    USE_WANDB = args.use_wandb

    if not os.path.exists(args.data_path):
        os.mkdir(args.data_path)

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    if USE_WANDB:
        wandb.init(project=f"CombinedDM-{args.dataset}", config={
            "ipc": args.ipc,
            "batch_size": args.batch_real,
            "batch_size_train": args.batch_train,
            "lr_img": args.lr_img,
            "lr_net": args.lr_net,
            "epoch_eval_train": args.epoch_eval_train,
            "num_eval": args.num_eval,
            "Iteration": args.Iteration,
            "vanilla_weight": args.vanilla_weight
            # "joint_weight": args.joint_weight,
            # "modality_gap_weight": args.modality_gap_weight
        })

    transform_spec = torchaudio.transforms.MelSpectrogram(
                        sample_rate=16000,
                        hop_length=512,
                        n_fft=2048,
                        n_mels=128
                    ).to(args.device)
    
    transform_mfcc = torchaudio.transforms.MFCC(
                    sample_rate=16000,
                    n_mfcc=13,
                    melkwargs={'n_fft': 2048, 'hop_length': 512, 'n_mels': 128}
                ).to(args.device)
    model_eval_pool = [args.model]

    wav_len = 16000
    if args.dataset == 'AUDIO_MNIST':
        wav_len = 16000
    elif args.dataset == 'UrbanSound8K':   
        wav_len = 32000

    eval_it_pool = np.arange(0, args.Iteration+1, 500).tolist()
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = get_dataset(args.dataset, args.data_path, args.feature)
    accs_all_exps = dict() # record performances of all experiments
    for key in model_eval_pool:
        accs_all_exps[key] = []

    data_save = []
    


    for exp in range(args.num_exp):
        print('\n================== Exp %d ==================\n '%exp)
        print('Hyper-parameters: \n', args.__dict__)
        print('Evaluation model pool: ', model_eval_pool)

        ''' organize the real dataset '''
        all_specs = []
        all_mfccs = []
        labels_all = []
        indices_class = [[] for c in range(num_classes)]
        file_name_processed_audio = f'cache/processed_audio_{args.dataset}_{args.feature}.pt'
        if os.path.exists(file_name_processed_audio):
            print('Loading processed audio from %s'%file_name_processed_audio)
            all_specs, all_mfccs, labels_all = torch.load(file_name_processed_audio)
        else:
            all_specs = [torch.unsqueeze(dst_train[i][0], dim=0) for i in range(len(dst_train))] 
            all_mfccs = [torch.unsqueeze(dst_train[i][1], dim=0) for i in range(len(dst_train))] 
            labels_all = [dst_train[i][2] for i in range(len(dst_train))]
            torch.save((all_specs,all_mfccs, labels_all), file_name_processed_audio)
        for i, lab in enumerate(labels_all):
            indices_class[lab].append(i)

        all_specs = torch.cat(all_specs, dim=0).to(args.device)
        all_mfccs = torch.cat(all_mfccs, dim=0).to(args.device)
        labels_all = torch.tensor(labels_all, dtype=torch.long, device=args.device)


        for c in range(num_classes):
            print('class c = %d: %d real images'%(c, len(indices_class[c])))

        ''' initialize the synthetic data '''
        audio_syn = torch.randn(size=(num_classes*args.ipc, wav_len), dtype=torch.float, requires_grad=True, device=args.device)
        label_syn = torch.tensor([np.ones(args.ipc)*i for i in range(num_classes)], dtype=torch.long, requires_grad=False, device=args.device).view(-1) # [0,0,0, 1,1,1, ..., 9,9,9]

        ''' training '''
        optimizer_img = torch.optim.SGD([audio_syn, ], lr=args.lr_img, momentum=0.5) # optimizer_img for synthetic data
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
                        net_eval = get_network(model_eval, channel, num_classes, MNIST_MEL_SPEC).to(args.device) # get a random model
                        audio_syn_eval, label_syn_eval = copy.deepcopy(audio_syn.detach()), copy.deepcopy(label_syn.detach()) # avoid any unaware modification
            
                        # transform audio to spectrogram
                        audio_syn_eval = transform_spec(audio_syn_eval)
                        # pad to make it im_size shape
                        if audio_syn_eval.shape[2] < im_size[1]:
                            padding = im_size[1] - audio_syn_eval.shape[2]
                            audio_syn_eval = nn.functional.pad(audio_syn_eval, (0, padding), mode='constant', value=0)
                        audio_syn_eval = audio_syn_eval.unsqueeze(1) # [N, 1, 128, 128]
                        _, acc_train, acc_test = evaluate_synset(it_eval, net_eval, audio_syn_eval, label_syn_eval, testloader, args)

                        accs.append(acc_test)
                        accs_train.append(acc_train)
                        
                        # Memory cleanup
                        del net_eval
                        del audio_syn_eval
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

                # ''' visualize and save '''
                # save_name = os.path.join(args.save_path, 'vis_%s_%s_%s_%dipc_exp%d_iter%d.png'%(args.method, args.dataset, args.model, args.ipc, exp, it))
                # image_syn_vis = copy.deepcopy(image_syn.detach().cpu())
                # for ch in range(channel):
                #     image_syn_vis[:, ch] = image_syn_vis[:, ch]  * std[ch] + mean[ch]
                # image_syn_vis[image_syn_vis<0] = 0.0
                # image_syn_vis[image_syn_vis>1] = 1.0
                # save_image(image_syn_vis, save_name, nrow=args.ipc) # Trying normalize = True/False may get better visual effects.



            ''' Train synthetic data '''
            net_mfcc = get_network(args.model, channel, num_classes, MNIST_MFCC).to(args.device)
            net_mfcc.train()

            net_spec = get_network(args.model, channel, num_classes, MNIST_MEL_SPEC).to(args.device)
            net_spec.train()

            for param in list(net_spec.parameters()):
                param.requires_grad = False
            
            for param in list(net_mfcc.parameters()):
                param.requires_grad = False
            
            embed_mfcc, embed_spec = net_mfcc.embed, net_spec.embed

            loss_avg = 0

            ''' update synthetic data '''
            loss = torch.tensor(0.0).to(args.device)
            for c in range(num_classes):
                spec_real = sample_class_data(c, args.batch_real, indices_class, all_specs)
                mfcc_real = sample_class_data(c, args.batch_real, indices_class, all_mfccs)
                
                
                a_syn = audio_syn[c*args.ipc:(c+1)*args.ipc]
                spec_syn = transform_spec(a_syn)
                mfcc_syn = transform_mfcc(a_syn)

                if mfcc_syn.shape[2] < MNIST_MFCC[1]:
                    padding = MNIST_MFCC[1] - mfcc_syn.shape[2]
                    mfcc_syn = nn.functional.pad(mfcc_syn, (0, padding), mode='constant', value=0)
                mfcc_syn = mfcc_syn.reshape((args.ipc, channel, mfcc_syn.shape[1], MNIST_MFCC[1]))

                if spec_syn.shape[2] < MNIST_MEL_SPEC[1]:
                    padding = MNIST_MEL_SPEC[1] - spec_syn.shape[2]
                    spec_syn = nn.functional.pad(spec_syn, (0, padding), mode='constant', value=0)
                spec_syn = spec_syn.reshape((args.ipc, channel, spec_syn.shape[1], MNIST_MEL_SPEC[1]))

                output_real_mfcc = embed_mfcc(mfcc_real).detach()
                output_real_spec = embed_spec(spec_real).detach()

                output_syn_mfcc = embed_mfcc(mfcc_syn)
                output_syn_spec = embed_spec(spec_syn)

                loss +=  args.vanilla_weight * DistillationLosses.vanilla_loss(output_real_spec, output_syn_spec)  
                loss += (1 - args.vanilla_weight) * DistillationLosses.vanilla_loss(output_real_mfcc, output_syn_mfcc)
            # if args.joint_weight > 0:
            #     loss += args.joint_weight * DistillationLosses.joint_matching_loss(output_real_mfcc, output_real_spec, output_syn_mfcc, output_syn_spec)
            # if args.modality_gap_weight > 0:
            #     loss += args.modality_gap_weight * DistillationLosses.modality_gap_loss(output_real_mfcc, output_real_spec, output_syn_mfcc, output_syn_spec)

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
                data_save.append([copy.deepcopy(audio_syn.detach().cpu()), copy.deepcopy(label_syn.detach().cpu())])
                torch.save({'data': data_save, 'accs_all_exps': accs_all_exps, }, os.path.join(args.save_path, 'res_%s_%s_%s_%dwpc.pt'%(args.feature.upper(), args.dataset, args.model, args.ipc)))

            # Memory cleanup
            del net_mfcc
            del net_spec
            del embed_mfcc
            del embed_spec
            
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


