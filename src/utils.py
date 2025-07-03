import time
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models import MLP, ConvNet, SimplifiedConvNet
from src.datasets import TensorDataset
from transformers import ASTFeatureExtractor, ASTModel, ASTConfig

def get_default_convnet_setting():
    net_width, net_depth, net_act, net_norm, net_pooling = 128, 3, 'relu', 'instancenorm', 'avgpooling'
    return net_width, net_depth, net_act, net_norm, net_pooling


def get_network(model, channel, num_classes, im_size=(32, 32), embedding_size = None):
    torch.random.manual_seed(int(time.time() * 1000) % 100000)
    net_width, net_depth, net_act, net_norm, net_pooling = get_default_convnet_setting()

    if model == 'MLP':
        net = MLP(channel=channel, num_classes=num_classes)
    elif model == 'ConvNet':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'AST':
        net = load_AST()
    elif model == 'SimplifiedConvNet':
        net = SimplifiedConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, im_size=im_size, embedding_size=embedding_size)
    else:
        net = None
        exit('unknown model: %s'%model)

    gpu_num = torch.cuda.device_count()
    if gpu_num>0:
        device = 'cuda'
        if gpu_num>1:
            net = nn.DataParallel(net)
    else:
        device = 'cpu'
    net = net.to(device)

    return net


def get_time():
    return str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))

def distance_wb(gwr, gws):
    shape = gwr.shape
    if len(shape) == 4: # conv, out*in*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2] * shape[3])
        gws = gws.reshape(shape[0], shape[1] * shape[2] * shape[3])
    elif len(shape) == 3:  # layernorm, C*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2])
        gws = gws.reshape(shape[0], shape[1] * shape[2])
    elif len(shape) == 2: # linear, out*in
        tmp = 'do nothing'
    elif len(shape) == 1: # batchnorm/instancenorm, C; groupnorm x, bias
        gwr = gwr.reshape(1, shape[0])
        gws = gws.reshape(1, shape[0])
        return torch.tensor(0, dtype=torch.float, device=gwr.device)

    dis_weight = torch.sum(1 - torch.sum(gwr * gws, dim=-1) / (torch.norm(gwr, dim=-1) * torch.norm(gws, dim=-1) + 0.000001))
    dis = dis_weight
    return dis

def match_loss(gw_syn, gw_real, args):
    dis = torch.tensor(0.0).to(args.device)

    if args.dis_metric == 'ours':
        for ig in range(len(gw_real)):
            gwr = gw_real[ig]
            gws = gw_syn[ig]
            dis += distance_wb(gwr, gws)

    elif args.dis_metric == 'mse':
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(gw_real)):
            gw_real_vec.append(gw_real[ig].reshape((-1)))
            gw_syn_vec.append(gw_syn[ig].reshape((-1)))
        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
        dis = torch.sum((gw_syn_vec - gw_real_vec)**2)

    elif args.dis_metric == 'cos':
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(gw_real)):
            gw_real_vec.append(gw_real[ig].reshape((-1)))
            gw_syn_vec.append(gw_syn[ig].reshape((-1)))
        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
        dis = 1 - torch.sum(gw_real_vec * gw_syn_vec, dim=-1) / (torch.norm(gw_real_vec, dim=-1) * torch.norm(gw_syn_vec, dim=-1) + 0.000001)

    else:
        exit('unknown distance function: %s'%args.dis_metric)

    return dis


def get_loops(ipc):
    # Get the two hyper-parameters of outer-loop and inner-loop.
    # The following values are empirically good.
    if ipc == 1:
        outer_loop, inner_loop = 1, 1
    elif ipc == 10:
        outer_loop, inner_loop = 10, 50
    elif ipc == 20:
        outer_loop, inner_loop = 20, 25
    elif ipc == 30:
        outer_loop, inner_loop = 30, 20
    elif ipc == 40:
        outer_loop, inner_loop = 40, 15
    elif ipc == 50:
        outer_loop, inner_loop = 50, 10
    else:
        outer_loop, inner_loop = 0, 0
        exit('loop hyper-parameters are not defined for %d ipc'%ipc)
    return outer_loop, inner_loop



def epoch(mode, dataloader, net, optimizer, criterion, args, aug):
    loss_avg, acc_avg, num_exp = 0, 0, 0
    net = net.to(args.device)
    criterion = criterion.to(args.device)

    if mode == 'train':
        net.train()
    else:
        net.eval()
    for i_batch, datum in enumerate(dataloader):
        img = datum[0].float().to(args.device)
        lab = datum[1].long().to(args.device)
        if(len(img.shape) == 5):
            batch_size, num_chunks, channels, height, width = img.shape
            img = img.view(batch_size * num_chunks, channels, height, width)
            lab = lab.repeat(num_chunks)

        n_b = lab.shape[0]
        try:
            output = net(img)
        except:
            breakpoint()
        
        loss = criterion(output, lab)
        acc = np.sum(np.equal(np.argmax(output.cpu().data.numpy(), axis=-1), lab.cpu().data.numpy()))

        loss_avg += loss.item()*n_b
        acc_avg += acc
        num_exp += n_b

        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    try:
        loss_avg /= num_exp
        acc_avg /= num_exp
    except:
        breakpoint()

    return loss_avg, acc_avg

def evaluate_synset(it_eval, net, images_train, labels_train, testloader, args):
    net = net.to(args.device)
    images_train = images_train.to(args.device)
    labels_train = labels_train.to(args.device)
    lr = float(args.lr_net)
    Epoch = int(args.epoch_eval_train)
    lr_schedule = [Epoch//2+1]
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
    criterion = nn.CrossEntropyLoss().to(args.device)

    dst_train = TensorDataset(images_train, labels_train)
    trainloader = torch.utils.data.DataLoader(dst_train, batch_size=args.batch_train, shuffle=True, num_workers=0)
    start = time.time()
    for ep in range(Epoch+1):
        loss_train, acc_train = epoch('train', trainloader, net, optimizer, criterion, args, aug = True)
        if ep in lr_schedule:
            lr *= 0.1
            optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
    time_train = time.time() - start
    loss_test, acc_test = epoch('test', testloader, net, optimizer, criterion, args, aug = False)
    print('%s Evaluate_%02d: epoch = %04d train time = %d s train loss = %.6f train acc = %.4f, test acc = %.4f' % (get_time(), it_eval, Epoch, int(time_train), loss_train, acc_train, acc_test))

    return net, acc_train, acc_test


def get_eval_pool(eval_mode, model, model_eval):
    if eval_mode == 'M': # multiple architectures
        model_eval_pool = ['MLP', 'ConvNet', 'LeNet', 'AlexNet', 'VGG11', 'ResNet18']
    elif eval_mode == 'B':  # multiple architectures with BatchNorm for DM experiments
        model_eval_pool = ['ConvNetBN', 'ConvNetASwishBN', 'AlexNetBN', 'VGG11BN', 'ResNet18BN']
    elif eval_mode == 'W': # ablation study on network width
        model_eval_pool = ['ConvNetW32', 'ConvNetW64', 'ConvNetW128', 'ConvNetW256']
    elif eval_mode == 'D': # ablation study on network depth
        model_eval_pool = ['ConvNetD1', 'ConvNetD2', 'ConvNetD3', 'ConvNetD4']
    elif eval_mode == 'A': # ablation study on network activation function
        model_eval_pool = ['ConvNetAS', 'ConvNetAR', 'ConvNetAL', 'ConvNetASwish']
    elif eval_mode == 'P': # ablation study on network pooling layer
        model_eval_pool = ['ConvNetNP', 'ConvNetMP', 'ConvNetAP']
    elif eval_mode == 'N': # ablation study on network normalization layer
        model_eval_pool = ['ConvNetNN', 'ConvNetBN', 'ConvNetLN', 'ConvNetIN', 'ConvNetGN']
    elif eval_mode == 'S': # itself
        if 'BN' in model:
            print('Attention: Here I will replace BN with IN in evaluation, as the synthetic set is too small to measure BN hyper-parameters.')
        model_eval_pool = [model[:model.index('BN')]] if 'BN' in model else [model]
    elif eval_mode == 'SS':  # itself
        model_eval_pool = [model]
    else:
        model_eval_pool = [model_eval]
    return model_eval_pool




def sample_negative_samples(class_id, num_samples, class_indices, all_real_data):
    """
    Randomly sample data points from classes other than the specified class_id.

    Args:
        class_id (int): The class ID to exclude.
        num_samples (int): The number of samples to retrieve.
        class_indices (dict): A dictionary where keys are class IDs and values are lists of indices for each class.
        all_real_data (np.ndarray): The dataset containing all data points.

    Returns:
        np.ndarray: An array of sampled data points from other classes.
    """
    # Collect indices from all classes except the specified class_id
    negative_indices = [idx for cls, indices in enumerate(class_indices) if cls != class_id for idx in indices]
    
    # Randomly sample the required number of indices
    sampled_indices = np.random.choice(negative_indices, size=num_samples, replace=False)
    
    # Return the sampled data points
    return all_real_data[sampled_indices]


def info_nce_loss(anchor, positive, negatives, device, temperature=0.1):
    """
    Compute the InfoNCE loss.

    Args:
        anchor (torch.Tensor): Anchor embeddings of shape (batch_size, embedding_dim).
        positive (torch.Tensor): Positive embeddings of shape (batch_size, embedding_dim).
        negatives (torch.Tensor): Negative embeddings of shape (num_negatives, embedding_dim).
        device (torch.device): The device to use (e.g., torch.device('cuda') or 'cpu').
        temperature (float): Temperature scaling factor.

    Returns:
        torch.Tensor: The computed InfoNCE loss.
    """
    batch_size = anchor.shape[0]
    embedding_dim = anchor.shape[1]
    num_negatives = negatives.shape[0]

    # Normalize all embeddings
    anchor = F.normalize(anchor, dim=1)
    positive = F.normalize(positive, dim=1)
    negatives = F.normalize(negatives, dim=1)

    # Compute similarity between anchor and positive: (batch_size,)
    pos_sim = torch.sum(anchor * positive, dim=1, keepdim=True)  # shape: (batch_size, 1)

    # Compute similarity between anchor and each negative
    # anchor: (batch_size, embedding_dim)
    # negatives.T: (embedding_dim, num_negatives)
    neg_sim = anchor @ negatives.T  # shape: (batch_size, num_negatives)

    # Concatenate positive and negative similarities: (batch_size, 1 + num_negatives)
    logits = torch.cat([pos_sim, neg_sim], dim=1)

    # Scale by temperature
    logits /= temperature

    # Create labels: positive is at index 0
    labels = torch.zeros(batch_size, dtype=torch.long, device=device)

    # Compute cross-entropy loss
    loss = F.cross_entropy(logits, labels)

    return loss

def sample_class_data(class_id, num_samples, indices_class, all_real_data):
    sampled_indices = np.random.permutation(indices_class[class_id])[:num_samples]
    return all_real_data[sampled_indices]

def sample_negative_samples(class_id, num_samples, indices_class, all_real_data):
    # Collect indices from all classes except the specified class_id
    negative_indices = [idx for cls, indices in enumerate(indices_class) if cls != class_id for idx in indices]
    # Randomly sample the required number of indices
    sampled_indices = np.random.choice(negative_indices, size=num_samples, replace=False)
    # Return the sampled data points
    return all_real_data[sampled_indices]

def load_AST():
  config = ASTConfig(max_length=128)
  AST_SAMPLE_RATE = 16000
  feature_extractor = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593", config=config)
  feature_extractor_orig = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
  model_orig = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
  model = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593", config=config,ignore_mismatched_sizes=True)
  model.embeddings.position_embeddings = torch.nn.Parameter(model_orig.embeddings.position_embeddings[0,:146,:])

  return model

def load_AST_feature_extractor():
  config = ASTConfig(max_length=128)
  feature_extractor = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593", config=config)
  return feature_extractor