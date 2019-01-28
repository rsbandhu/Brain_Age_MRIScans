
import torch.nn as nn
import torchvision
import numpy as np
import logging
import argparse
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from torchvision.models import resnet18

log_dir = '/home/hackathon/hackathon/Code/'

os.chdir(log_dir)

import  utils
import evaluate
import data_loader


def train(model, optimizer, loss_fn, dataloader, params, img_count, cuda_present):
    """Train the model on `num_steps` batches
    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """
    
    # set model to training mode
    model.train()

    # summary for current training loop and a running average object for loss
    epoch_metric_summ = []
    summary_batch = {}
    
    epoch_metrics = {}
    
    loss_avg = utils.RunningAverage()
    loss_class_wts = torch.tensor(params.wts, dtype=torch.float32)
    
    threshold = params.threshold #threshold value above which a class is considered present
    
    y_pred = torch.zeros(img_count, params.class_count)
    y_true = torch.zeros(img_count, params.class_count)
    
    if cuda_present:
        loss_class_wts = loss_class_wts.cuda()
    k= 0
    
    for i, (train_batch, labels_batch) in enumerate(dataloader):
        
        batch_size = labels_batch.size()[0] 
        
        #print(i, batch_size)
        y_true[k:k+ batch_size, :] = labels_batch #build entire array of predicted labels

        #If CUDA available, move data to GPU
        if cuda_present:
            train_batch = train_batch.cuda() #async=True)
            labels_batch = labels_batch.cuda() #async=True)
            
        # convert to torch Variables
        train_batch, labels_batch = Variable(train_batch), Variable(labels_batch)

        # compute model output and loss
        prim_out  = model(train_batch)

        #Compute primary, Aux and total weighted loss
        loss =loss_fn(prim_out, labels_batch)

        #send the primary output after thresholding for metrics calc
        yp = ((prim_out > threshold).int()*1).cpu()
        y_pred[k:k+ batch_size, :] = yp #build entire array of predicted labels
        k += batch_size

        # clear previous gradients, compute gradients of all variables wrt loss
        optimizer.zero_grad()
        loss.backward()

        # performs updates using calculated gradients
        optimizer.step()

        # update the average loss
        loss_avg.update(loss.item())
        
    #Calculate the metrics of the entire training dataset
    #epoch_metrics = metrics(y_pred, y_true, threshold)
    epoch_metrics['loss'] = loss_avg()
    
    logging.info("Training error {}".format(epoch_metrics['loss']))
    
    
def train_and_evaluate(params, dataloader, optimizer, scheduler, loss_fn, log_dir, cuda_present):
    """Train the model and evaluate every epoch.
    Args:
        model: (torch.nn.Module) the neural network
        train_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        val_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches validation data
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        model_dir: (string) directory containing config, weights and log
        restore_file: (string) optional- name of file to restore from (without its extension .pth.tar)
    """
    
    best_val_acc = 10000000 
    
    for epoch in range(params.num_epochs):
        
        t0 = time.time()
        '''Do the following for every epoch'''
        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))
        
        train_image_dict = dataloader.load_data("train", params)
        train_labels_dict = dataloader.load_labels("train", params)
        train_img_count = len(train_image_dict)
        train_data_generator = dataloader.data_iterator(params, "train", train_image_dict, train_labels_dict)
              
        train_data_generator = dataloader.data_iterator(params, "train", train_image_dict, train_labels_dict)
        
        # compute number of batches in one epoch (one full pass over the training set)
        #train(model, optimizer, loss_fn, train_data_generator, params, train_img_count, cuda_present)

        # Evaluate for one epoch on validation set
        val_image_dict = dataloader.load_data("val", params)
        val_labels_dict = dataloader.load_labels("val", params)
        val_img_count = len(val_image_dict)
        val_data_generator = dataloader.data_iterator(params, "val", val_image_dict, val_labels_dict)
        (val_metrics, y_true, y_pred) = evaluate.evaluate(model, loss_fn, val_data_generator, params, val_img_count, cuda_present)
        
        val_acc = val_metrics['loss']
        is_best = val_acc < best_val_acc

        logging.info("y_true {}".format(y_true))
        logging.info("y_pred {}".format(y_pred))
        
        best_file_name = 'train3_resnet18_eval'
        
        if (is_best):
            best_val_acc = val_acc
            utils.save_checkpoint({'epoch': epoch + 1,
                                   'state_dict': model.state_dict(),
                                   'optim_dict' : optimizer.state_dict()},
                                   is_best,
                                   log_dir, best_file_name)
        t1 = time.time()
        logging.info("Time taken for this epoch = {}".format(t1-t0))
        logging.info("Validation error after epoch {}, {}".format(epoch, val_acc))
        
    return (y_true, y_pred)
        
if __name__ == "__main__":
    
    
    print('first line in main')
    
    mymodel = resnet18(pretrained= True)
    mymodel.fc = nn.Linear(512, 1)    
    modelpath = '/home/hackathon/hackathon/Code/logs/best_weights_train3_resnet18'
    checkpoint = torch.load(modelpath)
    mymodel.load_state_dict(checkpoint['state_dict'])
    
    log_dir = '/home/hackathon/hackathon/Code/logs'
    json_path = '/home/hackathon/hackathon/Code/params3.json'
    # Set the logger
    utils.set_logger(os.path.join(log_dir, 'train3_resnet18_eval.log'))

    #Read params file
    params = utils.Params(json_path)

    #Generate Dataloader
    logging.info("Generating the dataloader")
    dataloader = data_loader.Dataloader(params)    
    logging.info("Done loading the Dataloader")

    # use GPU if available
    cuda_present = torch.cuda.is_available() #Boolean

    if cuda_present:
        logging.info("using CUDA")
    else:
        logging.info("cuda not available, using CPU")

    logging.info("Loading model and weights")

    # Change the following 1 lines for new models
    #model = net.myDensenet169(model_dir, params.class_count)
    model = mymodel
    logging.info("Transferring model to GPU if CUDA available")
    for param in model.parameters():
        param.requires_grad = True
    if cuda_present:
        model = model.cuda()

    #Specify the loss function Optimizer    
    optimizer = optim.Adam(model.parameters(), lr = params.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=3, gamma=0.7)
    
    #loss_fn = nn.BCEWithLogitsLoss()  # moving to net.py
    loss_fn = nn.MSELoss()

    # Train and Evaluate the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    #train_and_evaluate(params, image_dict, labels_dict)
    (y_true, y_pred) = train_and_evaluate(params, dataloader, optimizer, scheduler, loss_fn, log_dir, cuda_present)
