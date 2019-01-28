"""Evaluates the model"""

import argparse
import logging
import os

import numpy as np
import torch
from torch.autograd import Variable
import utils
#import data_loader

import sklearn
from sklearn.metrics import accuracy_score, hamming_loss, precision_score, recall_score, f1_score


def evaluate(model, loss_fn, dataloader, params, img_count, cuda_present):
    """Evaluate the model on `num_steps` batches.
    Args:
        model: (torch.nn.Module) the neural network
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """

    # set model to evaluation mode
    model.eval()

    # summary for current eval loop
    epoch_metric_summ = []
    epoch_metrics = {}
    summary_batch = {}
    loss_avg = utils.RunningAverage()

    y_pred = torch.zeros(img_count, params.class_count, dtype=torch.float32)
    y_true = torch.zeros(img_count, params.class_count)
    
    k= 0
    
    # compute metrics over the dataset
    with torch.no_grad():
        for i, (data_batch, labels_batch) in enumerate(dataloader):

            batch_size = labels_batch.size()[0] 
            y_true[k:k+ batch_size, :] = labels_batch #build entire array of predicted labels
        
            batchlabel = labels_batch

            # move to GPU if available
            if cuda_present:
                data_batch, labels_batch = data_batch.cuda(non_blocking=True), labels_batch.cuda(non_blocking=True)
            # fetch the next evaluation batch
            data_batch, labels_batch = Variable(data_batch), Variable(labels_batch)
        
            # compute model output and loss
            prim_out = model(data_batch)

            #Compute primary, Aux and total weighted loss
            loss =loss_fn(prim_out, labels_batch)
            
            y_pred[k:k+ batch_size, :] = prim_out #build entire array of predicted labels
            
            k += batch_size
        
            summary_batch['loss'] = loss.item()
            epoch_metric_summ.append(summary_batch)

            #print(summary_batch)
            loss_avg.update(loss.item())
    # compute epoch mean of all metrics in summary
    logging.info("Batch: {} : - Dev set average metrics: ".format(i) + str(summary_batch['loss']))
    
    #Calculate the metrics of the entire dev dataset
    epoch_metrics['loss'] = loss_avg()
    
    return (epoch_metrics, y_true, y_pred)