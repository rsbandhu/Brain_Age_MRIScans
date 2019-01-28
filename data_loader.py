import os
import numpy as np
from collections import defaultdict
from scipy import misc
import torch
from torch.autograd import Variable
from torchvision import transforms
from PIL import Image
import time

class Dataloader():
    
    def __init__(self, params):
        
        # loading dataset_params
        
        train_path = params.train_data_path
        print("train data path = ", train_path)
        assert os.path.isdir(train_path), "No directory found at {}".format(train_path)
        
        test_path = params.test_data_path
        print("Val data path = ",test_path)
        assert os.path.isdir(test_path), "No directory found at {}".format(test_path)
        
    
    def Create_Img_dict(self, data_type, params, ch_filter = ["red", "blue", "green"]):
        """
        Reads the image file names  
        Creates a dict, key is the image count, value is the image file name without the extension
        Args:
            data_type: 'train' or 'val'
            params: the loads from params.json
            channel filter: blue, green, red
        """
        ch_filter = params.ch_filter
        image_dict = {}
        img_stat = {}
        image_count = 0
        print(data_type)
        if (data_type == 'train'):
            current_path = params.train_data_path
            print("datatype is train, path = ", current_path)
        elif data_type == 'val':
            current_path = params.test_data_path
            print("datatype is val, path = ", current_path)
        #image_dict = {}
        image_dict = defaultdict(list)
        
        img_files = os.listdir(current_path)
        file_count = len(img_files)
        print("total image files = ", file_count, "total samples =  ", file_count//4)
        img_files_sorted = sorted(img_files)  # sort the file list by name
        
        for i in range(file_count):  # go through all the image files
            img_file = img_files_sorted[i] 
            img_id = img_file.split("_")[0]  # split the file name by id and channel color
            channel = (img_file.split("_")[1]).split(".")[0] #get the image color, strip off ".png"
            
            '''
            if channel in ch_filter: # keep only the channels that are in the list "ch_filter"
                if img_id not in image_dict:
                    image_dict[img_id] = [img_file]
                else:
                    image_dict[img_id].append(img_file)
            
                image_count += 1
            '''
            
            for input_ch in ch_filter:
                if (input_ch == channel):
                    image_dict[img_id].append(img_file)
                    image_count += 1
        print("total image id count = ", image_count)
        return image_dict
    
    def load_data(self, data_type, params, ch_filter = ["red", "blue", "green"]):
        """
        Reads the image file names  
        Creates a dict, key is the image count, value is the image file name without the extension
        Args:
            data_type: 'train' or 'val'
            params: the loads from params.json
            channel filter: blue, green, red
        """
     
        image_dict = {}
        img_stat = {}
        image_count = 0
        print(data_type)
        if (data_type == 'train'):
            current_path = params.train_data_path
            print("datatype is train, path = ", current_path)
        elif data_type == 'val':
            current_path = params.test_data_path
            print("datatype is val, path = ", current_path)
        
        img_files = os.listdir(current_path)
        file_count = len(img_files)
        print(file_count)
        #find the image id that can be used to get the label from the .csv file
        
        img_files_sorted = sorted(img_files)  # sort the file list by name
        
        for img_file in img_files:
            image_dict[img_file] = img_file
                                                              
            image_count += 1
        print("image id count = ", image_count)
        return image_dict
        
    
    
    def load_labels(self, data_type, params):
        
        """
        Loads the labels from their corresponding files. 
        Creates a dict, key is the image name, value is the list of labels
        Args:
            labels_file: each line contains the labels for the corresponding image
        """
        
        #label_file = open("../train.csv", 'r')
        label_dict = {}
        if data_type == "train":
            label_file = open(params.train_label_file, 'r')
        if data_type == "val":
            label_file = open(params.test_label_file, 'r')
        
        line = label_file.readline()
        line = label_file.readline()
        while line:
            line = line.strip()
            line_split = line.split(',')
            ytrain = float(line_split[1])
            image_id = line_split[2]
            
            image_id = image_id.split('/')[-1]
            label_dict[image_id] = ytrain  #create the dict item for the image
            line = label_file.readline()

        label_file.close()
        return label_dict
        
    
    def data_iterator_from_dict(self, params, data_type, image_dict, labels_dict, shuffle = True):
        """
        Returns a generator that yields batches data with labels. Batch size is params.batch_size. Expires after one
        pass over the data.
        Args:
        
            params: (Params) hyperparameters of the training process.
            image_dict: dictionary of image files, keis image id, value is image file name
            label_dict: dictionary of labels, key is image id, value is a tensor of multi-one hot of size 28
            shuffle: (bool) whether the data should be shuffled
        Yields:
            batch_data: (Variable) dimension batch_size x channels X image_size_X X image_size_Y 
            batch_labels: (Variable) dimension batch_size x 4 with the corresponding labels
        """
        
        # Shuffle the images for training 
        shuffle = True
        key_list = list(image_dict.keys())
        #Shuffle the order of the images if needed
        if shuffle:
            np.random.seed(int(time.time()))
            np.random.shuffle(key_list)
      
        image_count = len(image_dict)
        minibatch_size = params.batch_size
      
        num_input_features = params.features_in
        for i in range((image_count-1)//minibatch_size+1):
            # "i" denotes batch number
            mini_batch_index_start = i*minibatch_size
            if (i < (image_count//minibatch_size)):
                mini_batch_index_end = (i+1)*minibatch_size
            else:
                mini_batch_index_end = image_count # the last batch might be less than minibatch_size
                minibatch_size = mini_batch_index_end - mini_batch_index_start
            batch_image_index = key_list[mini_batch_index_start:mini_batch_index_end ]

            batch_data = torch.zeros([minibatch_size, num_input_features], dtype=torch.float32)
            batch_labels = torch.zeros([minibatch_size, 28], dtype=torch.float32)

            # Generate mini-batch data and labels
            for k in range(minibatch_size):
                img_id = key_list[mini_batch_index_start + k]
                batch_data[k,:] = image_dict[img_id][0]  # extra '[0]' for densenet121 feature ouput, remove for VGG19 feature out
                labels = torch.tensor(labels_dict[img_id], dtype= torch.float32)
                batch_labels[k,:] = labels
            
            yield batch_data, batch_labels
    
    def data_iterator(self, params, data_type, image_dict, label_dict, shuffle = True):
        """
        Returns a generator that yields batches data with labels. Batch size is params.batch_size. Expires after one
        pass over the data.
        Args:
        
            params: (Params) hyperparameters of the training process.
            image_dict: dictionary of image files, keis image id, value is image file name
            label_dict: dictionary of labels, key is image id, value is a tensor of multi-one hot of size 28
            shuffle: (bool) whether the data should be shuffled
        Yields:
            batch_data: (Variable) dimension batch_size x channels X image_size_X X image_size_Y 
            batch_labels: (Variable) dimension batch_size x 4 with the corresponding labels
        """
        if (data_type == 'train'):
            datadir = params.train_data_path
        if (data_type == 'val'):
            datadir = params.test_data_path
            
        image_count = len(image_dict)
        
        #read hyper-parameters from params file
        channels = params.channels
        class_count = int(params.class_count)
        size_X = params.im_size_X
        size_Y = params.im_size_Y
        print("output image size =  ", size_X, size_Y )
        
        order = list(image_dict.keys())
        #Shuffle the order of the images if needed
        if shuffle:
            np.random.seed(int(time.time()))
            np.random.shuffle(order)
        minibatch_size = params.batch_size
        
        print( image_count, (image_count+1)//minibatch_size, minibatch_size)
        # one pass over data
        for i in range((image_count-1)//minibatch_size+1):
            # "i" denotes batch number
            mini_batch_index_start = i*minibatch_size
            if (i < (image_count//minibatch_size)):
                mini_batch_index_end = (i+1)*minibatch_size
            else:
                mini_batch_index_end = image_count # the last batch might be less than minibatch_size
                minibatch_size = mini_batch_index_end - mini_batch_index_start
            batch_image_index = order[mini_batch_index_start:mini_batch_index_end ]
            
            # Create initial arrays of zeros for batch data and labels
            batch_data = torch.zeros([minibatch_size, channels, size_X, size_Y], dtype=torch.float32)
            batch_labels = torch.zeros([minibatch_size, class_count], dtype=torch.float32)
            
            #print("mini batch index = ", i)
            # one pass over all the images in the mini batch
            for k in range(minibatch_size):
                #j = batch_image_index[k]
                image_id = batch_image_index[k]
                #print(image_id)
                #raw_img = torch.zeros([channels, 512,512], dtype = torch.float32)
                for m in range(channels):
                    image_name = image_dict[image_id] # get the image file name
                    image_path = os.path.join(datadir,image_name)
                    
                    img = np.load(image_path)
                    img_mean = np.mean(img)
                    img_std = np.std(img)
                    img_normalized = (img - img_mean)/img_std
                    
                    #print(img.shape, image_path )
                    img_tansformed = torch.tensor(img_normalized)
                    
                    batch_data[k,m,:,:] = img_tansformed
                batch_labels[k, :] = torch.tensor(label_dict[image_id] , dtype = torch.float32)
            
            
            # convert them to Variables to record operations in the computational graph
            batch_data, batch_labels = Variable(batch_data), Variable(batch_labels)
    
            yield batch_data, batch_labels
        
    def image_transform(self, data_type, size):
        if (data_type == 'train'):
            transform = transforms.Compose([transforms.Resize(size),transforms.ToTensor()])
        else:
            transform = transforms.Compose([transforms.resize(size),transforms.ToTensor()])
        
        return transform
        
    
    
