import tensorflow as tf
import numpy as np
import time as timeR

RANDOM_SEED = 2308
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

tf_datatype = 'float64'
tf.keras.backend.set_floatx(tf_datatype)
np_datatype = np.float64

    
class Neural_Network_FeedForward_Regularizer:
    
    def __init__(self,input_data,target_data,bsz_data,input_data_val,target_data_val,bsz_data_val,validate,stab_epsilon,shuffle,layers,lr_rate,save_step,tf_datatype):
        
        self.input_data = input_data
        self.target_data = target_data
        self.bsz_data = bsz_data
        self.input_data_val = input_data_val
        self.target_data_val = target_data_val
        self.validate = validate
        self.bsz_data_val = bsz_data_val
        self.stab_epsilon = stab_epsilon
        self.input_mean = []
        self.input_std = []
        self.layers = layers
        self.loss_net_list = []
        self.loss_data_list = []
        self.loss_data_val_list = []
        self.loss_w_list = []
        self.save_step = save_step
        self.iters = 0
        self.training_time_in_min = None    
        self.lr_rate = lr_rate
        self.training_time_per_itr_in_sec = None
        self.shuffle = shuffle
        self.tf_datatype = tf_datatype
        self.target_dim = self.target_data.shape[1]
        if self.shuffle:
            self.train_dataset = tf.data.Dataset.from_tensor_slices((self.input_data,self.target_data)).shuffle(buffer_size = len(self.input_data)).batch(self.bsz_data)
        else:
            self.train_dataset = tf.data.Dataset.from_tensor_slices((self.input_data,self.target_data)).batch(self.bsz_data)
        
        if self.validate:
            self.val_dataset = tf.data.Dataset.from_tensor_slices((self.input_data_val,self.target_data_val)).batch(self.bsz_data_val)
            self.steps_val = len(self.val_dataset)
        
        self.steps =  len(self.train_dataset)
        self.model_NN = self.neural_network_output(self.layers,self.input_mean,self.input_std)
        self.ADAM_optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_rate)
            
    def neural_network_output(self,layers,input_mean,input_std): 
        model = tf.keras.Sequential()
        model.add(tf.keras.Input(shape=(layers[0],)))
        #scaling_layer = tf.keras.layers.Lambda(lambda x:  (x - input_mean)/input_std ) #tf.keras.layers.Lambda(lambda x: 2.0*(x - lb)/(ub - lb) - 1.0)
        #model.add(scaling_layer)
        for i in range(1,len(layers)-1):
            model.add(tf.keras.layers.Dense(layers[i],activation=tf.keras.activations.get('elu'), kernel_initializer='glorot_normal'))
        model.add(tf.keras.layers.Dense(layers[-1],activation=tf.keras.activations.get('linear'),use_bias=True,kernel_initializer='glorot_normal'))
        
        return model
    
    
    def net_output(self,inputs):
        
        outputs =  self.model_NN(inputs)
    
        return outputs
    

    def norm_w(self):
        
        train_var = self.train_variable_list()
        f = 1.0
        for i in range(len(train_var)//2):
            W_stable = train_var[2*i]
            f = f*tf.norm(W_stable)
    
        return f
    
    def loss_calculate_validate(self,input_data,target_data):
        
        output_data  = self.net_output(input_data)
        loss = tf.reduce_mean(tf.square(target_data - output_data))

        return loss

    
    def loss_calculate(self,input_data,target_data):
        
        output_data  = self.net_output(input_data)
        norm_W = self.norm_w()
        loss_data = tf.reduce_mean(tf.square(target_data - output_data))
        loss_w = self.stab_epsilon*norm_W
        loss_net = loss_data  + loss_w
        loss_info = (loss_data, loss_w)
        
        return loss_net , loss_info

    
    def train_variable_list(self):
        
        train_var = self.model_NN.trainable_variables 
        
        return train_var

    def get_grad(self,input_data,target_data):
        
        with tf.GradientTape() as tape:
            # tape.watch(self.train_variable_list())
            loss_net, loss_info = self.loss_calculate(input_data,target_data)
        grad_loss_net = tape.gradient(loss_net, self.train_variable_list())
        del tape

        return loss_net, loss_info, grad_loss_net
    
    
    def train_with_AdamOptimizer(self,iteration):
                
        @tf.function
        def train_step(input_data,target_data):
            
            loss_net, loss_info, grad_loss_net =  self.get_grad(input_data,target_data)
            self.ADAM_optimizer.apply_gradients(zip(grad_loss_net, self.train_variable_list()))

            return loss_net, loss_info

        

        def train_epoch():
            
            train_dataset_iter = iter(self.train_dataset)
            loss_net_avg  = 0.0         
            loss_data_avg = 0.0
            loss_w_avg = 0.0
            for i in range(self.steps):
                input_data, target_data = next(train_dataset_iter)
                loss_net, loss_info = train_step(input_data,target_data)
                loss_data, loss_w = loss_info
                loss_net_avg = loss_net_avg + loss_net
                loss_data_avg = loss_data_avg + loss_data
                loss_w_avg = loss_w_avg + loss_w
            loss_net_avg = loss_net_avg/self.steps
            loss_data_avg = loss_data_avg/self.steps
            loss_w_avg = loss_w_avg/self.steps
            loss_info_avg = (loss_data_avg,loss_w_avg)
        
            return loss_net_avg, loss_info_avg
            
            
        def validation_epoch():
            
            val_dataset_iter = iter(self.val_dataset)
            loss_avg  = 0.0         
            for i in range(self.steps_val):
                input_data, target_data = next(val_dataset_iter)
                loss = self.loss_calculate_validate(input_data,target_data)
                loss_avg = loss_avg + loss
      
            loss_avg = loss_avg/self.steps_val
            return loss_avg
        
        for itr in range(iteration):

            loss_net_avg, loss_info_avg = train_epoch()
            if self.iters % self.save_step ==0:
                loss_data_avg, loss_w_avg  = loss_info_avg
                loss_net_avg = loss_net_avg.numpy()
                loss_data_avg  = loss_data_avg.numpy()
                loss_w_avg = loss_w_avg.numpy()
                self.loss_net_list.append(loss_net_avg)
                self.loss_w_list.append(loss_w_avg)
                self.loss_data_list.append(loss_data_avg)    
                
                if self.validate:
                    loss_val = validation_epoch()
                    loss_val = loss_val.numpy()
                    self.loss_data_val_list.append(loss_val)
                    print('Epoch: ',self.iters ,'  net: ',loss_net_avg, ' data: ',loss_data_avg ,' w stab: ', loss_w_avg, ' val: ',loss_val) 
                else:
                    print('Epoch: ',self.iters ,' net: ',loss_net_avg, ' data: ',loss_data_avg ,' w stab: ', loss_w_avg)
                    
            self.iters = self.iters + 1


    def train_model(self,iteration):
        
        print('-'*100)
        print("Model Neural Network")
        self.model_NN.summary()
        print('-'*100)
        print('Adam Optimization Starts')
        time_start = timeR.time()
        self.train_with_AdamOptimizer(iteration)
        time_end = timeR.time()
        time_elapsed = time_end - time_start
        self.training_time_in_min = (time_elapsed)/60.0
        print('Adam Optimization Ends')
        print("Training Time for " + str(iteration)  + " Epochs in min: ",self.training_time_in_min)
        print("Training Time per Epoch in min: ",self.training_time_in_min/iteration)
        print('-'*100)
        

    def predict(self,inputs):
        
        inputs = tf.convert_to_tensor(inputs,self.tf_datatype)
        output = self.net_output(inputs)
        
        return output.numpy()
    
