import torch


class Model(torch.nn.Module):
    """ base class with default (overidable) training loop through 
    model.fit() and model.compile(), in the style of Tensorflow. 
    
    Note: 
    REQUIRED: DEFINE LAYERS IN __INIT__
    OPTIONAL TO OVERRIDE: self.train_step, self.fit
    """

    def __init__(self, dims=None, **kwargs):        
        
        super().__init__()
        self._build()

        ## DEFINE MODEL LAYERS HERE
        """ 
        # Example Layers
        self.Dense = torch.nn.Linear(10, 1)
        self.Dropout = torch.nn.Dropout(p=0.5)
        self.Dense = torch.nn.Linear(10, 1)
        self.Sigmoid = torch.nn.Sigmoid()
        """

    def _build(self):
        # Set CPU / GPU devices for training.
        self.device = "cuda" if torch.cuda.is_available() else "cpu"        
        
        # model status indicators
        self._is_compiled = False  # require compile before using train step

        # placeholders for compile
        self._loss_fn = None  
        self._optimizer = None
        self._callbacks = None  # note: not yet implements

        # progress trackers
        self._epoch_losses = []

    def forward(self, x, training=False):
        """ MUST OVERRIDE THIS METHOD! """
        
        ## MAKE SURE TO INCLUDE THESE LINES!
        # prepare for inference / training
        x = self._forward_prep(x)

        """
        Example usage:
        # build on first call
        if not self.is_built:
            self.build(x.shape)
            self.is_built = True

        # call
        x = self.Dense(x)

        if training:
            # optional training path variation
            pass

        output = x
        
        return output
        """
        raise NotImplementedError    

    def train_step(self, x, y_true):
        """ One Training Step.  OPTIONAL to override."""
        
        ## MAKE SURE TO INCLUDE THESE LINES!
        with torch.inference_mode(False): # REQUIRED LINE
                    
            # clear gradients
            self._optimizer.zero_grad()

            # put labels on same device as model and x
            y_true = y_true.to(self.device)    

            """ OPTIONAL to alter code below. """

            # forward pass
            y_pred = self(x, training=True)
            loss = self._loss_fn(y_pred, y_true)

            # backwards pass
            loss.backward()
            self._optimizer.step()

        return loss.item()

    def fit(self, x_data, y_true_data, num_epochs, reporting_freq=1):
        """ Default Training Loop
        OPTIONAL to override this method """

        self = self.to(self.device)

        # clear loss tracker
        self._epoch_losses = []

        # get params
        batch_size = x_data[0].shape[0]  # shape from first element in batched ds
        
        # required training components
        self._train_prep(x_data[0])

        # training loop        
        # iterate through epochs
        for epoch_num in range(num_epochs):

            if epoch_num % reporting_freq == reporting_freq-1:                
                print('Epoch:', epoch_num)
                            
            # iterate through batches
            running_loss = 0.0  # reset loss
            batch_num = 1
            
            for x, y_true in zip(x_data, y_true_data):       
                
                # train step
                loss = self.train_step(x, y_true)
                
                # update params
                running_loss += loss
                batch_num += 1

            # record results
            self._epoch_losses.append(running_loss / batch_num)

            if epoch_num % reporting_freq == reporting_freq-1:
                print('epoch loss:', loss)

        return self._epoch_losses

    def compile(self, **kwargs):
        return self._compile(**kwargs)

    def _compile(self, loss='crossentropy', optimizer_name='adam', callbacks=(),
                loss_kwargs={}, opt_kwargs={}):   
        """ Loads loss and optimizer into model. 
        User must call this method before model training. """

        if 'params' not in opt_kwargs:
            opt_kwargs['params'] = list(self.parameters())
            print(list(opt_kwargs['params']))
        
        # loss
        loss = Losses(loss, reduction='mean', **loss_kwargs)
        self._loss_fn = loss.loss_fn
        
        # optimizer
        opt = Optimizers(optimizer_name, **opt_kwargs)
        self._optimizer = opt.optimizer_fn
        
        # callbacks
        self._callbacks = callbacks

        # clear saved loss progress
        self.epoch_losses = []     

        # mark as compiled
        self._is_compiled = True

        return None

    def _forward_prep(self, x):

        # send to CPU / GPU
        self = self.to(self.device)
        x = x.to(self.device)

        return x

    def _train_prep(self, x):        
    
        # verify model is compiled
        if not self._is_compiled:
            raise AssertionError('Must compile Model')    
    
        return None


class Activations:
    def __init__(self, name=None, **kwargs):
        super().__init__(**kwargs)
        self.name = name

        # default: idenity
        self.active_fn = lambda x: x

        if name=='sigmoid':
            self.active_fn = torch.nn.Sigmoid()

        elif self.name == 'tanh':
            self.active_fn = torch.nn.Tanh()

        elif self.name == 'relu':
            self.active_fn = torch.nn.ReLU()
        
        elif self.name == 'softmax':
            self.active_fn = torch.nn.Softmax()

        elif self.name == 'swish':
            self.active_fn = torch.nn.SiLU()


class Losses:
    """ convenient names for commonly used losses """
    def __init__(self, name='crossentropy', reduction='none', **kwargs):

        loss_dict = {'crossentropy':torch.nn.CrossEntropyLoss,
                     'binary_crossentropy': torch.nn.BCELoss}
                     
        self.loss_fn = loss_dict[name](reduction=reduction)     


class Optimizers:
    """ convenient names for commonly used optimizers """
    def __init__(self, name='adam', **kwargs):

        opt_dict = {'adam': torch.optim.Adam,
                    'adamw': torch.optim.AdamW,
                    'sgd': torch.optim.SGD}

        self.optimizer_fn = opt_dict[name](**kwargs)


