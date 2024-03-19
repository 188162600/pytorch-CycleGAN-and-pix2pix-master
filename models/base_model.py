import os
import torch
from collections import OrderedDict
from abc import ABC, abstractmethod
from . import networks


class BaseModel(ABC):
    """This class is an abstract base class (ABC) for models.
    To create a subclass, you need to implement the following five functions:
        -- <__init__>:                      initialize the class; first call BaseModel.__init__(self, opt).
        -- <set_input>:                     unpack data from dataset and apply preprocessing.
        -- <forward>:                       produce intermediate results.
        -- <optimize_parameters>:           calculate losses, gradients, and update network weights.
        -- <modify_commandline_options>:    (optionally) add model-specific options and set default options.
    """

    def __init__(self, opt):
        """Initialize the BaseModel class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions

        When creating your custom class, you need to implement your own initialization.
        In this function, you should first call <BaseModel.__init__(self, opt)>
        Then, you need to define four lists:
            -- self.loss_names (str list):          specify the training losses that you want to plot and save.
            -- self.model_names (str list):         define networks used in our training.
            -- self.visual_names (str list):        specify the images that you want to display and save.
            -- self.optimizers (optimizer list):    define and initialize optimizers. You can define one optimizer for each network. If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_model.py for an example.
        """
        
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')  # get device name: CPU or GPU
        
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.project_name) # save all the checkpoints to save_dir
        if opt.preprocess != 'scale_width':  # with [scale_width], input images might have different sizes, which hurts the performance of cudnn.benchmark.
            torch.backends.cudnn.benchmark = True
        self.loss_names = []
        self.model_names = []
        self.visual_names = []
        self.optimizers = []
        self.image_paths = []
        self.metric = 0  # used for learning rate policy 'plateau'

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new model-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        return parser

    @abstractmethod
    def set_input(self, input,name):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): includes the data itself and its metadata information.
        """
        pass

    @abstractmethod
    def forward(self,name):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        pass

    @abstractmethod
    def optimize_parameters(self,name):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        pass

    def setup(self, opt):
        """Load and print networks; create schedulers

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        
        for name,data in self.all_data.items():
            data.task_G_A.setup()
            data.task_G_B.setup()
        for i,section in enumerate(self.generator_sections):
                #print("section",section)
                
                optimizer_A=torch.optim.Adam(section.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                next_steps_classifier_optimizer_A=torch.optim.Adam(section.classifier.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                # optimizer_B=torch.optim.Adam(section.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                # next_steps_classifier_optimizer_B=torch.optim.Adam(section.classifier.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                optimizer_B=optimizer_A
                next_steps_classifier_optimizer_B=next_steps_classifier_optimizer_A
                self.optimizers.append(optimizer_A)
                #self.optimizers.append(optimizer_B)
                self.optimizers.append(next_steps_classifier_optimizer_A)
                
                
                for name,data in self.all_data.items():
                    data.task_G_A.set_optimizer(i, optimizer_A,next_steps_classifier_optimizer_A)
                    data.task_G_B.set_optimizer(i, optimizer_B,next_steps_classifier_optimizer_B)
                self.optimizers.append(next_steps_classifier_optimizer_A)
                self.optimizers.append(next_steps_classifier_optimizer_B)
                # self.task_G_A.set_optimizer(i, optimizer_A,next_steps_classifier_optimizer_A)
                # self.task_G_B.set_optimizer(i, optimizer_B,next_steps_classifier_optimizer_B)
                section.to(self.device)
                section.classifier.to(self.device)
                #section.feature_adjustment.to(self.device)
                section.classifier.encoder.to(self.device)
                section.classifier.encoder.to(self.device)
                for layers in section.layers:
                    if isinstance(layers, list):
                        for layer in layers:
                            layer.to(self.device)
                    else:
                        layers.to(self.device)
        if self.isTrain:
            self.schedulers = [networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers]
        if not self.isTrain or opt.continue_train:
            load_suffix = 'iter_%d' % opt.load_iter if opt.load_iter > 0 else opt.epoch
            self.load_networks(load_suffix)
        self.print_networks(opt.verbose)
        
        # self.task_G_A.setup()
        # self.task_G_B.setup()
        # for i,section in enumerate(self.generator_sections):
        #         print("section",section)
                
        #         optimizer_A=torch.optim.Adam(section.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        #         next_steps_classifier_optimizer_A=torch.optim.Adam(section.classifier.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        #         optimizer_B=torch.optim.Adam(section.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        #         next_steps_classifier_optimizer_B=torch.optim.Adam(section.classifier.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        #         self.optimizers.append(optimizer_A)
        #         self.optimizers.append(optimizer_B)
        #         self.optimizers.append(next_steps_classifier_optimizer_A)
        #         self.optimizers.append(next_steps_classifier_optimizer_B)
        #         self.task_G_A.set_optimizer(i, optimizer_A,next_steps_classifier_optimizer_A)
        #         self.task_G_B.set_optimizer(i, optimizer_B,next_steps_classifier_optimizer_B)
        #         section.to(self.device)
        #         section.classifier.to(self.device)
        #         #section.feature_adjustment.to(self.device)
        #         section.classifier.encoder.to(self.device)
        #         section.classifier.encoder.to(self.device)
        #         for layers in section.layers:
        #             for layer in layers:
        #                 layer.to(self.device)
        if self.isTrain:
            self.schedulers = [networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers]
        if not self.isTrain or opt.continue_train:
            load_suffix = 'iter_%d' % opt.load_iter if opt.load_iter > 0 else opt.epoch
            self.load_networks(load_suffix)
        self.print_networks(opt.verbose)
        

        
        

    def eval(self):
        """Make models eval mode during test time"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.eval()

    def test(self):
        """Forward function used in test time.

        This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop
        It also calls <compute_visuals> to produce additional visualization results
        """
        with torch.no_grad():
            self.forward()
            self.compute_visuals()

    def compute_visuals(self):
        """Calculate additional output images for visdom and HTML visualization"""
        pass

    def get_image_paths(self):
        """ Return image paths that are used to load current data"""
        return self.image_paths

    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every epoch"""
        old_lr = self.optimizers[0].param_groups[0]['lr']
        for scheduler in self.schedulers:
            if self.opt.lr_policy == 'plateau':
                scheduler.step(self.metric)
            else:
                scheduler.step()

        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate %.7f -> %.7f' % (old_lr, lr))

    def get_current_visuals(self):
        """Return visualization images. train.py will display these images with visdom, and save the images to a HTML"""
        visual_ret = OrderedDict()
        # print("self.opt.names",self.opt.names)
        # print("self.visual_names",self.visual_names)
        for name in self.opt.names:
            data=self.all_data[name]
            # if data.empty:
            #     continue
            for visual_name in self.visual_names:
                if isinstance(name, str):
                    #print(torch.equal(getattr(data, '' + visual_name),data.dummy.to(getattr(data, '' + visual_name).device)),visual_name)
                    visual_ret[name+'_'+visual_name] =getattr(data, visual_name)
                    #print(getattr(data, visual_name).shape,visual_name)
                    
        return visual_ret
    @staticmethod
    def _get_loss(loss):
        if isinstance(loss, torch.Tensor):
            if loss.dim()==0:
                return float(loss.item())
            return float(torch.mean(loss).item())
        else:
            return loss
    def get_current_losses(self):
        """Return traning losses / errors. train.py will print out these errors on console, and save them to a file"""
        with torch.no_grad():
            errors_ret = OrderedDict()
            for name in self.opt.names:
                data=self.all_data[name]
                # if data.empty:
                #     continue
                for loss_name in self.loss_names:
                    if isinstance(name, str):
                        errors_ret[name+'_'+loss_name] = self._get_loss(getattr(data,  'loss_' + loss_name))
                  
       
        return errors_ret

    def save_networks(self, epoch):
        """Save all the networks to the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        saved_sections=set()
        for name,data in self.all_data.items():
            for model_name in self.model_names:
                if isinstance(name, str):
                    save_filename = '%s_net_%s.pth' % (epoch, name+"_"+model_name)
                    print("save_filename",save_filename,self.save_dir)
                    save_path = os.path.join(self.save_dir, save_filename)
                    #data=self.all_data[name]
                    net = getattr(data, 'net' + model_name)

                    if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                        torch.save(net.module.cpu().state_dict(), save_path)
                        torch.save(net.module.cpu(), save_path+"2")
                        torch.save(net, save_path +"3")
                        net.cuda(self.gpu_ids[0])
                    else:
                        torch.save(net.cpu().state_dict(), save_path)
                        torch.save(net.cpu(), save_path + "2")
                        torch.save(net, save_path + "3")
        
      
           
            for task_name in self.task_names:
                task=getattr(data, 'task_' + task_name)
                task.save_network(self.save_dir,epoch,saved_sections)
    
            

    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        """Fix InstanceNorm checkpoints incompatibility (prior to 0.4)"""
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and \
               (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)

    def load_networks(self, epoch):
        """Load all the networks from the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        loaded_sections=set()
        for name,data in self.all_data.items():
            
            for model_name in self.model_names:
                if isinstance(name, str):
                    load_filename = '%s_net_%s.pth' % (epoch, name+"_"+model_name)
                    load_path = os.path.join(self.save_dir, load_filename)
                    net = getattr(data, 'net' + model_name)
                    if isinstance(net, torch.nn.DataParallel):
                        net = net.module
                    print('loading the model from %s' % load_path)
                    # if you are using PyTorch newer than 0.4 (e.g., built from
                    # GitHub source), you can remove str() on self.device
                    state_dict = torch.load(load_path, map_location=str(self.device))
                    if hasattr(state_dict, '_metadata'):
                        del state_dict._metadata

                    # patch InstanceNorm checkpoints prior to 0.4
                    for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
                        self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
                    net.load_state_dict(state_dict)
           
            
            for task_name in self.task_names:
                task=getattr(data, 'task_' + task_name)
                task.load_network(self.save_dir,epoch,task.sections,loaded_sections)
     

    def print_networks(self, verbose):
        """Print the total number of parameters in the network and (if verbose) network architecture

        Parameters:
            verbose (bool) -- if verbose: print the network architecture
        """
        print('---------- Networks initialized -------------')
        for name in self.opt.names:
            data=self.all_data[name]
            for name in self.model_names:
                if isinstance(name, str):
                    net = getattr(data, 'net' + name)
                    num_params = 0
                    for param in net.parameters():
                        num_params += param.numel()
                    if verbose:
                        print(net)
                    print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
