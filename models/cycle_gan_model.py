import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import meta.task
import meta.section
import meta.network
import meta.next_steps

class CycleGANModel(BaseModel):
    class Data:
        def __init__(self,dummy) -> None:
            #self.empty=True
            self.dummy=dummy
            self.real_A=dummy
            self.real_B=dummy
            self.fake_A=dummy
            self.fake_B=dummy
            self.rec_A=dummy
            self.rec_B=dummy
            self.idt_A=dummy
            self.idt_B=dummy
            self.loss_D_A=0
            self.loss_G_A=0
            self.loss_cycle_A=0
            self.loss_idt_A=0
            self.loss_D_B=0
            self.loss_G_B=0
            self.loss_cycle_B=0
            self.loss_idt_B=0
            self.loss_G=0
            self.optimize_D=False
            self.optimize_C=False
            self.optimize_G=False
        def detach(self):
            self.loss_D_A=self.loss_D_A.detach()
            self.loss_G_A=self.loss_G_A.detach()
            self.loss_cycle_A=self.loss_cycle_A.detach()
            self.loss_idt_A=self.loss_idt_A.detach()
            self.loss_D_B=self.loss_D_B.detach()
            self.loss_G_B=self.loss_G_B.detach()
            self.loss_cycle_B=self.loss_cycle_B.detach()
            self.loss_idt_B=self.loss_idt_B.detach()
            self.loss_G=self.loss_G.detach()
    
            
            
         
    def enable_optimizer_D(self,name,enable=True):
        self.all_data[name].optimize_D=enable
    def enable_optimizer_C(self,name,enable=True):
        self.all_data[name].optimize_C=enable
    def enable_optimizer_G(self,name,enable=True):
        self.all_data[name].optimize_G=enable
            
            
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
        Dropout is not used in the original CycleGAN paper.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')

        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        
        
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_B', 'rec_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B']
        if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['D_A', 'D_B']
            self.task_names=['G_A','G_B']
        else:  # during test time, only load Gs
            self.model_names = []
            self.task_names=['G_A','G_B']

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        # self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
        #                                 not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        # self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
        #                                 not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        shape=(opt.input_nc,opt.crop_size,opt.crop_size)
        num_options=[opt.num_options_upsample,opt.num_options_blocks,opt.num_options_downsample]
        num_shared=[opt.num_shared_upsample,opt.num_shared_blocks,opt.num_shared_downsample]
        next_steps_old_new_fresh_distribution=[opt.next_steps_old_weight,opt.next_steps_new_weight,opt.next_steps_fresh_weight]
        
        self.generator_sections=meta.network.define_resnet_sections(num_options,num_shared,
                                                                     opt.input_nc,opt.output_nc,opt.ngf,opt.norm,not opt.no_dropout,8,"reflect",opt.init_type,opt.init_gain,self.gpu_ids)
        dummy_task=meta.task.Task(shape,self.device,[],self.opt.separate_classifier_backward)
        encoders=[opt.downsample_step_classifier_encoder,opt.blocks_step_classifier_encoder,opt.upsample_step_classifier_encoder]
        for i,section in enumerate( self.generator_sections):
            channels=dummy_task.dummy_features.size(1)
            #print(dummy_task.dummy_features.shape,"dummy_task.dummy_features.shape","channels",channels)
            section.classifier_encoder=meta.network.define_resnet_encoder(encoders[i],input_channels=channels)
            #print("section.classifier_encoder",section.name,section.classifier_encoder)
            section.steps_record=meta.next_steps.RestoredSteps(section.layers.num_layers_with_params,num_options[i],opt.num_tracking_samples,opt.next_steps_old_num_tracking,opt.next_steps_new_num_tracking,opt.next_steps_fresh_num_tracking,next_steps_old_new_fresh_distribution)
            dummy_task.append_section(section)
        
        self.all_data=dict()
        dummy=torch.zeros(1,*shape)
        if opt.separate_classifier_backward and opt.accurate_classifier_backward:
            
            reduction_mode="batch_mean"
        else:
         
            reduction_mode="mean"
            
        
        for name in opt.names:
            data=self.Data(dummy)
            self.all_data.update({name:data})
            data.task_G_A=meta.task.Task(shape,self.device,[],opt.separate_classifier_backward)
            data.task_G_B=meta.task.Task(shape,self.device,[],opt.separate_classifier_backward)
            data.task_G_A.extend_sections(self.generator_sections)
            data.task_G_B.extend_sections(self.generator_sections)
              
            if self.isTrain:  # define discriminators
                
                data.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                                opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
                data.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                                opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            if self.isTrain:
                if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                    assert(opt.input_nc == opt.output_nc)
                data.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
                data.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
                # define loss functions
            
                data.criterionGAN = networks.GANLoss(opt.gan_mode,reduction=reduction_mode).to(self.device)  # define GAN loss.
                data.criterionCycle = networks.L1Loss(reduction=reduction_mode).to(self.device)    # define cycle loss
                data.criterionIdt = networks.L1Loss(reduction=reduction_mode).to(self.device)
                # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
                #self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            
                data.optimizer_D = torch.optim.Adam(itertools.chain(data.netD_A.parameters(), data.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
                #self.optimizers.append(self.optimizer_G)
                self.optimizers.append(data.optimizer_D)
                

            
   

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        #print("input",input,input['name'])
        names=input['name']
        name=names[0]
        assert all(i==name for i in names)
        data=self.all_data[name]

        self.current_data=data
        AtoB = self.opt.direction == 'AtoB'
        data.real_A = input['A' if AtoB else 'B'].to(self.device)
        data.real_B = input['B' if AtoB else 'A'].to(self.device)
        data.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        # self.fake_B = self.netG_A(self.real_A)  # G_A(A)
        # self.rec_A = self.netG_B(self.fake_B)   # G_B(G_A(A))
        # self.fake_A = self.netG_B(self.real_B)  # G_B(B)
        # self.rec_B = self.netG_A(self.fake_A)   # G_A(G_B(B))
        # self.task_G_A.set_criterion(lambda x:self.criterionCycle(self.netD_A(x),True))
        # self.fake_B=self.task_G_A.forward(self.real_A,track_loss=True)
        # self.steps_fake_B=self.task_G_A.previous_steps
        
        # self.task_G_B.set_criterion(lambda x:self.criterionCycle(self.netD_B(x),True))
        # self.rec_A=self.task_G_B.forward(self.fake_B,track_loss=True)
        # self.steps_rec_A=self.task_G_B.previous_steps
        
        # self.task_G_B.set_criterion(lambda x:self.criterionCycle(self.netD_B(x),True))
        # self.fake_A=self.task_G_B.forward(self.real_B,track_loss=True)
        # self.steps_fake_A=self.task_G_B.previous_steps
        
        # self.task_G_A.set_criterion(lambda x:self.criterionCycle(self.netD_A(x),True))
        # self.rec_B=self.task_G_A.forward(self.fake_A,track_loss=True)
        # self.steps_rec_B=self.task_G_A.previous_steps
        #print("self.real_A",self.real_A.device)
        data=self.current_data
        data.empty=False
        data.fake_B=data.task_G_A.forward(data.real_A)
        #self.fake_B_each_section=self.task_G_A.get_results()
        data.fake_B_steps=data.task_G_A.previous_steps
        #print(data.fake_B)
        data.rec_A=data.task_G_B.forward(data.fake_B)
        #self.rec_A_each_section=self.task_G_B.get_results()
        data.rec_A_steps=data.task_G_B.previous_steps
        
        data.fake_A=data.task_G_B.forward(data.real_B)
        #self.fake_A_each_section=self.task_G_B.get_results()
        data.fake_A_steps=data.task_G_B.previous_steps
        
        data.rec_B=data.task_G_A.forward(data.fake_A)
        #self.rec_B_each_section=self.task_G_A.get_results()
        data.rec_B_steps=data.task_G_A.previous_steps
        
        # # GAN loss D_A(G_A(A))
        # self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        # # GAN loss D_B(G_B(B))
        # self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        # # Forward cycle loss || G_B(G_A(A)) - A||
        # self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # # Backward cycle loss || G_A(G_B(B)) - B||
        # self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        # # combined loss and calculate gradients

    def backward_D_basic(self, netD, real, fake,data):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = data.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = data.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        
        loss_D.backward()
        return loss_D

    def backward_D_A(self,data):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = data.fake_B_pool.query(data.fake_B)
        data.loss_D_A = self.backward_D_basic(data.netD_A, data.real_B, fake_B,data)
       
    def backward_D_B(self,data):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = data.fake_A_pool.query(data.fake_A)
        data.loss_D_B = self.backward_D_basic(data.netD_B, data.real_A, fake_A,data)
     

    # def backward_G(self,data):
    #     """Calculate the loss for generators G_A and G_B"""
    #     lambda_idt = self.opt.lambda_identity
    #     lambda_A = self.opt.lambda_A
    #     lambda_B = self.opt.lambda_B
    #     # Identity loss
    #     # if lambda_idt > 0:
    #     #     # G_A should be identity if real_B is fed: ||G_A(B) - B||
    #     #     self.idt_A = self.netG_A(self.real_B)
    #     #     self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
    #     #     # G_B should be identity if real_A is fed: ||G_B(A) - A||
    #     #     self.idt_B = self.netG_B(self.real_A)
    #     #     self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
    #     # else:
    #     #     self.loss_idt_A = 0
    #     #     self.loss_idt_B = 0
    #     #classifier_loss=0
    #     if lambda_idt > 0:
    #         # G_A should be identity if real_B is fed: ||G_A(B) - B||
    #         data.idt_A=data.task_G_A.forward(data.real_B)
    #        # self.idt_A_each_section=self.task_G_A.get_results()
            
    #         #self.idt_A_losses=[self.criterionIdt(self.idt_A_each_section[i],self.real_B)*(lambda_B*lambda_idt) for i in range(len(self.idt_A_each_section))]
    #         data.loss_idt_A= data.criterionIdt(data.idt_A,data.real_B)*(lambda_B*lambda_idt)
    #         data.idt_A_steps=data.task_G_A.previous_steps
            
            
    #         data.idt_B=data.task_G_B.forward(data.real_A)
    #         # self.idt_B_each_section=self.task_G_B.get_results()
    #         # self.idt_B_losses=[self.criterionIdt(self.idt_B_each_section[i],self.real_A)*(lambda_A*lambda_idt) for i in range(len(self.idt_B_each_section))]
    #         # self.idt_B_loss=self.idt_B_losses[-1]
    #         data.loss_idt_B= data.criterionIdt(data.idt_B,data.real_A)*(lambda_A*lambda_idt)
    #         data.idt_B_steps=data.task_G_B.previous_steps
    #         #print(data.real_B.shape,data.idt_A.shape)
    #         #if self.opt.batch_size==1:
    #         #classifier_loss=classifier_loss+data.loss_idt_A+data.loss_idt_B
    #         # else:
    #         #     with torch.no_grad():
    #         #         classifier_loss=classifier_loss+data.criterionIdt(data.idt_A[0],data.real_B[0])*(lambda_B*lambda_idt)
    #         #         classifier_loss=classifier_loss+data.criterionIdt(data.idt_B[0],data.real_A[0])*(lambda_A*lambda_idt)
          
    #     else:
    #         data.loss_idt_A = 0
    #         data.loss_idt_B = 0
            
        
    #     # # GAN loss D_A(G_A(A))
    #     # self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
    #     # # GAN loss D_B(G_B(B))
    #     # self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
    #     # # Forward cycle loss || G_B(G_A(A)) - A||
    #     # self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
    #     # # Backward cycle loss || G_A(G_B(B)) - B||
    #     # self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
    #     # # combined loss and calculate gradients
    #     # self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
    #     # self.loss_G.backward()
        
    #     # # GAN loss D_A(G_A(A))
    #     # self.losses_G_A=[self.criterionGAN(self.netD_A(fake_B_each_section),True) for fake_B_each_section in self.fake_B_each_section]
    #     # self.loss_G_A=self.losses_G_A[-1]f
    #     data.loss_G_A=data.criterionGAN(data.netD_A(data.fake_B),True)
        
    #     # GAN loss D_B(G_B(B))
    #     # self.losses_G_B=[self.criterionGAN(self.netD_B(fake_A_each_section),True) for fake_A_each_section in self.fake_A_each_section]
    #     # self.loss_G_B=self.losses_G_B[-1]
    #     data.loss_G_B=data.criterionGAN(data.netD_B(data.fake_A),True)
    #     # Forward cycle loss || G_B(G_A(A)) - A||
    #     # self.losses_cycle_A=[self.criterionCycle(rec_A_each_section,self.real_A) for rec_A_each_section in self.rec_A_each_section]
    #     # self.loss_cycle_A=self.losses_cycle_A[-1]
    #     data.loss_cycle_A=data.criterionCycle(data.rec_A,data.real_A)*lambda_A
    #     # Backward cycle loss || G_A(G_B(B)) - B||
    #     # self.losses_cycle_B=[self.criterionCycle(rec_B_each_section,self.real_B) for rec_B_each_section in self.rec_B_each_section]
    #     # self.loss_cycle_B=self.losses_cycle_B[-1]
    #     data.loss_cycle_B=data.criterionCycle(data.rec_B,data.real_B)*lambda_B
    #     #if self.opt.batch_size==1:
    #     # print(data.loss_cycle_A+data.loss_cycle_B+data.loss_idt_A+data.loss_idt_B,classifier_loss)
    #     # classifier_loss=classifier_loss+data.loss_cycle_A+data.loss_cycle_B+data.loss_idt_A+data.loss_idt_B
    #     # else:
    #     #     with torch.no_grad():
    #     #         classifier_loss=classifier_loss+self.loss_reduction_function(data.criterionGAN(data.netD_A(data.fake_B[0]),True))
    #     #         classifier_loss=classifier_loss+self.loss_reduction_function(data.criterionGAN(data.netD_B(data.fake_A[0]),True))
    #     #         classifier_loss=classifier_loss+self.loss_reduction_function(data.criterionCycle(data.rec_A[0],data.real_A[0])*lambda_A)
    #     #         classifier_loss=classifier_loss+self.loss_reduction_function(data.criterionCycle(data.rec_B[0],data.real_B[0])*lambda_B)
                
            
        
        
       
    #     # self.losses_G=[loss_G_A+loss_G_B+loss_cycle_A+loss_cycle_B+idt_A_loss+idt_B_loss for loss_G_A,loss_G_B,loss_cycle_A,loss_cycle_B,idt_A_loss,idt_B_loss in zip(self.losses_G_A,self.losses_G_B,self.losses_cycle_A,self.losses_cycle_B,self.idt_A_losses,self.idt_B_losses)]
    #     # self.loss_G=self.losses_G[-1]
    #     data.loss_G=data.loss_G_A+data.loss_G_B+data.loss_cycle_A+data.loss_cycle_B+data.loss_idt_A+data.loss_idt_B
    #     #print("loss_G",data.loss_G.shape)
    #     if data.optimize_G:
    #         data.task_G_A.optimize_layers(data.loss_G)
    #     if data.optimize_C:
    #         data.task_G_A.optimize_steps_classifiers(data.loss_G,[data.fake_B_steps,data.rec_B_steps,data.idt_A_steps,data.fake_A_steps,data.rec_A_steps,data.idt_B_steps])
    #     #data.task_G_B.optimize_steps_classifiers(data.loss_G,[])
    #     # #self.task_G_B.optimize_layers2(self.loss_G)
        
    #     # data.task_G_A.optimize_steps_classifier(data.loss_G,data.fake_B_steps)
    #     # data.task_G_A.optimize_steps_classifier(data.loss_G,data.rec_B_steps)
    #     # data.task_G_A.optimize_steps_classifier(data.loss_G,data.idt_A_steps)
        
    #     # data.task_G_B.optimize_steps_classifier(data.loss_G,data.fake_A_steps)
       
    
        
    #     # data.task_G_B.optimize_steps_classifier(data.loss_G,data.rec_A_steps)
        
    #     # data.task_G_B.optimize_steps_classifier(data.loss_G,data.idt_B_steps)
        
    #     # # self.task_G_A.optimize_parameters(self.losses_G_A,self.task_G_A.previous_steps)
        
     
    def backward_G(self,data):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        # Identity loss
        # if lambda_idt > 0:
        #     # G_A should be identity if real_B is fed: ||G_A(B) - B||
        #     self.idt_A = self.netG_A(self.real_B)
        #     self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
        #     # G_B should be identity if real_A is fed: ||G_B(A) - A||
        #     self.idt_B = self.netG_B(self.real_A)
        #     self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        # else:
        #     self.loss_idt_A = 0
        #     self.loss_idt_B = 0
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            data.idt_A=data.task_G_A.forward(data.real_B)
           # self.idt_A_each_section=self.task_G_A.get_results()
            
            #self.idt_A_losses=[self.criterionIdt(self.idt_A_each_section[i],self.real_B)*(lambda_B*lambda_idt) for i in range(len(self.idt_A_each_section))]
            data.loss_idt_A=data.criterionIdt(data.idt_A,data.real_B)*(lambda_B*lambda_idt)
            data.idt_A_steps=data.task_G_A.previous_steps
            
            
            data.idt_B=data.task_G_B.forward(data.real_A)
            # self.idt_B_each_section=self.task_G_B.get_results()
            # self.idt_B_losses=[self.criterionIdt(self.idt_B_each_section[i],self.real_A)*(lambda_A*lambda_idt) for i in range(len(self.idt_B_each_section))]
            # self.idt_B_loss=self.idt_B_losses[-1]
            data.loss_idt_B=data.criterionIdt(data.idt_B,data.real_A)*(lambda_A*lambda_idt)
            data.idt_B_steps=data.task_G_B.previous_steps
  
          
        else:
            data.loss_idt_A = 0
            data.loss_idt_B = 0
            
        
        # # GAN loss D_A(G_A(A))
        # self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        # # GAN loss D_B(G_B(B))
        # self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        # # Forward cycle loss || G_B(G_A(A)) - A||
        # self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # # Backward cycle loss || G_A(G_B(B)) - B||
        # self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        # # combined loss and calculate gradients
        # self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
        # self.loss_G.backward()
        
        # # GAN loss D_A(G_A(A))
        # self.losses_G_A=[self.criterionGAN(self.netD_A(fake_B_each_section),True) for fake_B_each_section in self.fake_B_each_section]
        # self.loss_G_A=self.losses_G_A[-1]
        data.loss_G_A=data.criterionGAN(data.netD_A(data.fake_B),True)
        
        # GAN loss D_B(G_B(B))
        # self.losses_G_B=[self.criterionGAN(self.netD_B(fake_A_each_section),True) for fake_A_each_section in self.fake_A_each_section]
        # self.loss_G_B=self.losses_G_B[-1]
        data.loss_G_B=data.criterionGAN(data.netD_B(data.fake_A),True)
        # Forward cycle loss || G_B(G_A(A)) - A||
        # self.losses_cycle_A=[self.criterionCycle(rec_A_each_section,self.real_A) for rec_A_each_section in self.rec_A_each_section]
        # self.loss_cycle_A=self.losses_cycle_A[-1]
        data.loss_cycle_A=data.criterionCycle(data.rec_A,data.real_A)*lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        # self.losses_cycle_B=[self.criterionCycle(rec_B_each_section,self.real_B) for rec_B_each_section in self.rec_B_each_section]
        # self.loss_cycle_B=self.losses_cycle_B[-1]
        data.loss_cycle_B=data.criterionCycle(data.rec_B,data.real_B)*lambda_B
      
        
        
       
        # self.losses_G=[loss_G_A+loss_G_B+loss_cycle_A+loss_cycle_B+idt_A_loss+idt_B_loss for loss_G_A,loss_G_B,loss_cycle_A,loss_cycle_B,idt_A_loss,idt_B_loss in zip(self.losses_G_A,self.losses_G_B,self.losses_cycle_A,self.losses_cycle_B,self.idt_A_losses,self.idt_B_losses)]
        # self.loss_G=self.losses_G[-1]
        data.loss_G=data.loss_G_A+data.loss_G_B+data.loss_cycle_A+data.loss_cycle_B+data.loss_idt_A+data.loss_idt_B
        
        
        data.task_G_A.track(data.loss_G_A,data.fake_B_steps)
        data.task_G_A.track(data.loss_cycle_A,data.rec_B_steps)
        data.task_G_A.track(data.loss_idt_A,data.idt_A_steps)
            
        data.task_G_B.track(data.loss_G_B,data.fake_A_steps)
        data.task_G_B.track(data.loss_cycle_A,data.rec_A_steps)
        data.task_G_B.track(data.loss_idt_B,data.idt_B_steps)
            
        if data.optimize_G:
            data.task_G_A.optimize_layers(data.loss_G)
        #self.task_G_B.optimize_layers2(self.loss_G)
        
        
        if data.optimize_C:
        
            data.task_G_A.optimize_steps_classifier2(data.loss_G_A,data.fake_B_steps)
            data.task_G_A.optimize_steps_classifier2(data.loss_cycle_A,data.rec_B_steps)
            data.task_G_A.optimize_steps_classifier2(data.loss_idt_A,data.idt_A_steps)
            
            data.task_G_B.optimize_steps_classifier2(data.loss_G_B,data.fake_A_steps)
            data.task_G_B.optimize_steps_classifier2(data.loss_cycle_A,data.rec_A_steps)
            data.task_G_B.optimize_steps_classifier2(data.loss_idt_B,data.idt_B_steps)
           
            
        
        
        # self.task_G_A.optimize_parameters(self.losses_G_A,self.task_G_A.previous_steps)
        
     

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
    
        data=self.current_data
        if not data.optimize_D and not data.optimize_C and not data.optimize_G:
            return
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        
        self.set_requires_grad([data.netD_A, data.netD_B], False)  # Ds require no gradients when optimizing Gs
        #self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G(data)             # calculate gradients for G_A and G_B
        #self.optimizer_G.step()       # update G_A and G_B's weights
        # D_A and D_B
        
        self.set_requires_grad([data.netD_A, data.netD_B], True)
        if data.optimize_D:
                
            data.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
            self.backward_D_A(data)      # calculate gradients for D_A
            self.backward_D_B(data)      # calculate graidents for D_B
            data.optimizer_D.step()  # update D_A and D_B's weights
        data.detach()