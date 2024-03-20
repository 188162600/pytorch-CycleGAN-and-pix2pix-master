# import torch
# import torch.nn as nn



# import torch
# import torch.nn as nn

# class NextSteps:
#     def __init__(self, tensor: torch.Tensor, num_step_classes, num_next_steps):
#         self.tensor = tensor
#         batch = tensor.size(0)
        
#         # Reshape tensor correctly assuming 'tensor' is [batch, num_step_classes*num_next_steps]
#         # Assuming tensor should be reshaped to [num_step_classes, batch, num_next_steps]
#         self.reshaped_tensor = tensor.permute(2, 0, 1)
#         #print(batch,num_step_classes,num_next_steps,self.reshaped_tensor.shape)
#         self.indices = torch.argmax(self.reshaped_tensor, dim=-1)
#         #print("self.indices.shape",self.indices.shape)
#         self.softmax = torch.softmax(self.reshaped_tensor, dim=-1)
#         #print("self.softmax.shape", self.softmax.shape)
#         # Correcting the expanded indices to match dimensions for gathering
#         expanded_indices = self.indices.unsqueeze(-1)
#             #.expand(-1, -1, num_next_steps)
        
#         # Gathering probabilities
#         self.probability = torch.gather(self.softmax, 2, expanded_indices)
#         #print("self.probability.shape", self.probability.shape)
#         #print(self.indices.shape, self.softmax.shape, self.probability.shape)

# class NextStepClassifier(nn.Module):
#     def __init__(self, in_features_shape, num_next_steps, num_step_classes,encoder):
#         super(NextStepClassifier, self).__init__()
        
#         self.encoder=encoder
#         dummy_input=torch.zeros(1,*in_features_shape)
#         # print("encode2",self.encoder)
#         # print(self.encoder(dummy_input))
#         # print("encoder",self.encoder)
#         # print("dummy_input",dummy_input.shape)
#         self.in_features_size=self.encoder(dummy_input).numel()
        
#         self.nets = nn.ModuleList(
#             [nn.LSTMCell(input_size= self.in_features_size, hidden_size=num_step_classes * num_next_steps) for _ in
#              range(num_next_steps)])
#         self.num_next_steps = num_next_steps
#         self.num_step_classes = num_step_classes
#         #print("init num step classes",num_step_classes)


#     def forward(self, features, previous:NextSteps, task):
#         # print("classifier vars",vars(self))
#         # print("")
#         # print("forward", self.nets, features)
#         #print("Next Steps Features",features.shape)
#         batch = features.size(0)
#         #print("encoder",self.encoder)
#         #print("features device",features.device)
#         features=self.encoder(features)
#         #print("Next Steps Features encode",features.shape)
#         features=features.view(batch,-1)
#         features_size=features[0].numel()
        
#         if features_size!=self.in_features_size:
#             features=nn.functional.pad(features,(0,0,0,self.in_features_size-features_size))
        
#         #print("feature shape after",features.shape)
#         hidden_long_term = task.hidden_long_term[self] if task.hidden_long_term.get(self) is not None else torch.zeros(
#             batch, self.num_step_classes * self.num_next_steps, device=features.device)

#         if previous is None:
#             hidden_short_term = torch.zeros(batch, self.num_step_classes , self.num_next_steps,
#                                             device=features.device)
#         else:
#             hidden_short_term = previous.tensor
     
#         hx = hidden_short_term
      
#         if hx.size(1)!=self.num_step_classes or hx.size(2)!= self.num_next_steps:
#             #print("self.num_step_classes",self.num_step_classes)
#             #hx=hx.view(batch,self.num_step_classes,-1)
#             #print(hx.shape,(batch,self.num_step_classes,))
#             #print("interpolating",hx.shape)
#             #print("hx.shape,(self.num_step_classes,self.num_next_steps ,)",hx.shape,(self.num_step_classes,self.num_next_steps ,))
#             hx=hx.unsqueeze(0)
#             #print("hx.shape,(self.num_step_classes,self.num_next_steps ,)",hx.shape,(self.num_step_classes,self.num_next_steps ,))
#             hx=torch.nn.functional.interpolate(hx,(self.num_step_classes,self.num_next_steps ,))
#             #print("interpolated",hx.shape)
#         hx=hx.view(batch,-1)
#         #
#         # previous_steps=hx.size(1)
#         # hx=hx[:,:,:]
#         cx = hidden_long_term
#         #print("hx cx",hx,cx,features)
#         #print("hidden", batch, self.num_step_classes, self.num_next_steps, self.num_previous_steps)
#         for net in self.nets:
#             #print("forward2", net, features.shape, hx.shape, cx.shape)
#             #print("forward2",features.shape,hx.shape,cx.shape)
#             hx, cx = net(features, (hx, cx))
#             #print("forward2 result", net, features.shape, hx.shape, cx.shape)
#         hx=hx.view(batch, self.num_step_classes, self.num_next_steps)
#         task.hidden_long_term[self] = cx.detach()

#         return NextSteps(hx, self.num_step_classes, self.num_next_steps)

import torch
import torch.nn as nn



import torch
import torch.nn as nn
from util.util import linear_interp
import math

# class NextSteps:
#     def __init__(self, tensor: torch.Tensor, num_step_classes, num_next_steps):
#         self.tensor = tensor
#         batch = tensor.size(0)

#         self.reshaped_tensor = tensor.permute(2, 0, 1)

#         self.probability=[]


#     def next_indices(self, index, num_options, num_indices):
#         # print("self.reshaped_tensor",self.reshaped_tensor.shape)
#         tensor = self.reshaped_tensor[0:num_options, index, ]

#         indices = torch.argsort(tensor, descending=True, dim=0)[0:num_indices]



#         softmax = torch.softmax(tensor, dim=0)[0:num_options]
#         # print("tensor", tensor)
#         # print("indices", indices)
#         # print("softmax",softmax)
#         # try:
#         #     print("probbility",torch.gather(softmax, 0, indices).shape)
#         # except:
#         #     print(softmax,indices,tensor,num_options,num_indices)
#         #     raise
#         #print(self.probability[-1].shape)
#         #print(softmax.shape,indices.shape,"softmax.shape,indices.shape")
#         self.probability.append(torch.gather(softmax, 0, indices))
#         #print("probablity",torch.gather(softmax, 0, indices))
#         return indices

# class NextSteps:
#     def __init__(self, tensor: torch.Tensor,index:torch.Tensor=None,softmax:torch.Tensor=None,probability:torch.Tensor=None,confidence:torch.Tensor=None,restored_step_index=None):
#         #print("tensor1",tensor.shape)
        
#         self.tensor = tensor
#         #print("tensor2",tensor.shape)
        
#         if index is None:
#             self.indices = torch.argmax(tensor,dim=1)
#         else :
#             self.indices=index
#         # print(self.indices)
#         # print("self.indices.shape,self.tensor.shape",self.indices.shape,self.tensor.shape)
#         expanded_indices = self.indices.unsqueeze(-1)
#         if softmax is None:
#             self.softmax = torch.softmax(self.tensor, dim=1)
#         else:
#             self.softmax=softmax
       
#         if probability is None:
#             self.probability = torch.gather(self.softmax, 1,expanded_indices)
#         else:
#             self.probability=probability
#         if confidence is None:
#             print("self.probability",self.probability.shape)
#             self.confidence=torch.sum(self.probability,dim=1)
#         else:
#             self.confidence=confidence
#         self.restored_step_index=restored_step_index
     
        
# class RestoredSteps:
#     def __init__(self,num_steps,num_options,num_old,num_new,num_tracking_sample,num_tracking) -> None:
#         self.softmax=torch.zeros(num_old+num_new+num_tracking,num_steps,num_options)
#         self.indices=torch.zeros(num_old+num_new+num_tracking,num_steps,dtype=torch.long)
        
#         self.losses=torch.zeros(num_old+num_new+num_tracking,num_tracking_sample)
#         self.occurrences=torch.zeros(num_old+num_new+num_tracking,dtype=torch.long)

        
#         self.num_sample=torch.zeros(num_old+num_new+num_tracking,dtype=torch.long)
        
        
        
#         self.tracking_index=num_new+num_old
        
         
#         torch.fill_(self.losses,math.inf)
       
#         self.num_tracking=num_tracking
#         self.num_old=num_old
#         self.num_new=num_new
#     def aggregate_losses(self):
#         # Unique occurrences
#         unique_occurrences, indices = torch.unique(self.occurrences, return_inverse=True)
#         # Initialize tensor for aggregated losses
#         aggregated_losses = torch.zeros_like(unique_occurrences, dtype=torch.float)

#         for i, occ in enumerate(unique_occurrences):
#             # Indices of all tracking events with the current occurrence
#             idx = (indices == i)
#             # Select the corresponding losses and sample counts
#             occ_losses = self.losses[idx]
#             occ_samples = self.num_sample[idx]

#             # Aggregate losses by calculating the mean for each occurrence
#             total_loss = 0
#             total_samples = 0
#             for j, samples in enumerate(occ_samples):
#                 if samples > 0:  # Ensure division is meaningful
#                     total_loss += occ_losses[j, :samples].sum()
#                     total_samples += samples
#             if total_samples > 0:  # Avoid division by zero
#                 aggregated_losses[i] = total_loss / total_samples
#             # else:
#             #     aggregated_losses[i]=math.inf

#         return unique_occurrences.float(), aggregated_losses

#     def linear_interp_loss(self, occurrence):
#         # Assuming torch_linear_interp is defined as before
#         occurrences, aggregated_losses = self.aggregate_losses()

#         # Make sure occurrences are sorted
#         sorted_indices = torch.argsort(occurrences)
#         sorted_occurrences = occurrences[sorted_indices]
#         sorted_losses = aggregated_losses[sorted_indices]

#         # Interpolate
#         interp_val = linear_interp(occurrence, sorted_occurrences, sorted_losses)
#         return interp_val
#     def get_losses(self):
#         pass 
#     def get_efficiency(self):
#         expected_loss=self.linear_interp_loss(self.occurrences)
#         diff= expected_loss-self.get_losses()
#         return diff
        
 
        
#     def track(self,next_steps:NextSteps,loss):
#         self.losses[self.tracking_index]=loss
        
#         batch=next_steps.tensor.size(0)
#         self.softmax[self.tracking_index:self.tracking_index+batch]=next_steps.probability
#         self.indices[self.tracking_index:self.tracking_index+batch]=next_steps.indices
#         self.losses[self.tracking_index:self.tracking_index+batch]=loss
#         self.tracking_index+=batch
#         if self.tracking_index%self.num_tracking==0:
#             self.tracking_index=self.num_old+self.num_new

#     def update(self,batch):
       
        
#         torch.fill_(self.losses[self.num_old+self.num_new:],math.inf)

        
#         pass 
#     def get(self,next_steps):
#         pass 

class NextSteps:
    def __init__(self, tensor: torch.Tensor):
        #print("tensor1",tensor.shape)
        
        self.tensor = tensor
        #print("tensor2",tensor.shape)
        
       
        self.indices = torch.argmax(tensor,dim=1)
       
        # print(self.indices)
        # print("self.indices.shape,self.tensor.shape",self.indices.shape,self.tensor.shape)
        expanded_indices = self.indices.unsqueeze(-1)
       
        self.softmax = torch.softmax(self.tensor, dim=1)
        
       
        
        self.probability = torch.gather(self.softmax, 1,expanded_indices) .squeeze(-1)
        #print("self.probability",self.probability.shape)
    
         
        self.confidence=torch.sum(self.probability,dim=1)
        #print("self.confidence",self.confidence.shape)
       
class NextStepClassifier(nn.Module):
    def __init__(self, in_features_shape, num_next_steps, num_step_classes,encoder):
        super(NextStepClassifier, self).__init__()
        
        self.encoder=encoder
        dummy_input=torch.zeros(1,*in_features_shape)
        # print("encode2",self.encoder)
        # print(self.encoder(dummy_input))
        # print("encoder",self.encoder)
        # print("dummy_input",dummy_input.shape)
        self.in_features_size=self.encoder(dummy_input).numel()
       
        
        
        self.nets = nn.ModuleList(
            [nn.LSTMCell(input_size= self.in_features_size, hidden_size=num_step_classes*num_next_steps ) for _ in
             range(num_next_steps)])
        self.num_next_steps = num_next_steps
        self.num_step_classes = num_step_classes
        #print("init num step classes",num_step_classes)


    def forward(self, features, previous:NextSteps, task):
        # print("classifier vars",vars(self))
        # print("")
        # print("forward", self.nets, features)
        #print("Next Steps Features",features.shape)
        batch = features.size(0)
        #print("encoder",self.encoder)
        #print("features device",features.device)
        features=self.encoder(features)
        #print("Next Steps Features encode",features.shape)
        features=features.view(batch,-1)
        features_size=features[0].numel()
        
        if features_size!=self.in_features_size:
            features=nn.functional.pad(features,(0,0,0,self.in_features_size-features_size))
        
        #print("feature shape after",features.shape)
        hidden_long_term = task.hidden_long_term[self] if task.hidden_long_term.get(self) is not None else torch.zeros(
            batch, self.num_step_classes *self.num_next_steps, device=features.device)

        if previous is None:
            hx = torch.zeros(batch, self.num_step_classes ,self.num_next_steps,
                                            device=features.device)
        else:
            hx = previous.tensor #.clone().detach()
     
        if hx.size(1)!=self.num_step_classes or hx.size(2)!=self.num_next_steps:
            hx=hx.unsqueeze(0)
            hx=torch.nn.functional.interpolate(hx,(self.num_step_classes,self.num_next_steps ,))
            #print("interpolated",hx.shape)
            #hx=hx[0]
        hx=hx.view(batch,-1)
      
        cx = hidden_long_term
        #print("hx",hx.shape,"cx",cx.shape,"features",features.shape)
       
        #result=[]
        
        for net in self.nets:
            #print("forward2", net, features.shape, hx.shape, cx.shape)
            #print("forward2",features.shape,hx.shape,cx.shape)
            hx, cx = net(features, (hx, cx))
            #result.append(hx)
            #print("forward2 result", net, features.shape, hx.shape, cx.shape)
        #result=torch.cat(result,dim=1)
      
        hx=hx.view(batch, self.num_step_classes, self.num_next_steps)
        task.hidden_long_term[self] = cx.detach()

        #print("result",result.shape)
        return NextSteps(hx)

# class NextStepClassifier(nn.Module):
#     def __init__(self, in_features_shape, num_next_steps, num_step_classes,num_picking_classes,encoder):
#         super(NextStepClassifier, self).__init__()
        
#         self.encoder=encoder
#         dummy_input=torch.zeros(1,*in_features_shape)
       
#         self.in_features_size=self.encoder(dummy_input).numel()
#         self.net=nn.LSTM( input_size =self.in_features_size,hidden_size = num_step_classes ,num_layers =num_next_steps)
#         # for param in self.net.parameters():
#         #     torch.nn.init.normal_(param,0.5)

#         #self.register_module("net",self.net)
#         self.num_next_steps = num_next_steps
#         self.num_step_classes = num_step_classes
#         self.num_picking_classes=num_picking_classes
      

#     def forward(self, features, previous:NextSteps, task):
#         #print("features",features)
#         batch = features.size(0)

#         features=self.encoder(features)
        
#         features=features.view(batch,-1)
#         features_size=features[0].numel()
        
#         if features_size!=self.in_features_size:
#             features=nn.functional.pad(features,(0,0,0,self.in_features_size-features_size))
#         features= torch.unsqueeze(features,1).repeat(1,self.num_next_steps,1)
#         features=features.view(self.num_next_steps,batch,-1)

#         hidden_long_term = task.hidden_long_term.get(self)
#         if hidden_long_term is None:
#             hidden_long_term=torch.zeros((self.num_next_steps,batch, self.num_step_classes), device=features.device)

#         if previous is None:

#             hidden_short_term = torch.zeros((batch,self.num_step_classes,self.num_next_steps ),device=features.device)
#         else:
#             hidden_short_term = previous.tensor.clone().detach()
    
#         hx = hidden_short_term
#         print("hx",hx.shape)
#         if hx.size(1)!= self.num_step_classes or hx.size(2)!= self.num_next_steps :
#             #hx = hx.permute(1, 0, 2)
#             hx=hx.unsqueeze(0)


#             hx=torch.nn.functional.interpolate(hx,(self.num_step_classes ,self.num_next_steps,))[0]
#         hx = hx.permute(2, 0, 1)

#         cx = hidden_long_term

#         output,(hx, cx) = self.net(features.clone().detach(), (hx.clone().detach(), cx))
#         #output, (hx, cx) = self.net(features, (hx, cx))
#             #print("forward2 result", net, features.shape, hx.shape, cx.shape)
#         #hx=hx.view(batch, self.num_step_classes, self.num_next_steps)
#         task.hidden_long_term[self] = cx.detach()
#         #print("output",output.shape)
#         #print("output.grad_fn",output.grad_fn)
#         print("output",output.shape)    
#         return NextSteps(output.permute(1,0,2),self.num_picking_classes)
