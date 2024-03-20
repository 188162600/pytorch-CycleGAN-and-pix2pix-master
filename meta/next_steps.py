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
from util.util import linear_interp,cosine_similarity_2d
import math
import random
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

class NextSteps:
    def __init__(self, tensor: torch.Tensor,index:torch.Tensor=None,softmax:torch.Tensor=None,probability:torch.Tensor=None,confidence:torch.Tensor=None,restored_step_index=None):
        #print("tensor1",tensor.shape)
        
        self.tensor = tensor
        #print("tensor2",tensor.shape)
        
        if index is None:
            self.indices = torch.argmax(tensor,dim=1)
        else :
            self.indices=index
        # print(self.indices)
        # print("self.indices.shape,self.tensor.shape",self.indices.shape,self.tensor.shape)
        expanded_indices = self.indices.unsqueeze(-1)
        if softmax is None:
            self.softmax = torch.softmax(self.tensor, dim=1)
        else:
            self.softmax=softmax
       
        if probability is None:
            self.probability = torch.gather(self.softmax, 1,expanded_indices).squeeze(-1)
        else:
            self.probability=probability
        if confidence is None:
            #print("self.probability",self.probability.shape)
            self.confidence=torch.sum(self.probability,dim=1)
        else:
            self.confidence=confidence
        self.restored_step_index=restored_step_index
     

# class NextSteps:
#     def __init__(self, tensor: torch.Tensor):
#         #print("tensor1",tensor.shape)
        
#         self.tensor = tensor
#         #print("tensor2",tensor.shape)
        
       
#         self.indices = torch.argmax(tensor,dim=1)
       
#         # print(self.indices)
#         # print("self.indices.shape,self.tensor.shape",self.indices.shape,self.tensor.shape)
#         expanded_indices = self.indices.unsqueeze(-1)
       
#         self.softmax = torch.softmax(self.tensor, dim=1)
        
       
        
#         self.probability = torch.gather(self.softmax, 1,expanded_indices) .squeeze(-1)
#         #print("self.probability",self.probability)
    
#         self.confidence=torch.sum(self.probability,dim=1)
#         #print("self.confidence",self.confidence.shape)
        
class RestoredSteps:
    def __init__(self,num_steps,num_options,num_samples,num_old,num_new,num_fresh,old_new_fresh_distribution=None) -> None:
        self.softmax=torch.zeros(num_old+num_new+num_fresh,num_options,num_steps)
        self.indices=torch.zeros(num_old+num_new+num_fresh,num_steps,dtype=torch.long)
        self.losses=torch.zeros(num_old+num_new+num_fresh,num_samples)
        self.occurrences=torch.zeros(num_old+num_new+num_fresh,dtype=torch.long)
        self.old_new_fresh_distribution=old_new_fresh_distribution
        
        
        self.num_sample=torch.zeros(num_old+num_new+num_fresh,dtype=torch.long)
        self.tracking_index=num_new+num_old
        
         
        torch.fill_(self.losses,math.inf)
       
        self.num_fresh=num_fresh
        self.num_old=num_old
        self.num_new=num_new
    def to(self,*args,**kwargs):
        self.softmax=self.softmax.to(*args,**kwargs)
        self.indices=self.indices.to(*args,**kwargs)
        self.losses=self.losses.to(*args,**kwargs)
        self.occurrences=self.occurrences.to(*args,**kwargs)
        self.num_sample=self.num_sample.to(*args,**kwargs)
        return self
    
    def aggregate_losses(self):
        # Unique occurrences
        unique_occurrences, indices = torch.unique(self.occurrences, return_inverse=True)
        # Initialize tensor for aggregated losses
        aggregated_losses = torch.zeros_like(unique_occurrences, dtype=torch.float)

        for i, occ in enumerate(unique_occurrences):
            # Indices of all tracking events with the current occurrence
            idx = (indices == i)
            # Select the corresponding losses and sample counts
            occ_losses = self.losses[idx]
            occ_samples = self.num_sample[idx]

            # Aggregate losses by calculating the mean for each occurrence
            total_loss = 0
            total_samples = 0
            for j, samples in enumerate(occ_samples):
                if samples > 0:  # Ensure division is meaningful
                    total_loss += occ_losses[j, :samples].sum()
                    total_samples += samples
            if total_samples > 0:  # Avoid division by zero
                aggregated_losses[i] = total_loss / total_samples
            # else:
            #     aggregated_losses[i]=math.inf

        return unique_occurrences.float(), aggregated_losses

    def linear_interp_loss(self, occurrence):
        # Assuming torch_linear_interp is defined as before
        occurrences, aggregated_losses = self.aggregate_losses()

        # Make sure occurrences are sorted
        sorted_indices = torch.argsort(occurrences)
        sorted_occurrences = occurrences[sorted_indices]
        sorted_losses = aggregated_losses[sorted_indices]

        # Interpolate
        interp_val = linear_interp(occurrence, sorted_occurrences, sorted_losses)
        return interp_val
    def get_losses(self):
        sum_loss=self.losses.sum(dim=1)
        return sum_loss/self.num_sample
    def get_efficiency(self):
        expected_loss=self.linear_interp_loss(self.occurrences)
        diff= expected_loss-self.get_losses()
        return diff
        
 
        
    def track(self,loss,next_steps:NextSteps):
        batch=next_steps.tensor.size(0)
        if next_steps.restored_step_index is not None:
            index_start=next_steps.restored_step_index
            index_end=(next_steps.restored_step_index+batch)%(self.num_old+self.num_new)
        else:
            index_start=self.tracking_index
            index_end=self.tracking_index+batch
            
            if index_end>=self.num_old+self.num_new+self.num_fresh:
               
                index_start=self.num_old+self.num_new
                index_start=index_start+batch
            self.tracking_index=index_end
        n=index_end-index_start
        if loss.dim()==0:
            self.losses[index_start:index_end]=loss
        else:
            self.losses[index_start:index_end]=loss[:n]
        
        self.softmax[index_start:index_end]=next_steps.softmax[:n]
        self.indices[index_start:index_end]=next_steps.indices[:n]
        # self.losses[index_start:index_end]=loss[:n]
        self.occurrences[index_start:index_end]+=1
       
        if next_steps.restored_step_index is None:
            self.tracking_index+=batch
            
    def reset_fresh(self):
        self.tracking_index=self.num_old+self.num_new
        self.occurrences[self.num_new+self.num_old:]=0
        self.num_sample[self.num_new+self.num_old:]=0
        self.losses[self.num_new+self.num_old:]=math.inf
    
       
        
    def update(self):
        # indices=torch.argsort(self.get_efficiency(),descending=True)[self.num_old+self.num_new:]
        # indices=torch.argsort(self.occurrences[indices],descending=True)
        sorted_indices_by_efficiency = torch.argsort(self.get_efficiency(), descending=True)

        # Select the subset of items, skipping the top self.num_old+self.num_new items.
        subset_indices = sorted_indices_by_efficiency[self.num_old+self.num_new:]

        # Now, get the occurrences of these selected items.
        subset_occurrences = self.occurrences[subset_indices]

        # Finally, sort these selected items by their occurrences in descending order.
        # Note: We sort subset_occurrences, but we need to sort subset_indices based on these occurrences.
        sorted_indices_by_occurrences = subset_indices[torch.argsort(subset_occurrences, descending=True)]

        self.softmax[self.num_old+self.num_new:]=self.softmax[sorted_indices_by_occurrences]
        self.indices[self.num_old+self.num_new]=self.indices[sorted_indices_by_occurrences]
        self.losses[self.num_old+self.num_new]=self.losses[sorted_indices_by_occurrences]
        self.occurrences[self.num_old+self.num_new]=self.occurrences[sorted_indices_by_occurrences]
        self.num_sample[self.num_old+self.num_new]=self.num_sample[sorted_indices_by_occurrences]
        self.reset_fresh()
        
        
        
    def get_new(self,next_steps:NextSteps):
       # print("self.softmax.shape,next_steps.softmax.shape",self.softmax.shape,next_steps.softmax.shape)
       
        softmax=self.softmax[self.num_old:self.num_old+self.num_new].view(self.num_new,-1)
        next_steps_softmax=next_steps.softmax.view(next_steps.softmax.size(0),-1)
      
        similarity=cosine_similarity_2d(softmax,next_steps_softmax)
       
        index=torch.argmax(similarity,dim=0)

       
        confidence=torch.gather(similarity,0,index.unsqueeze(0)).squeeze(0) *next_steps.confidence
        # print("softmax--",self.softmax[index].shape,next_steps.softmax.shape)
        # print("indices--",self.indices[index].shape,next_steps.indices.shape)
        # print("confidence",confidence.shape,next_steps.confidence.shape)
        
        return NextSteps(self.softmax[index],self.indices[index],self.softmax[index],None,confidence=confidence,restored_step_index=index)
    def get_old(self,next_steps:NextSteps):
        #print("self.softmax.shape,next_steps.softmax.shape",self.softmax.shape,next_steps.softmax.shape)
        softmax=self.softmax[:self.num_old].view(self.num_old,-1)
        next_steps_softmax=next_steps.softmax.view(next_steps.softmax.size(0),-1)
        
        similarity=cosine_similarity_2d(softmax,next_steps_softmax)
        
        index=torch.argmax(similarity,dim=0)
        
        confidence=torch.gather(similarity ,0,index.unsqueeze(0)).squeeze(0)*next_steps.confidence
        # print("softmax--",self.softmax[index].shape,next_steps.softmax.shape)
        # print("indices--",self.indices[index].shape,next_steps.indices.shape)
        # print("confidence",confidence.shape,next_steps.confidence.shape)
        return NextSteps(self.softmax[index],self.indices[index],self.softmax[index],None,confidence=confidence,restored_step_index=index)
    def get_fresh(self,next_steps):
        return next_steps
    def get_random(self, next_steps):
        weight=list(self.old_new_fresh_distribution)
        if self.losses[0][0].item() == math.inf:
            weight[0]=0
        if self.losses[self.num_old][0].item() == math.inf:
            weight[1]=0
        which=random.choices((self.get_old,self.get_new,self.get_fresh),weights=weight,k=1)[0]
        return which(next_steps)
        
    
class NextStepClassifier(nn.Module):
    def __init__(self, in_features_shape, num_next_steps, num_step_classes,encoder,):
        super(NextStepClassifier, self).__init__()
        
        self.encoder=encoder
        dummy_input=torch.zeros(1,*in_features_shape)
        # print("encode2",self.encoder)
        # print(self.encoder(dummy_input))
        # print("encoder",self.encoder)
        # print("dummy_input",dummy_input.shape)
        self.in_features_size=self.encoder(dummy_input).numel()
        #self.recorded_steps=RestoredSteps(num_steps=num_next_steps,num_options=num_step_classes,)
       
        
        
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
        #print("features",features.shape)
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
