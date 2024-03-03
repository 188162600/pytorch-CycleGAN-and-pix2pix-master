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
    def __init__(self, tensor: torch.Tensor, num_picking_classes):
        self.tensor = tensor
        batch = tensor.size(0)
        #print(num_picking_classes,"num_picking_classes")
        # Reshape tensor correctly assuming 'tensor' is [batch, num_step_classes*num_next_steps]
        # Assuming tensor should be reshaped to [num_step_classes, batch, num_next_steps]
        #self.reshaped_tensor = tensor.permute(2, 0, 1)
        #print("self.reshaped_tensor.shape",self.reshaped_tensor.shape)
        #print(batch,num_step_classes,num_next_steps,self.reshaped_tensor.shape)
        #hx=hx.view(batch, self.num_step_classes, self.num_next_steps)
        self.indices = torch.argsort(tensor,dim=1,descending=True)[:,0:num_picking_classes,:]
        #print("self.indices.shape",self.indices.shape)
        self.softmax = torch.softmax(self.tensor, dim=1)
       
        self.probability = torch.gather(self.softmax, 1,self. indices)
        #print(self.indices)
        #print("num_picking_classes",num_picking_classes,self.probability.shape,self.indices.shape,self.softmax.shape)
        #print("self.probability.shape", self.probability.shape)
        #print(self.indices.shape, self.softmax.shape, self.probability.shape)

class NextStepClassifier(nn.Module):
    def __init__(self, in_features_shape, num_next_steps, num_step_classes,num_picking_classes,encoder):
        super(NextStepClassifier, self).__init__()
        
        self.encoder=encoder
        dummy_input=torch.zeros(1,*in_features_shape)
        # print("encode2",self.encoder)
        # print(self.encoder(dummy_input))
        # print("encoder",self.encoder)
        # print("dummy_input",dummy_input.shape)
        self.in_features_size=self.encoder(dummy_input).numel()
        self.num_picking_classes=num_picking_classes
        
        
        self.nets = nn.ModuleList(
            [nn.LSTMCell(input_size= self.in_features_size, hidden_size=num_step_classes * num_next_steps) for _ in
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
            batch, self.num_step_classes * self.num_next_steps, device=features.device)

        if previous is None:
            hidden_short_term = torch.zeros(batch, self.num_step_classes , self.num_next_steps,
                                            device=features.device)
        else:
            hidden_short_term = previous.tensor
     
        hx = hidden_short_term
      
        if hx.size(1)!=self.num_step_classes or hx.size(2)!= self.num_next_steps:
            #print("self.num_step_classes",self.num_step_classes)
            #hx=hx.view(batch,self.num_step_classes,-1)
            #print(hx.shape,(batch,self.num_step_classes,))
            #print("interpolating",hx.shape)
            #print("hx.shape,(self.num_step_classes,self.num_next_steps ,)",hx.shape,(self.num_step_classes,self.num_next_steps ,))
            hx=hx.unsqueeze(0)
            #print("hx.shape,(self.num_step_classes,self.num_next_steps ,)",hx.shape,(self.num_step_classes,self.num_next_steps ,))
            hx=torch.nn.functional.interpolate(hx,(self.num_step_classes,self.num_next_steps ,))
            #print("interpolated",hx.shape)
        hx=hx.view(batch,-1)
        #
        # previous_steps=hx.size(1)
        # hx=hx[:,:,:]
        cx = hidden_long_term
        #print("hx cx",hx,cx,features)
        #print("hidden", batch, self.num_step_classes, self.num_next_steps, self.num_previous_steps)
        for net in self.nets:
            #print("forward2", net, features.shape, hx.shape, cx.shape)
            #print("forward2",features.shape,hx.shape,cx.shape)
            hx, cx = net(features.clone().detach(), (hx.clone().detach(), cx))
            #print("forward2 result", net, features.shape, hx.shape, cx.shape)
        hx=hx.view(batch, self.num_step_classes, self.num_next_steps)
        task.hidden_long_term[self] = cx.detach()

        return NextSteps(hx, self.num_picking_classes)

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
