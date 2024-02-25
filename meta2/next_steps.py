import torch
import torch.nn as nn

# class NextSteps:
#     def __init__(self,tensor:torch.Tensor,num_step_classes,num_next_steps ) -> None:
        
#         self.tensor=tensor
#         batch=tensor.size(0)
#         self.reshaped_tensor=tensor.view(-1,num_step_classes,num_next_steps).permute(1,0,2)
#         #batch first
#         self.indices=torch.argmax(self.reshaped_tensor,dim=-1).permute(1,0)
#         self.softmax=torch.softmax(self.reshaped_tensor,dim=-1).permute(1,0,2)
#         #expanded_indices = self.indices.unsqueeze(-1).expand(-1, -1, num_next_steps)
#         expanded_indices=self.indices.unsqueeze(-1).expand(-1,-1,num_next_steps)
#         # Then, gather probabilities. Since we've permuted reshaped_tensor to have steps as the first dimension, we gather along the second dimension (dim=1)
#         self.probability = torch.gather(self.softmax, 2, expanded_indices)
#         print(self.indices.shape,self.softmax.shape,self.probability.shape)

#         # print(self.indices.shape,self.softmax.shape)
#         # self.probability=torch.gather(self.softmax,0,self.indices.view(1,batch,num_next_steps))
#         #self.hidden_short_term=hidden_short_term
   
        
# class NextStepClassifier(torch.nn.Module):
#     def __init__(self,in_features,num_next_steps,num_step_classes) -> None:
#         super(NextStepClassifier, self).__init__()
#         self.nets=[nn.LSTMCell(input_size=in_features,hidden_size=num_step_classes*num_next_steps) for i in range(num_next_steps)]
#         self.num_next_steps=num_next_steps
#         self.num_step_classes=num_step_classes
#     def forward(self,features,previous:NextSteps,task:Task):
#         batch=features.size(0)
#         hidden_long_term=task.hidden_long_term[self]
#         if previous is None:
#             hidden_short_term=torch.zeros(batch,self.num_next_steps*self.num_step_classes)
#         else:
#             hidden_short_term=previous.tensor
#         if hidden_long_term is None:
#             hidden_long_term=torch.zeros(batch,self.num_step_classes*self.num_next_steps)
        
#         result=(hidden_short_term,hidden_long_term)
#         #outputs=[]
#         for net in self.nets:

#             result=net(features,result)
#             #outputs.append(result[0])
            
#         output,task.hidden_long_term[self]=result
#         #output=torch.stack(outputs,dim=1)
       
#         return NextSteps(output,self.num_step_classes,self.num_next_steps)

import torch
import torch.nn as nn

class NextSteps:
    def __init__(self, tensor: torch.Tensor, num_step_classes, num_next_steps):
        self.tensor = tensor
        batch = tensor.size(0)
        
        # Reshape tensor correctly assuming 'tensor' is [batch, num_step_classes*num_next_steps]
        # Assuming tensor should be reshaped to [num_step_classes, batch, num_next_steps]
        self.reshaped_tensor = tensor.view(batch, num_step_classes, num_next_steps).permute(2, 0, 1)
        print(batch,num_step_classes,num_next_steps,self.reshaped_tensor.shape)
        self.indices = torch.argmax(self.reshaped_tensor, dim=-1)
        print("self.indices.shape",self.indices.shape)
        self.softmax = torch.softmax(self.reshaped_tensor, dim=-1)
        print("self.softmax.shape", self.softmax.shape)
        # Correcting the expanded indices to match dimensions for gathering
        expanded_indices = self.indices.unsqueeze(-1)
            #.expand(-1, -1, num_next_steps)
        
        # Gathering probabilities
        self.probability = torch.gather(self.softmax, 2, expanded_indices)
        print("self.probability.shape", self.probability.shape)
        #print(self.indices.shape, self.softmax.shape, self.probability.shape)
#
# class NextStepClassifier(nn.Module):
#     def __init__(self, in_features, num_next_steps, num_step_classes,num_previous_steps):
#         super(NextStepClassifier, self).__init__()
#
#         self.nets = nn.ModuleList([nn.LSTMCell(input_size=in_features, hidden_size=num_step_classes * num_previous_steps) for _ in range(num_next_steps)])
#         self.num_next_steps = num_next_steps
#         self.num_step_classes = num_step_classes
#         self.num_previous_steps=num_previous_steps
#
#     def forward(self, features, previous: torch.Tensor|NextSteps, task):
#         print("forward",self.nets,features.shape)
#
#         batch = features.size(0)
#         hidden_long_term = task.hidden_long_term[self] if task.hidden_long_term.get(self) is not None else torch.zeros(batch, self.num_step_classes * self.num_previous_steps, device=features.device)
#
#         if previous is None:
#             hidden_short_term = torch.zeros(batch, self.num_step_classes * self.num_previous_steps, device=features.device)
#         else:
#             hidden_short_term = previous.tensor
#
#
#         hx = hidden_short_term
#         cx = hidden_long_term
#         print("hidden",batch,self.num_step_classes , self.num_next_steps,self.num_previous_steps)
#         for net in self.nets:
#             print("forward2",net,features.shape,hx.shape,cx.shape)
#
#             hx, cx = net(features, (hx, cx))
#             print("forward2 result", net, features.shape, hx.shape, cx.shape)
#
#         task.hidden_long_term[self] = cx.detach()
#
#         return NextSteps(hx, self.num_step_classes, self.num_next_steps)

class NextStepClassifier(nn.Module):
    def __init__(self, in_features, num_next_steps, num_step_classes):
        super(NextStepClassifier, self).__init__()

        self.nets = nn.ModuleList(
            [nn.LSTMCell(input_size=in_features, hidden_size=num_step_classes * num_next_steps) for _ in
             range(num_next_steps)])
        self.num_next_steps = num_next_steps
        self.num_step_classes = num_step_classes
        print("init num step classes",num_step_classes)


    def forward(self, features, previous: torch.Tensor | NextSteps, task):
        print("forward", self.nets, features.shape)

        batch = features.size(0)
        hidden_long_term = task.hidden_long_term[self] if task.hidden_long_term.get(self) is not None else torch.zeros(
            batch, self.num_step_classes * self.num_next_steps, device=features.device)

        if previous is None:
            hidden_short_term = torch.zeros(batch, self.num_step_classes * self.num_next_steps,
                                            device=features.device)
        else:
            hidden_short_term = previous.tensor
        hx = hidden_short_term

        if hx.size(1)!=self.num_step_classes * self.num_next_steps:
            print("self.num_step_classes",self.num_step_classes)
            hx=hx.view(batch,self.num_step_classes,-1)
            print(hx.shape,(batch,self.num_step_classes,))
            hx=torch.nn.functional.interpolate(hx,(self.num_step_classes ,))
            hx=hx.view(batch,-1)
        #
        # previous_steps=hx.size(1)
        # hx=hx[:,:,:]
        cx = hidden_long_term
        #print("hidden", batch, self.num_step_classes, self.num_next_steps, self.num_previous_steps)
        for net in self.nets:
            print("forward2", net, features.shape, hx.shape, cx.shape)

            hx, cx = net(features, (hx, cx))
            print("forward2 result", net, features.shape, hx.shape, cx.shape)

        task.hidden_long_term[self] = cx.detach()

        return NextSteps(hx, self.num_step_classes, self.num_next_steps)
