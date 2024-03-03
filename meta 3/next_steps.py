import torch
import torch.nn as nn



import torch
import torch.nn as nn

class NextSteps:
    def __init__(self, tensor: torch.Tensor, num_step_classes, num_next_steps):
        self.tensor = tensor
        batch = tensor.size(0)

        self.reshaped_tensor = tensor.permute(2, 0, 1)

        self.probability=[]
        self.confidence=None

    def next_indices(self, index, num_options, num_indices):
        # print("self.reshaped_tensor",self.reshaped_tensor.shape)
        tensor = self.reshaped_tensor[0:num_options, index, ]

        indices = torch.argsort(tensor, descending=True, dim=0)[0:num_indices]



        softmax = torch.softmax(tensor, dim=0)[0:num_options]
        # print("tensor", tensor)
        # print("indices", indices)
        # print("softmax",softmax)
        # try:
        #     print("probbility",torch.gather(softmax, 0, indices).shape)
        # except:
        #     print(softmax,indices,tensor,num_options,num_indices)
        #     raise
        #print(self.probability[-1].shape)
        #print(softmax.shape,indices.shape,"softmax.shape,indices.shape")
        self.probability.append(torch.gather(softmax, 0, indices))
        #print("probablity",torch.gather(softmax, 0, indices))
        return indices


class NextStepClassifier(nn.Module):
    def __init__(self, in_features_shape, num_next_steps, num_step_classes,encoder):
        super(NextStepClassifier, self).__init__()
        
        self.encoder=encoder
        dummy_input=torch.zeros(1,*in_features_shape)
       
        self.in_features_size=self.encoder(dummy_input).numel()
        self.net=nn.LSTM( input_size =self.in_features_size,hidden_size = num_step_classes ,num_layers =num_next_steps)
        # for param in self.net.parameters():
        #     torch.nn.init.normal_(param,0.5)

        self.register_module("net",self.net)
        self.num_next_steps = num_next_steps
        self.num_step_classes = num_step_classes
      

    def forward(self, features, previous:NextSteps, task):
        #print("features",features)
        batch = features.size(0)

        features=self.encoder(features)
        
        features=features.view(batch,-1)
        features_size=features[0].numel()
        
        if features_size!=self.in_features_size:
            features=nn.functional.pad(features,(0,0,0,self.in_features_size-features_size))
        features= torch.unsqueeze(features,1).repeat(1,self.num_next_steps,1)
        features=features.view(self.num_next_steps,batch,-1)

        hidden_long_term = task.hidden_long_term.get(self)
        if hidden_long_term is None:
            hidden_long_term=torch.zeros((self.num_next_steps,batch, self.num_step_classes), device=features.device)

        if previous is None:

            hidden_short_term = torch.zeros((self.num_next_steps,batch,self.num_step_classes ),device=features.device)
        else:
            hidden_short_term = previous.tensor.clone()
    
        hx = hidden_short_term
    
        if hx.size(0)!= self.num_next_steps or hx.size(2)!= self.num_step_classes:
            hx = hx.permute(1, 0, 2)
            hx=hx.unsqueeze(0)


            hx=torch.nn.functional.interpolate(hx,(self.num_next_steps,self.num_step_classes ,))[0]
            hx = hx.permute(1, 0, 2)

        cx = hidden_long_term

        output,(hx, cx) = self.net(features, (hx, cx))
        #output, (hx, cx) = self.net(features, (hx, cx))
            #print("forward2 result", net, features.shape, hx.shape, cx.shape)
        #hx=hx.view(batch, self.num_step_classes, self.num_next_steps)
        task.hidden_long_term[self] = cx.detach()
        #print("output",output.shape)
        #print("output.grad_fn",output.grad_fn)
        return NextSteps(output, self.num_step_classes, self.num_next_steps)
