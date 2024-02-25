import torch
import torch.nn as nn



import torch
import torch.nn as nn

class NextSteps:
    def __init__(self, tensor: torch.Tensor, num_step_classes, num_next_steps):
        self.tensor = tensor
        batch = tensor.size(0)
        
        # Reshape tensor correctly assuming 'tensor' is [batch, num_step_classes*num_next_steps]
        # Assuming tensor should be reshaped to [num_step_classes, batch, num_next_steps]
        self.reshaped_tensor = tensor.view(batch, num_step_classes, num_next_steps).permute(2, 0, 1)
        #print(batch,num_step_classes,num_next_steps,self.reshaped_tensor.shape)
        self.indices = torch.argmax(self.reshaped_tensor, dim=-1)
        #print("self.indices.shape",self.indices.shape)
        self.softmax = torch.softmax(self.reshaped_tensor, dim=-1)
        #print("self.softmax.shape", self.softmax.shape)
        # Correcting the expanded indices to match dimensions for gathering
        expanded_indices = self.indices.unsqueeze(-1)
            #.expand(-1, -1, num_next_steps)
        
        # Gathering probabilities
        self.probability = torch.gather(self.softmax, 2, expanded_indices)
        #print("self.probability.shape", self.probability.shape)
        #print(self.indices.shape, self.softmax.shape, self.probability.shape)

class NextStepClassifier(nn.Module):
    def __init__(self, in_features_shape, num_next_steps, num_step_classes,encoder):
        super(NextStepClassifier, self).__init__()
        
        self.encoder=encoder
        dummy_input=torch.zeros(1,*in_features_shape)
        # print("encode2",self.encoder)
        # print(self.encoder(dummy_input))
        print("encoder",self.encoder)
        print("dummy_input",dummy_input.shape)
        self.in_features_size=self.encoder(dummy_input).numel()
        
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

        batch = features.size(0)
        #print("encoder",self.encoder)
        #print("features device",features.device)
        features=self.encoder(features)
        features=features.view(batch,-1)
        features_size=features[0].numel()
        
        if features_size!=self.in_features_size:
            features=nn.functional.pad(features,(0,0,0,self.in_features_size-features_size))
        
        #print("feature shape after",features.shape)
        hidden_long_term = task.hidden_long_term[self] if task.hidden_long_term.get(self) is not None else torch.zeros(
            batch, self.num_step_classes * self.num_next_steps, device=features.device)

        if previous is None:
            hidden_short_term = torch.zeros(batch, self.num_step_classes * self.num_next_steps,
                                            device=features.device)
        else:
            hidden_short_term = previous.tensor
     
        hx = hidden_short_term
      
        if hx.size(1)!=self.num_step_classes * self.num_next_steps:
            #print("self.num_step_classes",self.num_step_classes)
            hx=hx.view(batch,self.num_step_classes,-1)
            #print(hx.shape,(batch,self.num_step_classes,))
            #print("interpolating",hx.shape)
            hx=torch.nn.functional.interpolate(hx,(self.num_next_steps ,))
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

            hx, cx = net(features, (hx, cx))
            #print("forward2 result", net, features.shape, hx.shape, cx.shape)

        task.hidden_long_term[self] = cx.detach()

        return NextSteps(hx, self.num_step_classes, self.num_next_steps)
