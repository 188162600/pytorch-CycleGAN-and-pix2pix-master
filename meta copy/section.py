import math
import typing
import torch
import torch.nn as nn
import copy
from meta.next_steps import NextSteps
from meta.next_steps import NextStepClassifier



class Section(nn.Module):
    def __init__(self,name,layers:typing.List[nn.Module],num_options_each_layer,is_encoder=lambda x:isinstance(x,nn.Conv2d)) -> None:
        super().__init__()
        self.base_layers=layers
        self.is_setup=False
        self.num_options_each_layer=num_options_each_layer
        self.name=name
        
        self.last_sections_encode_shapes=[]
        self.last_sections_steps=[]
        self.is_encoder=is_encoder
        self.step_classifier_encoder=None
     
        self.features=None
    def append_layer(self,layer):
        self.base_layers.append(layer)
    def append_layers(self,layers):
        self.base_layers.extend(layers)
    def set_step_classifier_encoder(self,encoder):
        self.step_classifier_encoder=encoder
    def setup(self):
        if self.is_setup:
            return 

        self.layers=[]
        self.is_layer_with_params=[]
        self.num_layers_with_params=0
        self.num_total_layers=0
        self.last_feature_index=None
        for layer in self.base_layers:
            self.layers.append([copy.deepcopy(layer) for i in range(self.num_options_each_layer)])
            for j,layer in  enumerate(self.layers[-1]):
                self.register_module(f"{self.num_total_layers},{j}",layer)

            if self.is_encoder(layer):
                self.last_feature_index=self.num_total_layers
            if len(list(layer.parameters()))>0:
                self.num_layers_with_params+=1
                self.is_layer_with_params.append(True)
            else:
                self.is_layer_with_params.append(False) 
            self.num_total_layers+=1    
         
        self.last_sections_encode_shapes.append((1,self.num_total_layers,self.num_options_each_layer))
        self.input_features_shape=[*max(self.last_sections_encode_shapes,key=math.prod)]
        
        
      
   
  
        self.last_sections_steps.append(self.num_total_layers)
        self.last_sections_step=max(self.last_sections_steps)
   


        self.classifier=NextStepClassifier(self.input_features_shape,self.num_total_layers,self.num_options_each_layer,self.step_classifier_encoder)
        #self.add_module ("next_steps_classifier",self.classifier)
        self.is_setup=True
    def dummy_forward(self,data):
        #print(self.base_layers)
        #print("data",data.shape)
        self.features=None
        for i,layer in enumerate(self.base_layers):
           
            data=layer(data)
            if self.is_encoder(layer):
                with torch.no_grad():
                    self.features=data
                    print("self.features.shape",self.features.shape)
        # print("features",self.features.shape)
        # print("data",data.shape)
            
        return data
        
    def forward(self,data:torch.Tensor,features:torch.Tensor,previous_steps:NextSteps,task):
        # print("section vars",vars(self))
        # print("-------------------------------------------------")
        assert self.is_setup
        #print(self.classifier)
        
        # batch=features.size(0)
        
        # features=features[0].view(batch,-1)
        #print("features",features.shape,self.input_features_size)
       # print("features.shape,self.input_features_size,self.feature_adjustment",features.shape,self.input_features_size,self.feature_adjustment)
        
        #print("features--",features)
       
     

        # features=features.detach()
        
      
        #print("features",features.shape)
        #print("features--",features)
       
        next_steps=self.classifier.forward(features,previous_steps,task)
        #print("after forward",next_steps.tensor.shape)
        self.next_steps=next_steps
        #print("len(self.layers),self.num_total_layers",len(self.layers),self.num_total_layers)
        layer_with_params_index=0
        for i in range(self.num_total_layers):
            
            if self.is_layer_with_params[i]:
                index=next_steps.indices[layer_with_params_index]
            else:
                index=0
               
            #print("index i",index,i)
            #print(next_steps.indices)
            #print(index)

            layer=self.layers[i][index]
            data=layer(data)
            if i==self.last_feature_index:
                # with torch.no_grad():
                self.features=data.detach()
            self.last_feature_index+=1
        return data
    def get_steps_classifier_loss(self,loss,next_steps:NextSteps):
        confidence=torch.sum(next_steps.probability)
        return confidence*(loss.detach())

            
        
        
        
        #self.layers=[]
        

    
    
    