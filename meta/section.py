import math
import typing
import torch
import torch.nn as nn
import copy
from meta.next_steps import NextSteps
from meta.next_steps import NextStepClassifier

from meta.conv2d import SelectiveConv2d

class Section(nn.Module):
    def __init__(self,name,layers,num_options_each_layer,is_encoder=lambda x:isinstance(x,(nn.Conv2d,SelectiveConv2d))) -> None:
        super().__init__()
        self.base_layers=[]
        # self.num_shared_layers=[]
        # self.shared_index=[]
        self.is_setup=False
        self.num_options_each_layer=num_options_each_layer
       
        self.name=name
        
        self.last_sections_encode_shapes=[]
        self.last_sections_steps=[]
        self.is_encoder=is_encoder
        self.step_classifier_encoder=None
        
       
        self.layers=[]
        self.features=None
        self.num_shared_layers=[]
        self.shared_index=[]
        self.extend_layers(layers)
  
    def save_network(self,path):
        #print(vars(self))
        data=copy.copy(vars(self))
        #print("is_encoder",self.is_encoder)
        del data["is_encoder"]
        #("is_encoder")
        
        torch.save(data,path)
    def load_network(self,path):
        states=torch.load(path)
        for key,value in states.items():
            #print(key)
            setattr(self,key,value)
       
        self.register_module("classifier",self.classifier)
        self.register_module("step_classifier_encoder",self.step_classifier_encoder)
        for i,layer in enumerate(self.layers):
            self.add_module(f"layer_{i}",layer)
           
                
                
    # def load_network(self,path):
    #     self.load_state_dict(torch.load(path))
        
    def append_layer(self,layer):
        self.base_layers.append(layer)
        self.num_shared_layers.append(0)
        self.shared_index.append(None)
       
    # def append_channelled_layer(self,layer):
    #     self.base_layers.append(layer)
      
    #     self.use_channels.append(True)
    def append_shared_layers(self,index,num_shared_layers):
        self.base_layers.append(self.base_layers[index])
        self.num_shared_layers.append(num_shared_layers)
        self.shared_index.append(index)
        
    def extend_layers(self,layers):
        for layer in layers:
            self.append_layer(layer)
    def extend_shared_layers(self,indices,num_shared_layers):
        for i in indices:
            self.append_shared_layers(i,num_shared_layers)
    def set_step_classifier_encoder(self,encoder):
        self.step_classifier_encoder=encoder
    def setup(self):
        
        if self.is_setup:
            return
        self.is_layer_with_params=[]
        self.layers=[]
        self.last_feature_index=None
        self.num_layers_with_params=0
        self.num_total_layers=len(self.base_layers)
        for i,layer in enumerate(self.base_layers):
            
            if self.is_encoder(layer):
                self.last_feature_index=i
                
            if len(list(layer.parameters()))>0: 
                
                    #print("append1",layer)
                layer=copy.deepcopy(layer)
                self.layers.append(layer)
                self.add_module(f"layer_{i}",layer)
                    #print("append2",layer)
                self.num_layers_with_params+=1
                self.is_layer_with_params.append(True)
            else:
                self.is_layer_with_params.append(False)
                layer=copy.deepcopy(layer)
                self.layers.append(layer)
                self.add_module(f"layer_{i}",layer)
                #print("append3",layer)
                
            
                
        
        self.is_setup=True 
        self.last_sections_encode_shapes.append((1,self.num_total_layers,self.num_options_each_layer))
        print(self.last_sections_encode_shapes)
        self.input_features_shape=[*max(self.last_sections_encode_shapes,key=math.prod)]
   
        self.last_sections_steps.append(self.num_total_layers)
        self.last_sections_step=max(self.last_sections_steps)
   


        self.classifier=NextStepClassifier(self.input_features_shape,self.num_total_layers,self.num_options_each_layer,self.step_classifier_encoder) 
        #self.add_module ("next_steps_classifier",self.classifier)
        del self.base_layers
        self.is_setup=True

    def dummy_forward(self,data):
        #print(self.base_layers)
        #print("data",data.shape)
        self.features=None
        for i,layer in enumerate(self.base_layers):
            if len(list(layer.parameters()))==0:
                data=layer(data)
            else:
                
                index=torch.zeros(1,dtype=torch.long)
                
                #print("index",index.shape)  
                #print("layer",layer)
                data=layer(data,index)
                
            if self.is_encoder(layer):
                    self.features=data.clone().detach()
                    print("self.features.shape",self.features.shape)
        # print("features",self.features.shape)
            #print("data",data.shape)
            
        return data
        
    def forward(self,data:torch.Tensor,features:torch.Tensor,previous_steps:NextSteps,task):
        
        assert self.is_setup
        
        
        next_steps=self.classifier.forward(features,previous_steps,task)
        #print("next_steps",next_steps.indices)
        self.next_steps=next_steps
      
        layer_with_params_index=0
        #print("len",len(self.is_layer_with_params),self.num_total_layers)
        for i in range(self.num_total_layers):
            
            if self.is_layer_with_params[i]:
                indices=next_steps.indices[:,layer_with_params_index]
                data=self.layers[i](data,indices)
            else:
                data=self.layers[i](data)
            if i==self.last_feature_index:
                self.features=data.clone().detach()
            layer_with_params_index+=1
               
         
        return data
    @staticmethod
    def get_steps_classifier_loss(loss,next_steps:NextSteps):
        #confidence=torch.sum(next_steps.probability)
        return next_steps.confidence*(loss.detach())

            
        
        
        
        #self.layers=[]
        

    
    
    