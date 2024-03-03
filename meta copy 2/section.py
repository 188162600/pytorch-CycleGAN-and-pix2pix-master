import math
import typing
import torch
import torch.nn as nn
import copy
from meta.next_steps import NextSteps
from meta.next_steps import NextStepClassifier



class Section(nn.Module):
    def __init__(self,name,num_options_each_layer,num_pick,is_encoder=lambda x:isinstance(x,nn.Conv2d)) -> None:
        super().__init__()
        self.base_layers=[]
        # self.num_shared_layers=[]
        # self.shared_index=[]
        self.is_setup=False
        self.num_options_each_layer=num_options_each_layer
        self.num_pick=num_pick
        self.name=name
        
        self.last_sections_encode_shapes=[]
        self.last_sections_steps=[]
        self.is_encoder=is_encoder
        self.step_classifier_encoder=None
        
        self.use_channels=[]
        
        self.features=None
  
    def save_network(self,path):
        print(vars(self))
        data=copy.copy(vars(self))
        #print("is_encoder",self.is_encoder)
        del data["is_encoder"]
        #("is_encoder")
        
        torch.save(data,path)
    def load_network(self,path):
        states=torch.load(path)
        for key,value in states.items():
            print(key)
            setattr(self,key,value)
       
        self.register_module("classifier",self.classifier)
        self.register_module("step_classifier_encoder",self.step_classifier_encoder)
        for i,layer in enumerate(self.layers):
            if isinstance(layer,list):
                for j,l in enumerate(layer):
                    self.register_module(f"layer_{i}_{j}",l)
            else:
                self.register_module(f"layer_{i}",layer)
           
                
                
    # def load_network(self,path):
    #     self.load_state_dict(torch.load(path))
        
    def append_layer(self,layer,use_channels):
        self.base_layers.append(layer)
        # self.num_shared_layers.append(0)
        # self.shared_index.append(None)
        self.use_channels.append(use_channels)
    
    # def append_shared_layers(self,index,num_shared_layers):
    #     self.base_layers.append(self.base_layers[index])
    #     self.num_shared_layers.append(num_shared_layers)
    #     self.shared_index.append(index)
        
    # def extend_layers(self,layers):
    #     for layer in layers:
    #         self.append_layer(layer)
    # def extend_shared_layers(self,indices,num_shared_layers):
    #     for i in indices:
    #         self.append_shared_layers(i,num_shared_layers)
    def set_step_classifier_encoder(self,encoder):
        self.step_classifier_encoder=encoder
    def setup(self):
        self.is_layer_with_params=[]
        if self.is_setup:
            return
        self.layers=[]
        self.last_feature_index=None
        self.num_layers_with_params=0
        self.num_total_layers=len(self.base_layers)
        for i,layer in enumerate(self.base_layers):
            
            if self.is_encoder(layer):
                self.last_feature_index=i
                
            if len(list(layer.parameters()))>0: 
                if self.use_channels[i]:
                    #print("append1",layer)
                    layer=copy.deepcopy(layer)
                    self.layers.append(layer)
                    self.register_module(f"layer_{i}",layer)
                else:
                   
                    layers=[]
                    for j in range(self.num_options_each_layer):
                        layer=copy.deepcopy(layer)
                        layers.append(layer)
                        self.register_module(f"layer_{i}_{j}",layer)
                    self.layers.append(layers)
                    #print("append2",layer)
                self.num_layers_with_params+=1
                self.is_layer_with_params.append(True)
            else:
                self.is_layer_with_params.append(False)
                layer=copy.deepcopy(layer)
                self.layers.append(layer)
                self.register_module(f"layer_{i}",layer)
                #print("append3",layer)
                
            
                
        
        self.is_setup=True 
        self.last_sections_encode_shapes.append((1,self.num_total_layers,self.num_options_each_layer))
        self.input_features_shape=[*max(self.last_sections_encode_shapes,key=math.prod)]
   
        self.last_sections_steps.append(self.num_total_layers)
        self.last_sections_step=max(self.last_sections_steps)
   


        self.classifier=NextStepClassifier(self.input_features_shape,self.num_total_layers,self.num_options_each_layer,self.num_pick,self.step_classifier_encoder)
        #self.add_module ("next_steps_classifier",self.classifier)
        self.is_setup=True
    def dummy_forward(self,data):
        #print(self.base_layers)
        #print("data",data.shape)
        self.features=None
        for i,layer in enumerate(self.base_layers):
            if len(list(layer.parameters()))==0:
                data=layer(data)
            else:
                if self.use_channels[i]:
                    index=torch.zeros(self.num_pick,dtype=torch.long)
                    
                    print("index",index.shape)  
                    print("layer",layer)
                    data=layer(data,index)
                else:
                    data=layer(data)
            if self.is_encoder(layer):
                    self.features=data.clone().detach()
                    print("self.features.shape",self.features.shape)
        # print("features",self.features.shape)
            print("data",data.shape)
            
        return data
        
    def forward(self,data:torch.Tensor,features:torch.Tensor,previous_steps:NextSteps,task):
        
        assert self.is_setup
        
        
        next_steps=self.classifier.forward(features,previous_steps,task)
       
        self.next_steps=next_steps
      
        layer_with_params_index=0
        for i in range(self.num_total_layers):
            
            if self.is_layer_with_params[i]:
               
                assert next_steps.indices.size(0)==1 and "batch size must be 1" 
                index=next_steps.indices[0]
                
                index=index[:,layer_with_params_index]
                if self.use_channels[i]:
                # print("next_steps.indices",next_steps.indices.shape)
                
                
               
                # print("index",index.shape,index)
                
                    data=self.layers[i](data,index)
                else:
                    data=self.layers[i][index[0]](data)
            else:
                data=self.layers[i](data)
            if i==self.last_feature_index:
                self.features=data.clone().detach()
               
         
        return data
    @staticmethod
    def get_steps_classifier_loss(loss,next_steps:NextSteps):
        #confidence=torch.sum(next_steps.probability)
        return next_steps.confidence*(loss.detach())

            
        
        
        
        #self.layers=[]
        

    
    
    