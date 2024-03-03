import math
import typing
import torch
import torch.nn as nn
import copy
from meta.next_steps import NextSteps
from meta.next_steps import NextStepClassifier
from meta.conv2d import Conv2d
import itertools

class Section(nn.Module):
    def __init__(self,name,is_encoder=lambda x:isinstance(x,Conv2d)) -> None:
        super().__init__()
        self.last_feature_index=None
        self.num_layers_with_params=None
        #self.num_options_each_layer=None
        self.num_total_layers=None
        self.layers=[]
        self.base_layers=[]
        self.classifier=None
        self.output_channels=[]

        self.num_options_each_layer=[]
      
        self.is_setup=False
        
        self.name=name
        
        self.last_sections_encode_shapes=[]
        self.last_sections_steps=[]
        self.is_encoder=is_encoder
        self.step_classifier_encoder=None
      
        self.is_layer_with_params=[]
        #self.extend_layers(layers)
        
        self.features=None
    def save_network(self,path):
        state={
            "num_layers_with_params":self.num_layers_with_params,
            "num_layers_with_params":self.num_layers_with_params,
            "last_feature_index":self.last_feature_index,
            "num_total_layers":self.num_total_layers,
            "layers":self.layers,
            "base_layers":self.base_layers,
            "classifier":self.classifier,
            "name":self.name,
            "output_channels":self.output_channels,
            "num_options_each_layer":self.num_options_each_layer,
            "is_setup":self.is_setup,
            "last_sections_encode_shapes":self.last_sections_encode_shapes,
            "last_sections_steps":self.last_sections_steps,
            "is_layer_with_params":self.is_layer_with_params,
            
        }
        torch.save(state,path)
    def load_network(self,path):
        state=torch.load(path)
        self.last_feature_index=state["last_feature_index"]
        self.num_layers_with_params=state["num_layers_with_params"]
        self.num_total_layers=state["num_total_layers"]
        self.layers=state["layers"]
        self.base_layers=state["base_layers"]
        self.classifier=state["classifier"]
        self.name=state["name"]
        self.output_channels=state["output_channels"]
        self.num_options_each_layer=state["num_options_each_layer"]
        self.is_setup=state["is_setup"]
        self.last_sections_encode_shapes=state["last_sections_encode_shapes"]
        self.last_sections_steps=state["last_sections_steps"]
        self.is_layer_with_params=state["is_layer_with_params"]
        self._register_models()
        
    def append_layer(self,layer:nn.Module,num_options_each_layer:int,output_channel:int=None):
        self.base_layers.append(layer)
        self.output_channels.append(output_channel)
        self.num_options_each_layer.append(num_options_each_layer)
        
        
 
       
        
        
    # def extend_layers(self,layers):
    #     for layer in layers:
    #         self.append_layer(layer)
    # def extend_shared_layers(self,indices,num_shared_layers):
    #     for i in indices:
    #         self.append_shared_layers(i,num_shared_layers)
    def set_step_classifier_encoder(self,encoder):
        self.step_classifier_encoder=encoder
    def _register_models(self):
        for i in range(len(self.layers)):
            if isinstance(self.layers[i],list):
                for j in range(len(self.layers[i])):
                    self.register_module(f"{i},{j}",self.layers[i][j])
            self.register_module(f"{i}",self.base_layers[i])
    def setup(self):
        if self.is_setup:
            return 

        self.layers=[]
        self.is_layer_with_params=[]
        self.num_layers_with_params=0
        self.num_total_layers=0
        self.last_feature_index=None
        for i,layer in enumerate(self.base_layers):
           
            if len(list(layer.parameters()))>0:
                self.num_layers_with_params+=1
                self.is_layer_with_params.append(True)
                num_options_each_layer=self.num_options_each_layer[i]
                out_channels=self.output_channels[i]
                if out_channels is not None:
                    layer=copy.deepcopy(layer)
                    self.layers.append(layer)
                    self.register_module(f'{i}',layer)
                else:
                    layers=[]
                    for j in  range(num_options_each_layer):
                        layer=copy.deepcopy(layer)
                        layers.append(layer)
                        self.register_module(f"{i},{j}",layer)
                    #layers=[copy.deepcopy(layer) for j in range(num_options_each_layer)]
                    self.layers.append(layers)
            else:
                self.is_layer_with_params.append(False) 
                self.layers.append(layer)
                
            #print(self.layers)
        
               
            
          

            if self.is_encoder(layer):
                self.last_feature_index=self.num_total_layers
            
            self.num_total_layers+=1    
        #print("end")
         
        #self.last_sections_encode_shapes.append((1,self.num_total_layers,self.num_options_each_layer))
       
        #print("self.last_sections_encode_shapes",self.last_sections_encode_shapes)
        self.input_features_shape=[*max(self.last_sections_encode_shapes,key=math.prod)]
        
        self.last_sections_steps.append(self.num_total_layers)
        self.last_sections_step=max(self.last_sections_steps)
      #  print("max",*filter( lambda x: x is not None,self.num_options_each_layer)),*filter(lambda x:x is not None,max(self.output_channels))
        #print("max")
        #print(self.num_options_each_layer,self.output_channels)
        classes=max(filter( lambda x: x is not None,(* self.num_options_each_layer,* self.output_channels)))
        #print("self.input_features_shape,self.num_total_layers,classes,self.step_classifier_encoder",self.input_features_shape,self.num_total_layers,classes,self.step_classifier_encoder)
        self.classifier=NextStepClassifier(self.input_features_shape,self.num_total_layers,classes,self.step_classifier_encoder)
        #print("end")
        #self.add_module ("next_steps_classifier",self.classifier)
        self.is_setup=True
    def forward(self,data:torch.Tensor,features:torch.Tensor,previous_steps:NextSteps,task):
       
        assert self.is_setup
        # if previous_steps is not None:
        #     previous_steps.tensor=previous_steps.tensor.clone()
        next_steps=self.classifier.forward(features.clone(),previous_steps,task)
        
        self.next_steps=next_steps
        
        layer_with_params_index=0
        self.features=None
        for i in range(self.num_total_layers):
            
            out_channels=self.output_channels[i]
            if self.is_layer_with_params[i]:
               
                out_channels=self.output_channels[i]
                if out_channels is not None:
                    indices=next_steps.next_indices(layer_with_params_index,self.num_options_each_layer[i],out_channels)
                    #indices[layer_with_params_index][0:out_channels]
                    #print("indices",indices.shape)
                    data=self.layers[i](data,indices[:,0])
                else:
                    index=next_steps.next_indices(layer_with_params_index,self.num_options_each_layer[i],1)
                    
                    data=self.layers[i][index](data)    
               
            else:
                data=self.base_layers[i](data)
           
           
            if i==self.last_feature_index:
                self.features=data.detach()
                
            #self.last_feature_index+=1
        return data
    def dummy_forward(self,data):
        #print(self.base_layers)
        #print("data",data.shape)
       
        
        layer_with_params_index=0
        self.features=None
        for i in range(len(self.base_layers)):
            
            out_channels=self.output_channels[i]
            if len(list(self.base_layers[i].parameters()))>0:
               
                out_channels=self.output_channels[i]
                if out_channels is not None:
                    indices=torch.zeros(out_channels,dtype=torch.long)
                    #print("index",indices)
                    data=self.base_layers[i](data,indices)
                else:
                    index=0

                    data=self.base_layers[i](data)
                
                
               
            else:
                data=self.base_layers[i](data)
           
           
            if self.is_encoder(self.base_layers[i]):
                self.features=data.detach()
                
                self.last_feature_index=i
        return data
        

    def get_steps_classifier_loss(self,loss,next_steps:NextSteps):
        #for i in  next_steps.probablity:
            #print("i",i.shape)
        confidence=torch.sum(torch.cat(next_steps.probability))
        #print("confience",confidence)
        # confidence=0
        #
        # for probability in next_steps.probability:
        #     #print(probability,"prob")
        #     #print(torch.mean(probability).shape,"torch.mean(probability).shape")
        #     confidence=torch.mean(probability)+confidence
        return confidence*(loss.detach())

            
        
        
        
        #self.layers=[]
        

    
    
    