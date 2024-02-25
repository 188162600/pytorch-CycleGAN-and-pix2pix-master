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
        self.feature_adjustment=None
        self.features=None
    def append_layer(self,layer):
        self.base_layers.append(layer)
    def append_layers(self,layers):
        self.base_layers.extend(layers)
    def set_feature_adjustment(self,adjustment):
        self.feature_adjustment=adjustment
    def setup(self):
        if self.is_setup:
            return 

        self.layers=[]
    
        self.num_total_layers=0
        self.last_feature_index=None
        for layer in self.base_layers:
            self.layers.append([copy.deepcopy(layer) for i in range(self.num_options_each_layer)])
            for j,layer in  enumerate(self.layers[-1]):
                self.register_module(f"{self.num_total_layers},{j}",layer)

            if self.is_encoder(layer):
                self.last_feature_index=self.num_total_layers
            self.num_total_layers+=1
        self.last_sections_encode_shapes.append((1,self.num_total_layers*self.num_options_each_layer))
        self.input_features_shape=[*max(self.last_sections_encode_shapes,key=math.prod)]
        feature_adjustments=[]
        channels=self.input_features_shape[0]
        dummy_feature=torch.zeros(1,*self.input_features_shape)
        while dummy_feature.numel()>25600:
            conv=torch.nn.Conv2d(channels,channels,7,3,3)
            dummy_feature=conv(dummy_feature)
            feature_adjustments.append(conv)
        self.feature_adjustment=torch.nn.Sequential(*feature_adjustments)
        self.input_features_size=dummy_feature.numel()
        self.last_sections_steps.append(self.num_total_layers)
        self.last_sections_step=max(self.last_sections_steps)
        print("input_features_size",self.input_features_size)



        self.classifier=NextStepClassifier(self.input_features_size,self.num_total_layers,self.num_options_each_layer)
        #self.add_module ("next_steps_classifier",self.classifier)
        self.is_setup=True
    def dummy_forward(self,data):
        print(self.base_layers)
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
        
    def forward(self,data:torch.Tensor,features:torch.Tensor,previopuus_steps:NextSteps,task):
        assert self.is_setup
        #print(self.classifier)
        
        # batch=features.size(0)
        
        # features=features[0].view(batch,-1)
        #print("features",features.shape,self.input_features_size)
        print("features",features.shape,self.input_features_size,self.feature_adjustment)
        batch=features.size(0)
        features=self.feature_adjustment(features)
        features= features.view(batch,-1)
        if features[0:].numel()!=self.input_features_size:
            #print("feature shape",features.shape,features.size(1))
            features=nn.functional.pad(features,(0,0,0,self.input_features_size-features.size(1)))
            #print("feature shape after",features.shape)


        
      
        #print("features",features.shape)
       
        next_steps=self.classifier.forward(features,previous_steps,task)
        #print("after forward",next_steps.tensor.shape)
        self.next_steps=next_steps
        print("len(self.layers),self.num_total_layers",len(self.layers),self.num_total_layers)
        for i in range(self.num_total_layers):
            
                
            index=next_steps.indices[i]
            print(index)
            #print(next_steps.indices)
            #print(index)

            layer=self.layers[i][index]
            data=layer(data)
            if i==self.last_feature_index:
                with torch.no_grad():
                    self.features=data
               
        return data
    def get_steps_classifier_loss(self,loss,next_steps:NextSteps):
        confidence=torch.sum(next_steps.probability)
        return confidence*(loss.detach())

            
        
        
        
        #self.layers=[]
        

    
    
    