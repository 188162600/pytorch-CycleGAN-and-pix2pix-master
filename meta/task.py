
import torch
import os
from meta.section import Section
from meta.next_steps import NextSteps
# from torchviz import make_dot


class Task:
    def __init__(self,input_shape,device,sections:list,separate_classifier_backward) -> None:
       
        #print(input_shape)
        self.device=device
        self.dummy_input=torch.zeros(1,*input_shape)
        self.dummy_features=self.dummy_input
        
       
        self.optimizers=[]
        self.steps_classifier_optimizers=[]
        self.sections=[]
        self.extend_sections(sections)  
        self.hidden_long_term=dict()
        self.separate_classifier_backward=separate_classifier_backward
        
    def append_section(self,section:Section):
        
        #section.to(self.device)
        self.sections.append(section)
        #print("dummy",self.dummy_features.numel() if self.dummy_features is not None else 0)
        if self.dummy_features is not None:
            section.last_sections_encode_shapes.append(self.dummy_features.shape[1:])
        else:
            section.last_sections_encode_shapes.append(self.dummy_input.shape[1:])
        self.dummy_input,self.dummy_features=section.dummy_forward(self.dummy_input)
        if self.dummy_features is  None:
            #print("dummy_features is None")
            self.dummy_features=self.dummy_input
        #print("dummy_features",self.dummy_features.shape,"dummy_input",self.dummy_input.shape)
        self.optimizers.append(None)
        self.steps_classifier_optimizers.append(None)
        
        
    def extend_sections(self,sections):
        for section in sections:
            
            self.append_section(section)
    def setup(self):
        #print("task setup")
        for section in self.sections:
            print(section.classifier_encoder.conv1.weight.device)
            section.setup()
         
            section.layers.to(self.device)
        
            section.steps_record.to(self.device)
            section.classifier_encoder.to(self.device)
        
         
            section.classifier.to(self.device)
            # for layer in section.layers:
            #     #for layer in layers:
            #     if isinstance(layer,list):
            #         for l in layer:
            #             l.to(self.device)
            #     else:
            #         layer.to(self.device)
    def update(self,updated_sections):
        for i in range(len(self.sections)):
            if self.sections[i].name in updated_sections:
                self.sections[i].update()
                
    def save_network(self,path,epoch,saved_sections:set):
        for section in self.sections:
            if not section.name in saved_sections:
                saved_sections.update(section.name)
                section.save_network(os.path.join(path,f"{epoch}_{section.name}"))
                #torch.save(section.state_dict(),os.path.join(path,f"{epoch}_section_{section.name}.pth"))
                #section.save_network(os.path.join(path,f"{epoch}_section_{section.name}.pth"))
          
    
    def load_network(self,path,epoch,sections:list,loaded_sections:set):
        for section in sections:
            if not section.name in loaded_sections:
                loaded_sections.update(section.name)
                section.load_network(os.path.join(path,f"{epoch}_{section.name}"))
                #section.load_state_dict(torch.load(os.path.join(path,f"{epoch}_section_{section.name}.pth")))
                #section.load_network(os.path.join(path,f"{epoch}_section_{section.name}.pth"))
                
                
            
               
            
    def eval(self):
        for section in self.sections:
            section.eval()

    
    def get_results(self):
        return self.results
   
    def forward(self,data):
        # print("task vars",vars(self))
        # print("----------------------------------")
        #
     
       
        # if previous is None:
        #     hidden_short_term = torch.zeros(batch, self.num_step_classes * self.num_next_steps, device=features.device)
        # else:
        #     hidden_short_term = previous.tensor
      
        self.previous_steps=[]
        #self.results=[]
        # last_features=torch.zeros(1,self.sections[0].input_features_size,device=self.device)
        last_features=data
        #print("last_features",last_features.device)
        last_section_steps=None
        #print("len(self.sections)",len(self.sections))
        for section in self.sections:

           # print("section.input_features_size",section.input_features_size)
            #data=data.detach()
            if self.separate_classifier_backward:
                # if last_section_steps is not None:
                #     last_section_steps.tensor=last_section_steps.tensor.clone().detach()
                
                if last_features is not None:
                    last_features=last_features.clone().detach()
                
            #print("task forward last_features",last_features.device)
            data,last_features,last_section_steps=section.forward(data,  last_features  ,last_section_steps,self )
            if last_features is None:
                last_features=data
           
        

            #self.results.append(data)
            #print("len(self.sections),len(self.results)",len(self.sections),len(self.results))
            self.previous_steps.append(last_section_steps)
        return data
  
    def set_optimizer(self,index,optimizer,steps_classifier_optimizer):
        self.optimizers[index]=optimizer
        self.steps_classifier_optimizers[index]=steps_classifier_optimizer
    @staticmethod
    def _backward(loss:torch.Tensor):
        if len(loss.shape)==0:
            loss.backward()
        else:
            #print("loss",loss.shape)
            assert len(loss.shape)==1
            loss.mean().backward()
            
    def optimize_layers(self,loss):
        for i in range(len(self.sections)):
            self.optimizers[i].zero_grad()
            if not  self.separate_classifier_backward:
                self.steps_classifier_optimizers[i].zero_grad()
        self._backward(loss)
        #loss.backward()
        for i in range(len(self.sections)):
            self.optimizers[i].step()
            if not  self.separate_classifier_backward:
                self.steps_classifier_optimizers[i].step()
    # def optimize_steps_classifier(self,loss,previous_steps):
        
    #     for i in range(len(self.sections)):
    #         self.steps_classifier_optimizers[i].zero_grad()
    #         classifier_loss=self.sections[i].get_steps_classifier_loss(loss.detach(),previous_steps[i])
    #         classifier_loss.backward()
    #         self.steps_classifier_optimizers[i].step()
    def optimize_steps_classifiers(self,loss,previous_steps):
        if  self.separate_classifier_backward:
             
        
            for i in range(len(self.sections)):
                self.steps_classifier_optimizers[i].zero_grad()
                #print(previous_steps)
                confidence=0
                for steps in previous_steps:
                #for i in range(len(self.sections)):
                    confidence=confidence+steps[i].confidence
                #print("loss",loss.shape)
                loss=loss.detach()*confidence
                #print("loss",loss.shape)
                self._backward(loss)
                #(loss.detach()*confidence).backward()
            #for i in range(len(self.sections)):
                self.steps_classifier_optimizers[i].step()
    # def optimize_steps_classifier2(self,loss,previous_steps):
    #     if self.separate_classifier_backward:
    #         for i in range(len(self.sections)):
    #             self.steps_classifier_optimizers[i].zero_grad()
    #             classifier_loss=self.sections[i].get_steps_classifier_loss(loss.detach(),previous_steps[i])
    #             self._backward( classifier_loss)
    #             self.steps_classifier_optimizers[i].step()
    def track(self,loss,previous_steps):
        for i in range(len(self.sections)):
            #print("track",i)
            self.sections[i].track(loss,previous_steps[i])
    def optimize_steps_classifier2(self,loss,previous_steps):
        if self.separate_classifier_backward:
            for i in range(len(self.sections)):
                self.steps_classifier_optimizers[i].zero_grad()
            #print("previous_steps[-1].confidence.shape,loss.detach().shape",previous_steps[-1].confidence.shape,loss.detach().shape)
            classifier_loss=previous_steps[-1].confidence*(loss.detach())
                
            self._backward( classifier_loss)
            for i in range(len(self.sections)):
                self.steps_classifier_optimizers[i].step()
   
    # def optimize_parameters(self,losses,previous_steps):
    #     for i in range(len(self.sections)):
    #         #print("optimized",i)
    #         self.optimizers[i].zero_grad()
    #         loss=losses[i]

    #         loss.backward()
            
    #         self.optimizers[i].step()
        
    #         self.steps_classifier_optimizers[i].zero_grad()
    #         classifier_loss=self.sections[i].get_steps_classifier_loss(loss,previous_steps[i])
    #         classifier_loss.backward()
    #         self.steps_classifier_optimizers[i].step()
            
            