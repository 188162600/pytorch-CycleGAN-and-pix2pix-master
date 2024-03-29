
import torch
import os
from meta.section import Section
from meta.next_steps import NextSteps
class Task:
    def __init__(self,input_shape,device,sections:list) -> None:
       
        #print(input_shape)
        self.device=device
        self.dummy_input=torch.zeros(1,*input_shape)
        self.dummy_features=None
        self.last_section_steps=0
        
       
        self.optimizers=[]
        self.steps_classifier_optimizers=[]
        self.sections=[]
        self.extend_sections(sections)  
        self.hidden_long_term=dict()
        
    def append_section(self,section:Section):
        section.to(self.device)
        self.sections.append(section)
        
        section.last_sections_encode_size.append(self.dummy_features.numel() if self.dummy_features is not None else 0)
        section.last_sections_steps.append(self.last_section_steps)
        self.dummy_input=section.dummy_forward(self.dummy_input)
        self.last_section_steps=section.num_total_layers
        if self.dummy_features is not None:
            self.dummy_features=section.features
       
        self.optimizers.append(None)
        self.steps_classifier_optimizers.append(None)

        
        
    def extend_sections(self,sections):
        for section in sections:
            self.append_section(section)
    def setup(self):
       
        for section in self.sections:
            section.setup()
        
            
    def save_network(self,path,saved_sections:set):
        for section in self.sections:
            if not section.name in saved_sections:
                saved_sections.append(section.name)
                torch.save(section,os.path.join(path,f"section_{section.name}"))
    
    def load_network(self,path,sections:list,loaded_sections:set):
        for section in sections:
            if not section in loaded_sections:
                section=torch.load(os.path.join(path,f"section_{section.name}"))
                loaded_sections.append(section.name)
               
            
    def eval(self):
        for section in self.sections:
            section.eval()
    def set_criterion(self,criterion,*args):
        self.criterion=criterion
        self.criterion_args=args
    
        
        
    def forward(self,data,track_loss):
        
        
     
       
        # if previous is None:
        #     hidden_short_term = torch.zeros(batch, self.num_step_classes * self.num_next_steps, device=features.device)
        # else:
        #     hidden_short_term = previous.tensor
            
        if track_loss:
            self.losses=[]
        self.previous_steps=[]
        last_features=torch.zeros(1,self.sections[0].input_features_size,device=self.device)
        
        last_section_steps=None
        
        for section in self.sections:
            
            if last_section_steps is not None:
                last_section_steps.tensor=last_section_steps.tensor.detach()
            data=section.forward(data.detach(),last_features.detach() if last_features is not None else None ,last_section_steps,self)
            if section.features is not None:
                last_features=section.features[0].view(1,-1)
            last_section_steps=section.next_steps
            if track_loss:
                self.losses.append(self.criterion(data,*self.criterion_args))
            self.previous_steps.append(last_section_steps)
        return data
    def get_losses(self):
        return self.losses
    def set_optimizer(self,index,optimizer,steps_classifier_optimizer):
        self.optimizers[index]=optimizer
        self.steps_classifier_optimizers[index]=steps_classifier_optimizer
    def optimize_layers(self,losses):
        for i in range(len(self.sections)):
            self.optimizers[i].zero_grad()
            losses[i].backward()
            self.optimizers[i].step()
    def optimize_steps_classifier(self,losses,previous_steps):
        for i in range(len(self.sections)):
            self.steps_classifier_optimizers[i].zero_grad()
            classifier_loss=self.sections[i].get_steps_classifier_loss(losses[i],previous_steps[i])
            classifier_loss.backward()
            self.steps_classifier_optimizers[i].step()
            
    def optimize_parameters(self,losses,previous_steps):
        for i in range(len(self.sections)):
            self.optimizers[i].zero_grad()
            loss=losses[i]
            loss.backward()
            
            self.optimizers[i].step()
        
            self.steps_classifier_optimizers[i].zero_grad()
            classifier_loss=self.sections[i].get_steps_classifier_loss(loss,previous_steps[i])
            classifier_loss.backward()
            self.steps_classifier_optimizers[i].step()
            
            