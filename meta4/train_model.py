from meta.base_model import BaseModel   
class TrainModel(BaseModel):
    def __init__(self,input_shape,device,sections:list) -> None:
        super().__init__(input_shape,device,sections)