import cv2
import torch
import torchvision
from pathlib import Path
from torch import nn
from torchvision import transforms
from torchvision.transforms import ToTensor
from torch.nn.modules.pooling import MaxPool2d
from PIL import Image
from collections import Counter    #for vote confidence
import time as timer



#--------------------------------------------------------------------------------------------------------------------------
vid=cv2.VideoCapture(0)
model_path=Path("Model1best_83_95.pth")  #u CAN change it as u want
class_names=["no_gesture","palm","one","peace","thumb_index","ok","three"]

frame_width=int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height=int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
resolution=(frame_width,frame_height)   #its 640,480 for experimenting
print(resolution)
x1, y1 = 220, 90
x2, y2 = 470, 390
print(torch.__version__)
#---------------------------------------------------------------------------------------------------------------------------

transform=transforms.Compose([
   transforms.Resize((128,128)),
   transforms.ToTensor()
])




#----------------------------------------------------------------------------------------------------------------------------
#So,to eval the input image using the trained model, i gotta pass in the architecture my model trained in .
class Hand_gesture(nn.Module):
  def __init__(self,input_features,hidden_units,output_features):
    super().__init__()  #to make sure parent class creates necessary variables
    self.conv_layer_1=nn.Sequential(
        nn.Conv2d(in_channels=input_features,out_channels=hidden_units,kernel_size=3,padding=1,stride=1),
        nn.BatchNorm2d(hidden_units),
        nn.ReLU(),
        nn.Conv2d(in_channels=hidden_units,out_channels=hidden_units,kernel_size=3,padding=1,stride=1),
        nn.BatchNorm2d(hidden_units),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2,stride=2)
    )
    self.conv_layer_2=nn.Sequential(
        nn.Conv2d(in_channels=hidden_units,out_channels=hidden_units,kernel_size=3,padding=1,stride=1),
        nn.BatchNorm2d(hidden_units),
        nn.ReLU(),
        nn.Conv2d(in_channels=hidden_units,out_channels=hidden_units,kernel_size=3,padding=1,stride=1),
        nn.BatchNorm2d(hidden_units),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2,stride=2)
    )
    self.conv_layer_3=nn.Sequential(
        nn.Conv2d(in_channels=hidden_units,out_channels=hidden_units,kernel_size=3,padding=1,stride=1),
        nn.BatchNorm2d(hidden_units),
        nn.ReLU(),
        nn.Conv2d(in_channels=hidden_units,out_channels=hidden_units,kernel_size=3,padding=1,stride=1),
        nn.BatchNorm2d(hidden_units),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2,stride=2)
    )
    self.Linear_layer=nn.Sequential(
        nn.Flatten(),
        nn.Linear(
            in_features=hidden_units*256,
            out_features=output_features
        )

    )
  def forward(self,X):
    x=self.conv_layer_1(X)
    x=self.conv_layer_2(x)
    x=self.conv_layer_3(x)
    x=self.Linear_layer(x)
    return x

  
#----------------------------------------------------------------------------------------------------------------
#loading trained_model
model_1=Hand_gesture(input_features=3,hidden_units=32,output_features=7)

model_1.load_state_dict(torch.load( (model_path) , map_location=torch.device("cpu") ))
model_1.eval()
#-------------------------------------------------------------------------------------------------------------------











#-----------------------------------------------------------------------------------------------------
predictions_5=[]
guessed_dict={}

key_press=False  #for countdown 
while True:
    condition,frame=vid.read()
    #frame_gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #frame_gray_blurred=cv2.GaussianBlur(frame_gray,(3,3),1.5)
    #clahe_instance=cv2.createCLAHE(clipLimit=2,tileGridSize=(7,7))
    #frame_gray_clahed=clahe_instance.apply(frame_gray_blurred)

    if guessed_dict:
       for key,value in guessed_dict.items():
          cv2.putText(frame,("Guessed_gesture:"+str(key)),(400,40),cv2.FONT_HERSHEY_COMPLEX,0.5,(0, 165, 255),1)
          cv2.putText(frame,(("Confidence:"+str(value))),(400,80),cv2.FONT_HERSHEY_COMPLEX,0.5,(0, 165, 255),1)
          

    

    

    if not condition:
        break

    cropped = frame[y1:y2, x1:x2]
    cropped_RGB=cv2.cvtColor(cropped,cv2.COLOR_BGR2RGB)


    

    
    
    frame_img=Image.fromarray(cropped_RGB)
    frame_tensor=transform(frame_img).unsqueeze(dim=0)

   


    if key_press:
       end=timer.time()
       timer_count=round( (3.0 - (end-start)),2)
       cv2.putText(frame,("Timer:"+str(timer_count)),(10,40),cv2.FONT_HERSHEY_COMPLEX,1,(0, 165, 255),1)

       with torch.inference_mode():
        logits=model_1(frame_tensor)

        predictions=logits.argmax(dim=1)
        ##
        '''

        pred = logits.argmax(1)
        probs = torch.softmax(logits, dim=1)[0]

        for name, p in zip(class_names, probs):
            print(f"{name:12} {p:.3f}")
        #''' #use this only when u want to check whts the probability for other classes
       

        predictions_5.append(predictions.item())
        if len(predictions_5)>70:
           predictions_5.pop(0)

        if len(predictions_5)==70:
           counts = Counter(predictions_5)
           predicted_class, votes = counts.most_common(1)[0]
           confidence=(votes/len(predictions_5)) *100
           
           if (confidence > 70) and print_count:
             print_count=False
             guessed_dict[class_names[predicted_class]]=confidence
             
             
             print(class_names[predicted_class], "confidence:",confidence)

             

        if (end-start)>=3:
           key_press=False
       



    
       

    
    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,250,0),2)
    cv2.imshow("w1",frame)
    cv2.imshow("w2",cropped)

    key=(chr(cv2.waitKey(1) & 0xFF) ).lower() #the reason why i didnt do this seperately for Z and Q is that its a 1 time consumption key.

    if key == 'q':
       break

    if (not key_press) and (key == 'z'):
       predictions_5.clear()
       guessed_dict.clear()
       start=timer.time()
       key_press=True
       print_count=True

vid.release()
cv2.destroyAllWindows()
