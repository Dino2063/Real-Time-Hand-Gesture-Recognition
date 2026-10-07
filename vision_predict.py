import torch
from torch import nn
from torchvision import transforms
import cv2
from collections import Counter
from pathlib import Path
from PIL import Image
import time as timer



def initialize_model(model_path,device):
    class Hand_gesture_recog(nn.Module):
        def __init__(self,in_features,hidden_conv_units,out_features,hidden_linear_units):
            super().__init__()
            self.conv_layer_1=nn.Sequential(
                nn.Conv2d(in_channels=in_features,out_channels=hidden_conv_units,kernel_size=(5,5),padding=2,stride=1),
                nn.BatchNorm2d(hidden_conv_units),
                nn.ReLU(),
                nn.Conv2d(in_channels=hidden_conv_units,out_channels=48,kernel_size=(5,5),padding=1,stride=1),
                nn.BatchNorm2d(48),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2,2),stride=2)
            )
            self.conv_layer_2=nn.Sequential(
                nn.Conv2d(in_channels=48,out_channels=48,kernel_size=(5,5),padding=1,stride=1),
                nn.BatchNorm2d(48),
                nn.ReLU(),
                nn.Conv2d(in_channels=48,out_channels=64,kernel_size=(5,5),padding=1,stride=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2,2),stride=2)
            )
            self.conv_layer_3=nn.Sequential(
                nn.Conv2d(in_channels=64,out_channels=64,kernel_size=(5,5),padding=1,stride=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(in_channels=64,out_channels=80,kernel_size=(5,5),padding=1,stride=1),
                nn.BatchNorm2d(80),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2,2),stride=2)
            )

            self.conv_layer_4=nn.Sequential(
                nn.Conv2d(in_channels=80,out_channels=80,kernel_size=(5,5),padding=1,stride=1),
                nn.BatchNorm2d(80),
                nn.ReLU(),
                nn.Conv2d(in_channels=80,out_channels=96,kernel_size=(5,5),padding=1,stride=1),
                nn.BatchNorm2d(96),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2,2),stride=2)
            )
            self.conv_layer_5=nn.Sequential(
                nn.Conv2d(in_channels=96,out_channels=108,kernel_size=(5,5),padding=1,stride=1),
                nn.BatchNorm2d(108),
                nn.ReLU(),
                nn.Conv2d(in_channels=108,out_channels=128,kernel_size=(5,5),padding=1,stride=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2,2),stride=2),
                nn.MaxPool2d(kernel_size=(2,2),stride=2)
            )
            self.linear_layer=nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_features=4608,out_features=hidden_linear_units),
                nn.ReLU(),
                nn.Linear(in_features=hidden_linear_units,out_features=out_features)
            )
        def forward(self,X_tensor):
            return self.linear_layer(self.conv_layer_5(self.conv_layer_4(self.conv_layer_3(self.conv_layer_2(self.conv_layer_1(X_tensor))))))
    

    in_features=3
    hidden_conv_features=32
    hidden_linear_features=38
    out_features=7

    model=Hand_gesture_recog(in_features=in_features,hidden_conv_units=hidden_conv_features,out_features=out_features,hidden_linear_units=hidden_linear_features)
    model.load_state_dict(torch.load(f=model_path,map_location=device))
    model.to(device)
    model.eval()
    
    return model

def predict(model,device):
    objective_list=["no_gesture","ok","one","palm","peace","three","thumb_index"]
    key_press=False
    print_key=False
    bounce=True
    size=(512,512)
    prediction_list=[]
    confidence=0
    gesture={}   
    frame_count=0
    
    
    
    vid=cv2.VideoCapture(0)
    x1, y1 = 220, 90   #rectangle frame creaton coordinates
    x2, y2 = 470, 390
    

    
    while True:
        frame_count+=1
        cond,frame=vid.read()
        
        if not cond: # to prevent the program from breaking if it fails to return a frame
            break
        
        
        
        cropped_img=frame[y1:y2,x1:x2]
        cropped_img_rgb=cv2.cvtColor(cropped_img,cv2.COLOR_BGR2RGB)  #it is cause pil img accepts from array but in default it expects an rgb, otherwise if we simply pass bgr , it will stay bgr in tensor form ,which wont work
        
        if gesture:
            for class_name,confidence in gesture.items():
                cv2.putText(frame,"Predicted_class: "+" "+class_name,(400,40),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,165,255),1)
                cv2.putText(frame,"Confidence_level: "+" "+str(confidence)+"%",(400,80),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,165,255),1)
                
        
        if key_press:
            end=timer.time()
            time_lapse=round( (end-start) ,2)
            cv2.putText(frame,"Time_passed:"+str(time_lapse),(10,40), cv2.FONT_HERSHEY_COMPLEX, 1,(0,165,255), 1)
            
            
            if frame_count%3==0: # only processes every 3 frames, to prevent lag 
                
                #pre processing 
                pil_img=Image.fromarray(cropped_img_rgb)
                pil_resized=transforms.functional.resize(pil_img,size)
                img_tensor=transforms.functional.to_tensor(pil_resized)
                img_batched=img_tensor.unsqueeze(0).to(device)
                
                with torch.inference_mode():
                    y_preds=model(img_batched).argmax(dim=1)
                prediction_list.append(y_preds.item())
                
                if len(prediction_list)>5:
                    prediction_list.pop(0)
                if len(prediction_list)==5:
                    frequency_counter=Counter(prediction_list)
                    class_idx,frequency=frequency_counter.most_common(1)[0]
                    confidence=(frequency/len(prediction_list)) *100
                    if print_key and confidence>=60:
                        key_press=False
                        print_key=False
                        bounce=True
                        class_name=objective_list[class_idx]
                        gesture[class_name]=confidence   #it will be classname:confidence level and will only store once per click and clear every press of z
            
            #for bounce correction : i.e we want the z key to be not pressed multiple times during the processing ,so we add a limit of 3 sec
            
            if time_lapse>4:
                key_press=False 
                print_key=False
                bounce=True
                
            
            
        
        
        key=chr(cv2.waitKey(1) & 0xFF)
        

        
        cv2.imshow("Magnified_view",cropped_img)
        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,250,0),2)
        cv2.imshow("w1",frame)
        if key.lower()=='z' and bounce==True:
            gesture.clear()
            prediction_list.clear()
            key_press=True
            print_key=True
            bounce=False
            start=timer.time()
        elif key.lower()=='q':
            break
        else:
            pass
        
    
    vid.release()
    cv2.destroyAllWindows()
    return
    
        
            
        
    
    


def main():
    
    model_path=Path("model_synthesized_94_97.pth")
    
    if not model_path.exists():
        print("Please put the learned model(.pth) file inside this folder. Otherwise it wont be able to analyze at all")
        return
    
    start_command=input("Welcome to my hand gesture recog program.\nYour webcam will turn on once u type X , when ur webcam turns on put your hand in the rectangular frame and make a gesture and press 'Z' in your keyboard to make a prediction.\nDo remember that we only have 7 different gestures that my model can determine at the moment,which includes (one,two/peace,three,ok,thumb_index,palm,no_gesture)\nPlease click on the w1 window and then press your keys\n")
    if start_command.lower() != 'x':
        print("You didnt type x, operation ended")
        return
    
    device="cuda" if torch.cuda.is_available() else "cpu"
    

    model=initialize_model(model_path,device)
    
    predict(model,device)
    
    


if __name__=="__main__":
    main()
    
