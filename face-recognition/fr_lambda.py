import os
import json
import boto3
import base64
import numpy as np
import torch
from PIL import Image

ASU_ID = '1220103989' 
ACCOUNT_ID = '976193259691'
sqs = boto3.client('sqs')  
response_queue_url = f'https://sqs.us-east-1.amazonaws.com/{ACCOUNT_ID}/{ASU_ID}-resp-queue'
request_queue_url = f'https://sqs.us-east-1.amazonaws.com/{ACCOUNT_ID}/{ASU_ID}-req-queue'

class face_recognition:
    def __init__(self) -> None:
        self.model_path  = "/var/task/resnetV1.pt"  
        self.model_wt_path = "/var/task/resnetV1_video_weights.pt"
        self.resnet = torch.jit.load(self.model_path)
        self.saved_data = torch.load(self.model_wt_path)
        self.embedding_list = self.saved_data[0] 
        self.name_list = self.saved_data[1]  
        
    def face_recognition_func(self, face_img_path) -> str:
        face_pil = Image.open(face_img_path).convert("RGB")
        face_numpy = np.array(face_pil, dtype=np.float32)  
        face_numpy /= 255.0 
        face_numpy = np.transpose(face_numpy, (2, 0, 1))
        face_tensor = torch.tensor(face_numpy, dtype=torch.float32)
        
        if face_tensor is not None:
            emb = self.resnet(face_tensor.unsqueeze(0)).detach() 
            dist_list = []  
            for idx, emb_db in enumerate(self.embedding_list):
                dist = torch.dist(emb, emb_db).item()
                dist_list.append(dist)

            idx_min = dist_list.index(min(dist_list))
            recognized_name = self.name_list[idx_min]
            return recognized_name

recognizer = face_recognition()

def lambda_handler(event, context) -> dict:  
    temp_dir = "/tmp"    
    os.makedirs(temp_dir, exist_ok=True)

    for i, record in enumerate(event.get('Records', [])):
        queue_message_body = json.loads(record['body'])                
        queue_message_request_id = queue_message_body.get('request_id')
        encoded_face_string = queue_message_body.get('face')        
        if not all([queue_message_request_id, encoded_face_string]):
            continue

        decoded_face_bytes = base64.b64decode(encoded_face_string)
        face_img_path = os.path.join(temp_dir, f"{queue_message_request_id}_face.jpg")
        with open(face_img_path, "wb") as f:
            f.write(decoded_face_bytes)
        
        recognized_name = recognizer.face_recognition_func(face_img_path)
        
        response_queue_message = {
            'request_id': queue_message_request_id,
            'result': recognized_name if recognized_name else "Unknown"
        }
        
        sqs_response = sqs.send_message(
            QueueUrl=response_queue_url,
            MessageBody=json.dumps(response_queue_message),
        )

        receipt_handle = record['receiptHandle']
        sqs.delete_message(
            QueueUrl=request_queue_url,
            ReceiptHandle=receipt_handle
        )
        
        if os.path.exists(face_img_path):
            os.remove(face_img_path)
            
    
    