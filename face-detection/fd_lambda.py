import os
import json
import boto3
import base64
from facenet_pytorch import MTCNN
from PIL import Image
import numpy as np


ASU_ID = '1220103989'
sqs = boto3.client('sqs')
ACCOUNT_ID = '976193259691'
request_queue_url = f'https://sqs.us-east-1.amazonaws.com/{ACCOUNT_ID}/{ASU_ID}-req-queue'

class face_detection:
    def __init__(self) -> None:
        self.mtcnn = MTCNN(image_size=240, margin=0, min_face_size=20)

    def face_detection_func(self, test_image_path, output_path) -> str:
        img = Image.open(test_image_path).convert("RGB")
        img_array = np.array(img)
        img = Image.fromarray(img_array)

        key = os.path.splitext(os.path.basename(test_image_path))[0].split(".")[0]

        face, prob = self.mtcnn(img, return_prob=True, save_path=None)

        if face is not None:
            os.makedirs(output_path, exist_ok=True)
            face_img = face - face.min()      
            face_img = face_img / face_img.max() 
            face_img = (face_img * 255).byte().permute(1, 2, 0).numpy()
            face_pil = Image.fromarray(face_img, mode="RGB")
            face_img_path = os.path.join(output_path, f"{key}_face.jpg")
            face_pil.save(face_img_path)  # Save as JPEG
            return face_img_path
        else:
            print("No face is detected")
            return None

detector = face_detection()

def lambda_handler(event, context) -> dict:  
    if 'body' in event:
        request_body = json.loads(event['body'])
        request_content = request_body.get('content') 
        request_id = request_body.get('request_id')  
        request_filename = request_body.get('filename')  
    else:
        request_content = event.get('content')
        request_id = event.get('request_id')
        request_filename = event.get('filename')

    if not all({request_content, request_id, request_filename}):
        return {
            'statusCode': 400,
            'body': json.dumps('Missing required parameters')
        }
    
    image_data = base64.b64decode(request_content)
    temp_dir = "/tmp"
    os.makedirs(temp_dir, exist_ok=True)
    file_basename = os.path.basename(request_filename)
    input_image_path = os.path.join(temp_dir, file_basename)
    with open(input_image_path, "wb") as f:
        f.write(image_data)
    output_dir = os.path.join(temp_dir, "faces")
    detected_face = detector.face_detection_func(input_image_path, output_dir)

    if detected_face is not None:
        with open(detected_face, "rb") as f:
            detected_face_bytes = f.read()
        encoded_detected_face = base64.b64encode(detected_face_bytes).decode('utf-8')

        request_queue_message = {
            'request_id': request_id,
            'face': encoded_detected_face
        }

        sqs_request = sqs.send_message(
            QueueUrl=request_queue_url,
            MessageBody=json.dumps(request_queue_message),
        )

        if os.path.exists(input_image_path):
            os.remove(input_image_path)
        if os.path.exists(detected_face):
            os.remove(detected_face)