# Cloud-Based IoT Video Face Detection and Recognition System

**Cloud-Based IoT Video Face Detection and Recognition System** is a hybrid cloud–edge platform engineered for real-time video analytics across connected devices. Built to handle the projected growth to over 40 billion IoT endpoints by 2030, it combines serverless cloud inference with edge processing to deliver high-accuracy face detection and recognition at low latency.

## Technologies Used

- **Serverless Compute**: AWS Lambda (containerized inference functions)
- **Container Registry**: Amazon ECR (PyTorch-based Docker images)
- **Messaging & Queuing**: Amazon SQS (cloud coordination), MQTT via AWS IoT Greengrass Core (edge messaging)
- **Edge Processing**: AWS IoT Greengrass on EC2 for local MTCNN face detection
- **Machine Learning Framework**: PyTorch (MTCNN for detection, InceptionResnetV1 for recognition)
- **Infrastructure**: AWS (Lambda, SQS, ECR, IoT Greengrass, EC2)
- **Monitoring & Logging**: Amazon CloudWatch (metrics, logs)
- **Security & Privacy**: Hybrid architecture minimizes bandwidth and keeps sensitive data local

