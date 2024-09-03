SHIP_IMAGE_CLASSIFICATION
<<<<<<< HEAD
WORKFLOW:
1. Update config.yaml
2. Update params.yaml
3. Update entity
4. Update configuration manager in src config
5. Update the components
6. Update the pipeline
7. Update the main.py
8. Update dvc.yaml

RUN:
1. Clone Repository - https://github.com/anishmaks/SHIP_Classification_using_Resnet.git
2. Create a conda environment after opening the repository
   conda create -n ship python=3.8 -y
   conda activate ship
3. Install requirements.txt
   pip install -r requirements.txt
4. Run app.py file

DVC:
1. dvc init
2. dvc repro
3. dvc dag

AWS-CI/CD with Github:
1. Login to AWS Console
2. Create an iam user
    EC2- virtual machine
    ECR- for saving docker image in AWS

3. Build docker image of the code
4. Push docker image to ECR
5. Launch and connect to EC2
6. Pull image from ECR to EC2 for the deployment

   #Policy:
   1. AmazonEC2ContainerRegistryFullAccess
   2. AmazonEC2FullAccess

7. Create ECR repo to store docker image
8. Create an EC2 ubuntu machine
9. Open EC2 and install docker by the following commands:
   sudo apt-get update -y
   sudo apt-get upgrade
   curl -fsSL https://get.docker.com -o get-docker.sh
   sudo sh get-docker.sh
   sudo usermod -aG docker ubuntu
   newgrp docker
10. Configure EC2 as self hosted runner
11. Setup secrets configuration in Github
12. edit inbound roles to update the port in AWS

Azure-CI/CD with Github:
1. Login azure account
2. Create a Container Registry and Save the passwords
3. Run below from terminal:
   docker build -t ship02.azurecr.io/ship02:latest .
   docker login ship02.azurecr.io
   docker push ship02.azurecr.io/ship02:latest

Deployment:
1. Build the Docker image of the Source Code
2. Push the Docker Image to Container Registry
3. Launch Web app server
