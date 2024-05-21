# Multi-class Image Segmentation using UNETR

⚡Combining the power of Transformers with UNet for state-of-the-art image segmentation task💪  <br><br>
**This is Module 2 of UNETR which covers backend development and deployment on the cloud**  
Module 1. [UNETR-MachineLearning](https://github.com/Taha0229/UNetR-MachineLearning)  
Module 2. [Develop and Deploy Backend of UNETR ](https://github.com/Taha0229/UNetR-slim-backend)   
Module 3. [Develop and Deploy Frontend of UNTER](https://github.com/Taha0229/UNetR-frontend)

# Project Brief

In October 2021, Ali Hatamizadeh et al. published a paper titled "UNETR: Transformers for 3D Medical Image Segmentation," introducing the UNETR architecture, which outperforms other segmentation models. In essence, UNETR utilizes a contracting-expanding pattern consisting of a stack of transformer as the encoder which is connected to the CNN-based decoder via skip connections, producing segmented image. 
<br><br>
This project aims to implement the UNETR architecture as described in the paper, training it on a custom multi-class dataset for facial feature segmentation. The project involves developing the machine learning model, backend, and frontend for the application. The UNETR model is served via a REST API using Django REST framework to a Next.js frontend, with the frontend and backend deployed separately on Vercel and AWS, respectively. This tech stack selection ensures high scalability, performance, and an excellent UI/UX.
<br><br>

# Module Overview
In this module, I have shown, how to develop and delpoy the backend using the mentioned tech to server the ML model built on Module 1. [UNETR-MachineLearning](https://github.com/Taha0229/UNetR-MachineLearning). This covers implementation from scratch and implementation by cloning this repo both in a very simple and descriptive step-by-step manner.  
## Tech Used
1. Django REST Framework (DRF):  To serve the model, I have opted for utilizing DRF due to its modular approach, high scalability, and off the shelf security.
2. Docker:
3. GitHub Actions: 
4. AWS EC2:
5. AWS ECR:

# How it works?
1. The `/inference/` route only accepts `POST` requests.
2. Incoming data is validated. If valid, processing continues; otherwise, `a Missing data in POST Request` response with a status code of 400 is sent.
3. The incoming data is processed by separating the Base64 image string and image name.
4. The `decodeImage()` function decodes the image and stores it in the `unetr_model_output\decode` directory. The function requires Base64 string and image name.
5. A prediction pipeline is instantiated, and the image name is passed to the instance. It automatically selects the decoded image based on the image name, performs inference on it, and stores the result in the `unetr_model_output\predict` directory.
6. The inferred image is encoded using the `encodeImageIntoBase64()` function to send the response to the client. The function take image name as its parameter and automatically picks the inferred image.

## Workflow

**Workflow for implementing from scratch:**
1. Setup Django REST Framework (from step 1-9)
2. Setup Docker (step 10)
3. Setup GitHub Workflows (step 11)
4. Setup AWS 
5. Setup GitHub Actions
6. Setup Outbound Rules

**Workflow for implementing by cloning:**
1. Setup Django REST Framework (from step 1-4)
2. Setup AWS 
3. Setup GitHub Actions
4. Setup Outbound Rules
 


# Implementing Backend from scratch
For this I will presume  you possess some basic knowledge of python, virtual environment, GitHub and Django. All you have to do is follow the commands.  

### Step 1:  Create Virtual Environment
To create the virtual env, I am gonna use conda, however any other method will work equally fine.   <br>
Open the directory where you want to develop this Django app on VS Code. 
Then run the follow following commands:   <br> <br>
`conda create --name unetr-backend python=3.9.19 -y`  
activate it with:  
`conda activate unetr-backend`   <br> <br>
**note:** I have chosen this particular version, because previously, I have tried to deploy the backend on Vercel, but because the serverless function had exceeded the unzipped maximum size of 250 MB. I had to shift to AWS and by that time, I had changed my python version and dependencies to match the python requirements on the Vercel. You can implement this with a later python version (I would recommend 3.10) just make sure to mention the correct dependency version in `requirements.txt`.  

### Step 2: Install and Setup Django
Install Django using:  
`pip install Django==4.2.13`   <br> <br>
Create a project using:  
django-admin startproject backend   <br> <br>
Change directory to backend  
`cd backend`
<br> <br>
Test Django installation:    
`python manage.py runserver` <br> <br>

Stop the server (if required):
`ctrl + c`

Migrate unapplied migrations:  
`python manage.py migrate`

Open the mentioned local host's link provided by the Django on the terminal. You should be able to see the initial Django screen with a rocket.


### Step 3: Install Dependencies
Create a `requirements.txt` file inside the root directory of Django project.

For our project we only require the following dependencies-
```
Django==4.2.13
django-cors-headers==4.3.1
djangorestframework==3.15.1
gunicorn==22.0.0
numpy==1.26.4
onnx==1.10.0
onnxruntime==1.10.0
opencv-python==4.9.0.80
patchify==0.2.3
```

Copy paste them to `requirements.txt` and run:  
`pip install -r requirements.txt`

### Step 4: Create unetr app 
In django, applications are organized into smaller, self-contained components called "django apps". In this approach, Django projects are composed of multiple apps, each responsible for a specific functionality or feature of the overall project.  <br><br>
Create unetr app for serving the UNETR model:  
`python manage.py startapp unetr`

### Step 5: Setup unetr app
1. Create a folder - `unetr_model` inside unetr app. Copy the `compatible_model.onnx` from our previous implementation of [UNETR Machine Learning module](https://github.com/Taha0229/UNetR-MachineLearning). This can be found in `artifacts > training > compatible_model.onnx`
2. Create a folder - `unetr_model_output` inside unetr app with two sub-folders - `decode` and `predict`
3. Create a file - `utils.py` inside unetr app. It should contain only two required functions from UNETR ML model, which can be found in `src > UNETRMultiClass > utils > common.py`. The two function are `decodeImage` and `encodeImageIntoBase64`, but they are implemented with a little modification from that of the ML model as follows:

```
import base64
from django.conf import settings  ## to access BASE_DIR
import os

def decodeImage(imgstring, fileName):
    filename_path = os.path.join(settings.BASE_DIR, "unetr", "unetr_model_output", "decode", fileName)  ## To point to appropriate directory
    imgdata = base64.b64decode(imgstring)
    with open(filename_path, "wb") as f:
        f.write(imgdata)
        f.close()
    print("decode done", "="*100) ## simple logging message


def encodeImageIntoBase64(croppedImagePath):
    filename_path = os.path.join(settings.BASE_DIR, "unetr", "unetr_model_output", "predict", croppedImagePath) ## To point to appropriate directory
    with open(filename_path, "rb") as f:
        print("encode done", "="*100) ## Simple logging message
        return base64.b64encode(f.read()) 
```

If you encounter `could not resolve from source` error or `module not found` error, then it's because wrong python interpreter is selected. To resolve the issue, while being on any python file, on the very bottom right of the VS Code, you can see Python as the selected language and its version followed by the interpreter. The interpreter should be `3.9.19 ('unetr-backend':conda)`. Click on the python version and a popup will appear with the list of python interpreters. 

4. Create a file `predict.py` which is again copied from the ML model, but implemented in the backend with a few modifications
```
import os
import cv2
import numpy as np
from patchify import patchify
import onnxruntime as ort
from django.conf import settings

class PredictionPipeline:
    def __init__(self):
        self.rgb_codes = [
            [0, 0, 0],
            [0, 153, 255],
            [102, 255, 153],
            [0, 204, 153],
            [255, 255, 102],
            [255, 255, 204],
            [255, 153, 0],
            [255, 102, 255],
            [102, 0, 51],
            [255, 204, 255],
            [255, 0, 102],
        ]

        self.classes = [
            "background",
            "skin",
            "left eyebrow",
            "right eyebrow",
            "left eye",
            "right eye",
            "nose",
            "upper lip",
            "inner mouth",
            "lower lip",
            "hair",
        ]

        self.onnx_model_path = os.path.join(settings.BASE_DIR,"unetr","unetr_model", "compatible_model.onnx")
        
        self.session = ort.InferenceSession(self.onnx_model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def grayscale_to_rgb(self, mask, rgb_codes):
        h, w = mask.shape[0], mask.shape[1]
        mask = mask.astype(np.int32)
        output = []

        enum = enumerate(mask.flatten())

        for i, pixel in enum:
            output.append(rgb_codes[pixel])

        output = np.reshape(output, (h, w, 3))

        return output

    def save_results(self, image_x, pred, save_image_path):

        pred = np.expand_dims(pred, axis=-1)
        pred = self.grayscale_to_rgb(pred, self.rgb_codes)

        line = np.ones((image_x.shape[0], 10, 3)) * 255

        cat_images = np.concatenate([image_x, line, pred], axis=1)

        cv2.imwrite(save_image_path, cat_images)

    def predict(self, filename):
        cf = {}
        cf["image_size"] = 256
        cf["num_classes"] = 11
        cf["num_channels"] = 3
        cf["num_layers"] = 12
        cf["hidden_dim"] = 128
        cf["mlp_dim"] = 32
        cf["num_heads"] = 6
        cf["dropout_rate"] = 0.1
        cf["patch_size"] = 16
        cf["num_patches"] = (cf["image_size"] ** 2) // (cf["patch_size"] ** 2)
        cf["flat_patches_shape"] = (
            cf["num_patches"],
            cf["patch_size"] * cf["patch_size"] * cf["num_channels"],
        )

        image_name = os.path.join(settings.BASE_DIR, "unetr", "unetr_model_output", "decode", filename)  ## To point to appropriate directory
        display_name = image_name.split("\\")[-1].split(".")[0]         ## Splits on behalf of back slash
        print("display_name: ", display_name)

        input_img = cv2.imread(image_name, cv2.IMREAD_COLOR)
        input_img = cv2.resize(input_img, (cf["image_size"], cf["image_size"]))
        norm_input_img = input_img / 255.0

        patch_shape = (cf["patch_size"], cf["patch_size"], cf["num_channels"])
        patches = patchify(norm_input_img, patch_shape, cf["patch_size"])
        patches = np.reshape(patches, cf["flat_patches_shape"])
        patches = patches.astype(np.float32)  # [...]
        patches = np.expand_dims(patches, axis=0)  # [1, ...]

        """ Prediction """

        input_dict = {self.input_name: patches}
        outputs = self.session.run([self.output_name], input_dict)
        pred_1 = np.argmax(outputs, axis=-1)  ## [0.1, 0.2, 0.1, 0.6] -> 3
        pred_1 = pred_1.astype(np.int32)
        pred_1 = np.reshape(pred_1, (256, 256))

        print("saving...")
        
        save_image_path = os.path.join(settings.BASE_DIR, "unetr", "unetr_model_output", "predict" , filename) ## To point to appropriate directory
        self.save_results(input_img, pred_1, save_image_path)

        return save_image_path
```

### Step 6: Implement views
Our API is very compact and doesn't require fancy view implementation. 
1. First we will import encoder and decoder functions from the `utils.py` and `PredictPipeline` from `predict.py` along with two DRF imports.
2. In the post method, we will store the incoming data in `input_data` which will contain name of the image as `imagename` and the base64 encoded image string as `image` if the client has sent required data. 
3. Send the image string and its name to the decoder function. This function will store the image in `unetr_model_output > decode` folder. 
4. Instantiate the `PredictionPipeline` class followed by providing the name of the image to the instance. This will automatically, grab the image stored in the `unetr_model_output > decode` folder and run inference on it.
5. Encode the inference image back to base64 string to send back to the frontend. This is done using the encoding function.
6. Finally we respond the client with the encoded base64 image string if the client had sent required data otherwise, we will send status 400 with a message "Missing data in POST Request"

```
from rest_framework.response import Response
from rest_framework.decorators import api_view
from .utils import encodeImageIntoBase64, decodeImage
from .predict import PredictionPipeline



@api_view(["POST"])
def run_prediction(request, *args, **kwargs):
    method = request.method
    if method == "POST":
        input_data = request.data
        
        image_name = input_data.get("imgname")
        base64_image_string = input_data.get("image")
        if image_name is not None and base64_image_string is not None:
            decodeImage(input_data["image"], image_name)
            predict = PredictionPipeline()
            predict.predict(filename=image_name)
            output_encoded = encodeImageIntoBase64(image_name)
            return Response({"output": output_encoded})
        else:
            return Response({"output": "Missing data in POST Request"}, status=400)
```

### Step 7: Configure routes 
In `backend > urls.py ` we need to add the route for our unetr app.  
1. import the view from unetr  
`from unetr.views import run_prediction`
2. append the list of `urlpatterns` with `inference` route:  
```
    urlpatterns = [
    ...
    path("inference/", run_prediction, name="inference"),
]
```
You can use the shortcut `ctrl + p` to list and search any file in the project

### Step 8: Configure settings.py
Finally we will configure the `settings.py` as follows
1. Add `ALLOWED_HOSTS`  
`ALLOWED_HOSTS = ["*"]`
2. Append the list of `INSTALLED_APPS`:
```
INSTALLED_APPS = [

    ...    

    "corsheaders",
    "rest_framework",
    "unetr",
]
```

3. Add corsheader middleware in `MIDDLEWARE` list  
```
MIDDLEWARE = [
    ...,
    "django.contrib.sessions.middleware.SessionMiddleware", # for context
    "corsheaders.middleware.CorsMiddleware", # add this only
    "django.middleware.common.CommonMiddleware", # add this only
    "django.contrib.sessions.middleware.SessionMiddleware", # for context
    ...,
]
```

4. Add `CORS_ALLOWED_ORIGIN_REGEXES` and `CORS_ALLOWED_ORIGIN` below `ROOT_URLCONF`:  
```
CORS_ALLOWED_ORIGIN_REGEXES = [  
    r"^http:\/\/localhost:*([0-9]+)?$",  
    r"^https:\/\/localhost:*([0-9]+)?$",  
    r"^http:\/\/127.0.0.1:*([0-9]+)?$",  
    r"^https:\/\/127.0.0.1:*([0-9]+)?$",     
]  

CORS_ALLOWED_ORIGIN = [
    "http://localhost:3000",
    "https://unetr-frontend.vercel.app",
]
```

### Step 9: Test the server
Since, the frontend is not ready yet, we cannot test the model over API but we can check for any error while running the server:
`python manage.py runserver`


### Step 10: Setup Docker 
For the deployment we are gonna use Docker to containerize and deploy the model. Make sure you have docker installed on your machine.

1. Run `docker init`

In the terminal for the follow-up questions choose the followings

What application platform does your project use?  
`Python`  
What version of Python do you want to use? (3.11.9)  
`3.9.19`  
What port do you want your app to listen on? (8000)  
`press enter`  
What is the command you use to run your app? (gunicorn 'backend.wsgi' --bind=0.0.0.0:8000)  
`press enter`  

2. Modify `Dockerfile` as follows:  
When we use `docker init` it creates a Dockerfile with an unprivileged user to encourage security. We need to provide this user with limited privilege to write on our `ml_model_output` folder in unetr app.  
Secondly, in ubuntu, for opencv's python bindings, there are few internal dependencies errors. This is can be resolved by installing the following dependencies:  
`libgl1 libglib2.0-0 libsm6 libxrender1 libxext6`

So, the modified `Dockerfile` would look like:
```
# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.9.19
FROM python:${PYTHON_VERSION}-slim as base

# Prevents Python from writing pyc files.
ENV PYTHONDONTWRITEBYTECODE=1

# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Create a non-privileged user that the app will run under.
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/nonexistent" \
    --shell "/sbin/nologin" \
    --no-create-home \
    --uid "${UID}" \
    appuser

# Download dependencies as a separate step to take advantage of Docker's caching.
# Leverage a cache mount to /root/.cache/pip to speed up subsequent builds.
# Leverage a bind mount to requirements.txt to avoid having to copy them into
# into this layer.
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=requirements.txt,target=requirements.txt \
    python -m pip install -r requirements.txt

# This resolves dependencies error due to open-cv
RUN apt-get update && apt-get install libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 -y

# Copy the source code into the container.
COPY . .

# Provide only necessary permissions to the appuser
RUN chown -R appuser:appuser /app/unetr/unetr_model_output

# Switch to the non-privileged user to run the application.
USER appuser    



# Expose the port that the application listens on.
EXPOSE 8000

# Run the application.
CMD gunicorn 'backend.wsgi' --bind=0.0.0.0:8000
```
3. Test the server with docker:
We can use either `docker run` command or `docker compose` command, in my opinion, the later is better but for single service i.e. our Django server, `docker run` can also be used.
use:  
`docker build -t unetr .`  
`docker run unetr`  

or 

`docker compose up` - this will build the image and run the container itself. 
alternatively you can use `docker compose up -d`, the `-d` flag is for detached mode i.e. runs container on the background.  
Test the server on `http://127.0.0.1:8000/` rather than on `http://0.0.0.0:8000`.

### Step 11: Setup GitHub Workflows
For implement CI/CD pipelines, I have used GitHub Actions. How it works? When something is pushed to the branch which is being watched, GitHub Actions automatically executes the pre-defined jobs. These job are defined in yaml file.  
For this project I have three pipelines which are: 
1. Continuous Integration: Checks for updates in the code.
2. Continuous Delivery: updates the Ubuntu, configure AWS Credentials, build a docker image and push it to AWS ECR.
3. Continuous Deployment: This runs on EC2 instance, pulls a docker image from AWS ECR and runs the docker container.  

In the root directory of our Django app we need to create a folder `.github` with a sub-folder `workflows`. Inside `workflows` create a file - `main.yaml`. Copy the configuration from [here](https://github.com/Taha0229/UNetR-slim-backend/blob/master/.github/workflows/main.yaml) and paste it inside the `main.yaml`, because the file is comparatively big and can be used in other applications as well. 

**note:**
1. The workflow is configured to watch `master` branch rather than the `main` branch. Modify it as per your needs. 
2. The third job from the bottom **must** be commented for initial deployment. It checks, if a container is running and removes it if it is running. Because there is no container in the first deployment, hence it will throw an error. You can uncomment from the second deployment onwards.
```
# - name: Run Docker Image to serve users
      #   run: |
      #    docker run -d -p 8000:8000 --name=unetr -e 'AWS_ACCESS_KEY_ID=${{ secrets.AWS_ACCESS_KEY_ID }}' -e 'AWS_SECRET_ACCESS_KEY=${{ secrets.AWS_SECRET_ACCESS_KEY }}' -e 'AWS_REGION=${{ secrets.AWS_REGION }}'  ${{secrets.AWS_ECR_LOGIN_URI}}/${{ secrets.ECR_REPOSITORY_NAME }}:latest
```
3. For the continuous deployment, if `docker build` is being used, then we need to provide the environment variables (code is already provided in the config file i.e. `main.yaml`), if `docker compose up -d` is used then `compose.yaml` file needs to be updated to add environment variables as follows:
```
services:
  server:
    build:
      context: .
    ports:
      - 8000:8000
    environment: # this is added 
      - AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
      - AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
      - AWS_REGION=${AWS_REGION}
```

After setting up the workflow, push your code in GitHub. Use:  
`git add .`  
`git commit -m "commit message"`  
`git push`




# Implementing Backend by Cloning the Repo  
For this I will presume that you possess some basic knowledge of python, virtual environment and Django. 

### Step 1: Fork and Clone the repo  
Fork the repo and clone it on your local machine using:  
`git clone [url of forked repo]`  <br><br>
Change directory to UNetR-slim-backend  
`cd UNetR-slim-backend`

### Step 2: Create Virtual Environment
Create virtual env as described in the above section  
`conda create --name unetr-backend python=3.9.19 -y`  
`conda activate unetr-backend`   

If you encounter `could not resolve from source` error or `module not found` error, then it's because wrong python interpreter is selected. To resolve the issue, while being on any python file, on the very bottom right of the VS Code, you can see Python as the selected language and its version followed by the interpreter. The interpreter should be `3.9.19 ('unetr-backend':conda)`. Click on the python version and a popup will appear with the list of python interpreters. 

### Step 3: Install Dependencies
`pip install -r requirements.txt`  
This command will install all the dependencies mentioned in the requirements.txt

### Step 4: Test the server
If you clone the repo, everything is pre-implemented. However, you can test server:  
**Test without docker:**  
`python manage.py migrate`  
`python manage.py runserver`   
**Test with docker:**  
`docker compose up`  

For both the cases open the local host from `http://127.0.0.1:8000/`  

These commands should be able to run the server without any error other than 404 for the root route with two available routes.


## Setup AWS 
1. Login to your AWS account 
2. Search for IAM in services
3. Create User: In the left menu of IAM, click on `Users`. Here, click on `Create user`. Just set the user name then click `Next`. Then select `Attach policies directly`.
4. Attach policies: Search and attach the following policies-   
i. `AmazonEC2ContainerRegistryFullAccess`  
ii. `AmazonEC2FullAccess`.   
Click `Next`and then finally click on `Create user`.
5. Now, we need to get the access keys of the user.  
Open the user and in the right side of the summary, click on `Create access key`.  
Select `CLI` the click on `Next`, skip the optional part by clicking on `Create access key`.  
Now download the `Access key` and `Secret access key` by clicking on `Download .csv file`.

5. Search for ECR in services, then create a repository in it.  
Name anything you like such as: `unetr-ecr`.  
Finally click on `Create repository`. Lastly copy the URI and store it somewhere as it is required in the next section.  
It looks something like: `11143526****.dkr.ecr.us-east-1.amazonaws.com/unetr-ecr` 
6. Search for EC2 in services.
7. Launch Instance: In the EC2 dashboard, click on `Launch instance`. 
8. Name the server such as `unetr-server`.  
Select `Ubuntu` as the OS and select the free tier AMI.  
Afterwards, select a free tier instance type i.e. t2.micro, which is enough for serving the API but definitely not for model training.  
Now, Generate a key-pair by setting a name and rest as default, click on `Create a key pair`.  
In the network settings, select `Create security group` and check `Allow SSH traffic from`, `Allow HTTPS traffic from the internet` and `Allow HTTP traffic from the internet`.  
Finally, Configure the storage, 8 GB should be enough for this project. After all of this configuration, click on `Launch instance`.
9. Wait until the instance is initialized. 
10. Connect to the instance.
11. Run: `clear` on the Ubuntu machine to clear the screen
12. Run the following command one-by-one to update Ubuntu and install docker.
```
sudo apt-get update -y

sudo apt-get upgrade -y

curl -fsSL https://get.docker.com -o get-docker.sh

sudo sh get-docker.sh

sudo usermod -aG docker ubuntu

newgrp docker
```
With this, we have completed the AWS setup but we will be needed to execute a few more lines of code which will be provided by GitHub Actions, more on it in the next section.

## Setup GitHub Action
1. Open the settings of your repo, in the left menu, click on `Actions > Runner` inside `Code and automation`. 
2. Click on `New self-hosted runner`
3. Select `Linux` as the Runner image 
4. Now copy and paste each line in our ec2 terminal, sequentially.  
Starting with `mkdir actions-runner && cd actions-runner` command  
when `./config.sh --url https://github.com/Taha0229/UNetR-MachineLearning --token A3CYKLZXDHEK6PSR6P7B553GJNR5M` is executed, it will ask a few runner registration questions. Do as follows:  <br><br>
Enter the name of runner group to add this runner to:  
`press Enter`  
Enter the name of runner:  
`self-hosted`  
This runner will have the following labels: 'self-hosted', 'Linux', 'x64' Enter any additional labels:  
`press Enter`  
Enter name of work folder:  
`Press Enter`

Finally copy and executed the last command:   `./run.sh`.  <br><br>
Now, if we go back to the `Runner` we should be able to see a self-hosted runner with "online" status.
5. Lastly, we need to configure our Secrets for AWS in GitHub Actions. On the left menu, expand `Secrets and variables` present inside `Security`, select `Actions`. Here we need to add `New repository secret`.  
Now, we will add the secrets one by one.
```
AWS_ACCESS_KEY_ID= ## from the 5th step of Setup AWS

AWS_SECRET_ACCESS_KEY= ## from the 5th step of Setup AWS

AWS_REGION = us-east-1 ## check your region 

AWS_ECR_LOGIN_URI = (example) 11143526****.dkr.ecr.us-east-1.amazonaws.com ## only up to .com

ECR_REPOSITORY_NAME = unetr-ecr ## after .com
``` 
With this everything is good to go, We can push our code to GitHub and it will automatically deploy it on the AWS. Subsequently, we can manually trigger the CI/CD pipeline from `Actions` in the GitHub.

## Setup Outbound Rules
The very last step is set outbound rules, so we can connect with our server remotely. Navigate to the dashboard of the running EC2 instance, scroll down to select `Security` tab. Click on the `Security groups`. On the bottom, inside `Inbound rules`, click on `Edit inbound rules`. Click `Add rule` and add the following rule:  
```
Type: Custom TCP
Port range: 8000
Source: Anywhere-IPv4
```
Finally, click on `Save rules`.
Now, we can access our server from the provided `Public IPv4` followed by port `8000`. Example- `http://54.9x.24x.16x:8000 `


### GitHub Commit message format
Feat– feature

Fix– bug fixes

Docs– changes to the documentation like README

Style– style or formatting change 

Perf – improves code performance

Test– test a feature