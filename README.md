# Text Generation API (Local LLM Inference)

> This project presents an API that simplifies the locally setup of a large language model
> and access to the inference funtionalities using prompt engineering.

## 1. Installing

**Configure a local token for gitlab authentication:** 
> https://docs.gitlab.com/ee/user/profile/personal_access_tokens.html

```bash
# Download via ssh
git clone ssh://git@gitlab.estig.ipb.pt:4589/_cedri/paulo_leitao/TextGenerationAPI.git
```
```bash
# Download via https
git clone https://gitlab.estig.ipb.pt/_cedri/paulo_leitao/TextGenerationAPI.git
```

## 2. Configuring the environment

**Install Python:** If not already installed, download and install Python from python.org (Version 3.8.10).

### Using venv (Python)

**Create a Virtual Environment:** Open a terminal and navigate to the project clone directory. 

```bash
# Replace myenv with your preferred environment name.
python -m venv myenv
# Activate the myenv enviroment
.\myenv\Scripts\activate
# Installing the dependencies from requirements.txt
pip install -r requirements.txt
```

### Using Anaconda
**Install Anaconda:** Download and install Anaconda from anaconda.com.

**Create a Virtual Environment:** Open a terminal or Anaconda Prompt and navigate to the project clone directory.

```bash
# Replace myenv with your preferred environment name.
conda create -n myenv python=3.8.10
# Activate the myenv enviroment
conda activate myenv
# Installing the dependencies from requirements.txt
pip install -r requirements.txt
```

## 3. Running the API server

Navigate to the project top directory and configure your own token from huggingface in the .env file. 

### If you are using venv python enviroment

```bash
.\myenv\Scripts\activate
python "project\path\TextGenerationAPI\main.py"
```

### If you are using an anaconda enviroment

```bash
conda activate myenv
python "project\path\TextGenerationAPI\main.py"
```

## 4. Testing

With the API server up, you just need to set up a client to start to access the endpoints.

Current endpoints:

- **/** -> Retrieves the Local LLM information, and the device that it is running.
- **/generate_inference/** -> Require a body message with information about the prompt,
num_tokens, temperature, skip_special_tokens and retrieves the llm response and additional 
information about the request.


The examples below used the Postman Client as an example to demonstrate the endpoints results:

![inference_demo](UsageSamples/info_example_postman.jpg)

![inference_demo](UsageSamples/inference_example_postman.jpg)

## 5. Extra

The LLMsAPIsTesting directory contains a project that tested 3 major LLM inference APIs for a study basis.

## 6. Cautions

This project uses LLM models download from hugging face, this models can be chosed using the enviroment
variables, this models can be very large, and require a lot of VRam, Ram, CPU and Storage even for inference
manners.