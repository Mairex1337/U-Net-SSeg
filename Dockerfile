FROM python:3.12-slim


WORKDIR /code

#it needs this otherwise opencv doesn't run
RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 && rm -rf /var/lib/apt/lists/*


COPY ./requirements_api.txt /code/requirements_api.txt


RUN pip install --no-cache-dir --upgrade -r /code/requirements_api.txt


COPY ./api /code/api


COPY ./src /code/src


COPY ./cfg.yaml /code/cfg.yaml


COPY ./outputs /code/outputs


CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "80"]
