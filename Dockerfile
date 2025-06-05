FROM python:3.12


WORKDIR /code


RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y


COPY ./requirements.txt /code/requirements.txt


RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt


COPY ./api /code/api


CMD ["fastapi", "run", "api/main.py", "--port", "80"]