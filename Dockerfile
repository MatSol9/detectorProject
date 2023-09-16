FROM datamachines/cudnn_tensorflow_opencv:10.1_2.1.0_4.3.0-20200423
EXPOSE 5000
RUN mkdir /tmp/build/
#RUN apt-get update
#RUN apt-get install pip -y
#RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
#RUN apt-get install libglu1-mesa-dev -y
#RUN apt-get install libgl1-mesa-glx -y
#RUN apt-get install -y libglib2.0-0 -y
ADD requirements.txt /tmp/build/
RUN pip install -r /tmp/build/requirements.txt
COPY . /tmp/build
RUN find /tmp/build
CMD ["python3", "/tmp/build/src/main.py"] 
