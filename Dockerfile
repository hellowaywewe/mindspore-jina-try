FROM mindspore/mindspore-cpu:1.0.0

# setup the workspace
COPY . /workspace
WORKDIR /workspace

# fix: add to path
ENV PATH="/root/.local/bin:${PATH}"

# install the third-party requirements
RUN apt-get update -y && \
    apt-get install apt-utils -y && \
    apt-get install libgl1-mesa-glx -y && \
    apt-get install libglib2.0-0 -y && \
    pip install --upgrade pip && \
    pip install --user -r requirements.txt

# download dataset and train
RUN sh get_data.sh && \
    cd lenet && \
    python train.py --data_path /tmp/jina/mnist/ --ckpt_path ckpt --device_target="CPU" && \
    cd -

RUN  python generate.py

# for testing the image
RUN pip install --user pytest && pytest -s

RUN python app.py -t index -n 10000


ENTRYPOINT ["python", "app.py", "-t", "query"]