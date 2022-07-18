FROM pytorch/pytorch

COPY /home/ben/ucl/project/mnist-tests .
RUN cd mnist-tests

RUN python3 -m pip install --upgrade pip &&\
    pip install -r requirements.txt

