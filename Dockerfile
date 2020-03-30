FROM nvcr.io/nvidia/tensorflow:19.12-tf2-py3
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

RUN rm /etc/bash.bashrc

WORKDIR /root

RUN pip install scikit-learn pandas matplotlib torch torchvision pathos fairseq tensorflow_text tensorflow_hub
RUN pip install git+https://github.com/huggingface/transformers.git@master#egg=transformers
RUN apt-get update && apt-get install -y openssh-server screen
RUN mkdir /var/run/sshd
RUN echo 'root:testdocker' | chpasswd
RUN sed -i 's/.*PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config

# SSH login fix. Otherwise user is kicked off after login
RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

RUN git clone https://github.com/NVIDIA/apex
RUN cd apex
RUN sed -i 's/check_cuda_torch_binary_vs_bare_metal(torch.utils.cpp_extension.CUDA_HOME)/pass/g' /root/apex/setup.py
RUN pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" /root/apex/

RUN git clone https://github.com/facebookresearch/fastText.git
WORKDIR /root/fastText
RUN pip install .

ENV NOTVISIBLE "in users profile"
RUN echo "export VISIBLE=now" >> /etc/profile

WORKDIR /root
COPY . .

EXPOSE 22
CMD ["bash", "-c", "/usr/sbin/sshd && jupyter notebook"]