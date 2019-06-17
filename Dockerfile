FROM tensorflow/tensorflow:18.04-gpu-py3-jupyter

RUN pip3 install jupytext --upgrade

RUN pip3 install pandas --upgrade

RUN pip3 install pillow --upgrade

RUN pip3 install sklearn --upgrade

RUN pip3 install xarray --upgrade

#CMD ["bash", "-c", "source /etc/bash.bashrc && jupyter notebook --notebook-dir=/tf --ip 0.0.0.0 --no-browser --allow-root"]