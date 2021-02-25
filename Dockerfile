FROM python:3.8
FROM tensorflow/tensorflow:latest-gpu-jupyter
WORKDIR /
RUN jupyter notebook --generate-config --allow-root
EXPOSE 8888
CMD ["jupyter", "notebook", "--allow-root", "--notebook-dir=/", "--ip=0.0.0.0", "--port=8888", "--no-browser"]
