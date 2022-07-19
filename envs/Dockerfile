FROM tensorflow/tensorflow:2.2.2-gpu-py3
RUN apt-get update ; apt-get install vim git wget -y
RUN python3 -m pip install --upgrade pip
RUN pip install numpy pandas pydot matplotlib rasterio scikit-learn scikit-image graphviz tqdm pydotplus ray optuna
RUN pip install kaleido plotly
#RUN apt-get --yes install graphviz
