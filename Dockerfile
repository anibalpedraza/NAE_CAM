# Usa la imagen base de TensorFlow con soporte para GPU
FROM tensorflow/tensorflow:2.19.0-gpu

RUN mkdir /NAE_CAM

# Actualiza pip
RUN pip install --timeout=300 --upgrade pip

ADD env_Keras3_DockerGPU.txt /NAE_CAM

# Establece el directorio de trabajo
WORKDIR /NAE_CAM

# Instala Keras 3 (que es parte de tensorflow 2.19)
RUN pip install -r env_Keras3_DockerGPU.txt --verbose

# Comando predeterminado para ejecutar Python cuando inicias el contenedor
CMD /bin/bash