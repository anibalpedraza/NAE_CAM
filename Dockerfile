# Usa la imagen base de TensorFlow con soporte para GPU
FROM tensorflow/tensorflow:2.7.4-gpu

RUN mkdir /NAE_CAM

# Actualiza pip
RUN pip install --timeout=300 --upgrade pip

ADD env_Keras2_DockerGPU.txt /NAE_CAM

# Establece el directorio de trabajo
WORKDIR /NAE_CAM

# Instala dependencias
RUN pip install -r env_Keras2_DockerGPU.txt --verbose

# Comando predeterminado para ejecutar Python cuando inicias el contenedor
CMD /bin/bash