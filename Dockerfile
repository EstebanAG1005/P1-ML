# Usa una imagen base de Python
FROM python:3.8-slim

# Instala las dependencias necesarias
RUN pip install pandas numpy scikit-learn tensorflow matplotlib scikeras

# Establece el directorio de trabajo
WORKDIR /proyect
# Copia el contenido del proyecto al contenedor
COPY ./src src

COPY ./data data


# Comando por defecto para ejecutar el script de pipeline
CMD ["python", "src/model_tunning_pipeline.py", "--data", "data/train.csv"]
