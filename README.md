# metadata-sadosky-santander
Clasificación de preguntas de clientes


## [1] Build the Docker image
docker build -t <image_name>:<image_tag> -f Dockerfile .

## [2] Docker run 
### [cpu]
docker run -v $(pwd):/meta/ -p <port>:<port> -it <image_name>:<image_tag>
### [gpu]
nvidia-docker run -v $(pwd):/meta/ -p <port>:<port> -it <image_name>:<image_tag>

## In order to run the jupyter notebooks:
jupyter notebook --allow-root --ip=0.0.0.0 --port <port>

## In order to run the pipelin:
### This will pre-process the data, perform data augmentation, train the model and predict
make all

### To perform indiduals commands run:
make help
