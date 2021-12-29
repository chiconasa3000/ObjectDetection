import os
#os.system("pip install wget")

import wget

CUSTOM_MODEL_NAME = 'my_ssd_mobnet'
PRETRAINED_MODEL_NAME = 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8'
PRETRAINED_MODEL_URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz'
TF_RECORD_SCRIPT_NAME = 'generate_tfrecord.py'
LABEL_MAP_NAME = 'label_map.pbtxt'


# In[3]:


paths = {
    'WORKSPACE_PATH': os.path.join('Tensorflow', 'workspace'),
    'SCRIPTS_PATH': os.path.join('Tensorflow','scripts'),
    'APIMODEL_PATH': os.path.join('Tensorflow','models'),
    'ANNOTATION_PATH': os.path.join('Tensorflow', 'workspace','annotations'),
    'IMAGE_PATH': os.path.join('Tensorflow', 'workspace','images'),
    'MODEL_PATH': os.path.join('Tensorflow', 'workspace','models'),
    'PRETRAINED_MODEL_PATH': os.path.join('Tensorflow', 'workspace','pre-trained-models'),
    'CHECKPOINT_PATH': os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME),
    'OUTPUT_PATH': os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'export'),
    'TFJS_PATH':os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'tfjsexport'), 
    'TFLITE_PATH':os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'tfliteexport'),
    'PROTOC_PATH':os.path.join('Tensorflow','protoc')
 }

files = {
    'PIPELINE_CONFIG':os.path.join('Tensorflow', 'workspace','models', CUSTOM_MODEL_NAME, 'pipeline.config'),
    'TF_RECORD_SCRIPT': os.path.join(paths['SCRIPTS_PATH'], TF_RECORD_SCRIPT_NAME),
    'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], LABEL_MAP_NAME)
}

#creacion de rutas
for path in paths.values():
    if not os.path.exists(path):
        os.system("mkdir -p {}".format(path))

#Clonando framework de detection de objetos de tensorflow
if not os.path.exists(os.path.join(paths['APIMODEL_PATH'], 'research', 'object_detection')):
    os.system("git clone https://github.com/tensorflow/models {}".format(paths['APIMODEL_PATH']))


#Instalando Tensorflow Object detection
#compilacion de los archivos protoc de object detection
#os.system("apt-get install protobuf-compiler")
#os.system("cd Tensorflow/models/research && protoc object_detection/protos/*.proto --python_out=. && cp object_detection/packages/tf2/setup.py . && python -m pip install .")

#Verificacion de instalacion de archivos protoc de object detection
#Podria salir mal algunas dependencias pero por lo general si fue instalado bien y compilado bien los protocs y tengas
#correctamente instalado las librerias graficas el test debe salir bien
VERIFICATION_SCRIPT = os.path.join(paths["APIMODEL_PATH"],"research",'object_detection',"builders","model_builder_tf2_test.py")
#os.system("python3 {}".format(VERIFICATION_SCRIPT))
#os.system("pip install tensorflow --upgrade")
#os.system("pip uninstall protobuf matplotlib -y")
#os.system("pip install protobuf matplotlib==3.2")

import object_detection

#Consiguiendo el modelo preentrenado de ssd con backboen de mobilenet
#actuando sobre imagenes 320x320
#os.system("wget {}".format(PRETRAINED_MODEL_URL))
#os.system("mv {}.tar.gz {}".format(PRETRAINED_MODEL_NAME,paths["PRETRAINED_MODEL_PATH"]))
#os.system("cd {} && tar -zxvf {}.tar.gz".format(paths["PRETRAINED_MODEL_PATH"], PRETRAINED_MODEL_NAME))

#Creando un labelmap (clases)
#labels = [{"name":"Taxi", "id":1}, {"name":"Billboard","id":2},{"name":"Persons","id":3}]

#Escribir los archivos con las clases o labelmap
# with open(files["LABELMAP"],"w") as f:
#     for label in labels:
#         f.write("item{ \n")
#         f.write("\tname:\'{}\'\n".format(label["name"]))
#         f.write("\tid:{}\n".format(label["id"]))
#         f.write("}\n")


#Creando los TF records
# #Descomprimiendo si subiste un archive conteniendo los train y test directorios
# #de las imagenes
# ARCHIVE_FILES = os.path.join(paths["IMAGE_PATH"],"archive.tar.gz")
# if os.path.exists(ARCHIVE_FILES):
#     os.system("tar -zxvf {}".format(ARCHIVE_FILES))

#blob rescatar rutas de imagenes
import glob
import numpy as np
from sklearn.model_selection import train_test_split

dir_labels = np.array(np.sort(glob.glob("/root/mylabels/labels/*")))
dir_images = np.array(np.sort(glob.glob("/root/mylabels/data/*")))

# dir_all =  zip(dir_labels,dir_images)
# for dir_tmp in dir_all:
#     print(dir_tmp)

#division de las rutas en train y test
x_train, x_test, y_train, y_test = train_test_split(dir_images,dir_labels,test_size=0.30, random_state=42)
# for idtmp in range(len(x_train)):
#     print("{},{}".format(x_train[idtmp],y_train[idtmp]))

#creacion de rutas para la data
paths_data = {
    "TRAIN_DIR" : os.path.join(paths["IMAGE_PATH"],"train"),
    "TEST_DIR" : os.path.join(paths["IMAGE_PATH"],"test")
}

for path in paths_data.values():
    if not os.path.exists(path):
        os.system("mkdir -p {}".format(path))

#mover los respectivas rutas a los directorios de test y train
#Para el caso training labels (x) e imagenes (y)
# for idtrain in range(len(x_train)):
#     os.system("cp {} {}".format(x_train[idtrain],paths_data["TRAIN_DIR"]))
#     os.system("cp {} {}".format(y_train[idtrain],paths_data["TRAIN_DIR"]))
# for idtest in range(len(x_test)):
#     os.system("cp {} {}".format(x_test[idtest],paths_data["TEST_DIR"]))
#     os.system("cp {} {}".format(y_test[idtest],paths_data["TEST_DIR"]))

#ToDo: Analizar el generador de tf_records
if not os.path.exists(files["TF_RECORD_SCRIPT"]):
    os.system("git clone https://github.com/nicknochnack/GenerateTFRecord {}".format(paths["SCRIPTS_PATH"]))

#Generacion de tfrecord para train y test
os.system("python3 {} -x {} -l {} -o {}".format(files["TF_RECORD_SCRIPT"],os.path.join(paths["IMAGE_PATH"],"train"),files["LABELMAP"],os.path.join(paths["ANNOTATION_PATH"],"train.record")))
os.system("python3 {} -x {} -l {} -o {}".format(files["TF_RECORD_SCRIPT"],os.path.join(paths["IMAGE_PATH"],"test"),files["LABELMAP"],os.path.join(paths["ANNOTATION_PATH"],"test.record")))
