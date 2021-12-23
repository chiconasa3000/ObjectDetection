import os

#Instalacion de fiftyone
#os.system("pip install fiftyone")

import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone import ViewField as F

dataset = foz.load_zoo_dataset("open-images-v6",
split="validation",
label_type=["detections"],
classes = ["Taxi", "Billboard", "Persons"],
max_samples=60,)

#Solo coger las etiquetas que corresponden a las clases
#aunque seria tambien excelente recolectar etiquetas negativas
#print(dataset)
#view = dataset.filter_labels("ground_truth",F("label").is_in(["Taxi","Billboard","Persons"]))

#Exportamos
dataset.export(export_dir="/root/mylabels/",dataset_type=fo.types.VOCDetectionDataset)
print("Existen {} clases".format(len(dataset.default_classes)))


#comenzando a exportar
# dataset.default_classes = None
# view.export(
#     labels_path="/root/fiftyone",
#     dataset_type=fo.types.VOCDetectionDataset,
# )

