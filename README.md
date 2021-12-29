# ObjectDetection
Object Detection Self-Learning

## Papers Review:

### CenterNet:
Keypoint Triplets for Object Detection 
Kaiwen Duan, Song Bai, Lingxi Xie, Honggang Qi, Qingming Huang, Qi Tian

Realiza una deteccion de objetos en base a 3 puntos clave sobre la imagen, de acuerdo a otros propuestas tales como R-CNN que al usar un bounding box se requiere de un punto central asi como su ancho
y altura a ser aproximados, pero en en este propuesta se toman las esquinas opuestas asi como el centro como un tercer punto a diferencia de CornerNet que trabaja con solo las dos esquinas. Una gran
ventaja es que estos vectores cambian conforme a las caracteristicas internas de la imagen, para el punto central se enfoca especialmente en zonas donde exista mayor cantidad de informacion, para las
esquinas el enfoque es en los bordes y las zonas que dependen de dichos bordes formando una relacion en cascada, esto conocido como mapas de calor el "Corner Heatmaps" y el "Center Hearmaps".

Estos dos modulos evitan y filtran incorrectos bounding box reduciendo la tasa de falsos descubrimientos, además en cuanto al area central del bounding box crece dependiendo de cuando amplio o pequeño
es el area que cubre el bounding box si es grande la region central sera pequeña, si el area del bounding box es pequeña el area central tiene oportunidad de abarcar mayor area, lo que permite
detectar objetos bastante pequeños al capturar mayor cantidad de caracteristicas en una zona pequeña

Tres caracteristicas principales exploracion de zona central, pooling de los centros, y pooling de las esquinas en cascada permite a la red tener una buen mAP de 41.3  y tiempo rapido de inferencia de
270ms cuando se enfoca en una arquitectura sencilla como CenterNet511-52.

### MaskRcnn
Kaiming He, Georgia Gkioxari, Piotr Dollar, Ross Girshick

Esta arquitectura realizar una segmentacion de instancias que permite no solo detectar los objetos encapsulados en un bounding box, tambien segmentarlos con una mascaras que permiten conocer los
bordes o boundary del objeto. La propuesta se basa en la red Faster RCNN adicionando una rama para la prediccion de las mascaras. Una de las caracteristicas importantes es el trabajo para las regiones
de interes, en la Faster RCNN y Fast RCNN, se usaba el RoiPool, RoiWarp. RoiPool era encargado de seleccionar las caracteristicas especificas a la zona de interes ya detectada, mientras el RoiWarp se
encargaba de reducir el bounding box a una tamanio fijo para la entrada a una CNN, el RoiWarp usaba metodos de interpolacion, la MaskRcnn usa un RoiAlign que realiza el alineamiento de las caracteristicas extraidas con la imagen mediante  interpolacion, evita cualquier calculo de cuantizacion sobre la imagen, esto tambien logra mejorar la seleccion de caracteristicas a diferencia de RoiWarp y RoiPool que pueden producir mala alineaciones del ROI. Otra caracteristica importante de la arquitectura es la separacion de prediccion entre las mascaras y las clases, en la maskrccn se puede generar mascaras por cada clase sin que exista una competencia entre estas (cada uno es independiente), lo comun en segmentacion de instancias era usar un softmax por pixel y un multinomial cross-entropy loss, pero ahora con una funcion sigmoid por pixel y un binary loss, aplicado al usar FCN en la capa final, permite este desacoplamiento entre segmentacion y clasificacion.







