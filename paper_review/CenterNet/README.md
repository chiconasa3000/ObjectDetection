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


