
### MaskRcnn
Kaiming He, Georgia Gkioxari, Piotr Dollar, Ross Girshick

Esta arquitectura realizar una segmentacion de instancias que permite no solo detectar los objetos encapsulados en un bounding box, tambien segmentarlos con una mascaras que permiten conocer los
bordes o boundary del objeto. La propuesta se basa en la red Faster RCNN adicionando una rama para la prediccion de las mascaras. Una de las caracteristicas importantes es el trabajo para las regiones
de interes, en la Faster RCNN y Fast RCNN, se usaba el RoiPool, RoiWarp. RoiPool era encargado de seleccionar las caracteristicas especificas a la zona de interes ya detectada, mientras el RoiWarp se
encargaba de reducir el bounding box a una tamanio fijo para la entrada a una CNN, el RoiWarp usaba metodos de interpolacion, la MaskRcnn usa un RoiAlign que realiza el alineamiento de las caracteristicas extraidas con la imagen mediante  interpolacion, evita cualquier calculo de cuantizacion sobre la imagen, esto tambien logra mejorar la seleccion de caracteristicas a diferencia de RoiWarp y RoiPool que pueden producir mala alineaciones del ROI. Otra caracteristica importante de la arquitectura es la separacion de prediccion entre las mascaras y las clases, en la maskrccn se puede generar mascaras por cada clase sin que exista una competencia entre estas (cada uno es independiente), lo comun en segmentacion de instancias era usar un softmax por pixel y un multinomial cross-entropy loss, pero ahora con una funcion sigmoid por pixel y un binary loss, aplicado al usar FCN en la capa final, permite este desacoplamiento entre segmentacion y clasificacion.







