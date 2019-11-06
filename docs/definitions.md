# Assumptions
 - Los datasets elegidos se encuetran bien etiquetados y son representativos a la escena a la que pertenecen (REI - 9g).
 - El hardware funcionará correctamente.
 - Los resultados que se muestran en los papers son reales y son replicables.
 - Las redes preentrenadas a descargar (ImageNet, PlacesCNN) fueron entrenadas sólo con las imágenes de su dataset.

# Limits
 - El análisis se centrará en imágenes de las siguientes escenas de propiedades: cocina, comedor, baño, dormitorio, exterior, living, otros interior.
 - Tanto los tiempos de entrenamiento y predicción como el tamaño de las redes, estarán restringidos al hardware con el que se cuenta.
 - El trabajo intentara aceptar o refutar las hipótesis determinadas, quedando excluidas del alcance del mismo posibles investigaciones que surjan a partir del mismo.

# Hypothesis
 - H1 - Cada escena que se clasifica depende de lo que contiene en ella, y sobretodo de un top N de objetos de cada una (por ejemplo camas en dormitorios, bacha en baño), ¿qué pasa si se remueven o se agrega ruido sobre estos objetos? ¿la red mantiene sus métricas? ¿cuánto caen? La idea de esto es medir la capacidad de análisis que se pierde en habitaciones que al momento de tomar la foto no cuenten con todos sus muebles interiores. Analizar con dos habitaciones diferentes. Incluye object detection y localization, generación de ruido en su posición. Clasificación previa con análisis de features aprendidas o objetos que emergen. 
 
 - H2 - La detección de escenas con las redes actuales aún no alcanza un estado del arte en el cual, sin importar la calidad o la habitación con que se pruebe se clasifique correctamente (ver figura métricas por dataset y red del paper de la creación de PlacesCNN). Esto es por un lado por la gran cantidad de categorías con las que se suele probar y por otro por la dificultad del problema mismo. 
 Sería coherente realizar un análisis de las razones por las cuales estas redes fallan ante escenas que en otros casos es capaz de reconocer, con el fin de poder mejorar esa arista en el set de entrenamiento y alcanzar mejores resultados. El análisis se realizaría sobre las 7 etiquetas más comunes con el fin de determinar, en las últimas capas, qué se activa en los verdaderos positivos y qué en los falsos negativos. Estas diferencias deberían dar lugar a mejoras en los datasets de manera de completarlos o agregarles la diversidad suficiente para que la red sea capaz de reconocer estos casos.

 - H3 - Mejorada la diversidad de las imágenes para las etiquetas elegidas, lo más probable es que hacer transfer learning mejore las predicciones de los modelos en relación a predecir directamente con la red preentrenada. 
 
 - H4 - Agregar el filtro CLAHE debería derivar en mejores resultados de clasificación, es esto cierto?
 
# Experiments
 - H1: a partir de las red PlacesCNN se observarían las activaciones más frecuentes de una, dos o tres escenas diferentes. Según se vió en uno de los papers, para cada escena emergen objetos (uno o varios) en las últimas capas. Se localizaría a los mismos para posteriormente poder agregarles ruido encima. Si quitamos/tapamos estos objetos la red debería funcionar peor al clasificarlos, pero si se reentrena la red quitando/tapando estos objetos más importantes entonces se podría lograr que otras features tomen relevancia.

 - H2: a partir de la red PlacesCNN, predecir completo el dataset 9g, y obtener dos sub-datasets: los verdaderos positivos y los falsos negativos para cada etiqueta. A partir del sub-dataset de verdaderos positivos, analizar las activaciones de las últimas capas. Luego, comparar lo que emerge en estas últimas capas con los falsos negativos, de manera de obtener insights de las razones por las cuales las imágenes son mal etiquetadas. Además, para obtener insights, otra posibilidad es comparar con el dataset de entrenamiento (está publicado y etiquetado, no debería ser difícil). Reentrenar con imágenes que contengan lo que hace que los falsos negativos, y ver si mejora. De ser así, es porque al dataset PlacesCNN en el caso de Real Estate images, le hace falta diversidad, o al entrenarse, nunca vió casos similares a los etiquetados como falsos negativos.
 
 - tradicional vs cnn, debería mejorar cnn
 - cnn con más imágenes
 - una cnn clásica básica por etiqueta mejora una sola cnn con todas las etiquetas?
 - Feature extraction vs cnn
 - Transfer learning vs cnn
 -
 - H3: Transfer Learning con datasets REI y 9g, más imágenes generadas en pasos previos.
 - H4: Agregar CLAHE a imágenes de entrenamiento de H3 y hacer Transfer Learning.
 
 - Nota: H3 y H4 se harían con PlacesCNN
 
# Metrics
    - Balanced accuracy
    - Accuracy
