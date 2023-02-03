
# Project-Brazil

# <center> Clasificación de Defectos de Café Verde
Actividad Final Clase Vision  Artificial
  
**Universidad Nacional de Colombia**

El café colombiano es de tipo **arábica** nativo de la península península arábiga, el cual se cultiva desde el año 1 D.C. En Colombia _563.000_ familias (aproximadamente 3.5 millones) se dedican a la producción del café. Estados Unidos representa el mayor mercado mundial de café, seguido de Brasil, siendo este país asimismo el mayor productor de este cultivo en el mundo. Los países escandinavos y Finlandia son donde se consume más café por número de habitantes. Colombia es el segundo exportador de café, para el año 2019 exporto 14.8 millones de sacos de café. Brasil es el primer exportador de café, para el año 2019 exporto 40.6 millones de sacos de café. Un saco de café equivale a 60 kilos. Uno de los pilares fundamentales que ha hecho posible la conservación de los mercados y la fidelidad de los clientes en el exterior es la calidad del Café. Esta confianza y seguridad se basa en el hecho cierto que el producto entregado satisface las expectativas de clientes y consumidores en cuanto a sus características físicas y sensoriales. Las características del café colombiano son la calidad, frescura, mantenimiento y preservación.

**Diseño de experimento:**

A cada grano de café se realizó la toma de dos fotos, una por el frente y otra por el reverso, sobre una hoja clara, la cual cuenta con tres cuadros de referencia de color que miden 1 por lado.

Se realizara un clasificador de defectos de Café Verde. Los defectos tenidos en cuenta y las caracteristicas son:

* Buenos:
  * Grano grande.
  * Plano y parejo.
  * Cumplimiento de estandares de calidad dados por la Green Coffee Association of New York City, Inc. (Mallas)
* Negro total o parcial:
  * Grano con coloración del pardo al negro.
  * Encogido.
  * Arrugado.
  * Cara plana hundida.
  * Hendidura muy abierta.
* Vinagre o parcialmente vinagre:
  * Grano con coloración del crema al carmelita oscuro.
  * Hendidura libre de tegumentos.
  * Película plateada puede tender a coloraciones pardo rojizas
* Mordido:
  * Grano con una herida o cortada.
  * Color oxidado.
* Inmaduro:
  * Grano de color verdoso o gris claro.
  * La cutícula no desprende.
  * Superficie marchita.
  * Tamaño menor que el normal.
  * En este grupo se incluye el grano del paloteo.
* Arrugados:
  * Desarrollo pobre del cafeto por sequía.
  * Debilidad del cafeto por falta de fertilizantes.
* Broca:
  * Grano con pequeños orificios.
* Decolorado:
  * Grano con alteraciones en su color normal, presenta colores que van desde el blanqueado, crema, amarillo hasta el carmelita.

Se tienen 8 tipos de cafe para el experimento, en total son 1089 fotos.

    * 72 Arrugados.
    * 90 Broca.
    * 191 Buenos.
    * 200 Decolorados.
    * 200 Inmaduro.
    * 123 Mordido.
    * 101 Negro.
    * 113 Vinagre.

El experimento es no balanceado!!!

Dimensiones: 740 x 480 pixeles, 2256 x 1504 pixeles, hasta 4272 x 2848, entre otros <br>
Peso: 800 kb hasta 3.2 MB  <br>
Formato: jpeg

# Problema

EL COMITE NACIONAL DE CAFETEROS mediante la RESOLUCION NUMERO 5 DE 2002, dicta las medidas conducentes a garantizar la calidad del café de Exportación, las cuales deberán ser observadas, tanto por la Federación Nacional de Cafeteros de Colombia como por los exportadores privados.

Los requisitos mínimos de calidad para la exportación de café verde en almendra los siguientes: 
* Excelso de Exportación: Debe tener por lo menos un 50% de granos retenidos sobre la malla quince (15).
* Humedad: No debe sobrepasar el 12%.
* Defectos: Hasta 72 granos defectuosos por 500 gramos de muestra, sin exceder de 12 granos del primer grupo. 
* Compensación por grano brocado.
* Infestación: El café deberá estar libre de todo insecto vivo.
* Olor: El café deberá tener su olor característico. 
* Color: El café deberà tener una apariencia uniforme en color.
* Prueba de taza: El café deberá tener sabor y aroma característico.
* Marcaciones adicionales: Medido por malla:
    * Premium: Malla 18
    * Supremo: Malla 17
    * Extra: Malla 16
    * Maragogipe: Malla 17 <br>

[Video: Clasificacion por mallas](https://www.youtube.com/watch?v=yeMeBAQGTUc "YouTube")

# Objetivo

* Desarrollar un prototipo capaz de clasificar los granos del café. 
* Reducir el porcentaje de error de error de clasificación para evitar realizar una compensación o la devolución del cargamento. 

# Resultados y conclusiones

Se Ajustaron 3 modelos, una Red Neuronal Convolucional, un Bosque Aleatorio y una Regresión Logística. Se calificaron con las métricas de f1_score, accuracy, precision_score, recall_score y matriz de confusión, dándole más relevancia a esta última.

Inicialmente se tiene que el modelo CNN tiene problemas ya que las redes neuronales son demandantes en imagenes por lo cual este modelo presenta sobreajuste, la Regresión Logística es un modelo no es demandante en muestras. Al comparar las metricas númericas la Regresión Logística presenta mejores metricas sobre todos los modelos tambien presenta la mejor matriz de confusion, por ultimo el tiempo de ajuste de modelos el modelo CNN tarda aproximadamente 15 minutos en ser entrenado, la Regresión Logística tarda solo tarda 1 segundo, por estas razones se suguiere lo siguiente:

* Realizar un nuevo diseño de experimento.
* Aumentar las muestras, ya que estos modelos pueden ser potencializados con mayores y mejores imagenes.
* Se elige la Regresión Logística por presentar mejores tiempos y metricas.

Como proyecto futuro se podria utilizar la funcion VideoCapture de CV2 para capturar en tiempo real y clasificar inmediatamente las imagenes. 

Figura 1: Matriz de Confusión del ajuste de la Regresión Logística.

![alt text](https://github.com/oecorrechag/Project-Brazil/blob/main/images/matriz.png)

# Poster

![alt text](https://github.com/oecorrechag/Project-Brazil/blob/main/images/poster.png)

### Bibliografia

cafedecolombia. Defectos de Café Verde. Recuperado de: http://www.cafedecolombia.com/clientes/es/regulacion_nacional/exportadores/2831_calidades_de_exportacion/

cafesocialcoffee. (2013, Enero 2). Tillado Café Social en Pergamino y Selección en Mallas Números 18-17-16-15-14 y 13. [Archivo de video]. Recuperado de: https://www.youtube.com/watch?v=yeMeBAQGTUc

EFE.(2020). Brasil exportó 3,22 millones de sacos de café en enero, 7,2 % menos que 2019. diariolibre. Sao Paulo. Recuperado de: https://www.diariolibre.com/actualidad/internacional/brasil-exporto-322-millones-de-sacos-de-cafe-en-enero-72-menos-que-2019-GG17007076

Ly, k. 7 Defectos de Café Verde que Tostadores y Productores Deben Reconocer. Recuperado de: https://www.perfectdailygrind.com

Resolucion No. 02 de 2002. Comite Nacional de Cafeteros. Bogota, Colombia, 6 de Junio de 2002.

Silvarolla, Maria B.; Mazzafera, Paulo; Fazuoli, Luiz C. (2004). «A naturally decaffeinated arabica coffee». Nature 429 (6994): 826. PMID 15215853. doi:10.1038/429826a.
Weinberg, Bennet Alan; Bealer, Bonnie K. (2001). The World of Caffeine: The Science and Culture of the World's Most Popular Drug. Nueva York: Routledge. ISBN 0-415-92722-6.

Vanegas, F. (2017). Estructura del fruto del café. [Figura]. Recuperado de https://www.yoamoelcafedecolombia.com/