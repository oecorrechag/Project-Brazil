import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, classification_report


def read_images_drive(imgpath):

    images = []                 
    directories = []             
    dircount = []                
    prevRoot = ''
    cant = 0                     

    labels = []
    indice = 0

    cafes = []
    indicec = 0

    for root, dirnames, filenames in os.walk(imgpath):                 
        for filename in filenames:                                     
            filepath = os.path.join(root, filename)                   
            image = plt.imread(filepath)                              
            
            images.append(image)                                        
            cant = cant + 1                                             
            
            if prevRoot != root:                                        
                prevRoot = root                                       
                directories.append(root)
                dircount.append(cant)
                cant = 0
    dircount.append(cant)
    dircount = dircount[1:]
    dircount[-1] = dircount[-1] + 1

    for cantidad in dircount:
        for i in range(cantidad):                            
            labels.append(indice)                            
        indice = indice + 1                                  

    for directorio in directories:                          
        name = directorio.split(os.sep)                     
        cafes.append(name[len(name) -1])                  
        indicec = indicec + 1                                  

    print('Directorios leidos: ', len(directories))         
    print('Imagenes en cada directorios', dircount)
    print('Suma total de imagenes en subdirs', sum(dircount))
    print('Cantidad etiquetas creadas: ', len(labels))

    return images, directories, dircount, labels, cafes


# Funciones auxiliares

def evaluate_model(y_test, y_pred, model = None):
    """
    Aqui se pondran las cuatro metricas que se piden en el reporte
    debido a que cada modelo requiere las mismas metricas es probable que se pueda ejecutar el mismo para todas
    """
    f1 = print('El f1 score es de: ', f1_score(y_test, y_pred, average = 'weighted'))
    acc = print('El acc score es de: ', accuracy_score(y_test, y_pred))
    ps = print('La precision es de: ', precision_score(y_test, y_pred, average = 'micro'))
    re = print('El f1 score es de: ', recall_score(y_test, y_pred, average = 'micro'))
    
    return f1, acc, ps, re



def matconfision(y_test, y_pred):
    '''EN ESTA FUNCION ENTRAN LAS Y ORIGINALES Y LAS PREDICCIONES PARA CREAR UNA MATRIZ DE CONFUSION MAS BONITA'''
    
    cm = confusion_matrix(y_test, y_pred)
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    target_names = set(y_test)
    
    fig, ax = plt.subplots(figsize = (10,10))
    sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Predicted')
    
    return plt.show(block=False)