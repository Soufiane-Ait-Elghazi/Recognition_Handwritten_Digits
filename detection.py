#!/usr/bin/env python
#-*- coding: utf-8 -*-

import os
from math import sqrt, fabs
from numpy import argmax
from PIL import Image
from random import shuffle, randint, seed
from sklearn.neural_network import MLPRegressor

def mean(tab):
    """
        Fonction qui calcul la moyenne d'un tableau
    """
    try:
        val = 0
        tot = 0
        for k in tab:
            val+=k
            tot+=1
        return val/tot
    except:
        print("Division by 0, return 0")
        return 0
         
def instance(img):
    """
        Création des instances
        On quadrille l'image suivant différente distances
        
            --Pour chaque case créée à partir de quadrillage
            --on calcul la moyenne des pixels contenus dans la case
    """
    tab = []
    dis = [4,5,10,20]
    for k in dis:
        pas = int(100/k)
        for i in range(0,100,pas):
            for j in range(0,100,pas):
                new = []
                for k in range(i,i+pas):
                    for l in range(j,j+pas):
                        new.append( mean( img.getpixel((k,l)) ) )
                tab.append(mean(new))
    return tab
                    
def creer_instance(fichier):
    """
        Pour chaque fichier on créé le couple label/features
        Le tableau de label se trouvera à l'index 0 de la sortie, le reste sera les features
    """
    try:
        print(fichier)
        img = Image.open("char/"+fichier)
        tab = []
        num = [0 for i in range(0,10)]
        ele = int(fichier.split("_")[0])
        num[ele] = 10000
        tab.append(num)
        tab = tab + instance(img)
        return tab
    except Exception as e:
        print(e)
        
def main():
    try:
        seed(0)
	#L'ensemble des fichiers se trouvent dans ce répertoire
        fichiers = os.listdir("char/")
        tab = []

        #Pour chaque fichier dans le répertoire, on créé ses instances
        print("Création des instances..")
        for k in fichiers:
            if("test" not in k):
                tab.append(creer_instance(k))
        shuffle(tab)

        #Création des features et des label
        feat = []
        lab = []
        for k in tab:
            feat.append(k[1:])
            lab.append(k[0])

        #Phase d'apprentissage
        print("Phase d'apprentissage..")
        mod = MLPRegressor(
            hidden_layer_sizes=(666,666,666),
            max_iter=2000
            )
        mod.fit(feat,lab)

        #On réalise un test de notre modèle en fonction d'une image test
        print("Test :")
        fichier = "char/test.png"
        img = Image.open(fichier)
        print("Fichier de référence :",fichier)
        pred = instance(img)
        result = mod.predict([pred])
        
        for k in range(0,len(result[0])):
            print(k," : ",result[0][k])
        
        print("Valeur prédite : ",argmax(result))

        #os.system("pause")
        
    except Exception as e:
        print(e)
    
    


if __name__=="__main__":
    main()
