import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import folium

df = pd.read_excel("C:\\Users\\Geksmode\\Desktop\\data\\liste-officielle-museesdf-20220127-data.xlsx")

# Créer une carte centrée sur une position GPS
carte_points = folium.Map(location=[46.820379, 1.677096], zoom_start=6.5)

for i in df.index :
    folium.Marker([ df["Latitude"][i],df["Longitude"][i]], popup=df["Nom officiel du musée"][i]).add_to(carte_points)
carte_points.save("carte_points.html")

dep = pd.unique(df['Département'])


df = df.astype({'Identifiant Muséofile':"string",'Département':"string",'Région administrative':"string",'Commune':"string",'Nom officiel du musée':"string",'Adresse':"string",'Lieu':"string",'Téléphone':"string",'URL':"string"})



def compter_musee_departement(dep,df):
    Liste =[]
    for i in range (len(dep)):
        compt = 0
        for j in df.index:
            if dep[i] == df["Département"][j]:
                compt = compt + 1
        Liste.append([dep[i],compt]) 
    return Liste


histogramme = compter_musee_departement(dep,df)

print(histogramme)

hist = df["Département"].hist()

plt.savefig("pandas_hist_03.png", bbox_inches = "tight",dpi=1000)

