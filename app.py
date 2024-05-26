import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import folium
from streamlit_folium import st_folium
sc=StandardScaler()
import warnings



def tempvshumid(df):
    df.plot(kind='scatter', x='Temperature', y='Humidity', s=32, alpha=.8)
    plt.xlabel("Temperature (C)")
    plt.ylabel("Humidity")
    st.pyplot(plt)

def rentvsarea(df):
    a=df
    a=a.drop_duplicates(subset=['Area'])
    a=a['Area']
    a=pd.DataFrame(a)
    rn=[]
    for i in a['Area']:
        fil=df[df['Area']==i]
        rn.append(int(fil['Rent'].mean()))
    plt.figure(figsize=(12,10))
    plt.bar(a['Area'],rn,width=0.4)
    plt.xticks(rotation=90)
    plt.xlabel("Area")
    plt.ylabel("Rent")
    st.pyplot(plt)

def nbhkvsrent(df):
    b=df
    b=b.drop_duplicates(subset=['nBHK'])
    nBHK=[]
    for i in b['nBHK']:
        fil=df[df['nBHK']==i]
        nBHK.append(int(fil['Rent'].mean()))
    plt.figure(figsize=(12,10))
    plt.bar(b['nBHK'],nBHK,width=0.4)
    plt.xlabel("nBHK")
    plt.ylabel("Rent")
    st.pyplot(plt)

def boxplt(df):
    plt.figure(figsize=(12,10))
    df.boxplot()
    plt.xticks(rotation=90)
    plt.ylabel("Values")
    st.pyplot(plt)

def calvals(df):
    a=df
    a=a.drop_duplicates(subset=['Area'])
    a=a['Area']
    a=pd.DataFrame(a)
    fam=[]
    bac=[]
    tem=[]
    rf=[]
    flo=[]
    hum=[]
    saf=[]
    con=[]
    n=[]
    lat=[]
    lon=[]
    rn=[]
    for i in a['Area']:
        fil=df[df['Area']==i]
        fam.append(int(fil['Family'].sum()))
        bac.append(int(fil['Bachelors'].sum()))
        tem.append(int(fil['Temperature( C)'].mean()))
        hum.append(int(fil['Humidity'].mean()))
        rf.append(int(fil['Rainfall'].mean()))
        flo.append(int(fil['Flooding'].mean()))
        rn.append(int(fil['Rent'].mean()))
        saf.append(int(fil['Safety'].mean()))
        con.append(int(fil['Connectivity'].mean()))
        n.append(int(fil['nBHK'].mean()))
        lat.append(fil['Latitude'].mean())
        lon.append(fil['Longitude'].mean())
    ndf={
        'Area': a['Area'],
        'Temperature': tem,
        'Humidity': hum,
        'Rainfall': rf,
        'Flooding': flo,
        'nBHK': n,
        'Rent': rn,
        'Safety':saf,
        'Connectivity': con,
        'Family': fam,
        'Bachelors': bac,
        'Latitude': lat,
        'Longitude': lon
    }
    new_df=pd.DataFrame(ndf)
    return new_df, a

def encodearea(df):
    j=1
    an=[]
    for i in df['Area']:
        an.append(j)
        j+=1
    df.insert(0,'Area No.',an)
    df=df.drop('Area',axis=1)
    sf=sc.fit_transform(df)
    sdf=pd.DataFrame(sf)
    return sdf, df

def kmeansclustering(sdf, df):
    km=KMeans(n_clusters=3,random_state=101,max_iter=300)
    df['Cluster']=km.fit_predict(sdf) 
    return df

def locmap(new_df):
    m1=folium.Map(location=(17.38405,78.45636),zoom_start=11.2)
    cluster_color={0: 'red', 1: 'orange', 2: 'green'}
    cluster_groups={0: folium.FeatureGroup(name="Bad"),
                    1: folium.FeatureGroup(name="Average"),
                    2: folium.FeatureGroup(name="Good")}
    for _,i in new_df.iterrows():
        c=i['Cluster']
        if i['Bachelors']>10 and i['Family']>10:
            type1="Bachelor & Family Friendly"
        elif i['Bachelors']>10:
            type1="Bachelor Friendly"
        else:
            type1="Family Friendly"
        folium.Marker(
            location=(i['Latitude'],i['Longitude']),
            icon=folium.Icon(color=cluster_color[c]),
            popup=f"Area: {i['Area']}, Rent: {i['Rent']}, Temp: {i['Temperature']}, Humidity: {i['Humidity']}, Rainfall: {i['Rainfall']}, {type1}"
        ).add_to(cluster_groups[c])

    for i in cluster_groups.values():
        i.add_to(m1)
    folium.LayerControl().add_to(m1)
    st_data = st_folium(m1, width=1000, height=1000)


warnings.filterwarnings('ignore')
st.set_page_config(layout="wide")
options = st.sidebar.radio('Options', ['Map', 'Data Visualisations', 'Dataset'])

data=pd.read_csv('projdata1.csv')

a=['Unnamed: 14','Unnamed: 15','Unnamed: 16']
data=data.drop(a, axis=1)

data.loc[data['s.no']==447,'Rent']=23000
df=data[data['nBHK']<=3]

df, a=calvals(df)
sdf, df=encodearea(df)
df=kmeansclustering(sdf, df)
df.insert(1,'Area',a['Area'])

if options=='Map':
    st.title('Hyderabad Rent Prediction')
    locmap(df)

elif options=='Data Visualisations':
    st.title('Data Visualisations')
    tempvshumid(df)
    rentvsarea(df)
    nbhkvsrent(df)
    boxplt(df)
else:
    st.title('Datasets')
    st.write("original dataset")
    st.write(data)
    st.write("dataset after preprocessing and averaging")
    st.write(df)
