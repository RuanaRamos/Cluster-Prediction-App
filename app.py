import streamlit as st
import pandas as pd
import joblib


encoder = joblib.load('encoder.pkl')
scaler = joblib.load('scaler.pkl')
kmeans = joblib.load('kmeans.pkl')



st.title('Marketing-Interessengruppen') # Grupos de interesse para marketing

st.write("""
         In diesem Projekt haben wir den K-Means-Clustering-Algorithmus angewendet, um Nutzerinteressengruppen zu identifizieren und vorherzusagen, mit dem Ziel, Marketingkampagnen effektiver auszurichten.
         Durch diese Analyse konnten wir das Publikum in "Interessensblasen" segmentieren. Dies ermöglicht die Erstellung personalisierter und zielgerichteter Kampagnen basierend auf den Verhaltensmustern und Vorlieben jeder Gruppe.
         """)

# Uploader de arquivo
up_file = st.file_uploader('Wählen Sie eine CSV-Datei für die Vorhersage aus', type='csv')

def processar_prever(df):
    encoded_sexo = encoder.transform(df[['sexo']])
    encoded_df = pd.DataFrame(encoded_sexo, columns=encoder.get_feature_names_out(['sexo']))
    dados = pd.concat([df.drop('sexo', axis=1), encoded_df], axis=1)

    dados_escalados = scaler.transform(dados)

    cluster = kmeans.predict(dados_escalados)

    return cluster

if up_file is not None:
    st.write("""
                ### Beschreibung der Gruppen:
                - **Gruppe 0**: Fokus auf ein junges Publikum mit starkem Interesse an Mode, Musik und Aussehen.
                - **Gruppe 1**: Stark assoziiert mit Sport, insbesondere American Football, Basketball und kulturellen Aktivitäten wie Band und Rockmusik.
                - **Gruppe 2**: Ausgewogener, mit Interessen an Musik, Tanz und Mode.
             """)
    
    df = pd.read_csv(up_file)
    cluster = processar_prever(df)
    df.insert(0, 'Gruppen', cluster) # Nome da coluna traduzido para 'Gruppen'

    st.write('Ergebnisansicht (die ersten 10 Datensätze):')
    st.write(df.head(10))

    csv = df.to_csv(index=False)
    # Botão de download
    st.download_button(
        label='Vollständige Ergebnisse herunterladen', 
        data=csv, 
        file_name='Interessengruppen.csv', 
        mime='text/csv'
    )

    
