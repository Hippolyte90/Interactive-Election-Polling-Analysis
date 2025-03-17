import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

def prepare_dataframe_for_graph(df):
    """
    Prépare le dataframe pour l'analyse et la visualisation en appliquant plusieurs transformations :
    - Supprime la colonne "polling_organization" si elle est présente.
    - Identifie et stocke les valeurs uniques des colonnes "identity_candidate_i" et "political_learning_candidate_i".
    - Remplace les valeurs des colonnes "identity_candidate_i" et "political_learning_candidate_i" par des valeurs numériques.
    - Agrège les données par "poll_date" en prenant la moyenne des valeurs numériques.
    - Remplace les valeurs numériques par leurs équivalents textuels d'origine.
    
    Args:
        df (pd.DataFrame): Le dataframe d'entrée.
    
    Returns:
        pd.DataFrame: Le dataframe transformé.
    """

    # Supprimer la colonne 'polling_organization' si elle existe
    if "polling_organization" in df.columns:
        df = df.drop(columns=["polling_organization"])

    # Trouver les colonnes correspondant aux identités des candidats
    identity_cols = [col for col in df.columns if col.lower().startswith("identity_candidate")]
    L = len(identity_cols)  # Nombre de candidats

    # Trouver les colonnes correspondant aux affiliations politiques des candidats
    political_cols = [col for col in df.columns if col.lower().startswith("political_learning_candidate")]

    # Création des dictionnaires de correspondance
    identity_dict = {col: df[col].iloc[0] for col in identity_cols}
    political_learning_dict = {col: df[col].iloc[0] for col in political_cols}

    # Remplacement des valeurs des colonnes identitaires et politiques par 0 et conversion en float
    for col in identity_cols + political_cols:
        df[col] = 0.0

    # Agrégation des données par "poll_date"
    df = df.groupby("poll_date", as_index=False).mean(numeric_only=True)

    # Remplacement des valeurs dans les colonnes identitaires et politiques par leurs équivalents textuels
    for col in identity_cols:
        df[col] = str(identity_dict[col])

    for col in political_cols:
        df[col] = str(political_learning_dict[col])

    return df


def organize_poll_data_and_sample(df):
    """
    - Vérifie la présence de la colonne 'sample_size' dans le DataFrame.
    - Convertit la colonne 'poll_date' au format datetime (mois/année).
    - Tronque la colonne 'sample_size' pour ne garder que la partie avant la virgule (partie entière).
    - Retourne le DataFrame modifié.
    """
    # 2. Conversion de 'poll_date' au format datetime
    #    Si poll_date est déjà en datetime, cette ligne ne posera pas de problème.
    #    Si poll_date est sous forme 'MM/YYYY', on utilise format='%m/%Y'
    df['poll_date'] = pd.to_datetime(df['poll_date'], format='%m/%Y', errors='coerce')
    
    # 3. Conserver uniquement la partie avant la virgule pour 'sample_size'
    #    Si 'sample_size' est déjà un float, un simple astype(int) suffit pour
    #    tronquer la partie décimale. Si c'est une chaîne, on remplace la virgule
    #    ou le point, puis on convertit.

     # 1. Vérification de la présence de 'sample_size'
    if 'sample_size' in df.columns:
        df['sample_size'] = (
        df['sample_size']
        .astype(str)                
        .str.replace(',', '.', regex=False)  
        .astype(float)             
        .astype(int) )
        
    # 1) Identifier les colonnes qui contiennent 'prediction' dans leur nom
    prediction_cols = [col for col in df.columns if 'prediction' in col]

    
    # 2) Supprimer les lignes pour lesquelles une de ces colonnes vaut 0.0
    df_filtre = df[(df[prediction_cols] != 0.0).all(axis=1)]

    df_filtre = df_filtre.sort_values(by='poll_date')
    # 4. Retour du DataFrame
    return df_filtre




# Charger les fichiers disponibles
data_files = {
    "UK": {
        "1997": "UK_1997_general_election.xlsx",
        "2005": "UK_2005_general_election.xlsx",
        "2010": "UK_2010_general_election.xlsx",
        "2015": "UK_2015_general_election.xlsx",
        "2017": "UK_2017_general_election.xlsx",
        "2019": "UK_2019_general_election.xlsx",
        "2024": "UK_2024_general_election.xlsx"
    }
}

# Interface utilisateur
st.title("📊 Évolution des prédictions avant les élections")

# Disposition en colonnes : 2 colonnes (Menu à gauche, Graphique à droite)
col1, col2 = st.columns([1, 3])

with col1:
    st.subheader("🔧 Paramètres")
    
    # Sélection du pays
    selected_country = st.selectbox("🌍 Sélectionnez un pays :", list(data_files.keys()))

    # Sélection de l'année des élections
    selected_year = st.selectbox("📅 Sélectionnez une année d'élection :", list(data_files[selected_country].keys()))

    # Charger les données
    file_path = f"{data_files[selected_country][selected_year]}"
    
    if os.path.exists(file_path):
        df = pd.read_excel(file_path)
        # Préparation et transformation des données pour le graphique
        df = prepare_dataframe_for_graph(df)
        # Convertir la colonne poll_date en format datetime
        df = organize_poll_data_and_sample(df)
        
        # Sélection de la période avec un calendrier interactif
        min_date = df["poll_date"].min()
        max_date = df["poll_date"].max()

        selected_dates = st.date_input(
            "📆 Sélectionnez une période :",
            [min_date, max_date],
            min_value=min_date,
            max_value=max_date
        )

        # Sélection du niveau de zoom avec un slider
        y_min, y_max = df[[col for col in df.columns if "prediction_result_candidate_" in col]].min().min(), df[[col for col in df.columns if "prediction_result_candidate_" in col]].max().max()
        zoom_level = st.slider("🔍 Zoom sur l'axe Y (%)", min_value= 0.0, max_value=float(y_max + 20), value=(0.0, float(y_max + 5)))

# Colonne 2 : Affichage du graphique
with col2:
    if os.path.exists(file_path):
        if len(selected_dates) == 2:
            start_date, end_date = pd.to_datetime(selected_dates)
            filtered_df = df[(df["poll_date"] >= start_date) & (df["poll_date"] <= end_date)]

            if filtered_df.empty:
                st.warning("⚠️ Aucune donnée disponible pour la période sélectionnée !")
            else:
                # Graphique des prédictions
                fig, ax = plt.subplots(figsize=(10, 5))
                
                candidate_columns = [col for col in df.columns if "prediction_result_candidate_" in col]
                identity_columns = [col for col in df.columns if "identity_candidate_" in col]

                for i, col in enumerate(candidate_columns):
                    candidate_name = df[identity_columns[i]].iloc[0] if identity_columns else f"Candidat {i+1}"
                    ax.plot(filtered_df["poll_date"], filtered_df[col], label=candidate_name)

                ax.set_xlabel("Date du sondage", fontsize = 12)
                ax.set_ylabel("Résultat de prédiction (%)", fontsize = 12)
                ax.set_ylim(zoom_level)  # Appliquer le zoom de l'utilisateur
                ax.grid(True, linestyle='--', alpha=0.7)
                ax.tick_params(axis='x', rotation=45)
                ax.legend()
                st.pyplot(fig)

                # Affichage des statistiques descriptives sous le graphique
                st.subheader("📊 Statistiques descriptives")
                st.write(filtered_df[candidate_columns].describe())

                # Option de téléchargement des données filtrées
                st.subheader("📥 Télécharger les données filtrées")
                csv = filtered_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📂 Télécharger en CSV",
                    data=csv,
                    file_name=f"predictions_{selected_year}_{selected_country}.csv",
                    mime="text/csv"
                )
    else:
        st.error("❌ Le fichier de données sélectionné n'existe pas.")
        
    
    
