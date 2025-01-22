import streamlit as st
from sklearn.svm import SVC
from sentence_transformers import SentenceTransformer
import pandas as pd
import matplotlib.pyplot as plt
from langdetect import detect
from joblib import load

# --- Charger le modèle SentenceTransformer ---
@st.cache_resource
def load_model():
    """Charge le modèle SentenceTransformer"""
    return SentenceTransformer('bert-base-nli-mean-tokens')

# --- Charger le modèle SVM ---
@st.cache_resource
def load_svm_model():
    """Charge le modèle SVM entraîné"""
    return load('svm_model.pkl')  # Chemin vers votre modèle SVM

# --- Charger les données ---
@st.cache_resource
def load_data():
    """Charge les données depuis un fichier Excel"""
    path = "avis_1_traduit.xlsx"  # Chemin vers votre fichier Excel
    df = pd.read_excel(path)
    df.columns = df.columns.str.lower().str.strip()  # Convertir en minuscules et enlever les espaces

    return df

# --- Fonction pour prédire une note ou un sentiment ---
def predict_category(review, transformer, svm_model):
    """Effectue une prédiction sur un avis donné"""
    embedding = transformer.encode([review])  # Génère l'embedding
    prediction = svm_model.predict(embedding)  # Prédiction avec le modèle SVM
    return prediction[0]

# --- Interface utilisateur Streamlit ---
def main():
    st.title("Streamlit - Prédiction d'avis")
    st.markdown("""
    ### Fonctionnalités :
    - Entrez un avis **en anglais** pour obtenir une prédiction.
    - Explorez les avis existants regroupés par assureur.
    - Affichez des statistiques et des prédictions pour des exemples issus du dataset.
    """)

    # Charger les données
    df = load_data()

    # --- Filtrer et afficher les avis par assureur ---
    st.sidebar.title("Filtres")
    assureurs = df["assureur"].unique()  # Remplacez 'assureur' par la colonne exacte
    selected_assureur = st.sidebar.selectbox("Sélectionnez un assureur :", ["Tous"] + list(assureurs))

    if selected_assureur == "Tous":
        filtered_df = df
    else:
        filtered_df = df[df["assureur"] == selected_assureur]

    # Créer deux colonnes : une pour le tableau, l'autre pour le graphique
    col1, col2 = st.columns(2)

# Afficher le tableau dans la première colonne
    with col1:
        st.write(f"### Avis pour l'assureur : {selected_assureur}")
        st.dataframe(filtered_df[["date_publication", "produit", "avis_en"]])  # Ajustez les colonnes pertinentes

# Afficher le graphique dans la deuxième colonne
    with col2:
        if "note" in filtered_df.columns:  # Vérifiez si la colonne 'note' existe
            st.write("### Distribution des notes pour l'assureur sélectionné")
            fig, ax = plt.subplots()
            filtered_df["note"].value_counts().sort_index().plot(kind="bar", ax=ax)
            ax.set_xlabel("Notes")
            ax.set_ylabel("Nombre d'avis")
            st.pyplot(fig)


    # --- Prédiction d'un avis ---
    st.markdown("### Prédiction d'un Avis")
    examples = filtered_df["avis_en"].head(10).tolist()  # Obtenez des exemples d'avis (en anglais)
    selected_example = st.selectbox("Sélectionnez un exemple d'avis :", [""] + examples)
    user_input = st.text_area("Entrez votre avis (en anglais) :", selected_example if selected_example else "")

    if st.button("Prédire"):
        if user_input.strip():
            try:
                # Vérifier la langue
                language = detect(user_input)
                if language != 'en':
                    st.warning("Veuillez entrer un avis rédigé en anglais.")
                else:
                    # Charger le modèle et effectuer la prédiction
                    transformer = load_model()
                    svm_model = load_svm_model()
                    prediction = predict_category(user_input, transformer, svm_model)
                    st.success(f"Le modèle prédit : **{prediction}**")
            except Exception as e:
                st.error(f"Erreur : {e}")
        else:
            st.warning("Veuillez entrer un avis avant de prédire.")

    # --- Prédictions pour des exemples existants ---
    if st.checkbox("Afficher les prédictions pour des exemples du dataset"):
        transformer = load_model()
        svm_model = load_svm_model()
        filtered_df["Predictions"] = filtered_df["avis_en"].apply(
            lambda x: svm_model.predict(transformer.encode([x]))[0]
        )
        st.write("### Comparaison entre les avis et les prédictions")
        st.dataframe(filtered_df[["assureur", "avis_en", "note", "Predictions"]])  # Ajustez les colonnes selon votre dataset


# --- Exécuter l'application ---
if __name__ == "__main__":
    main()
