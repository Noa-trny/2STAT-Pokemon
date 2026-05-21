import json

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Séance 1 : Analyse Descriptive du Dataset Pokémon\n",
                "\n",
                "Ce notebook a pour but de commencer l'étude du dataset **Pokemons.csv** (source : Kaggle) et de se familiariser concrètement avec les librairies Python les plus courantes en analyse de données. Nous utiliserons :\n",
                "- **pandas** pour charger, nettoyer et analyser les données.\n",
                "- **seaborn** et **matplotlib** pour les représentations graphiques.\n",
                "- **wordcloud** pour créer un nuage de mots représentant la fréquence des types de Pokémon.\n",
                "\n",
                "Chaque représentation graphique fera l'objet d'une mise en forme soignée (titre, nom des axes, taille de figure, etc.) et les résultats seront interprétés."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Imports des librairies requises\n",
                "import pandas as pd\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "from wordcloud import WordCloud\n",
                "\n",
                "# Configuration globale des représentations graphiques\n",
                "sns.set_theme(style=\"whitegrid\")\n",
                "plt.rcParams['figure.figsize'] = (10, 6)\n",
                "plt.rcParams['font.size'] = 11\n",
                "\n",
                "print(\"Librairies importées avec succès et prêtes à l'emploi !\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Chargement du Dataset et Exploration Initiale\n",
                "\n",
                "Nous utilisons la commande `read_csv` de pandas pour charger le fichier de données. Comme le fichier contient des caractères accentués (par exemple, le 'é' dans Pokémon ou des accents dans les noms de catégories), nous spécifions l'encodage `utf-8`."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Chargement du dataset\n",
                "df = pd.read_csv('Pokemons.csv', encoding='utf-8')\n",
                "\n",
                "# Dimensions du dataset\n",
                "n_obs, n_vars = df.shape\n",
                "print(f\"Dimensions du dataset : {n_obs} observations et {n_vars} variables.\")\n",
                "\n",
                "# Affichage des informations de base (noms des colonnes, types, valeurs non nulles)\n",
                "df.info()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 1.1 Valeurs Manquantes\n",
                "Identifions la quantité de valeurs manquantes par variable pour comprendre la qualité globale des données."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Recherche des valeurs manquantes\n",
                "valeurs_manquantes = df.isna().sum()\n",
                "valeurs_manquantes = valeurs_manquantes[valeurs_manquantes > 0].sort_values(ascending=False)\n",
                "\n",
                "if len(valeurs_manquantes) > 0:\n",
                "    print(\"Variables contenant des valeurs manquantes et leur proportion :\")\n",
                "    for col, count in valeurs_manquantes.items():\n",
                "        percentage = (count / len(df)) * 100\n",
                "        print(f\"- {col:18} : {count:3} valeurs manquantes ({percentage:.2f}%)\")\n",
                "else:\n",
                "    print(\"Aucune valeur manquante n'a été trouvée !\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Pré-traitement des Variables Qualitatives\n",
                "\n",
                "### 2.1 Remplacement des Variables de Génération Binaires\n",
                "Le dataset original contient 8 variables binaires (`is_generation_1` à `is_generation_8`). Nous allons les regrouper en une seule variable `generation` qui contiendra un entier de 1 à 8 désignant la génération à laquelle appartient chaque Pokémon, puis nous supprimerons ces 8 variables."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Liste des variables de génération binaires\n",
                "gen_cols = [f'is_generation_{i}' for i in range(1, 9)]\n",
                "\n",
                "# Vérification que chaque Pokémon appartient à exactement une seule génération\n",
                "appartient_a_une_gen = df[gen_cols].sum(axis=1)\n",
                "assert (appartient_a_une_gen == 1).all(), \"Certains Pokémon n'ont pas exactement une génération !\"\n",
                "\n",
                "# Création de la variable qualitative 'generation'\n",
                "# idxmax(axis=1) extrait le nom de la colonne de génération qui vaut True pour chaque ligne\n",
                "df['generation'] = df[gen_cols].idxmax(axis=1).str.replace('is_generation_', '').astype(int)\n",
                "\n",
                "# Suppression des anciennes colonnes binaires\n",
                "df = df.drop(columns=gen_cols)\n",
                "\n",
                "print(\"Les 8 variables binaires ont été supprimées et remplacées par la variable qualitative 'generation'.\")\n",
                "print(\"Fréquences par génération :\")\n",
                "print(df['generation'].value_counts().sort_index())"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 2.2 Scission de la Variable `types`\n",
                "La variable `types` contient des chaînes de la forme `'type_1, type_2'` ou `'type_1'`. Nous allons la scinder en 3 nouvelles variables :\n",
                "- `type_number` : nombre de types (1 ou 2).\n",
                "- `type_1` : premier type du Pokémon.\n",
                "- `type_2` : second type du Pokémon (ou `NaN` s'il n'existe pas)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Scission de la variable 'types'\n",
                "split_types = df['types'].str.split(', ')\n",
                "\n",
                "# Création des nouvelles variables\n",
                "df['type_number'] = split_types.apply(len)\n",
                "df['type_1'] = split_types.str[0]\n",
                "df['type_2'] = split_types.str[1] # Rempli par NaN si le second type n'existe pas\n",
                "\n",
                "print(\"Variable 'types' scindée avec succès !\")\n",
                "print(df[['name', 'types', 'type_number', 'type_1', 'type_2']].head(5))"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 2.3 Scission de la Variable `egg_types`\n",
                "De la même façon, la variable `egg_types` contient les types d'œufs. Nous la scindons en 3 variables :\n",
                "- `egg_type_number` : nombre de groupes d'œufs (0, 1 ou 2).\n",
                "- `egg_type_1` : premier groupe d'œufs.\n",
                "- `egg_type_2` : second groupe d'œufs (ou `NaN` si inexistant).\n",
                "\n",
                "Note : Nous gérons ici proprement les 3 valeurs manquantes (`NaN`) de la colonne `egg_types` d'origine pour lesquelles le nombre de types d'œufs doit être 0."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Nombre de types d'oeufs (0 si NaN, sinon nombre de types séparés par des virgules)\n",
                "df['egg_type_number'] = df['egg_types'].apply(lambda x: 0 if pd.isna(x) else len(str(x).split(', ')))\n",
                "\n",
                "# Scission en type 1 et type 2\n",
                "split_eggs = df['egg_types'].str.split(', ')\n",
                "df['egg_type_1'] = split_eggs.apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else pd.NA)\n",
                "df['egg_type_2'] = split_eggs.apply(lambda x: x[1] if isinstance(x, list) and len(x) > 1 else pd.NA)\n",
                "\n",
                "print(\"Variable 'egg_types' scindée avec succès !\")\n",
                "# Afficher des exemples avec 2 types d'oeufs, 1 seul, et aucun (NaN d'origine)\n",
                "indices_exemples = [0, 13, 1040]\n",
                "print(df[['name', 'egg_types', 'egg_type_number', 'egg_type_1', 'egg_type_2']].iloc[indices_exemples])"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 2.4 Autres Pré-traitements à Effectuer\n",
                "\n",
                "#### A. Nettoyage de `growth_rate`\n",
                "La variable `growth_rate` possède une unique valeur manquante (`NaN`) pour le Pokémon *Galarian Darmanitan Zen Mode*. Ses autres formes standard et de Galar possédant une vitesse de croissance de type `\"Medium Slow\"`, nous imputons de manière logique cette valeur à la ligne manquante."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Remplacement de la valeur manquante pour Galarian Darmanitan Zen Mode\n",
                "df.loc[df['name'] == 'Galarian Darmanitan Zen Mode', 'growth_rate'] = 'Medium Slow'\n",
                "print(\"Valeurs manquantes restantes dans growth_rate :\", df['growth_rate'].isna().sum())"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "#### B. Conversion de variables quantitatives chargées sous forme de texte (`height` et `weight`)\n",
                "Les colonnes `height` et `weight` sont chargées comme chaînes de caractères (ex: `2'04''` pour la hauteur, `15.21 lbs` pour le poids). Bien qu'il s'agisse de variables quantitatives, leur format textuel empêche tout calcul statistique. Nous allons les convertir en variables numériques réelles :\n",
                "- `height_m` (taille en mètres, calculée à partir des pieds et pouces: 1 pied = 0.3048m, 1 pouce = 0.0254m)\n",
                "- `weight_kg` (poids en kilogrammes: 1 lb = 0.45359237 kg)\n",
                "\n",
                "Nous supprimons aussi la colonne `Unnamed: 0` qui sert simplement de doublon d'index."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Conversion de height (ex: 2'04'') en mètres\n",
                "def convert_height(h_str):\n",
                "    if pd.isna(h_str):\n",
                "        return np.nan\n",
                "    try:\n",
                "        # Retirer les doubles guillemets de la fin et couper au niveau du simple guillemet\n",
                "        parts = str(h_str).replace(\"''\", \"\").split(\"'\")\n",
                "        feet = int(parts[0])\n",
                "        inches = int(parts[1])\n",
                "        meters = feet * 0.3048 + inches * 0.0254\n",
                "        return round(meters, 2)\n",
                "    except Exception:\n",
                "        return np.nan\n",
                "\n",
                "df['height_m'] = df['height'].apply(convert_height)\n",
                "\n",
                "# Conversion de weight (ex: 15.21 lbs) en kg\n",
                "def convert_weight(w_str):\n",
                "    if pd.isna(w_str):\n",
                "        return np.nan\n",
                "    try:\n",
                "        val_lbs = float(str(w_str).replace(\" lbs\", \"\"))\n",
                "        return round(val_lbs * 0.45359237, 2)\n",
                "    except Exception:\n",
                "        return np.nan\n",
                "\n",
                "df['weight_kg'] = df['weight'].apply(convert_weight)\n",
                "\n",
                "# Suppression de la colonne d'index inutile Unnamed: 0\n",
                "if 'Unnamed: 0' in df.columns:\n",
                "    df = df.drop(columns=['Unnamed: 0'])\n",
                "\n",
                "print(\"Conversion effectuée avec succès !\")\n",
                "print(df[['name', 'height', 'height_m', 'weight', 'weight_kg']].head(5))"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Étude des Principales Variables Qualitatives\n",
                "\n",
                "Nous allons analyser le profil des variables qualitatives principales :\n",
                "1. `generation` (Génération du Pokémon)\n",
                "2. `type_1` (Premier type de combat)\n",
                "3. `status` (Rareté : Normal, Sub Legendary, Legendary, Mythical)\n",
                "4. `growth_rate` (Vitesse d'apprentissage/croissance)\n",
                "\n",
                "Pour chacune d'elles, nous allons générer la table des effectifs, la table des fréquences relatives (pourcentages), et déterminer le mode."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def analyse_qualitative(column, label):\n",
                "    # Table des effectifs\n",
                "    effectifs = df[column].value_counts(dropna=False)\n",
                "    # Table des fréquences (%)\n",
                "    frequences = df[column].value_counts(normalize=True, dropna=False) * 100\n",
                "    \n",
                "    # Création du DataFrame bilan\n",
                "    table = pd.DataFrame({\n",
                "        'Effectifs': effectifs,\n",
                "        'Fréquences (%)': frequences.round(2)\n",
                "    })\n",
                "    \n",
                "    mode = df[column].mode()[0]\n",
                "    \n",
                "    print(f\"===================================================\")\n",
                "    print(f\" ANALYSE QUALITATIVE : {label.upper()} (Variable: {column})\")\n",
                "    print(f\"===================================================\")\n",
                "    print(table)\n",
                "    print(f\"Mode : {mode}\\n\")\n",
                "    return table\n",
                "\n",
                "stats_gen = analyse_qualitative('generation', 'Génération')\n",
                "stats_type1 = analyse_qualitative('type_1', 'Premier Type')\n",
                "stats_status = analyse_qualitative('status', 'Statut de rareté')\n",
                "stats_growth = analyse_qualitative('growth_rate', 'Vitesse de croissance')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 3.1 Visualisation des Distributions\n",
                "\n",
                "#### A. Répartition par Génération\n",
                "Nous utilisons un diagramme en barres Seaborn personnalisé."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Configuration de la figure\n",
                "plt.figure(figsize=(10, 6))\n",
                "\n",
                "# Création du diagramme en barres avec seaborn\n",
                "ax = sns.countplot(\n",
                "    data=df, \n",
                "    x='generation', \n",
                "    hue='generation', \n",
                "    palette='viridis', \n",
                "    legend=False\n",
                ")\n",
                "\n",
                "# Personnalisation des axes et du titre\n",
                "ax.set_title(\"Distribution des Pokémon par Génération\", fontsize=15, fontweight='bold', pad=15)\n",
                "ax.set_xlabel(\"Génération\", fontsize=12, labelpad=10)\n",
                "ax.set_ylabel(\"Nombre de Pokémon (Effectifs)\", fontsize=12, labelpad=10)\n",
                "\n",
                "# Configuration des graduations (ticks)\n",
                "ax.set_xticks(range(8))\n",
                "ax.set_xticklabels([f\"Génération {i}\" for i in range(1, 9)], rotation=0)\n",
                "\n",
                "# Affichage des valeurs au-dessus de chaque barre pour plus de clarté\n",
                "for container in ax.containers:\n",
                "    ax.bar_label(container, fmt='%d', label_type='edge', padding=3)\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "#### B. Répartition du Premier Type\n",
                "Nous utilisons un diagramme en barres horizontal, trié par effectif décroissant, pour faciliter la comparaison des types."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "plt.figure(figsize=(12, 8))\n",
                "\n",
                "# Définition de l'ordre d'affichage (décroissant selon les effectifs)\n",
                "order_types = df['type_1'].value_counts().index\n",
                "\n",
                "ax = sns.countplot(\n",
                "    data=df, \n",
                "    y='type_1', \n",
                "    order=order_types, \n",
                "    hue='type_1', \n",
                "    palette='Set2', \n",
                "    legend=False\n",
                ")\n",
                "\n",
                "ax.set_title(\"Distribution du Premier Type des Pokémon\", fontsize=15, fontweight='bold', pad=15)\n",
                "ax.set_xlabel(\"Nombre de Pokémon\", fontsize=12, labelpad=10)\n",
                "ax.set_ylabel(\"Premier Type\", fontsize=12, labelpad=10)\n",
                "\n",
                "# Ajout des étiquettes de valeurs\n",
                "for container in ax.containers:\n",
                "    ax.bar_label(container, fmt='%d', padding=5)\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "#### C. Statut de Rareté des Pokémon\n",
                "La variable `status` possède peu de modalités distinctes. Un diagramme circulaire (pie chart) est particulièrement adapté pour montrer la part de chaque statut."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "plt.figure(figsize=(8, 8))\n",
                "\n",
                "counts = df['status'].value_counts()\n",
                "labels = counts.index\n",
                "sizes = counts.values\n",
                "colors = ['#5dade2', '#f4d03f', '#eb984e', '#af7ac5']\n",
                "explode = (0.05, 0.1, 0.1, 0.1) # Légère mise en valeur des statuts rares\n",
                "\n",
                "plt.pie(\n",
                "    sizes, \n",
                "    labels=labels, \n",
                "    autopct='%1.1f%%', \n",
                "    startangle=140, \n",
                "    colors=colors, \n",
                "    explode=explode, \n",
                "    textprops={'fontsize': 11, 'weight': 'bold'},\n",
                "    wedgeprops={'edgecolor': 'white', 'linewidth': 1.5}\n",
                ")\n",
                "\n",
                "plt.title(\"Proportion des Pokémon selon leur Statut\", fontsize=16, fontweight='bold', pad=20)\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "#### D. Vitesse de Croissance\n",
                "Nous représentons les différentes vitesses de croissance à l'aide d'un diagramme circulaire."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "plt.figure(figsize=(8, 8))\n",
                "\n",
                "counts_growth = df['growth_rate'].value_counts()\n",
                "labels_growth = counts_growth.index\n",
                "sizes_growth = counts_growth.values\n",
                "colors_growth = sns.color_palette('pastel', len(labels_growth))\n",
                "\n",
                "plt.pie(\n",
                "    sizes_growth, \n",
                "    labels=labels_growth, \n",
                "    autopct='%1.1f%%', \n",
                "    startangle=90, \n",
                "    colors=colors_growth,\n",
                "    textprops={'fontsize': 11},\n",
                "    wedgeprops={'edgecolor': 'white', 'linewidth': 1}\n",
                ")\n",
                "\n",
                "plt.title(\"Proportion des Pokémon selon la Vitesse de Croissance\", fontsize=16, fontweight='bold', pad=20)\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 3.2 Interprétation des Résultats\n",
                "\n",
                "- **Génération (Mode = Génération 1, 151 Pokémon)** : Les générations de Pokémon présentent des tailles variées. La génération 1 (le mode de cette variable) est la plus représentée du fait de la présence de nombreuses formes alternatives et méga-évolutions rattachées à la première génération. Les générations 3, 5 et 8 sont également très fournies (plus de 140 Pokémon chacune). À l'inverse, la génération 6 est la plus réduite avec seulement 85 Pokémon.\n",
                "- **Premier Type (Mode = Water, 134 Pokémon)** : Le type Eau (`Water`) est le plus fréquent, suivi des types `Normal` (115) et `Grass` (Plante, 94). Les types les plus rares en tant que type principal sont le type Vol (`Flying`, 8) et Glace (`Ice`, 38).\n",
                "- **Statut de Rareté (Mode = Normal, 87.1 %)** : Une écrasante majorité des Pokémon du dataset appartient au statut `Normal` (910 individus). Les Pokémon exceptionnels de type `Sub Legendary` (6.2 %), `Legendary` (4.4 %) et `Mythical` (2.3 %) constituent environ 13 % de la population globale.\n",
                "- **Vitesse de Croissance (Mode = Medium Fast, 41.1 %)** : La majorité des Pokémon possède des vitesses de croissance modérées (`Medium Fast` à 41.1 % et `Medium Slow` à 33.6 %). Les profils de croissance atypiques comme `Erratic` (2.5 %) ou `Fluctuating` (2.8 %) sont très minoritaires."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Nuage de Mots (Word Cloud) des Types de Pokémon\n",
                "\n",
                "Pour représenter la fréquence des différents types sous une forme esthétique de nuage de mots, nous allons cumuler les apparitions de chaque type à la fois dans la variable `type_1` et dans `type_2` (pour les Pokémon possédant deux types). Cela nous donnera une image globale et fidèle de la représentativité de chaque type de Pokémon dans tout le Pokédex."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Regroupement des types de Pokémon (on omet les valeurs manquantes de type_2)\n",
                "t1_list = df['type_1'].dropna().tolist()\n",
                "t2_list = df['type_2'].dropna().tolist()\n",
                "t_total = t1_list + t2_list\n",
                "\n",
                "# Comptage des fréquences absolues\n",
                "from collections import Counter\n",
                "frequences_types = Counter(t_total)\n",
                "\n",
                "# Génération du nuage de mots\n",
                "# Le colormap 'tab10' ou 'Set1' offre une variété de couleurs adaptées à l'univers Pokémon\n",
                "wordcloud = WordCloud(\n",
                "    width=900, \n",
                "    height=550, \n",
                "    background_color='white', \n",
                "    colormap='Set1', \n",
                "    max_words=30, \n",
                "    prefer_horizontal=0.85,\n",
                "    random_state=42\n",
                ").generate_from_frequencies(frequences_types)\n",
                "\n",
                "# Visualisation graphique\n",
                "plt.figure(figsize=(11, 7))\n",
                "plt.imshow(wordcloud, interpolation='bilinear')\n",
                "plt.axis('off') # On désactive la graduation des axes pour l'esthétique du nuage\n",
                "plt.title(\"Nuage de Mots des Types de Pokémon (Cumul Type 1 & Type 2)\", fontsize=16, fontweight='bold', pad=15)\n",
                "\n",
                "# Sauvegarde de l'image du nuage de mots\n",
                "plt.savefig('pokemon_types_wordcloud.png', dpi=300, bbox_inches='tight')\n",
                "plt.show()\n",
                "\n",
                "# Affichage trié des fréquences totales cumulées\n",
                "print(\"Fréquences totales cumulées par type (type_1 + type_2) :\")\n",
                "for t, count in frequences_types.most_common():\n",
                "    print(f\"- {t:10} : {count:3} occurrences\")"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3 (ipykernel)",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

# Écriture du fichier notebook .ipynb
with open('Pokemons_Analysis.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)

print("Notebook 'Pokemons_Analysis.ipynb' créé avec succès !")
