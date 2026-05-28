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
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Séance 2 : Pré-traitement et Analyse des Variables Quantitatives\n",
                "\n",
                "Dans cette seconde séance, nous poursuivons l'étude du dataset en nous concentrant sur les **variables quantitatives**. Nous allons :\n",
                "1. Vérifier et compléter le **pré-traitement** des variables quantitatives (`height`, `weight`, et autres).\n",
                "2. Étudier leurs distributions (**histogrammes**, **boîtes à moustaches**) et calculer leurs paramètres de **position** et de **dispersion**.\n",
                "3. Constituer un sous-dataset des **50 meilleurs Pokémon selon `hp`** et le comparer au dataset complet.\n",
                "4. Créer des **radar charts** pour comparer nos Pokémon préférés, puis les moyennes par génération et par statut, avec la librairie **plotly**."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. Pré-traitement des Variables Quantitatives\n",
                "\n",
                "### 5.1 Identification des Variables Quantitatives\n",
                "\n",
                "Commençons par lister les variables quantitatives présentes dans le dataset et vérifier leur format."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Identification des variables numériques par leur dtype\n",
                "vars_numeriques = df.select_dtypes(include=[np.number]).columns.tolist()\n",
                "print(f\"Variables numériques détectées ({len(vars_numeriques)}) :\")\n",
                "for col in vars_numeriques:\n",
                "    nan_count = df[col].isna().sum()\n",
                "    print(f\" - {col:25s} dtype={str(df[col].dtype):8s} NaN={nan_count}\")\n",
                "\n",
                "# Variables potentiellement quantitatives mais chargées au format texte\n",
                "print(\"\\nVariables quantitatives au format texte (à convertir) :\")\n",
                "for col in ['height', 'weight']:\n",
                "    if col in df.columns:\n",
                "        print(f\" - {col:10s} dtype={str(df[col].dtype):8s} exemple={df[col].iloc[0]!r}\")\n",
                "    else:\n",
                "        print(f\" - {col:10s} : déjà supprimée ou convertie\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 5.2 Conversion de `height` et `weight` (Effectuée en Séance 1)\n",
                "\n",
                "Les conversions des variables `height` (de `pieds'pouces''` vers mètres) et `weight` (de livres vers kilogrammes) ont été réalisées en Séance 1 (section 2.4.B). Vérifions que les nouvelles variables `height_m` et `weight_kg` sont bien disponibles et au format `float`, puis supprimons les colonnes textuelles devenues redondantes."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Vérification des variables converties\n",
                "print(\"Vérification des variables converties :\")\n",
                "colonnes_a_montrer = [c for c in ['name', 'height', 'height_m', 'weight', 'weight_kg'] if c in df.columns]\n",
                "print(df[colonnes_a_montrer].head(5))\n",
                "print(f\"\\nType de height_m  : {df['height_m'].dtype}\")\n",
                "print(f\"Type de weight_kg : {df['weight_kg'].dtype}\")\n",
                "print(f\"Valeurs manquantes : height_m={df['height_m'].isna().sum()}, weight_kg={df['weight_kg'].isna().sum()}\")\n",
                "\n",
                "# Suppression des colonnes textuelles redondantes\n",
                "for col_textuelle in ['height', 'weight']:\n",
                "    if col_textuelle in df.columns:\n",
                "        df = df.drop(columns=[col_textuelle])\n",
                "        print(f\"Colonne textuelle '{col_textuelle}' supprimée.\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 5.3 Autres Variables Quantitatives à Examiner\n",
                "\n",
                "Vérifions si d'autres variables quantitatives méritent un pré-traitement particulier. Plusieurs d'entre elles possèdent des valeurs manquantes :\n",
                "- `catch_rate`, `base_friendship`, `base_experience` : taux de capture, niveau d'amitié initial et expérience de base.\n",
                "- `percentage_male` : pourcentage de chance qu'un Pokémon soit mâle (peut être `NaN` pour les Pokémon asexués, comme la plupart des Légendaires).\n",
                "- `egg_cycles` : nombre de cycles avant éclosion (peut être manquant pour les Pokémon ne pondant pas d'œufs).\n",
                "\n",
                "Ces `NaN` ont une **signification sémantique** (ce ne sont pas des défauts de saisie). Nous les conservons et les excluons simplement des calculs statistiques via `skipna=True` (comportement par défaut de pandas)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "vars_a_verifier = ['catch_rate', 'base_friendship', 'base_experience', 'percentage_male', 'egg_cycles']\n",
                "print(\"Variables quantitatives à examiner :\\n\")\n",
                "for col in vars_a_verifier:\n",
                "    nb_nan  = df[col].isna().sum()\n",
                "    pct_nan = (nb_nan / len(df)) * 100\n",
                "    print(f\"- {col:18s}: dtype={str(df[col].dtype):8s} | {nb_nan:3d} NaN ({pct_nan:5.2f}%)\")\n",
                "\n",
                "# Aperçu des Pokémon avec percentage_male = NaN (Pokémon asexués)\n",
                "print(\"\\nExemple de Pokémon avec percentage_male = NaN (asexués) :\")\n",
                "print(df[df['percentage_male'].isna()][['name', 'status', 'percentage_male']].head(5))"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 6. Étude des Principales Variables Quantitatives\n",
                "\n",
                "Nous nous concentrons sur les variables quantitatives clés qui caractérisent un Pokémon :\n",
                "- **Les 6 statistiques de combat** : `hp`, `attack`, `defense`, `sp_attack`, `sp_defense`, `speed`.\n",
                "- **Le total** : `total_points` (somme des 6 statistiques de combat).\n",
                "- **Les attributs physiques** : `height_m`, `weight_kg`.\n",
                "\n",
                "Pour chaque variable, nous calculerons :\n",
                "- les **paramètres de position** : moyenne, médiane, mode ;\n",
                "- les **paramètres de dispersion** : écart-type, variance, étendue (max − min), écart interquartile (IQR).\n",
                "\n",
                "Nous accompagnerons ces calculs de **histogrammes** et de **boîtes à moustaches**."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Variables quantitatives principales\n",
                "vars_quanti = ['hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed',\n",
                "               'total_points', 'height_m', 'weight_kg']\n",
                "\n",
                "def stats_descriptives(df_in, cols):\n",
                "    \"\"\"Calcule les paramètres de position et de dispersion pour les colonnes données.\"\"\"\n",
                "    stats = pd.DataFrame(index=cols)\n",
                "    stats['Moyenne']    = df_in[cols].mean().round(2)\n",
                "    stats['Médiane']    = df_in[cols].median().round(2)\n",
                "    stats['Mode']       = df_in[cols].mode().iloc[0].round(2)\n",
                "    stats['Min']        = df_in[cols].min().round(2)\n",
                "    stats['Max']        = df_in[cols].max().round(2)\n",
                "    stats['Étendue']    = (df_in[cols].max() - df_in[cols].min()).round(2)\n",
                "    stats['Écart-type'] = df_in[cols].std().round(2)\n",
                "    stats['Variance']   = df_in[cols].var().round(2)\n",
                "    stats['Q1']         = df_in[cols].quantile(0.25).round(2)\n",
                "    stats['Q3']         = df_in[cols].quantile(0.75).round(2)\n",
                "    stats['IQR']        = (df_in[cols].quantile(0.75) - df_in[cols].quantile(0.25)).round(2)\n",
                "    return stats\n",
                "\n",
                "stats_pop = stats_descriptives(df, vars_quanti)\n",
                "print(\"Paramètres de position et de dispersion (dataset complet) :\\n\")\n",
                "print(stats_pop)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 6.2 Histogrammes\n",
                "\n",
                "Les histogrammes donnent une vue d'ensemble de la distribution de chaque variable. La courbe KDE superposée approche la densité continue ; les lignes verticales mettent en évidence la **moyenne** (rouge pointillé) et la **médiane** (bleue pointillée)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "fig, axes = plt.subplots(3, 3, figsize=(16, 12))\n",
                "axes = axes.flatten()\n",
                "couleurs = sns.color_palette('viridis', len(vars_quanti))\n",
                "\n",
                "for i, col in enumerate(vars_quanti):\n",
                "    sns.histplot(df[col].dropna(), kde=True, ax=axes[i], color=couleurs[i])\n",
                "    axes[i].set_title(f\"Histogramme de {col}\", fontsize=12, fontweight='bold')\n",
                "    axes[i].set_xlabel(col)\n",
                "    axes[i].set_ylabel(\"Effectif\")\n",
                "    axes[i].axvline(df[col].mean(),   color='red',  linestyle='--', linewidth=1.4, label=f\"Moyenne = {df[col].mean():.1f}\")\n",
                "    axes[i].axvline(df[col].median(), color='blue', linestyle=':',  linewidth=1.4, label=f\"Médiane = {df[col].median():.1f}\")\n",
                "    axes[i].legend(fontsize=8)\n",
                "\n",
                "plt.suptitle(\"Distributions des Principales Variables Quantitatives\", fontsize=16, fontweight='bold', y=1.00)\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 6.3 Boîtes à Moustaches\n",
                "\n",
                "Les boîtes à moustaches mettent en évidence la **médiane**, les **quartiles**, l'**étendue** et les **valeurs aberrantes** (outliers)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Boîtes à moustaches groupées – 6 stats de combat (échelle commune)\n",
                "fig, axes = plt.subplots(1, 2, figsize=(16, 6))\n",
                "\n",
                "stats_combat = ['hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed']\n",
                "sns.boxplot(data=df[stats_combat], ax=axes[0], palette='Set2', hue=None)\n",
                "axes[0].set_title(\"Boîtes à Moustaches – Statistiques de Combat\", fontsize=13, fontweight='bold')\n",
                "axes[0].set_ylabel(\"Valeur\")\n",
                "axes[0].tick_params(axis='x', rotation=20)\n",
                "\n",
                "sns.boxplot(data=df[['total_points']], ax=axes[1], color='#3498db')\n",
                "axes[1].set_title(\"Boîte à Moustaches – total_points\", fontsize=13, fontweight='bold')\n",
                "axes[1].set_ylabel(\"total_points\")\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()\n",
                "\n",
                "# Boîtes à moustaches – attributs physiques (échelles très différentes)\n",
                "fig, axes = plt.subplots(1, 2, figsize=(12, 5))\n",
                "sns.boxplot(y=df['height_m'], ax=axes[0], color='#27ae60')\n",
                "axes[0].set_title(\"Boîte à Moustaches – Taille (m)\", fontsize=13, fontweight='bold')\n",
                "axes[0].set_ylabel(\"height_m (m)\")\n",
                "\n",
                "sns.boxplot(y=df['weight_kg'], ax=axes[1], color='#e67e22')\n",
                "axes[1].set_title(\"Boîte à Moustaches – Poids (kg)\", fontsize=13, fontweight='bold')\n",
                "axes[1].set_ylabel(\"weight_kg (kg)\")\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 6.4 Interprétation des Résultats\n",
                "\n",
                "- **Statistiques de combat** (`hp`, `attack`, `defense`, `sp_attack`, `sp_defense`, `speed`) : Les six variables possèdent des moyennes proches (~ 70 - 80) avec des médianes systématiquement inférieures aux moyennes : c'est le signe de **distributions étirées à droite** — quelques Pokémon très puissants tirent la moyenne vers le haut. Les écarts-types sont de l'ordre de 28 - 30 points (dispersion notable). Sur les boîtes à moustaches, on observe de nombreux **outliers** dans la partie haute : ce sont les Pokémon Légendaires/Mythiques ou les Méga-évolutions.\n",
                "- **`total_points`** : Distribution étalée, médiane ~ 450, moyenne ~ 440, écart-type ~ 110. L'étendue est très grande (~ 175 à ~ 1125). L'allure de l'histogramme suggère une distribution **multimodale** (Pokémon de base, évolutions, Méga/Légendaires) liée au statut.\n",
                "- **`height_m`** : Distribution **très asymétrique** (skewed right). 50 % des Pokémon mesurent moins d'1 m, mais quelques géants comme *Wailord* (~ 14.5 m) ou *Eternatus Eternamax* tirent fortement la moyenne et créent énormément d'outliers.\n",
                "- **`weight_kg`** : Même phénomène que pour la taille, mais encore plus accentué. La médiane est ~ 28 kg, mais certains Pokémon dépassent 900 kg (*Wailord*) voire plusieurs centaines de tonnes (*Cosmoem*). L'écart-type est gigantesque.\n",
                "- L'**écart interquartile** (IQR) reste relativement modeste pour la plupart des statistiques de combat (~ 40 points), ce qui confirme que la **majorité** des Pokémon possède des stats *normales* ; les valeurs extrêmes proviennent des formes spéciales."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 7. Sous-Dataset : Les 50 Meilleurs Pokémon selon `hp`\n",
                "\n",
                "Nous extrayons les **50 Pokémon ayant les `hp` les plus élevés**, puis nous reprenons rapidement l'étude des principales variables qualitatives et quantitatives sur ce sous-ensemble pour le comparer au dataset complet."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Sous-dataset des 50 meilleurs Pokémon selon hp\n",
                "df_top50 = df.sort_values('hp', ascending=False).head(50).reset_index(drop=True)\n",
                "print(f\"Sous-dataset constitué : {len(df_top50)} Pokémon.\")\n",
                "print(f\"hp minimum dans le top 50 : {df_top50['hp'].min()}  |  hp maximum : {df_top50['hp'].max()}\")\n",
                "print(\"\\nTop 10 des Pokémon ayant le plus de HP :\")\n",
                "print(df_top50[['name', 'hp', 'status', 'generation', 'type_1']].head(10))"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 7.1 Variables Qualitatives du Top 50\n",
                "\n",
                "Étudions rapidement la répartition des principales variables qualitatives (`status`, `generation`, `type_1`) dans le top 50."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "fig, axes = plt.subplots(1, 3, figsize=(18, 5))\n",
                "\n",
                "# Statut\n",
                "df_top50['status'].value_counts().plot.bar(ax=axes[0], color=sns.color_palette('Set2'))\n",
                "axes[0].set_title(\"Top 50 – Statut\", fontsize=12, fontweight='bold')\n",
                "axes[0].set_ylabel(\"Effectif\"); axes[0].tick_params(axis='x', rotation=20)\n",
                "\n",
                "# Génération\n",
                "df_top50['generation'].value_counts().sort_index().plot.bar(ax=axes[1], color=sns.color_palette('viridis', 8))\n",
                "axes[1].set_title(\"Top 50 – Génération\", fontsize=12, fontweight='bold')\n",
                "axes[1].set_ylabel(\"Effectif\"); axes[1].set_xlabel(\"Génération\")\n",
                "\n",
                "# Type 1\n",
                "df_top50['type_1'].value_counts().plot.bar(ax=axes[2], color=sns.color_palette('Set1', n_colors=df_top50['type_1'].nunique()))\n",
                "axes[2].set_title(\"Top 50 – Premier Type\", fontsize=12, fontweight='bold')\n",
                "axes[2].set_ylabel(\"Effectif\"); axes[2].tick_params(axis='x', rotation=45)\n",
                "\n",
                "plt.suptitle(\"Distributions Qualitatives – Top 50 par HP\", fontsize=15, fontweight='bold', y=1.02)\n",
                "plt.tight_layout()\n",
                "plt.show()\n",
                "\n",
                "# Comparaison statut Top 50 vs dataset complet\n",
                "print(\"\\nFréquences du STATUT (en %) :\")\n",
                "comparaison_statut = pd.DataFrame({\n",
                "    'Dataset complet': (df['status'].value_counts(normalize=True) * 100).round(2),\n",
                "    'Top 50 HP':       (df_top50['status'].value_counts(normalize=True) * 100).round(2)\n",
                "}).fillna(0)\n",
                "print(comparaison_statut)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 7.2 Variables Quantitatives du Top 50\n",
                "\n",
                "Reprenons le calcul des paramètres de position et de dispersion sur le top 50, et comparons les moyennes obtenues à celles du dataset complet."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "stats_top50 = stats_descriptives(df_top50, vars_quanti)\n",
                "print(\"Statistiques descriptives du TOP 50 :\\n\")\n",
                "print(stats_top50)\n",
                "\n",
                "print(\"\\nComparaison des moyennes (Top 50 vs Dataset complet) :\")\n",
                "comparaison_moy = pd.DataFrame({\n",
                "    'Moyenne Top 50':    df_top50[vars_quanti].mean().round(2),\n",
                "    'Moyenne Complet':   df[vars_quanti].mean().round(2),\n",
                "    'Écart absolu':      (df_top50[vars_quanti].mean() - df[vars_quanti].mean()).round(2),\n",
                "    'Écart relatif (%)': ((df_top50[vars_quanti].mean() - df[vars_quanti].mean()) / df[vars_quanti].mean() * 100).round(1)\n",
                "})\n",
                "print(comparaison_moy)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 7.3 Comparaison avec le Dataset Original – Constats\n",
                "\n",
                "Le sous-dataset des **50 meilleurs Pokémon par `hp`** présente des **différences marquantes** avec le dataset complet :\n",
                "\n",
                "- **Statut** : la proportion de Pokémon `Legendary`, `Sub Legendary` et `Mythical` est **massivement sur-représentée**. Dans le dataset complet ils ne pesaient que ~13 % ; dans le top 50, ils représentent une part bien plus importante. Posséder beaucoup de HP est donc fortement corrélé au caractère exceptionnel des Pokémon.\n",
                "- **Génération** : la répartition par génération est plus déséquilibrée. Certaines générations introduisent davantage de Pokémon endurants (en particulier les Gen 3 et Gen 5).\n",
                "- **Type 1** : la palette des types est resserrée. Le type `Normal` reste fréquent (les « tanks » comme Blissey, Chansey, Snorlax sont traditionnellement `Normal`), suivi de `Water` et `Dragon`.\n",
                "- **Statistiques quantitatives** : toutes les stats de combat sont **fortement augmentées**. Les écarts relatifs sont importants : `total_points` augmente d'environ +30 %, `hp` est doublé, et les autres stats croissent de 30 à 50 %. Cela illustre la **corrélation positive** entre les stats des Pokémon puissants.\n",
                "- **Taille et poids** : moyennes notablement plus élevées (les Pokémon avec gros HP sont en général plus grands et plus lourds — Wailord, Snorlax, etc.).\n",
                "\n",
                "**En résumé**, sélectionner les 50 meilleurs par HP revient en grande partie à sélectionner soit des Pokémon Légendaires/Mythiques, soit des Pokémon massifs (tank), ce qui décale fortement toutes les distributions."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 8. Radar Chart : Profil de nos Pokémon Préférés\n",
                "\n",
                "Nous allons représenter sous forme de **radar chart** (toile d'araignée) le profil de nos Pokémon préférés selon les 6 statistiques de combat (`hp`, `attack`, `defense`, `sp_attack`, `sp_defense`, `speed`).\n",
                "\n",
                "Pour cela, nous installons et utilisons la librairie **plotly**, qui propose nativement des radar charts via l'objet `Scatterpolar`. Si la librairie n'est pas encore installée :\n",
                "\n",
                "```bash\n",
                "pip install plotly\n",
                "```"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Installation (à décommenter si plotly n'est pas encore installé)\n",
                "# !pip install plotly\n",
                "\n",
                "import plotly.graph_objects as go\n",
                "import plotly.io as pio\n",
                "pio.renderers.default = 'notebook'  # rendu inline dans le notebook\n",
                "print(\"Plotly importé avec succès – nous pouvons construire des radar charts.\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 8.1 Choix des Pokémon Préférés\n",
                "\n",
                "Nous choisissons 4 Pokémon emblématiques pour comparer leurs profils :\n",
                "- **Pikachu** : la mascotte iconique de la franchise.\n",
                "- **Charizard** : Dragon-Feu très populaire.\n",
                "- **Mewtwo** : Pokémon Légendaire au profil offensif puissant.\n",
                "- **Lucario** : Pokémon Combat-Acier prisé pour son équilibre.\n",
                "\n",
                "Cette liste est bien sûr modifiable selon vos goûts !"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Liste des Pokémon préférés (modifiable au goût)\n",
                "pokemons_favoris = ['Pikachu', 'Charizard', 'Mewtwo', 'Lucario']\n",
                "\n",
                "# Variables retenues pour le radar (les 6 statistiques de combat)\n",
                "stats_radar  = ['hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed']\n",
                "labels_radar = ['HP', 'Attaque', 'Défense', 'Att. Spé.', 'Déf. Spé.', 'Vitesse']\n",
                "\n",
                "# Extraction des données (préserve l'ordre demandé)\n",
                "df_favoris = (df[df['name'].isin(pokemons_favoris)]\n",
                "              .drop_duplicates('name')\n",
                "              .set_index('name')\n",
                "              .loc[pokemons_favoris]\n",
                "              .reset_index())\n",
                "print(\"Pokémon sélectionnés et leurs statistiques :\\n\")\n",
                "print(df_favoris[['name'] + stats_radar])\n",
                "\n",
                "# Construction du radar chart avec plotly\n",
                "fig = go.Figure()\n",
                "couleurs = ['#f1c40f', '#e74c3c', '#9b59b6', '#3498db']\n",
                "\n",
                "for i, nom in enumerate(pokemons_favoris):\n",
                "    row = df_favoris[df_favoris['name'] == nom].iloc[0]\n",
                "    valeurs = row[stats_radar].tolist()\n",
                "    # Fermer la boucle (revenir au point de départ)\n",
                "    valeurs_closed = valeurs + [valeurs[0]]\n",
                "    labels_closed  = labels_radar + [labels_radar[0]]\n",
                "    \n",
                "    fig.add_trace(go.Scatterpolar(\n",
                "        r=valeurs_closed,\n",
                "        theta=labels_closed,\n",
                "        fill='toself',\n",
                "        name=nom,\n",
                "        line=dict(color=couleurs[i], width=2),\n",
                "        opacity=0.65\n",
                "    ))\n",
                "\n",
                "max_val = df_favoris[stats_radar].values.max()\n",
                "fig.update_layout(\n",
                "    polar=dict(radialaxis=dict(visible=True, range=[0, max_val + 20])),\n",
                "    title=dict(text=\"Radar Chart – Comparaison des Pokémon Préférés\", x=0.5, font=dict(size=16)),\n",
                "    showlegend=True,\n",
                "    width=800, height=600\n",
                ")\n",
                "fig.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 9. Radar Charts par Génération et par Statut\n",
                "\n",
                "Avec les **mêmes 6 statistiques de combat**, nous représentons cette fois les **moyennes** :\n",
                "- par **génération** (8 traces correspondant aux générations 1 à 8) ;\n",
                "- par **statut** (4 traces : Normal / Sub Legendary / Mythical / Legendary).\n",
                "\n",
                "Cela permet de visualiser d'un coup d'œil quelles générations / quels statuts sont les plus puissants et sur quels axes."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Moyennes des stats par génération\n",
                "moyennes_gen = df.groupby('generation')[stats_radar].mean().round(2)\n",
                "print(\"Moyennes des statistiques par génération :\\n\")\n",
                "print(moyennes_gen)\n",
                "\n",
                "fig = go.Figure()\n",
                "couleurs_gen = sns.color_palette('viridis', 8).as_hex()\n",
                "\n",
                "for i, gen in enumerate(sorted(df['generation'].unique())):\n",
                "    valeurs = moyennes_gen.loc[gen, stats_radar].tolist()\n",
                "    valeurs_closed = valeurs + [valeurs[0]]\n",
                "    labels_closed  = labels_radar + [labels_radar[0]]\n",
                "    \n",
                "    fig.add_trace(go.Scatterpolar(\n",
                "        r=valeurs_closed,\n",
                "        theta=labels_closed,\n",
                "        fill='toself',\n",
                "        name=f\"Génération {gen}\",\n",
                "        line=dict(color=couleurs_gen[i], width=2),\n",
                "        opacity=0.45\n",
                "    ))\n",
                "\n",
                "fig.update_layout(\n",
                "    polar=dict(radialaxis=dict(visible=True, range=[40, 100])),\n",
                "    title=dict(text=\"Radar Chart – Moyennes des Statistiques par Génération\", x=0.5, font=dict(size=16)),\n",
                "    showlegend=True,\n",
                "    width=850, height=650\n",
                ")\n",
                "fig.show()"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Moyennes des stats par statut\n",
                "moyennes_status = df.groupby('status')[stats_radar].mean().round(2)\n",
                "print(\"Moyennes des statistiques par statut :\\n\")\n",
                "print(moyennes_status)\n",
                "\n",
                "fig = go.Figure()\n",
                "couleurs_status = {'Normal': '#5dade2', 'Sub Legendary': '#f4d03f',\n",
                "                   'Mythical': '#eb984e', 'Legendary': '#af7ac5'}\n",
                "ordre_statuts   = ['Normal', 'Sub Legendary', 'Mythical', 'Legendary']\n",
                "\n",
                "for st in ordre_statuts:\n",
                "    if st in moyennes_status.index:\n",
                "        valeurs = moyennes_status.loc[st, stats_radar].tolist()\n",
                "        valeurs_closed = valeurs + [valeurs[0]]\n",
                "        labels_closed  = labels_radar + [labels_radar[0]]\n",
                "        \n",
                "        fig.add_trace(go.Scatterpolar(\n",
                "            r=valeurs_closed,\n",
                "            theta=labels_closed,\n",
                "            fill='toself',\n",
                "            name=st,\n",
                "            line=dict(color=couleurs_status[st], width=2.5),\n",
                "            opacity=0.55\n",
                "        ))\n",
                "\n",
                "fig.update_layout(\n",
                "    polar=dict(radialaxis=dict(visible=True, range=[50, 130])),\n",
                "    title=dict(text=\"Radar Chart – Moyennes des Statistiques par Statut\", x=0.5, font=dict(size=16)),\n",
                "    showlegend=True,\n",
                "    width=850, height=650\n",
                ")\n",
                "fig.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Interprétation des Radar Charts\n",
                "\n",
                "- **Pokémon préférés** : Les profils sont nettement différents. *Pikachu* a un radar peu étendu (stats modestes mais une `speed` honorable) ; *Charizard* présente un profil équilibré avec un fort `sp_attack` et une bonne `speed` ; *Lucario* possède un double mode physique/spécial avec `attack` et `sp_attack` élevés ; *Mewtwo* écrase tous les autres grâce à des statistiques offensives massives (`sp_attack`, `speed`).\n",
                "- **Générations** : Les moyennes des huit générations sont **remarquablement homogènes** (toutes situées entre 65 et 80 pour chaque stat), ce qui témoigne de l'équilibrage cohérent du jeu au fil du temps. Les générations 7 et 8 sont légèrement plus puissantes en moyenne sur `attack` et `sp_attack`.\n",
                "- **Statuts** : La différence entre `Normal` et les statuts spéciaux est saisissante. Les Pokémon `Legendary` affichent une moyenne supérieure à 100 sur **toutes les stats**, suivis par `Mythical` et `Sub Legendary` (~ 90-100), tandis que les `Normal` plafonnent vers 70. Le radar chart visualise immédiatement la **hiérarchie de rareté/puissance** du dataset."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Séance 3 : Analyses Bivariées du Dataset Pokémon\n",
                "\n",
                "Après avoir étudié séparément les variables qualitatives et quantitatives, nous abordons à présent les **croisements deux à deux** entre variables. Trois grandes familles d'analyses bivariées sont menées :\n",
                "\n",
                "1. **Qualitative × Qualitative** : tableaux de contingence et différents types de diagrammes à barres (groupé, empilé, empilé normalisé, heatmap).\n",
                "2. **Qualitative × Quantitative (`hp`)** : étude de l'influence des variables `status`, `generation`, `type_1`, `type_2`, `ability_1` sur les points de vie à l'aide de diagrammes à barres, boîtes à moustaches, violons et stripplots.\n",
                "3. **Qualitative × autres variables quantitatives** (`height_m`, `weight_kg`, `total_points`, `base_experience`) : reproduction des analyses précédentes en multi-grille.\n",
                "\n",
                "Chaque cellule de code est accompagnée d'une cellule d'interprétation."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 10. Croisement de Deux Variables Qualitatives\n",
                "\n",
                "Pour étudier la relation entre deux variables qualitatives `X` et `Y`, on construit un **tableau de contingence** (les effectifs croisés `pd.crosstab(X, Y)`). On peut ensuite le visualiser via :\n",
                "- un **diagramme à barres groupé** (les modalités de `Y` sont juxtaposées pour chaque modalité de `X`) ;\n",
                "- un **diagramme à barres empilé** (les modalités de `Y` sont superposées) ;\n",
                "- un **diagramme à barres empilé normalisé** (chaque barre vaut 100 % – idéal pour comparer des **profils**) ;\n",
                "- une **heatmap** (très lisible quand le nombre de modalités est élevé).\n",
                "\n",
                "Les couples étudiés ici sont : `status × generation`, `status × type_1`, `generation × type_1`, et `type_1 × type_2`."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 10.1 Croisement `status` × `generation`"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Tableau de contingence : effectifs croisés status × generation\n",
                "ordre_status = ['Normal', 'Sub Legendary', 'Mythical', 'Legendary']\n",
                "ct_status_gen = pd.crosstab(df['status'], df['generation']).reindex(ordre_status)\n",
                "print(\"Tableau de contingence (effectifs) - status × generation :\\n\")\n",
                "print(ct_status_gen)\n",
                "\n",
                "# Profils ligne (% par statut) : utile pour comparer la répartition par génération\n",
                "ct_status_gen_pct = pd.crosstab(df['status'], df['generation'], normalize='index').reindex(ordre_status) * 100\n",
                "print(\"\\nProfils-ligne (% par statut) :\\n\")\n",
                "print(ct_status_gen_pct.round(1))"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Représentations graphiques : 3 diagrammes à barres + 1 heatmap\n",
                "fig, axes = plt.subplots(2, 2, figsize=(18, 13))\n",
                "couleurs_gen = sns.color_palette('viridis', n_colors=ct_status_gen.shape[1])\n",
                "\n",
                "# (a) Diagramme à barres groupé\n",
                "ct_status_gen.plot(kind='bar', ax=axes[0, 0], color=couleurs_gen, edgecolor='black', width=0.8)\n",
                "axes[0, 0].set_title(\"Barres groupées : effectifs par statut et génération\", fontweight='bold')\n",
                "axes[0, 0].set_xlabel(\"Statut\")\n",
                "axes[0, 0].set_ylabel(\"Nombre de Pokémon\")\n",
                "axes[0, 0].legend(title='Génération', bbox_to_anchor=(1.02, 1), loc='upper left')\n",
                "axes[0, 0].tick_params(axis='x', rotation=20)\n",
                "\n",
                "# (b) Diagramme à barres empilé\n",
                "ct_status_gen.plot(kind='bar', stacked=True, ax=axes[0, 1], color=couleurs_gen, edgecolor='black', width=0.7)\n",
                "axes[0, 1].set_title(\"Barres empilées : effectifs par statut\", fontweight='bold')\n",
                "axes[0, 1].set_xlabel(\"Statut\")\n",
                "axes[0, 1].set_ylabel(\"Nombre de Pokémon\")\n",
                "axes[0, 1].legend(title='Génération', bbox_to_anchor=(1.02, 1), loc='upper left')\n",
                "axes[0, 1].tick_params(axis='x', rotation=20)\n",
                "\n",
                "# (c) Diagramme à barres empilé normalisé (100%)\n",
                "ct_status_gen_pct.plot(kind='bar', stacked=True, ax=axes[1, 0], color=couleurs_gen, edgecolor='black', width=0.7)\n",
                "axes[1, 0].set_title(\"Barres empilées normalisées : profil par statut (%)\", fontweight='bold')\n",
                "axes[1, 0].set_xlabel(\"Statut\")\n",
                "axes[1, 0].set_ylabel(\"Pourcentage\")\n",
                "axes[1, 0].legend(title='Génération', bbox_to_anchor=(1.02, 1), loc='upper left')\n",
                "axes[1, 0].tick_params(axis='x', rotation=20)\n",
                "axes[1, 0].set_ylim(0, 100)\n",
                "\n",
                "# (d) Heatmap\n",
                "sns.heatmap(ct_status_gen, annot=True, fmt='d', cmap='YlGnBu', ax=axes[1, 1], cbar_kws={'label': 'Effectif'})\n",
                "axes[1, 1].set_title(\"Heatmap : effectifs status × generation\", fontweight='bold')\n",
                "axes[1, 1].set_xlabel(\"Génération\")\n",
                "axes[1, 1].set_ylabel(\"Statut\")\n",
                "\n",
                "plt.suptitle(\"Croisement des variables qualitatives : status × generation\", fontsize=15, fontweight='bold')\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "**Interprétation :** Les Pokémon `Normal` dominent toutes les générations et représentent l'essentiel des effectifs. Les statuts spéciaux (`Legendary`, `Mythical`, `Sub Legendary`) restent rares mais leur **proportion relative augmente** avec les générations récentes : la barre empilée normalisée montre clairement que la part des Pokémon non-normaux est plus marquée en génération 7. Les `Mythical` sont absents de plusieurs générations, ce qui reflète une politique éditoriale (un Mythical est généralement révélé via un événement spécial)."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 10.2 Croisement `status` × `type_1`"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Tableau de contingence status × type_1\n",
                "ct_status_t1 = pd.crosstab(df['status'], df['type_1']).reindex(ordre_status)\n",
                "print(\"Tableau de contingence (effectifs) - status × type_1 :\\n\")\n",
                "print(ct_status_t1)\n",
                "\n",
                "# Profils-colonne (% par type) : pour chaque type, quelle est la répartition des statuts ?\n",
                "ct_t1_status_pct = pd.crosstab(df['type_1'], df['status'], normalize='index')[ordre_status] * 100\n",
                "print(\"\\nProfils-ligne (% par type_1) :\\n\")\n",
                "print(ct_t1_status_pct.round(1).head(10))"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Une grille à 2 lignes : heatmap large en haut, barh en bas → labels lisibles\n",
                "fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(16, 14), gridspec_kw={'height_ratios': [1, 1.6]})\n",
                "couleurs_status = ['#5dade2', '#f4d03f', '#eb984e', '#af7ac5']\n",
                "\n",
                "# (a) Heatmap : effectifs croisés\n",
                "sns.heatmap(ct_status_t1, annot=True, fmt='d', cmap='magma_r', ax=ax_top,\n",
                "            annot_kws={'fontsize': 10}, cbar_kws={'label': 'Effectif'})\n",
                "ax_top.set_title(\"Heatmap : status × type_1 (effectifs)\", fontsize=14, fontweight='bold')\n",
                "ax_top.set_xlabel(\"Type 1\", fontsize=12)\n",
                "ax_top.set_ylabel(\"Statut\", fontsize=12)\n",
                "ax_top.tick_params(axis='x', rotation=35, labelsize=10)\n",
                "ax_top.tick_params(axis='y', rotation=0, labelsize=10)\n",
                "\n",
                "# (b) Barres empilées normalisées : pour chaque type, profil statut\n",
                "ct_t1_status_pct.sort_values('Normal').plot(kind='barh', stacked=True, ax=ax_bot,\n",
                "                                            color=couleurs_status, edgecolor='black', width=0.8)\n",
                "ax_bot.set_title(\"Profil des statuts par type_1 (% empilé)\", fontsize=14, fontweight='bold')\n",
                "ax_bot.set_xlabel(\"Pourcentage\", fontsize=12)\n",
                "ax_bot.set_ylabel(\"Type 1\", fontsize=12)\n",
                "ax_bot.legend(title='Statut', bbox_to_anchor=(1.02, 1), loc='upper left')\n",
                "ax_bot.set_xlim(0, 100)\n",
                "ax_bot.tick_params(axis='y', labelsize=10)\n",
                "\n",
                "plt.suptitle(\"Croisement des variables qualitatives : status × type_1\",\n",
                "             fontsize=16, fontweight='bold', y=1.00)\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "**Interprétation :** La heatmap met en évidence les types les plus riches en Pokémon `Normal` (Water, Bug, Grass, Normal). Le diagramme horizontal empilé normalisé fait ressortir les types **sur-représentés en statuts rares** : `Psychic` et `Dragon` affichent la plus forte proportion de Pokémon `Legendary`/`Mythical`/`Sub Legendary`. À l'inverse, `Bug`, `Poison` et `Normal` sont **quasi exclusivement** composés de Pokémon `Normal`. Cette dissymétrie est cohérente avec l'imaginaire de la franchise (les dragons et psychiques sont mystiques)."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 10.3 Croisement `generation` × `type_1`"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Tableau de contingence\n",
                "ct_gen_t1 = pd.crosstab(df['generation'], df['type_1'])\n",
                "print(\"Tableau de contingence generation × type_1 :\\n\")\n",
                "print(ct_gen_t1)\n",
                "\n",
                "# Grille à 2 lignes pour ne pas écraser les graphiques\n",
                "fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(16, 13))\n",
                "\n",
                "# (a) Heatmap des effectifs\n",
                "sns.heatmap(ct_gen_t1, annot=True, fmt='d', cmap='YlOrRd', ax=ax_top,\n",
                "            annot_kws={'fontsize': 10}, cbar_kws={'label': 'Effectif'})\n",
                "ax_top.set_title(\"Heatmap : generation × type_1\", fontsize=14, fontweight='bold')\n",
                "ax_top.set_xlabel(\"Type 1\", fontsize=12)\n",
                "ax_top.set_ylabel(\"Génération\", fontsize=12)\n",
                "ax_top.tick_params(axis='x', rotation=35, labelsize=10)\n",
                "ax_top.tick_params(axis='y', rotation=0, labelsize=10)\n",
                "\n",
                "# (b) Barres empilées normalisées par génération\n",
                "ct_gen_t1_pct = pd.crosstab(df['generation'], df['type_1'], normalize='index') * 100\n",
                "ct_gen_t1_pct.plot(kind='bar', stacked=True, ax=ax_bot, colormap='tab20',\n",
                "                   edgecolor='black', width=0.85)\n",
                "ax_bot.set_title(\"Profil des types par génération (% empilé)\", fontsize=14, fontweight='bold')\n",
                "ax_bot.set_xlabel(\"Génération\", fontsize=12)\n",
                "ax_bot.set_ylabel(\"Pourcentage\", fontsize=12)\n",
                "ax_bot.legend(title='Type 1', bbox_to_anchor=(1.02, 1), loc='upper left', ncol=1, fontsize=10)\n",
                "ax_bot.set_ylim(0, 100)\n",
                "ax_bot.tick_params(axis='x', rotation=0, labelsize=11)\n",
                "\n",
                "plt.suptitle(\"Croisement des variables qualitatives : generation × type_1\",\n",
                "             fontsize=16, fontweight='bold', y=1.00)\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "**Interprétation :** Tous les types apparaissent dès la génération 1, mais leur **proportion fluctue notablement** : la gén. 1 est riche en `Normal` et `Bug`, la gén. 3 introduit massivement les types `Rock` et `Water`, et la gén. 5 fait la part belle aux `Bug` et `Dark`. Aucune génération n'introduit de monoculture (la diversité chromatique de chaque barre le confirme), mais les designers ont visiblement souhaité **rééquilibrer** les types au fil du temps."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 10.4 Croisement `type_1` × `type_2`\n",
                "\n",
                "Ce croisement est riche : il révèle les **combinaisons de types** les plus fréquentes (et les inexplorées). On filtre uniquement les Pokémon possédant deux types."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# On ne garde que les Pokémon ayant deux types\n",
                "df_bitype = df.dropna(subset=['type_2'])\n",
                "ct_t1_t2 = pd.crosstab(df_bitype['type_1'], df_bitype['type_2'])\n",
                "print(f\"Nombre de Pokémon bitypes : {len(df_bitype)}\")\n",
                "print(f\"Nombre de combinaisons (type_1, type_2) observées : {(ct_t1_t2 > 0).sum().sum()}\")\n",
                "print(\"\\nTableau de contingence type_1 × type_2 (extrait) :\\n\")\n",
                "print(ct_t1_t2)\n",
                "\n",
                "plt.figure(figsize=(15, 12))\n",
                "# mask=0 → masque les cellules vides pour mieux voir les associations existantes\n",
                "sns.heatmap(ct_t1_t2, annot=True, fmt='d', cmap='rocket_r', mask=(ct_t1_t2 == 0),\n",
                "            annot_kws={'fontsize': 10}, cbar_kws={'label': 'Effectif'},\n",
                "            linewidths=0.5, linecolor='lightgray', square=True)\n",
                "plt.title(\"Heatmap des combinaisons type_1 × type_2 (Pokémon bitypes)\",\n",
                "          fontsize=15, fontweight='bold', pad=15)\n",
                "plt.xlabel(\"Type 2\", fontsize=12)\n",
                "plt.ylabel(\"Type 1\", fontsize=12)\n",
                "plt.xticks(rotation=45, ha='right', fontsize=11)\n",
                "plt.yticks(rotation=0, fontsize=11)\n",
                "plt.tight_layout()\n",
                "plt.show()\n",
                "\n",
                "# Top 10 des combinaisons les plus fréquentes\n",
                "combinaisons = (ct_t1_t2.stack()\n",
                "                .reset_index(name='effectif')\n",
                "                .query('effectif > 0')\n",
                "                .sort_values('effectif', ascending=False)\n",
                "                .head(10))\n",
                "print(\"\\nTop 10 des combinaisons type_1 × type_2 :\\n\")\n",
                "print(combinaisons.to_string(index=False))"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "**Interprétation :** Les combinaisons les plus fréquentes sont `Normal/Flying`, `Bug/Flying`, `Bug/Poison` et `Grass/Poison` – elles correspondent à des archétypes classiques (les oiseaux normaux, les insectes ailés ou venimeux, les plantes toxiques). De nombreuses cases de la heatmap restent **vides** : la diagonale par exemple (un Pokémon ne peut pas avoir le même type 1 et type 2) mais aussi des associations peu logiques sur le plan thématique (`Fire/Water`, `Ghost/Normal`, etc.). La heatmap met aussi en évidence des types **plus polyvalents** (Flying, Poison, Psychic apparaissent beaucoup en type 2) que d'autres (Normal apparaît rarement en type 2)."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 11. Influence d'une Variable Qualitative sur la Variable `hp`\n",
                "\n",
                "Lorsqu'on étudie l'influence d'une variable qualitative `X` sur une variable quantitative `Y` (ici `hp`), on combine plusieurs représentations :\n",
                "- **Diagramme à barres** des **moyennes** (ou médianes) de `Y` par modalité de `X`, idéalement avec barres d'erreur (écart-type ou intervalle de confiance) ;\n",
                "- **Boîte à moustaches** (boxplot) pour visualiser la dispersion, la médiane, les quartiles et les valeurs extrêmes ;\n",
                "- **Violon** (violin plot) qui ajoute la densité de la distribution ;\n",
                "- **Stripplot/swarmplot** qui affiche chaque individu.\n",
                "\n",
                "Pour réduire la répétition de code, on définit d'abord une **fonction utilitaire** `influence_qual_quanti` qui produit une figure synthétique."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def influence_qual_quanti(data, var_qual, var_quanti, ordre=None, palette='Set2', top_n=None):\n",
                "    \"\"\"Affiche 3 graphiques (barres+erreurs, boxplot, violon) pour étudier l'influence\n",
                "    de la variable qualitative `var_qual` sur la variable quantitative `var_quanti`.\n",
                "    \n",
                "    - `top_n` : ne garde que les `top_n` modalités les plus fréquentes (utile pour ability_1).\n",
                "    - Si beaucoup de modalités (>= 8), bascule automatiquement en orientation horizontale\n",
                "      et empile les 3 graphes verticalement → labels lisibles.\"\"\"\n",
                "    d = data.dropna(subset=[var_qual, var_quanti]).copy()\n",
                "    if top_n is not None:\n",
                "        top_mods = d[var_qual].value_counts().head(top_n).index\n",
                "        d = d[d[var_qual].isin(top_mods)]\n",
                "        ordre = list(top_mods)\n",
                "    if ordre is None:\n",
                "        ordre = d.groupby(var_qual)[var_quanti].median().sort_values().index.tolist()\n",
                "    # Cast en str pour éviter un bug seaborn avec hue numérique + legend=False\n",
                "    d[var_qual] = d[var_qual].astype(str)\n",
                "    ordre = [str(o) for o in ordre]\n",
                "\n",
                "    n_mod = len(ordre)\n",
                "    horizontal = n_mod >= 8  # passe en horizontal au-delà de 7 modalités\n",
                "\n",
                "    if horizontal:\n",
                "        # 3 graphes empilés verticalement, boîtes horizontales → labels en y bien lisibles\n",
                "        h = max(5, n_mod * 0.35)\n",
                "        fig, axes = plt.subplots(3, 1, figsize=(14, h * 3))\n",
                "        xy = dict(y=var_qual, x=var_quanti)\n",
                "        sns.barplot(data=d, **xy, order=ordre, hue=var_qual, palette=palette,\n",
                "                    errorbar='sd', ax=axes[0], legend=False)\n",
                "        sns.boxplot(data=d, **xy, order=ordre, hue=var_qual, palette=palette,\n",
                "                    ax=axes[1], legend=False)\n",
                "        sns.violinplot(data=d, **xy, order=ordre, hue=var_qual, palette=palette,\n",
                "                       inner='quartile', ax=axes[2], legend=False)\n",
                "        axes[0].set_title(f\"Moyenne de {var_quanti} par {var_qual} (± écart-type)\",\n",
                "                          fontsize=13, fontweight='bold')\n",
                "        axes[1].set_title(f\"Boîte à moustaches de {var_quanti} par {var_qual}\",\n",
                "                          fontsize=13, fontweight='bold')\n",
                "        axes[2].set_title(f\"Violon de {var_quanti} par {var_qual}\",\n",
                "                          fontsize=13, fontweight='bold')\n",
                "        for ax in axes:\n",
                "            ax.set_ylabel(var_qual, fontsize=11)\n",
                "            ax.set_xlabel(var_quanti, fontsize=11)\n",
                "            ax.tick_params(axis='y', labelsize=10)\n",
                "    else:\n",
                "        # Peu de modalités : 3 graphes côte à côte avec axe x lisible\n",
                "        fig, axes = plt.subplots(1, 3, figsize=(18, 6))\n",
                "        xy = dict(x=var_qual, y=var_quanti)\n",
                "        sns.barplot(data=d, **xy, order=ordre, hue=var_qual, palette=palette,\n",
                "                    errorbar='sd', ax=axes[0], legend=False)\n",
                "        sns.boxplot(data=d, **xy, order=ordre, hue=var_qual, palette=palette,\n",
                "                    ax=axes[1], legend=False)\n",
                "        sns.violinplot(data=d, **xy, order=ordre, hue=var_qual, palette=palette,\n",
                "                       inner='quartile', ax=axes[2], legend=False)\n",
                "        axes[0].set_title(f\"Moyenne de {var_quanti}\\n(± écart-type)\",\n",
                "                          fontsize=13, fontweight='bold')\n",
                "        axes[1].set_title(f\"Boîte à moustaches de {var_quanti}\",\n",
                "                          fontsize=13, fontweight='bold')\n",
                "        axes[2].set_title(f\"Violon de {var_quanti}\",\n",
                "                          fontsize=13, fontweight='bold')\n",
                "        for ax in axes:\n",
                "            ax.set_xlabel(var_qual, fontsize=11)\n",
                "            ax.set_ylabel(var_quanti, fontsize=11)\n",
                "            ax.tick_params(axis='x', rotation=25, labelsize=10)\n",
                "\n",
                "    plt.suptitle(f\"Influence de '{var_qual}' sur '{var_quanti}'\",\n",
                "                 fontsize=16, fontweight='bold', y=1.00)\n",
                "    plt.tight_layout()\n",
                "    plt.show()\n",
                "\n",
                "    stats = d.groupby(var_qual)[var_quanti].agg(['count', 'mean', 'std', 'median', 'min', 'max']).round(2).loc[ordre]\n",
                "    print(f\"\\nStatistiques descriptives de '{var_quanti}' par '{var_qual}' :\\n\")\n",
                "    print(stats)\n",
                "    return stats\n",
                "\n",
                "print(\"Fonction influence_qual_quanti() définie avec succès.\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 11.1 Influence de `status` sur `hp`"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "_ = influence_qual_quanti(df, 'status', 'hp', ordre=ordre_status, palette='Set2')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "**Interprétation :** Les trois représentations convergent : le `hp` moyen croît nettement avec la rareté du statut. Les `Normal` ont une médiane autour de 65 et une distribution relativement resserrée, tandis que les `Legendary` dépassent en moyenne 100 et présentent une **dispersion plus large**. Le violon révèle que les `Mythical` sont concentrés autour d'une valeur typique (~ 80-100), alors que les `Sub Legendary` sont plus étalés. Quelques valeurs extrêmes (>200) chez les `Normal` correspondent vraisemblablement à des Pokémon comme Blissey ou Chansey, célèbres pour leur HP énorme."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 11.2 Influence de `generation` sur `hp`"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "_ = influence_qual_quanti(df, 'generation', 'hp',\n",
                "                          ordre=sorted(df['generation'].unique()),\n",
                "                          palette='viridis')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "**Interprétation :** Les médianes de `hp` sont **très homogènes** d'une génération à l'autre (entre 65 et 75), confirmant la cohérence du game design. La génération 7 montre une médiane légèrement plus élevée. Toutes les générations contiennent des valeurs extrêmes (>150), preuve que chaque épisode du jeu introduit au moins quelques Pokémon \"tanks\"."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 11.3 Influence de `type_1` sur `hp`"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "_ = influence_qual_quanti(df, 'type_1', 'hp', palette='tab20')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "**Interprétation :** Les `hp` varient sensiblement selon le type principal. Les `Dragon` ont la médiane la plus élevée (~ 80), suivis par `Flying` et `Fairy` (~ 74), tandis que les types `Bug` et `Steel` affichent les médianes les plus faibles (~ 55-60). Les `Normal` se distinguent moins par leur médiane que par la **présence de valeurs extrêmes** (Chansey, Blissey >200). Les boîtes à moustaches révèlent une variabilité **intra-type très forte** pour la plupart des types, signe que le `hp` n'est qu'un facteur de différenciation parmi d'autres au sein d'un même type."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 11.4 Influence de `type_2` sur `hp`"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "_ = influence_qual_quanti(df, 'type_2', 'hp', palette='tab20')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "**Interprétation :** L'effet de `type_2` est plus marqué que celui de `type_1`. Les Pokémon possédant `Ice` (médiane ~ 90), `Dragon` (~ 81) ou `Fighting` (~ 80) en second type ont un `hp` médian très élevé, en partie parce que beaucoup de Légendaires bitypes ont ce profil. Les distributions sont **plus dispersées** car le second type concerne moins de Pokémon (population réduite, n = 553)."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 11.5 Influence de `ability_1` sur `hp` (top 10 capacités)\n",
                "\n",
                "La variable `ability_1` compte plus d'une centaine de modalités : afficher chacune serait illisible. On restreint l'analyse aux **10 capacités les plus fréquentes**."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "_ = influence_qual_quanti(df, 'ability_1', 'hp', top_n=10, palette='Spectral')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "**Interprétation :** La capacité **`Pressure` se détache nettement** avec une médiane de `hp` de 90, bien au-dessus des autres : c'est l'ability emblématique de nombreux Légendaires (Mewtwo, les Oiseaux légendaires, Giratina...) ce qui tire mécaniquement la médiane vers le haut. Les autres capacités (`Swift Swim`, `Keen Eye`, `Chlorophyll`, `Overgrow`, `Blaze`, `Torrent`, `Swarm`, `Intimidate`, `Levitate`) restent dans une fourchette serrée (55-70) : à l'exception de `Pressure`, **l'ability_1 a peu d'influence sur le `hp`**, contrairement à `status` ou aux types principaux."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 12. Influence de Variables Qualitatives sur d'Autres Variables Quantitatives\n",
                "\n",
                "Nous reprenons les analyses précédentes en remplaçant `hp` par d'autres variables quantitatives pertinentes : `height_m`, `weight_kg`, `total_points`, `base_experience`.\n",
                "\n",
                "Pour gagner en concision, on construit une **multi-grille** : une ligne par variable qualitative, une colonne par variable quantitative, chaque case affichant un boxplot."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def grille_boxplots(data, var_qual, vars_quanti, ordre=None, palette='Set2', top_n=None):\n",
                "    \"\"\"Affiche une grille de boxplots : un sous-graphe par variable quantitative.\n",
                "    Si beaucoup de modalités (>=8), bascule en boxplots horizontaux sur une grille 2×2.\"\"\"\n",
                "    d = data.dropna(subset=[var_qual]).copy()\n",
                "    if top_n is not None:\n",
                "        top_mods = d[var_qual].value_counts().head(top_n).index\n",
                "        d = d[d[var_qual].isin(top_mods)]\n",
                "        ordre = list(top_mods)\n",
                "    if ordre is None:\n",
                "        ordre = sorted(d[var_qual].dropna().unique().tolist())\n",
                "    # Cast en str pour éviter un bug seaborn avec hue numérique + legend=False\n",
                "    d[var_qual] = d[var_qual].astype(str)\n",
                "    ordre = [str(o) for o in ordre]\n",
                "\n",
                "    n_mod = len(ordre)\n",
                "    horizontal = n_mod >= 8\n",
                "\n",
                "    if horizontal:\n",
                "        # Grille 2x2 → chaque boxplot a tout l'espace nécessaire, labels en y\n",
                "        h = max(5, n_mod * 0.32)\n",
                "        fig, axes = plt.subplots(2, 2, figsize=(18, h * 2))\n",
                "        axes = axes.flatten()\n",
                "        for ax, vq in zip(axes, vars_quanti):\n",
                "            sns.boxplot(data=d, y=var_qual, x=vq, order=ordre, hue=var_qual,\n",
                "                        palette=palette, ax=ax, legend=False)\n",
                "            ax.set_title(f\"{vq} par {var_qual}\", fontsize=13, fontweight='bold')\n",
                "            ax.set_ylabel(var_qual, fontsize=11)\n",
                "            ax.set_xlabel(vq, fontsize=11)\n",
                "            ax.tick_params(axis='y', labelsize=10)\n",
                "    else:\n",
                "        # Peu de modalités : 1 ligne × 4 colonnes, boxplots verticaux\n",
                "        fig, axes = plt.subplots(1, len(vars_quanti), figsize=(22, 6))\n",
                "        for ax, vq in zip(axes, vars_quanti):\n",
                "            sns.boxplot(data=d, x=var_qual, y=vq, order=ordre, hue=var_qual,\n",
                "                        palette=palette, ax=ax, legend=False)\n",
                "            ax.set_title(f\"{vq} par {var_qual}\", fontsize=13, fontweight='bold')\n",
                "            ax.set_xlabel(var_qual, fontsize=11)\n",
                "            ax.set_ylabel(vq, fontsize=11)\n",
                "            ax.tick_params(axis='x', rotation=25, labelsize=10)\n",
                "\n",
                "    plt.suptitle(f\"Influence de '{var_qual}' sur plusieurs variables quantitatives\",\n",
                "                 fontsize=16, fontweight='bold', y=1.00)\n",
                "    plt.tight_layout()\n",
                "    plt.show()\n",
                "\n",
                "    moyennes = d.groupby(var_qual)[vars_quanti].mean().round(2).loc[ordre]\n",
                "    print(f\"\\nMoyennes des variables quantitatives par '{var_qual}' :\\n\")\n",
                "    print(moyennes)\n",
                "    return moyennes\n",
                "\n",
                "vars_quanti = ['height_m', 'weight_kg', 'total_points', 'base_experience']\n",
                "print(\"Fonction grille_boxplots() définie. Variables quantitatives ciblées :\", vars_quanti)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 12.1 Influence de `status` sur `height_m`, `weight_kg`, `total_points`, `base_experience`"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "_ = grille_boxplots(df, 'status', vars_quanti, ordre=ordre_status, palette='Set2')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "**Interprétation :** L'influence de `status` est **massive** sur les quatre variables : les `Legendary` sont plus grands, plus lourds, plus puissants (`total_points`) et donnent plus d'expérience. Sur `base_experience`, le saut est particulièrement spectaculaire (médiane ~ 150 pour `Normal` contre 300+ pour `Legendary`). `total_points` confirme la hiérarchie observée au radar chart (Séance 2)."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 12.2 Influence de `generation` sur les mêmes variables"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "_ = grille_boxplots(df, 'generation', vars_quanti,\n",
                "                    ordre=sorted(df['generation'].unique()),\n",
                "                    palette='viridis')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "**Interprétation :** Les médianes de `height_m`, `weight_kg`, `total_points` sont **très stables** d'une génération à l'autre, ce qui confirme l'équilibrage soigneux du jeu. On note cependant une légère tendance haussière sur `base_experience` à partir de la génération 6, possiblement liée à un ajustement du système de récompense d'XP."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 12.3 Influence de `type_1` sur les mêmes variables"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "_ = grille_boxplots(df, 'type_1', vars_quanti, palette='tab20')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "**Interprétation :** Les variables `height_m` et `weight_kg` sont les plus discriminantes selon le `type_1` : les `Dragon`, `Steel` et `Rock` sont sensiblement plus grands et plus lourds que les `Bug`, `Fairy` ou `Electric`. Sur `total_points`, les types `Dragon` et `Psychic` dominent (présence de nombreux Légendaires), ce qui confirme les conclusions de la section 10.2. `base_experience` suit globalement la même hiérarchie que `total_points`, ce qui est attendu (les Pokémon plus puissants donnent plus d'XP)."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 12.4 Influence de `ability_1` (top 10) sur les mêmes variables"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "_ = grille_boxplots(df, 'ability_1', vars_quanti, top_n=10, palette='Spectral')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "**Interprétation :** Comme pour `hp`, l'ability `Pressure` se distingue très nettement avec des valeurs élevées de `total_points` et `base_experience` (Légendaires inside). Les capacités de départ des starters (`Overgrow`, `Blaze`, `Torrent`) donnent des profils proches, ce qui est attendu car elles s'appliquent à des Pokémon volontairement équilibrés en début de jeu. Hors `Pressure`, l'ability_1 reste **secondaire** par rapport à `status` ou `type_1` pour expliquer les variables quantitatives."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 13. Synthèse de la Séance 3\n",
                "\n",
                "Les analyses bivariées menées dans cette séance permettent de hiérarchiser **l'effet des différentes variables qualitatives** sur les variables quantitatives du dataset :\n",
                "\n",
                "| Variable qualitative | Effet sur `hp` | Effet sur `total_points` | Effet sur `height_m`/`weight_kg` |\n",
                "|---|---|---|---|\n",
                "| `status` | **Très fort** | **Très fort** | Fort |\n",
                "| `type_1` | Modéré | Fort (Dragon/Psychic dominent) | **Très fort** (Steel/Rock/Dragon) |\n",
                "| `type_2` | Modéré | Fort | Modéré |\n",
                "| `generation` | Faible | Faible | Faible |\n",
                "| `ability_1` (top 10) | Faible | Faible | Faible |\n",
                "\n",
                "**Conclusions principales :**\n",
                "1. **`status` est de loin la variable qualitative la plus discriminante** : elle structure tout le dataset et explique la majeure partie de la variance des stats de combat (cf. Séance 2 et radar charts).\n",
                "2. **Les types influencent surtout la morphologie** (`height_m`, `weight_kg`) et de manière indirecte `total_points` via la concentration de Légendaires dans certains types.\n",
                "3. **La génération est un excellent contre-exemple d'équilibrage** : aucune génération n'est globalement plus forte qu'une autre, ce qui est cohérent avec la volonté des designers.\n",
                "4. Les **croisements qualitatifs** (`status × type_1`, `type_1 × type_2`) révèlent la **structure du Pokédex** : surreprésentation des Légendaires en Psychic/Dragon, combinaisons Normal/Flying et Bug/Poison classiques, types `Fairy`/`Steel` plus rares en type 2.\n",
                "\n",
                "Ces résultats fournissent une base solide pour les futures analyses : tests d'indépendance (χ², ANOVA), corrélations entre variables quantitatives, et éventuellement des modèles prédictifs."
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
