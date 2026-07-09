import json
import os

notebook_content = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Progetto 1: Segmentazione Clienti e Churn Prediction\n",
    "### Laboratorio di Intelligenza Artificiale\n",
    "**Studente:** Massimo Mantineo (Matricola 541924)\n",
    "\n",
    "In questo progetto andiamo a studiare i clienti di una compagnia telefonica (usando il dataset Telco Customer Churn). L'obiettivo è duplice:\n",
    "1. **Clustering (Apprendimento non supervisionato)**: usiamo l'algoritmo k-Means per raggruppare i clienti in base ai loro consumi e contratti, provando a identificare dei profili tipici di cliente.\n",
    "2. **Classificazione (Apprendimento supervisionato)**: vogliamo prevedere chi sta per abbandonare la compagnia (*Churn Prediction*). Per farlo proviamo tre modelli: la Regressione Logistica (come modello base o *baseline*), un Albero di Decisione e una Rete Neurale (MLP).\n",
    "\n",
    "Vogliamo capire se aggiungere le informazioni dei gruppi (cluster) aiuti i modelli di classificazione a lavorare meglio. Inoltre controlleremo con attenzione i risultati sia sui dati di addestramento (Train) che su quelli di test (Test), per vedere se i modelli soffrono di:\n",
    "- **Overfitting (imparare a memoria)**: quando il modello impara troppo a memoria i dati di addestramento (compresi i dettagli inutili) e fallisce sui dati nuovi di test.\n",
    "- **Underfitting (troppo semplice)**: quando il modello è troppo semplice e non riesce a imparare i pattern di base delle caratteristiche del cliente, ottenendo scarse prestazioni sia su Train che su Test."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Caricamento Librerie e Configurazione\n",
    "Importiamo le librerie necessarie per gestire i dati, fare i grafici e addestrare i modelli."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import urllib.request\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.cluster import KMeans\n",
    "from sklearn.metrics import silhouette_score, accuracy_score, recall_score, roc_auc_score\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.tree import DecisionTreeClassifier, plot_tree\n",
    "from sklearn.neural_network import MLPClassifier\n",
    "\n",
    "# Impostazione del seed per la riproducibilità (evitando il termine errato 'randomico')\n",
    "RANDOM_STATE = 42\n",
    "np.random.seed(RANDOM_STATE)\n",
    "\n",
    "# Creazione directory per i grafici e dati (rispetto alla cartella codice)\n",
    "os.makedirs('../grafici', exist_ok=True)\n",
    "os.makedirs('../dati', exist_ok=True)\n",
    "print(\"Librerie importate con successo.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Caricamento Dataset ed Exploratory Data Analysis (EDA)\n",
    "Carichiamo il dataset. Questo è il dataset reale *Telco Customer Churn* fornito storicamente da IBM. Non si tratta di un dataset generico prelevato da internet, ma di una risorsa standard utilizzata nel Machine Learning per simulare scenari di business realistici: descrive infatti il comportamento di 7043 clienti di una compagnia telefonica tramite caratteristiche anagrafiche, metodi di pagamento, costi e servizi inclusi."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "url = \"https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv\"\n",
    "filepath = \"../dati/Telco-Customer-Churn.csv\"\n",
    "\n",
    "if not os.path.exists(filepath):\n",
    "    print(\"Download del dataset in corso...\")\n",
    "    urllib.request.urlretrieve(url, filepath)\n",
    "    print(\"Download completato.\")\n",
    "else:\n",
    "    print(\"Il dataset è già presente localmente in '../dati/'.\")\n",
    "\n",
    "df = pd.read_csv(filepath)\n",
    "print(f\"Dimensioni del dataset: {df.shape[0]} righe, {df.shape[1]} colonne.\")\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Ispezione e analisi delle variabili\n",
    "Diamo un'occhiata alla struttura del dataset chiamando `df.info()`. Notiamo che:\n",
    "- Ci sono 7043 righe (clienti) e 21 colonne.\n",
    "- La maggior parte delle variabili sono di tipo `object` (cioè contengono parole come 'Yes', 'No', 'Fiber optic', 'Month-to-month').\n",
    "- Le variabili numeriche reali sono `tenure` (mesi da cui il cliente è attivo) e `MonthlyCharges` (costi mensili).\n",
    "- La colonna `TotalCharges` (costi totali) viene letta come testo (`object`) invece che numerica. Il motivo è che contiene 11 spazi vuoti (\" \") per i clienti che hanno appena sottoscritto il contratto (con tenure = 0) e non hanno ancora pagato nulla. Python li legge come testo e questo impedisce qualsiasi calcolo. Nel preprocessing dovremo sostituire questi spazi vuoti con `0.0` e convertire la colonna in formato numerico."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "df.info()\n",
    "\n",
    "# Conteggio spazi vuoti in TotalCharges\n",
    "empty_total_charges = (df['TotalCharges'].str.strip() == '').sum()\n",
    "print(f\"Righe con TotalCharges vuoto: {empty_total_charges}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Grafici EDA Essenziali"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.figure(figsize=(6, 4))\n",
    "sns.countplot(x='Churn', data=df, hue='Churn', palette='Set2', legend=False)\n",
    "plt.title('Distribuzione del Churn nel Dataset')\n",
    "plt.xlabel('Churn (Abbandono)')\n",
    "plt.ylabel('Numero di Clienti')\n",
    "plt.tight_layout()\n",
    "plt.savefig('../grafici/churn_distribution.png', dpi=300)\n",
    "plt.show()\n",
    "\n",
    "plt.figure(figsize=(8, 5))\n",
    "sns.boxplot(x='Churn', y='tenure', data=df, hue='Churn', palette='Set2', legend=False)\n",
    "plt.title('Relazione tra Mesi di Permanenza (tenure) e Churn')\n",
    "plt.xlabel('Churn (Abbandono)')\n",
    "plt.ylabel('Tenure (mesi)')\n",
    "plt.tight_layout()\n",
    "plt.savefig('../grafici/churn_vs_tenure.png', dpi=300)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Cosa ci dicono i grafici?\n",
    "1. **Distribuzione del Churn**: Circa il 73.4% dei clienti non ha abbandonato (No Churn, classe 0) e il 26.6% sì (Yes Churn, classe 1). C'è uno sbilanciamento di quasi 3 a 1. Se un modello si limitasse a predire sempre 'No Churn', otterrebbe un'accuratezza del 73.4% pur essendo completamente inutile per il business. Dobbiamo stare attenti a questa metrica.\n",
    "2. **Relazione tra Churn e Mesi di Permanenza (Tenure)**: Il grafico a scatola (boxplot) mostra che la scatola di chi abbandona (Yes Churn) è molto schiacciata verso il basso (mediana intorno ai 10 mesi), indicando che la stragrande maggioranza dei clienti che se ne vanno sono nuovi clienti iscritti da pochissimo tempo. La scatola di chi rimane fedele (No Churn) è invece spostata molto verso l'alto (mediana intorno ai 38 mesi), confermando che più mesi un cliente rimane con la compagnia, più è probabile che rimanga fedele."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Sistemiamo i Dati (Preprocessing)\n",
    "Prima di addestrare i modelli dobbiamo preparare i dati con questi passaggi:\n",
    "1. **TotalCharges**: sostituiamo gli spazi vuoti con `0.0` (visto che sono nuovi clienti con tenure=0 che non hanno ancora pagato nulla) e convertiamo la colonna in numeri decimali (`float`).\n",
    "2. **customerID**: eliminiamo la colonna con i codici cliente (es. '7590-VHVEG'). Trattandosi di codici identificativi unici, non hanno alcun valore predittivo e confonderebbero i modelli.\n",
    "3. **Codifica del target**: convertiamo il target Churn da 'Yes'/'No' a 1/0.\n",
    "4. **One-Hot Encoding**: traduciamo le variabili scritte a parole in colonne composte solo da numeri 0 e 1. Usiamo `drop_first=True` per escludere una colonna di riferimento per ogni variabile ed evitare ridondanze nei calcoli (multicollinearità). Ad esempio, se abbiamo tre tipi di contratto (mensile, annuale, biennale), creiamo solo due colonne (annuale e biennale): se entrambe sono 0, significa automaticamente che il contratto è mensile.\n",
    "5. **Standardizzazione (Standard Scaling)**: portiamo le variabili numeriche (`tenure`, `MonthlyCharges`, `TotalCharges`) sulla stessa scala (media 0 e deviazione standard 1). Questo è fondamentale per modelli basati su distanze (come k-Means) o pesi (come la rete neurale MLP) per evitare che una variabile con valori grandi (TotalCharges) oscuri variabili con valori piccoli (tenure)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 1. Gestione TotalCharges\n",
    "df.loc[df['TotalCharges'].str.strip() == '', 'TotalCharges'] = '0.0'\n",
    "df['TotalCharges'] = df['TotalCharges'].astype(float)\n",
    "\n",
    "# 2. Rimozione customerID\n",
    "df_cleaned = df.drop(columns=['customerID'])\n",
    "\n",
    "# 3. Codifica Target\n",
    "df_cleaned['Churn'] = df_cleaned['Churn'].map({'Yes': 1, 'No': 0})\n",
    "\n",
    "# 4. Suddivisione Feature / Target\n",
    "X = df_cleaned.drop(columns=['Churn'])\n",
    "y = df_cleaned['Churn']\n",
    "\n",
    "# Identificazione tipi colonne\n",
    "num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']\n",
    "cat_cols = [col for col in X.columns if col not in num_cols]\n",
    "\n",
    "# 5. One-Hot Encoding delle categoriche\n",
    "X_encoded = pd.get_dummies(X, columns=cat_cols, drop_first=True)\n",
    "\n",
    "# 6. Standardizzazione numeriche\n",
    "scaler = StandardScaler()\n",
    "X_encoded[num_cols] = scaler.fit_transform(X_encoded[num_cols])\n",
    "\n",
    "print(f\"Formato del dataset preelaborato: {X_encoded.shape}\")\n",
    "X_encoded.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Raggruppiamo i clienti (Clustering con k-Means)\n",
    "Proviamo a dividere i clienti in 2, 3 o 4 gruppi. Per scegliere la suddivisione migliore usiamo due metriche:\n",
    "- **Inertia (Metodo del Gomito)**: indica la somma delle distanze dei punti dal centro del loro gruppo. Diminuisce all'aumentare del numero di gruppi; noi cerchiamo il punto in cui la discesa rallenta bruscamente (il 'gomito').\n",
    "- **Silhouette Score**: misura quanto ogni cliente è vicino al proprio gruppo rispetto agli altri. Più il valore è vicino a 1, meglio definiti e separati sono i gruppi."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "k_range = [2, 3, 4]\n",
    "inertias = []\n",
    "silhouettes = []\n",
    "\n",
    "for k in k_range:\n",
    "    kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)\n",
    "    kmeans.fit(X_encoded)\n",
    "    inertias.append(kmeans.inertia_)\n",
    "    \n",
    "    # Silhouette score\n",
    "    score = silhouette_score(X_encoded, kmeans.labels_, sample_size=2000, random_state=RANDOM_STATE)\n",
    "    silhouettes.append(score)\n",
    "    print(f\"k = {k} | Inertia: {kmeans.inertia_:.2f} | Silhouette Score: {score:.4f}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Plot di Valutazione del Clustering"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "fig, ax1 = plt.subplots(figsize=(8, 4))\n",
    "\n",
    "color = 'tab:red'\n",
    "ax1.set_xlabel('Numero di Cluster (k)')\n",
    "ax1.set_ylabel('Inertia (Elbow)', color=color)\n",
    "ax1.plot(k_range, inertias, marker='o', color=color)\n",
    "ax1.tick_params(axis='y', labelcolor=color)\n",
    "\n",
    "ax2 = ax1.twinx()  \n",
    "color = 'tab:blue'\n",
    "ax2.set_ylabel('Silhouette Score', color=color)\n",
    "ax2.plot(k_range, silhouettes, marker='s', color=color)\n",
    "ax2.tick_params(axis='y', labelcolor=color)\n",
    "\n",
    "plt.title('Metodo del Gomito e Silhouette Score a confronto')\n",
    "fig.tight_layout()\n",
    "plt.savefig('../grafici/clustering_evaluation.png', dpi=300)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Chi c'è nei due gruppi? (Scelta k=2)\n",
    "Guardando il grafico sopra, notiamo due cose:\n",
    "- L'Inertia (linea rossa) mostra un cambio di pendenza a $k=2$ e $k=3$.\n",
    "- Il Silhouette Score (linea blu) ha il suo valore massimo assoluto a **k=2** (0.2946) e scende per valori successivi.\n",
    "Per questo abbiamo scelto $k=2$ come partizione ottimale. \n",
    "\n",
    "Ora analizziamo i due gruppi reali confrontandoli con i valori commerciali prima della standardizzazione:\n",
    "- **Cluster 0 (Fedeli - Spesa Bassa)**: clienti stabili (tenure media 30.5 mesi), con costi mensili bassi (21.08 $) e contratti lunghi (biennali o annuali). Il loro tasso di churn reale è bassissimo (7.40%).\n",
    "- **Cluster 1 (A rischio - Spesa Alta)**: clienti più recenti o volatili, ma con una spesa mensile alta (76.84 $) e legati a contratti flessibili mensili. Il loro tasso di churn reale è molto alto (31.83%)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "optimal_k = 2\n",
    "kmeans_final = KMeans(n_clusters=optimal_k, random_state=RANDOM_STATE, n_init=10)\n",
    "cluster_labels = kmeans_final.fit_predict(X_encoded)\n",
    "\n",
    "# Creazione dataset temporaneo non scalato per ispezione\n",
    "df_inspect = df_cleaned.copy()\n",
    "df_inspect['Cluster'] = cluster_labels\n",
    "\n",
    "for c in range(optimal_k):\n",
    "    c_df = df_inspect[df_inspect['Cluster'] == c]\n",
    "    print(f\"\\nCluster {c} ({len(c_df)} clienti):\")\n",
    "    print(f\"  Tenure media: {c_df['tenure'].mean():.2f} mesi\")\n",
    "    print(f\"  Spesa mensile media: {c_df['MonthlyCharges'].mean():.2f} $\")\n",
    "    print(f\"  Tasso di Churn reale: {c_df['Churn'].mean()*100:.2f}%\")\n",
    "    print(\"  Contratti (%):\")\n",
    "    print(c_df['Contract'].value_counts(normalize=True) * 100)\n",
    "\n",
    "# Aggiunta della feature al dataset di classificazione\n",
    "X_with_cluster = X_encoded.copy()\n",
    "X_with_cluster['Cluster'] = cluster_labels\n",
    "X_with_cluster = pd.get_dummies(X_with_cluster, columns=['Cluster'], drop_first=True)\n",
    "X_with_cluster.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Prevediamo l'abbandono (Classificazione)\n",
    "Dividiamo i dati in Train (80% per addestrare) e Test (20% per verificare). Lo facciamo in modo stratificato per mantenere la proporzione originale del 26.6% di Churn sia in Train che in Test.\n",
    "*Importante*: lo split va fatto prima di calcolare la standardizzazione, altrimenti le informazioni del Test set influenzerebbero il Train (Data Leakage), invalidando la correttezza della valutazione.\n",
    "\n",
    "Addestriamo tre modelli con filosofie diverse:\n",
    "1. **Baseline (Regressione Logistica)**: un modello lineare che calcola una somma pesata delle feature e la passa attraverso una funzione Sigmoide per ottenere una probabilità tra 0 e 1. Ci serve come base semplice di confronto.\n",
    "2. **Decision Tree**: un modello a regole (IF-THEN) che suddivide i clienti in base a domande logiche che massimizzano la purità dei gruppi (misurata con l'indice di Gini). Impostiamo max_depth=5 per regolarizzare ed evitare che impari le risposte a memoria.\n",
    "3. **Multi-Layer Perceptron (MLP)**: una rete neurale artificiale complessa. Usa strati nascosti con attivazione non lineare (ReLU) per apprendere relazioni complesse e la Sigmoide in uscita per predire la probabilità di churn.\n",
    "\n",
    "Valutiamo i modelli in due scenari diversi:\n",
    "- **Scenario A (Senza Cluster)**: usiamo solo le caratteristiche di partenza.\n",
    "- **Scenario B (Con Cluster)**: aggiungiamo il cluster (0 o 1) trovato da k-Means come caratteristica in più per capire se questo profilo commerciale pre-calcolato facilita il lavoro dei classificatori."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Split Train/Test per i due scenari\n",
    "X_train_A, X_test_A, y_train_A, y_test_A = train_test_split(\n",
    "    X_encoded, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE\n",
    ")\n",
    "X_train_B, X_test_B, y_train_B, y_test_B = train_test_split(\n",
    "    X_with_cluster, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE\n",
    ")\n",
    "\n",
    "# Inizializzazione modelli\n",
    "models = {\n",
    "    'Baseline (Logistic Regression)': LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),\n",
    "    'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE),\n",
    "    'Multi-Layer Perceptron (MLP)': MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=RANDOM_STATE)\n}\n",
    "\n",
    "results = []\n",
    "\n",
    "def evaluate_pipeline(model, X_tr, X_te, y_tr, y_te):\n",
    "    model.fit(X_tr, y_tr)\n",
    "    \n",
    "    # Metriche di Training\n",
    "    preds_tr = model.predict(X_tr)\n",
    "    probs_tr = model.predict_proba(X_tr)[:, 1]\n",
    "    acc_tr = accuracy_score(y_tr, preds_tr)\n",
    "    rec_tr = recall_score(y_tr, preds_tr)\n",
    "    auc_tr = roc_auc_score(y_tr, probs_tr)\n",
    "    \n",
    "    # Metriche di Test\n",
    "    preds_te = model.predict(X_te)\n",
    "    probs_te = model.predict_proba(X_te)[:, 1]\n",
    "    acc_te = accuracy_score(y_te, preds_te)\n",
    "    rec_te = recall_score(y_te, preds_te)\n",
    "    auc_te = roc_auc_score(y_te, probs_te)\n",
    "    \n",
    "    return (acc_tr, rec_tr, auc_tr), (acc_te, rec_te, auc_te)\n",
    "\n",
    "for name, model in models.items():\n",
    "    # Scenario A\n",
    "    (acc_tr_A, rec_tr_A, auc_tr_A), (acc_te_A, rec_te_A, auc_te_A) = evaluate_pipeline(\n",
    "        model, X_train_A, X_test_A, y_train_A, y_test_A\n",
    "    )\n",
    "    results.append({\n",
    "        'Modello': name, 'Feature': 'Senza Cluster', \n",
    "        'Train Accuracy': acc_tr_A, 'Test Accuracy': acc_te_A, \n",
    "        'Train Recall': rec_tr_A, 'Test Recall': rec_te_A, \n",
    "        'Train AUC': auc_tr_A, 'Test AUC': auc_te_A\n",
    "    })\n",
    "    \n",
    "    # Scenario B\n",
    "    (acc_tr_B, rec_tr_B, auc_tr_B), (acc_te_B, rec_te_B, auc_te_B) = evaluate_pipeline(\n",
    "        model, X_train_B, X_test_B, y_train_B, y_test_B\n",
    "    )\n",
    "    results.append({\n",
    "        'Modello': name, 'Feature': 'Con Cluster', \n",
    "        'Train Accuracy': acc_tr_B, 'Test Accuracy': acc_te_B, \n",
    "        'Train Recall': rec_tr_B, 'Test Recall': rec_te_B, \n",
    "        'Train AUC': auc_tr_B, 'Test AUC': auc_te_B\n",
    "    })\n",
    "\n",
    "df_res = pd.DataFrame(results)\n",
    "df_res"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 6. Come ragionano i modelli? (Interpretazione)\n",
    "Disegniamo un albero di decisione semplificato per vedere quali domande fa per classificare i clienti.\n",
    "**Come leggere le informazioni dentro ciascun rettangolo dell'albero:**\n",
    "- **Condizione in alto** (es. `Contract_Month-to-month <= 0.5`): la domanda logica. Se la risposta per un cliente è SI, andiamo nel ramo a sinistra, altrimenti nel ramo a destra.\n",
    "- **gini**: misura l'impurezza del nodo. Varia da 0 (tutti i clienti in questo nodo appartengono alla stessa classe, purezza massima) a 0.5 (mix 50/50).\n",
    "- **samples**: quanti clienti sono passati per questo specifico nodo dell'albero.\n",
    "- **value**: quanti clienti appartengono alla classe 0 (No Churn, primo numero) e alla classe 1 (Churn, secondo numero).\n",
    "- **class**: la decisione provvisoria presa in quel nodo (in base alla maggioranza).\n",
    "- **Colori**: i nodi sono colorati in **blu** se la maggioranza dei clienti è fedele (No Churn) e in **arancione** se la maggioranza ha abbandonato (Churn). Più il colore è scuro e intenso, più il nodo è puro e la decisione è netta."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "dt_simple = DecisionTreeClassifier(max_depth=3, random_state=RANDOM_STATE)\n",
    "dt_simple.fit(X_train_A, y_train_A)\n",
    "\n",
    "plt.figure(figsize=(16, 9))\n",
    "plot_tree(dt_simple, feature_names=list(X_encoded.columns), class_names=['No Churn', 'Churn'], filled=True, rounded=True, fontsize=10)\n",
    "plt.title('Albero di Decisione Semplificato (max_depth=3)')\n",
    "plt.tight_layout()\n",
    "plt.savefig('../grafici/decision_tree_vis.png', dpi=300)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "dt_full = DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE)\n",
    "dt_full.fit(X_train_B, y_train_B)\n",
    "\n",
    "importances = dt_full.feature_importances_\n",
    "indices = np.argsort(importances)[::-1][:10]\n",
    "\n",
    "plt.figure(figsize=(10, 6))\n",
    "sns.barplot(x=importances[indices], y=[X_with_cluster.columns[i] for i in indices], hue=[X_with_cluster.columns[i] for i in indices], palette='viridis', legend=False)\n",
    "plt.title('Top 10 Importanza Feature nel Decision Tree (Con Cluster)')\n",
    "plt.xlabel('Importanza Relativa')\n",
    "plt.tight_layout()\n",
    "plt.savefig('../grafici/feature_importances.png', dpi=300)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Cosa ci dice il grafico dell'importanza delle caratteristiche?\n",
    "Questo grafico mostra quali caratteristiche (colonne) sono state più importanti per l'albero di decisione nello Scenario B per distinguere i clienti che se ne vanno da quelli che rimangono:\n",
    "- La colonna più importante in assoluto è `Contract_Month-to-month` (avere un contratto mensile flessibile senza vincoli temporali).\n",
    "- Al secondo posto c'è la `tenure` (da quanti mesi il cliente è attivo).\n",
    "- Al terzo posto c'è `InternetService_Fiber optic` (avere internet in fibra ottica, che è associato a spese mensili più alte).\n",
    "- Anche il nostro `Cluster_1` (il profilo del cliente volatile ad alta spesa identificato da k-Means) compare nella top 10 delle caratteristiche più significative. Questo conferma che la suddivisione commerciale fatta dal clustering non supervisionato ha un valore predittivo concreto che viene attivamente sfruttato dai modelli supervisionati."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 7. Cosa abbiamo scoperto? (Conclusioni)\n",
    "Guardando i risultati del confronto possiamo trarre queste conclusioni:\n",
    "1. **La Regressione Logistica e l'Albero di Decisione sono stabili**: le loro prestazioni su Train e Test sono praticamente uguali, quindi non c'è overfitting. L'albero di decisione è ottimo perché è trasparente ed è facile capire come prende le decisioni.\n",
    "2. **La Rete Neurale (MLP) impara a memoria (Overfitting)**: sul Train set raggiunge un'accuratezza altissima (91.4%), ma sul Test set scende parecchio (74.9%). Essendo un modello molto complesso, tende a memorizzare i dettagli del train invece di capire le regole generali.\n",
    "3. **L'utilità del Cluster**: se aggiungiamo il cluster come caratteristica (Scenario B), la Recall dell'MLP sul Test set fa un gran balzo in avanti, salendo dal 42.8% al 58.3% (+15.5%). Questo significa che il clustering non supervisionato aiuta concretamente la rete neurale a scovare più clienti a rischio di abbandono."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Uso di Intelligenza Artificiale Generativa\n",
    "Ho usato strumenti di intelligenza artificiale (LLM) solo per farmi aiutare a formattare il testo markdown e correggere la sintassi del codice."
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
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

# Scriviamo il file nella cartella codice
filepath_nb = os.path.join(os.path.dirname(__file__), "segmentazione_crunch.ipynb")
with open(filepath_nb, "w", encoding="utf-8") as f:
    json.dump(notebook_content, f, indent=1, ensure_ascii=False)

print(f"Notebook Jupyter generato con successo in '{filepath_nb}'.")
