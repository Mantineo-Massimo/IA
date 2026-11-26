# Foglio 3 — Confronto Completo Slide per Slide con il Codice

**Slide:** [`relazione/presentazione.pdf`](../relazione/presentazione.pdf) (10 slide)  
**Codice:** [`notebooks/churn_analysis.ipynb`](../notebooks/churn_analysis.ipynb) (31 celle: `Cella 0` → `Cella 30`, 14 celle eseguibili)  
**Studente:** Massimo Mantineo (Matricola: 541924)

---

## Mappa Rapida di Tracciabilità 1:1 (Slide ↔ Celle Notebook ↔ Immagini)

| Slide | Titolo Slide | Celle Notebook Corrispondenti | File Immagine in `relazione/images/` | Tipo di Corrispondenza |
| :---: | :--- | :--- | :--- | :--- |
| **1** | Copertina / Titolo | `Cella 0` (MD), `Cella 2` (Setup) | `logo_unime_orizontale.png` | Intestazione & configurazione globale |
| **2** | Definizione IA & Agente PEAS | `Cella 0` (MD) | Diagrammi TikZ (in `presentazione.tex`) | Quadro teorico (Lez. 01–02) |
| **3** | Task ML & Approccio Probabilistico | `Cella 0`, `Cella 2`, `Cella 10`, `Cella 19` | — (box LaTeX) | Formalizzazione teorica implementata nelle celle |
| **4** | Dataset, Preprocessing & Sbilanciamento | `Cella 4`, `Cella 6`, `Cella 8`, `Cella 10` | `boxplot_permanenza_costi.png` *(e `distribuzione_churn.png` per relazione)* | **100% diretto** (cleaning, split, scaling, boxplot) |
| **5** | Clustering K-Means & Scelta di $K$ | `Cella 2`, `Cella 12`, `Cella 14` | `clustering_scelta_k.png` | **100% diretto** (WCSS, Silhouette, scelta $K=3$) |
| **6** | Segmentazione PCA & Profili | `Cella 14`, `Cella 16` | `clustering_visualizzazione_pca.png` *(e `distribuzione_churn_cluster.png`)* | **100% diretto** (tabella profili, test check, PCA 2D) |
| **7** | Pipeline Sperimentale Supervisionata | `Cella 2`, `Cella 10`, `Cella 19` | Diagramma TikZ (in `presentazione.tex`) | **100% diretto** (30D vs 32D, 3 modelli, iperparametri) |
| **8** | Confronto Prestazioni & Soglia | `Cella 19`, `Cella 21`, `Cella 23`, `Cella 25`, `Cella 27` | `confronto_prestazioni_barre.png` *(e `curve_roc.png` per relazione)* | **100% diretto** (tabella, barre dinamiche, CV soglia) |
| **9** | Spiegabilità Business: Albero CART | `Cella 10`, `Cella 19`, `Cella 29` | `albero_decisionale.png` | **100% diretto** (CART non scalato, `plot_tree`, regole) |
| **10** | Discussione Critica & Conclusioni | `Cella 30` (MD) + sintesi `Celle 14, 21, 23, 29` | — | Sintesi critica dei risultati del notebook |

---

## Slide 1 — Titolo e Copertina

### Cosa compare sulla Slide
- Intestazione UniMe — Dipartimento MIFT, Corso di Laurea in Informatica (L-31), *Laboratorio di Intelligenza Artificiale — Progetto 1*.
- Titolo: **Churn Prediction e Segmentazione Comportamentale** (*Dataset IBM Telco Customer Churn*), Massimo Mantineo (541924).

### Dove si trova nel Codice (`Cella 0` e `Cella 2`)
La `Cella 0` (Markdown) introduce identità del progetto, obiettivi (non supervisionato + supervisionato) e vincolo *Zero Data Leakage*. La `Cella 2` centralizza tutte le costanti globali che governano la riproducibilità dell'intero esperimento:

```python
# Cella 2 — Configurazione Globale (riproducibilità)
IMAGES_DIR = "../relazione/images" if os.path.exists("../relazione") else "relazione/images"
RANDOM_STATE = 42
TEST_SIZE = 0.20

# Parametri K-Means (valutazione per K in {2, 3, 4, 5, 6})
K_MIN = 2
K_MAX = 7
SELECTED_K = 3
KMEANS_N_INIT = 10

# Parametri MLP (Lezione 08: regolarizzazione L2 esplicita + early stopping)
MLP_HIDDEN_LAYERS = (64, 32)
MLP_ALPHA = 0.1
MLP_MAX_ITER = 500
```

**Cosa sapere se il prof chiede di questa parte:** Tutte le costanti stanno in un unico blocco in cima al notebook (`RANDOM_STATE = 42` ovunque), così eseguendo *Restart & Run All* vengono rigenerati esattamente gli stessi numeri e gli stessi PNG inseriti nelle slide.

---

## Slide 2 — Inquadramento Teorico: Definizione di IA e Agente PEAS

### Cosa compare sulla Slide
- **Sinistra:** i 4 approcci di Russell & Norvig con evidenza su **Agire Razionalmente** (agente con apprendimento basato su utilità).
- **Destra:** modello **PEAS** (`P` valore netto di fidelizzazione vs CAC; `E` clientela Telco; `A` offerte/sconti/supporto; `S` log CRM) + diagramma del ciclo chiuso.

### Collegamento con il Codice (`Cella 0`, `Cella 4`, `Cella 19`, `Cella 23`)
Questa slide è **teorica** (Lezioni 01–02), quindi i diagrammi sono disegnati in TikZ dentro `relazione/presentazione.tex`, ma ogni componente del PEAS ha un corrispettivo concreto nel codice:
- **Sensori ($S$):** le 19 colonne predittive caricate da `Telco-Customer-Churn.csv` nella `Cella 4` (`X_raw` in `Cella 10`).
- **Elemento di Apprendimento:** il fitting di `KMeans` (`Cella 14`) e dei 3 classificatori (`Cella 19`) su `X_train`.
- **Elemento Esecutivo + Utilità (ponte verso Attuatori $A$):** le probabilità `predict_proba(X)[:, 1]` (`Cella 19`) combinate con la soglia calibrata sul CAC (`test_probs_lr >= 0.35` e `class_weight='balanced'` nella `Cella 23`).

---

## Slide 3 — Formulazione dei Task ML e Approccio Probabilistico

### Cosa compare sulla Slide
- Sistema **Batch** e **Model-Based** che unisce:
  1. **Task Supervisionato (Mitchell $\langle T, E, P \rangle$):** classificazione binaria probabilistica (`Churn`), `5.634` istanze di training su `7.043`, metriche *Recall, Precision, F1, AUC-ROC*.
  2. **Task Non Supervisionato:** K-Means su `30` feature senza `Churn` (WCSS + Silhouette) + PCA 2D.
- Approccio **probabilistico discriminativo** $P(\text{Churn}=1 \mid \mathbf{x}) \in [0, 1]$.

### Corrispondenza nel Codice (`Cella 2`, `Cella 10`, `Cella 12`, `Cella 19`)
- **Esperienza $E$ (`5.634` righe):** nasce in `Cella 10` con `train_test_split(..., test_size=0.20, stratify=y)`.
- **Metriche $P$ (Mitchell):** importate in `Cella 2` (`accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `roc_auc_score`) e calcolate in `Cella 19`.
- **Stima diretta $P(\text{Churn}=1 \mid \mathbf{x})$:** in `Cella 19`, tutti e 3 i modelli espongono `.predict_proba(X_te)[:, 1]`, restituendo lo score probabilistico continuo usato per ROC-AUC e calibrazione di soglia.

---

## Slide 4 — Dataset Telco: Preprocessing e Sbilanciamento del Target

### Cosa compare sulla Slide
- **Banner:** `7.043` clienti, `21` colonne (`19` feature utili + `customerID` rimosso + `Churn` al **26.54%**).
- **Sinistra:**
  - 11 valori vuoti su `TotalCharges` imputati a `0.0 €` (`tenure = 0`);
  - Split stratificato 80/20 **prima** di scaling e clustering (Zero Data Leakage);
  - `StandardScaler` + `OneHotEncoder(drop='first')` $\rightarrow$ **30 feature**;
  - Sbilanciamento `26.54%` (dummy: `73.46%` Acc, `0%` Recall).
- **Destra:** grafico `boxplot_permanenza_costi.png` con mediane `tenure` **10.0 vs 38.0 mesi** e `MonthlyCharges` **79.65 € vs 64.43 €**.

### Codice Corrispondente (`Celle 4, 6, 8, 10`)

```python
# Cella 4 — Caricamento dataset (7043 righe, 21 colonne)
df = pd.read_csv(DATA_PATH)

# Cella 6 — Pulizia dati: 11 spazi vuoti -> 0.0€, drop customerID, target 0/1
blank_rows = df[df['TotalCharges'].astype(str).str.strip() == ""]
# Output verificato: 11 righe, valori unici di tenure = [0]
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].astype(str).str.strip().replace("", "0"))
df = df.drop(columns=['customerID'])
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Cella 8 — Calcolo sbilanciamento, mediane e salvataggio boxplot per le slide
churn_rate = df['Churn'].mean()                                   # 0.265369... -> 26.54% (1869 / 7043)
med_stats = df.groupby('Churn')[['tenure', 'MonthlyCharges']].median()
# Output: Churn=0 -> tenure=38.0, MonthlyCharges=64.43 | Churn=1 -> tenure=10.0, MonthlyCharges=79.65
plt.savefig(f'{IMAGES_DIR}/boxplot_permanenza_costi.png', dpi=300, bbox_inches='tight')

# Cella 10 — Split stratificato 80/20 PRIMA di scaling/encoding (Zero Data Leakage)
X_raw = df.drop(columns=['Churn'])
y = df['Churn']
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
cat_cols = [col for col in X_raw.columns if col not in num_cols]  # 16 colonne categoriali

preprocessor_scaled = ColumnTransformer(transformers=[
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(drop='first', sparse_output=False), cat_cols)
])
X_train_scaled = preprocessor_scaled.fit_transform(X_train_raw)   # FIT solo sul Train -> (5634, 30)
X_test_scaled  = preprocessor_scaled.transform(X_test_raw)        # SOLO TRANSFORM sul Test -> (1409, 30)
```

### Verifica Output Reale
- `Train Set: (5634, 30) (Churn=26.54%)` | `Test Set: (1409, 30) (Churn=26.54%)`
- Parametri appresi da `StandardScaler` solo su Train: `tenure` ($\mu=32.49, \sigma=24.57$), `MonthlyCharges` ($\mu=64.93, \sigma=30.14$), `TotalCharges` ($\mu=2299.33, \sigma=2279.00$).
- `3` numeriche + `27` colonne binarie one-hot = **30 feature esatte**.

---

## Slide 5 — Apprendimento Non Supervisionato: Clustering K-Means

### Cosa compare sulla Slide
- Segmentazione dei `5.634` clienti del Training Set (`30` feature) senza `Churn`.
- Confronto $K \in \{2, 3, 4, 5, 6\}$ con grafico a doppio asse `clustering_scelta_k.png`:
  - $K=2$: $s=0.293$, Inerzia `33.485`;
  - **$K=3$ (Scelto):** gomito Inerzia (`33.485 -> 24.642`), $s=0.282$;
  - $K \ge 4$: crollo Silhouette (`0.230 -> 0.185`).

### Codice Corrispondente (`Cella 12`)

```python
# Cella 12 — Ciclo su K in {2, 3, 4, 5, 6} SOLO su X_train_scaled
k_range = list(range(K_MIN, K_MAX))   # [2, 3, 4, 5, 6]
for k in k_range:
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=KMEANS_N_INIT)
    labels_k = km.fit_predict(X_train_scaled)
    sil = silhouette_score(X_train_scaled, labels_k, sample_size=2000, random_state=RANDOM_STATE)
    inertias.append(km.inertia_)
    silhouettes.append(sil)

# Grafico a doppio asse (ax1 = Inerzia blu, ax2 = ax1.twinx() Silhouette arancione)
plt.savefig(f'{IMAGES_DIR}/clustering_scelta_k.png', dpi=300, bbox_inches='tight')
```

### Verifica Output Reale (`cluster_results_df`)
| $K$ | Inerzia WCSS (Notebook) | Silhouette (Notebook) | Riportato in Slide 5 |
| :---: | :---: | :---: | :--- |
| 2 | `33485.3863` | `0.2930` | `33.485`, `s = 0.293` |
| **3** | **`24642.3214`** | **`0.2817`** | **`24.642`, `s = 0.282` (Scelto)** |
| 4 | `22599.7888` | `0.2304` | `0.230` |
| 5 | `21178.5254` | `0.2093` | (in curva) |
| 6 | `19983.5212` | `0.1848` | `0.185` |

**Dettaglio importante da ricordare:** `KMeans` in `scikit-learn` utilizza di default `init='k-means++'`, e `n_init=10` esegue 10 inizializzazioni diverse scegliendo quella con WCSS minima.

---

## Slide 6 — Segmentazione PCA e Profilo dei Segmenti

### Cosa compare sulla Slide
- **Tabella profili (Training Set):**
  - `0 (Volatili)`: `2.557` clienti, `15.2` mesi, `67.87 €`, `1005.53 €`, **43.4%** churn (`43.3%` sul Test);
  - `1 (Fedeli High-Value)`: `1.863` clienti, `57.7` mesi, `89.44 €`, `5145.23 €`, `15.9%` churn;
  - `2 (Low-Cost)`: `1.214` clienti, `30.2` mesi, `21.12 €`, `657.12 €`, `7.2%` churn.
- **Grafico PCA 2D (`clustering_visualizzazione_pca.png`):** `57.9%` varianza spiegata (`PC1 = 39.4%` Spesa e Servizi; `PC2 = 18.5%` Anzianità contrattuale).

### Codice Corrispondente (`Celle 14 e 16`)

```python
# Cella 14 — Fitting K=3 su Train, predict induttivo su Test e riordino deterministico ID
kmeans_chosen = KMeans(n_clusters=SELECTED_K, random_state=RANDOM_STATE, n_init=KMEANS_N_INIT)
raw_train_clusters = kmeans_chosen.fit_predict(X_train_scaled)
raw_test_clusters  = kmeans_chosen.predict(X_test_scaled)   # Out-of-sample, ZERO leakage

# Riordino deterministico degli ID per tasso di churn decrescente sul Training Set
cluster_order = (
    pd.Series(y_train.values).groupby(raw_train_clusters)
    .mean().sort_values(ascending=False).index.tolist()
)
cluster_remap  = {old_id: new_id for new_id, old_id in enumerate(cluster_order)} # {2:0, 1:1, 0:2}
train_clusters = np.array([cluster_remap[c] for c in raw_train_clusters])
test_clusters  = np.array([cluster_remap[c] for c in raw_test_clusters])

# Aggregazione tabella profili su df_train_clustered + verifica diagnostica su df_test_clustered
cluster_profile = df_train_clustered.groupby('Cluster').agg({
    'tenure': 'mean', 'MonthlyCharges': 'mean', 'TotalCharges': 'mean', 'Churn': ['mean', 'count']
})

# Cella 16 — Proiezione 2D PCA e ispezione loadings
pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_train_pca = pca.fit_transform(X_train_scaled)
var_ratio = pca.explained_variance_ratio_  # PC1 = 39.37%, PC2 = 18.50%, Totale = 57.87%
for i in range(2):
    loadings = pd.Series(pca.components_[i], index=feature_names).sort_values()
plt.savefig(f'{IMAGES_DIR}/clustering_visualizzazione_pca.png', dpi=300, bbox_inches='tight')
```

### Verifica Output Reale
- **Verifica Test Set (Cella 14):** Cluster 0 = `43.3%` (`677` clienti), Cluster 1 = `13.3%` (`420`), Cluster 2 = `8.0%` (`312`).
- **Top Loadings PCA (Cella 16):**
  - `PC1`: positivi `MonthlyCharges (+0.501)`, `TotalCharges (+0.491)`, `tenure (+0.338)`, `StreamingMovies_Yes (+0.173)`; negativi `InternetService_No (-0.166)`.
  - `PC2`: positivi `tenure (+0.612)`, `TotalCharges (+0.338)`, `Contract_Two year (+0.221)`; negativi `MonthlyCharges (-0.277)`, `Electronic check (-0.180)`, `Fiber optic (-0.163)`.

---

## Slide 7 — Pipeline Sperimentale Supervisionata

### Cosa compare sulla Slide
- Diagramma TikZ: Split 80/20 $\rightarrow$ Preprocessing & K-Means solo su Train (`Orig` 30D vs `+Clust` 32D) $\rightarrow$ 3 classificatori:
  1. **Regressione Logistica** (baseline lineare);
  2. **Decision Tree CART (`max_depth=5`)** su feature numeriche **non scalate**;
  3. **MLP `(64, 32)`** ReLU, output sigmoideo, Adam, L2 (`alpha=0.1`) ed *early stopping*.

### Codice Corrispondente (`Celle 10 e 19`)

```python
# Cella 10 — Seconda pipeline SENZA scaling per il Decision Tree CART
preprocessor_tree = ColumnTransformer(transformers=[
    ('num', 'passthrough', num_cols),   # tenure, MonthlyCharges, TotalCharges in mesi ed Euro!
    ('cat', OneHotEncoder(drop='first', sparse_output=False), cat_cols)
])
X_train_tree = preprocessor_tree.fit_transform(X_train_raw)
X_test_tree  = preprocessor_tree.transform(X_test_raw)

# Cella 19 — Creazione feature One-Hot dei cluster (+2 colonne -> 32 feature totali)
ohe_cluster = OneHotEncoder(drop='first', sparse_output=False)
cl_train_dummies = pd.DataFrame(ohe_cluster.fit_transform(train_clusters.reshape(-1, 1)),
                                columns=['Cluster_1', 'Cluster_2'], index=X_train_raw.index)
cl_test_dummies  = pd.DataFrame(ohe_cluster.transform(test_clusters.reshape(-1, 1)),
                                columns=['Cluster_1', 'Cluster_2'], index=X_test_raw.index)

X_train_scaled_clust = pd.concat([X_train_scaled_df, cl_train_dummies], axis=1) # (5634, 32)
X_train_tree_clust   = pd.concat([X_train_tree_df,   cl_train_dummies], axis=1) # (5634, 32)

# Specifiche dei 3 classificatori
model_specs = [
    ('Regressione Logistica (Baseline)',
     LogisticRegression(random_state=42, max_iter=1000),
     X_train_scaled_df, X_test_scaled_df, X_train_scaled_clust, X_test_scaled_clust),

    ('Decision Tree (CART)',
     DecisionTreeClassifier(random_state=42, max_depth=5),
     X_train_tree_df, X_test_tree_df, X_train_tree_clust, X_test_tree_clust),

    ('MLP (Rete Neurale)',
     MLPClassifier(random_state=42, hidden_layer_sizes=(64, 32), alpha=0.1,
                   early_stopping=True, max_iter=500),
     X_train_scaled_df, X_test_scaled_df, X_train_scaled_clust, X_test_scaled_clust),
]
```

**Dettaglio importante sui default di `MLPClassifier`:** `activation='relu'`, `solver='adam'`, `out_activation_='logistic'` (sigmoide automatica per classificazione binaria), `validation_fraction=0.1` (10% del Train per early stopping), `n_iter_no_change=10`. L'MLP si arresta in modo autonomo a **17 epoche** (Base, `4.097` parametri) e **19 epoche** (+Cluster, `4.225` parametri).

---

## Slide 8 — Confronto delle Prestazioni Sperimentali

### Cosa compare sulla Slide
- **Sinistra:** `confronto_prestazioni_barre.png` (Recall a $\tau=0.50$ e AUC-ROC, blu = Base, rosso = +Cluster).
- **Destra:** tabella Recall, F1, AUC + Box Rasoio di Occam + Box Trade-off Soglia ($\tau=0.35$ o `class_weight='balanced'` $\rightarrow$ Recall `70.3%–78.3%`, F1 `0.614`).

### Codice Corrispondente (`Celle 19, 21, 23, 27`)

```python
# Cella 19 — Addestramento clonato Base vs +Cluster e calcolo metriche sul Test Set
for name, base_model, X_tr_o, X_te_o, X_tr_c, X_te_c in model_specs:
    m_orig  = clone(base_model).fit(X_tr_o, y_train)
    y_pred_o = m_orig.predict(X_te_o)
    y_prob_o = m_orig.predict_proba(X_te_o)[:, 1]   # Probabilità classe Churn=1

    m_clust = clone(base_model).fit(X_tr_c, y_train)
    y_pred_c = m_clust.predict(X_te_c)
    y_prob_c = m_clust.predict_proba(X_te_c)[:, 1]
    # Calcolo accuracy_score, precision_score, recall_score, f1_score, roc_auc_score -> results_df

# Cella 21 — Stampa results_df e matrici di confusione Base
cm = confusion_matrix(y_test, fitted_models[name]['y_pred_o'])

# Cella 23 — Calibrazione soglia in 5-fold CV su Training (ZERO leakage) + confronto su Test
oof_probs = cross_val_predict(
    LogisticRegression(random_state=42, max_iter=1000),
    X_train_scaled_df, y_train, cv=5, method='predict_proba'
)[:, 1]
# CV su Train (τ=0.35): Recall=0.7043, Precision=0.5649, F1=0.6270

test_probs_lr = fitted_models['Regressione Logistica (Baseline)']['y_prob_o']
lr_balanced = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced').fit(
    X_train_scaled_df, y_train
)
# Valutazione su Test: (test_probs_lr >= 0.50), (test_probs_lr >= 0.35), lr_balanced.predict(...)

# Cella 27 — Grafico a barre generato DINAMICAMENTE da results_df (nessun numero hardcoded!)
recall_base  = (results_df['Recall (Orig)'] * 100).tolist()
recall_clust = (results_df['Recall (Clust)'] * 100).tolist()
auc_base     = results_df['AUC (Orig)'].tolist()
auc_clust    = results_df['AUC (Clust)'].tolist()
plt.savefig(f'{IMAGES_DIR}/confronto_prestazioni_barre.png', dpi=300, bbox_inches='tight')
```

### Verifica Output Reale (`Celle 21 e 23` vs Slide 8)
- **Confronto Base $\rightarrow$ +Cluster (Test Set, $\tau=0.50$):**
  - LogReg: Acc `0.8062 -> 0.8034`, Rec `0.5588 -> 0.5508`, F1 `0.6049 -> 0.5980`, AUC `0.8422 -> 0.8420` (Base: `TN=927, FP=108, FN=165, TP=209`).
  - CART: Acc `0.7942`, Rec `0.5401`, F1 `0.5821`, AUC `0.8267` identici (Base: `TN=917, FP=118, FN=172, TP=202`).
  - MLP: Acc `0.7999 -> 0.8034`, Prec `0.6554 -> 0.6655`, Rec `0.5187 -> 0.5214`, F1 `0.5791 -> 0.5847`, AUC `0.8424 -> 0.8429` (Base: `TN=933, FP=102, FN=180, TP=194`).
- **Calibrazione Soglia LogReg (Test Set, Cella 23):**
  - $\tau = 0.50$: Acc `0.8062`, Prec `0.6593`, Rec `0.5588`, F1 `0.6049` (`TP=209, FN=165, FP=108`).
  - $\tau = 0.35$: Acc `0.7651`, Prec `0.5445`, Rec `0.7032`, F1 **`0.6138`** (`TP=263, FN=111, FP=220`).
  - `balanced`: Acc `0.7381`, Prec `0.5043`, Rec **`0.7834`**, F1 `0.6136` (`TP=293, FN=81, FP=288`).

---

## Slide 9 — Spiegabilità Business: Regole dell'Albero Decisionale (CART)

### Cosa compare sulla Slide
- Grafico `albero_decisionale.png` (primi 3 livelli, Training Set `N = 5.634`, dati non scalati).
- Box 1: Indice di Gini $G_i = 1 - \sum_k p_{i,k}^2 \in [0, 0.5]$.
- Box 2: Arancione = Rimasto, Blu = Abbandono, radice `value = [4139, 1495]`.
- Box 3: Regola critica: `tenure <= 16.5` $\rightarrow$ Fibra (`68.3%`) + `TotalCharges <= 120 €` $\rightarrow$ `165/193` (`85.5%`, $G=0.248$); senza `TechSupport`: `87.6%` (`162/185`).

### Codice Corrispondente (`Cella 29`)

```python
# Cella 29 — Estrazione modello CART Base (addestrato a max_depth=5) e visualizzazione primi 3 livelli
dt_model_orig = fitted_models['Decision Tree (CART)']['orig']
tree_labels = [c.replace('_', '\n', 1) for c in X_train_tree_df.columns]

fig, ax = plt.subplots(figsize=(24, 8.2))
anns = plot_tree(
    dt_model_orig,
    max_depth=3,                        # Mostra i primi 3 livelli di split per leggibilità
    feature_names=tree_labels,
    class_names=['Rimasto', 'Abbandono'],
    filled=True, rounded=True, fontsize=12, ax=ax
)
# Pulizia visiva: nasconde i nodi segnaposto '(...)' dei rami troncati al livello 4-5
for a in anns:
    if a.get_text().strip() == '(...)':
        a.set_visible(False)
        if getattr(a, 'arrow_patch', None) is not None:
            a.arrow_patch.set_visible(False)

plt.savefig(f'{IMAGES_DIR}/albero_decisionale.png', dpi=300, bbox_inches='tight')
print(export_text(dt_model_orig, feature_names=list(X_train_tree_df.columns), max_depth=3))
```

### Nota Tecnica di Precisione (Importantissima per l'Orale!)
1. **Addestramento vs Visualizzazione:** Il modello `dt_model_orig` è **addestrato con `max_depth=5`** (`Cella 19`, 32 foglie finali), mentre nella `Cella 29` `plot_tree(..., max_depth=3)` **visualizza solo i primi 3 livelli** per mantenere il grafico nitido e leggibile sulla slide (e nasconde i box grigi `(...)`).
2. **Lettura esatta del nodo `165/193` (85,5%):**
   - Radice (livello 0): `tenure <= 16.5`, `samples = 5634`, `value = [4139, 1495]`, `gini = 0.39`.
   - Livello 1 (True): `InternetService_Fiber optic <= 0.5`, `samples = 2042`, `value = [1120, 922]`.
   - Livello 2 (False = **Fibra Ottica**): `TotalCharges <= 120.0`, `samples = 868`, `value = [275, 593]` (**68.3%** churn).
   - Livello 3 (True = **`TotalCharges <= 120.0`**): `samples = 193`, `value = [28, 165]` (**85.5%** churn, `gini = 0.248`), il cui **successivo split** stampato in testa al nodo è `TechSupport_Yes <= 0.5`.
   - Livello 4 (nascosto dal `max_depth=3` del plot): andando nel figlio True (`TechSupport_Yes <= 0.5`, cioè **senza supporto tecnico**), il nodo scende a `185` clienti di cui `162` in churn (**87.6%**, `gini = 0.218`); nei soli `8` clienti *con* supporto tecnico il churn crolla al `37.5%` (`3/8`)!

---

## Slide 10 — Discussione Critica, Conclusioni & Sviluppi Futuri

### Cosa compare sulla Slide
1. Ruolo reale del clustering (descrittivo + lieve boost MLP `0.843` AUC).
2. Razionalità economica & soglia CAC (Recall da `55.9%` fino a `78.3%`).
3. Confronto modelli (LogReg per Rasoio di Occam) & limiti (cluster sferici, dataset statico, GDPR/fairness).
4. Sviluppi futuri (Random Forest/XGBoost, Soft Clustering `transform()`, SMOTE e Cost-Sensitive Learning).

### Corrispondenza nel Codice (`Cella 30` + Sintesi Sperimentale)
La `Cella 30` (Markdown finale del notebook) riassume esattamente questi punti sulla base di tutti gli output prodotti in precedenza:
- **Ruolo descrittivo del clustering:** confermato da `Cella 14` (`43.4%` Train / `43.3%` Test sul Cluster 0) e dal contrasto tra i risultati Base e `+Clust` in `Cella 21`.
- **Rasoio di Occam:** confermato confrontando `results_df` (`Cella 21`) e il rapporto di complessità (`31` parametri per LogReg contro `4.097–4.225` parametri per MLP).
- **Razionalità economica sul CAC:** dimostrata numericamente nella `Cella 23` (`FN` dimezzati da `165` a `81` con `class_weight='balanced'`).
- **Immagini generate dal notebook e usate solo nella Relazione (non nelle Slide):** se il prof le chiede, ricorda che `Cella 8` genera anche `distribuzione_churn.png`, `Cella 14` genera `distribuzione_churn_cluster.png` e `Cella 25` genera `curve_roc.png` — tutte salvate automaticamente in `relazione/images/` e incluse in `relazione/relazione.pdf`.
