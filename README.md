# Progetto Laboratorio IA 2025/2026 - Segmentazione Clienti e Churn Prediction

Questo repository contiene il codice e la documentazione per il **Progetto 1: Segmentazione clienti + churn prediction (Telco Customer Churn)**, sviluppato da Massimo Mantineo (Matricola: 541924) per il corso di Laboratorio di Intelligenza Artificiale (Università degli Studi di Messina).

## 📂 Struttura del Progetto
```
IA/
│
├── data/
│   └── Telco-Customer-Churn.csv       # Dataset originale IBM (7.043 clienti, 21 colonne)
│
├── notebooks/
│   └── churn_analysis.ipynb           # Pipeline completa eseguita (zero data leakage)
│
├── lezioni/                           # Slide ufficiali del corso e compendio teorico
│   ├── README.md                      # Compendio dettagliato delle 8 lezioni
│   └── *.pdf                          # Slide delle lezioni 01-08
│
├── relazione/
│   ├── images/                        # Grafici generati direttamente dal notebook
│   ├── relazione.tex / relazione.pdf  # Relazione finale in LaTeX
│   └── presentazione.tex / .pdf       # Slide d'esame (10 slide Beamer)
│
├── documenti_progetto/                # Linee guida e schede ufficiali del corso
│   ├── Progetti.pdf
│   └── Descrizione_progetti.pdf
│
├── requirements.txt                   # Dipendenze Python richieste
└── README.md                          # Questo file
```

---

## 🚀 Come Riprodurre il Progetto

Il progetto è stato sviluppato in ambiente Linux utilizzando **Python 3** (verificato su **Python 3.15**, `venv/` locale) ed è completamente riproducibile eseguendo i passi seguenti.

### 1. Prerequisiti
Assicurati di avere Python 3 e `pip` installati sul sistema.

### 2. Configurazione dell'Ambiente Virtuale
```bash
# Creazione dell'ambiente virtuale
python3 -m venv venv

# Attivazione dell'ambiente virtuale
source venv/bin/activate

# Installazione delle dipendenze
pip install -r requirements.txt
```

### 3. Esecuzione del Jupyter Notebook
```bash
jupyter notebook notebooks/churn_analysis.ipynb
```
Seleziona **Kernel -> Restart & Run All** (oppure esegui `jupyter nbconvert --to notebook --execute --inplace notebooks/churn_analysis.ipynb`). Tutti i grafici in `relazione/images/` e tutte le tabelle dei risultati vengono rigenerati in modo deterministico (`random_state=42`) senza *data leakage* (split stratificato 80/20 eseguito prima di scaling, K-Means e PCA).

---

## 📈 Sintesi dei Risultati Principali

1. **Segmentazione (K-Means sul Training Set, $K=3$, Silhouette $= 0.2817$):**
   * **Cluster 0 (Volatili):** Permanenza media ~15.2 mesi, spesa ~67.87€/mese, tasso di churn critico (**43.4%** su train, 43.3% su test).
   * **Cluster 1 (Fedeli High-Value):** Lunga permanenza (~57.7 mesi), spesa elevata (~89.44€/mese), churn contenuto (**15.9%**).
   * **Cluster 2 (Low-Cost):** Servizi base (~21.12€/mese), permanenza media (~30.2 mesi), churn minimo (**7.2%**).
2. **Classificazione (Test Set, 1.409 istanze, soglia standard $\tau=0.50$):**
   * **Regressione Logistica (Baseline):** Miglior modello complessivo alla soglia standard (**Accuracy 80.62%**, **Recall 55.88%**, **F1-Score 0.6049**, **AUC-ROC 0.8422**). Con calibrazione della soglia ($\tau=0.35$) o `class_weight='balanced'`, la Recall sale al **70.32%--78.34%** (`F1 ~ 0.614`).
   * **Decision Tree CART (`max_depth=5`, feature non scalate):** **Accuracy 79.42%**, **Recall 54.01%**, **AUC-ROC 0.8267**; fornisce regole *White-Box* leggibili in mesi ed Euro (nuovi clienti in Fibra Ottica con `tenure <= 16.5` mesi e `TotalCharges <= 120€` raggiungono l'**85.5% di churn**, salendo all'**87.6%** senza supporto tecnico).
   * **MLP (`(64, 32)`, L2 $\alpha=0.1$, *early stopping*):** Raggiunge **AUC-ROC 0.8424** (base) e **0.8429** (+Cluster, miglior AUC assoluta; Recall `51.87% -> 52.14%`, Accuracy `79.99% -> 80.34%`).
   * **Impatto del Clustering:** Il valore principale del K-Means è descrittivo/strategico; come feature aggiuntiva apporta un lieve miglioramento sull'MLP mentre risulta ridondante su Regressione Logistica e CART (confermando la preferenza per il modello più semplice e stabile: la **Regressione Logistica**).
