# Progetto Laboratorio IA 2025/2026 - Segmentazione Clienti e Churn Prediction

Questo repository contiene il codice e la relazione per il **Progetto 1: Segmentazione clienti + churn prediction (Telco Customer Churn)**, sviluppato da Massimo Mantineo per il corso di Laboratorio di Intelligenza Artificiale.

## 📂 Struttura del Progetto
La struttura del repository è organizzata come segue:
```
IA2026/
│
├── data/
│   └── Telco-Customer-Churn.csv   # Dataset originale IBM
│
├── notebooks/
│   └── churn_analysis.ipynb       # Jupyter Notebook con l'intera pipeline eseguita
│
├── relazione/
│   ├── relazione.tex              # Codice sorgente LaTeX della relazione finale
│   └── presentazione.tex          # Codice sorgente LaTeX Beamer per le slide d'esame
│
├── documenti_progetto/             # PDF con le specifiche e le linee guida del corso
│   ├── Progetti.pdf
│   └── Descrizione_progetti.pdf
│
├── requirements.txt               # Dipendenze Python richieste
└── README.md                      # Questo file
```

---

## 🚀 Come Riprodurre il Progetto

Il progetto è stato sviluppato in ambiente Linux utilizzando **Python 3.12** ed è completamente riproducibile seguendo i passi indicati di seguito.

### 1. Prerequisiti
Assicurati di avere Python 3 e il gestore pacchetti `pip` installati sul tuo sistema.

### 2. Configurazione dell'Ambiente Virtuale
Si consiglia l'uso di un virtual environment per non interferire con le librerie globali del sistema:

```bash
# Creazione dell'ambiente virtuale
python3 -m venv venv

# Attivazione dell'ambiente virtuale
source venv/bin/activate

# Installazione delle dipendenze
pip install -r requirements.txt
```

### 3. Esecuzione del Jupyter Notebook
Avvia il server Jupyter e apri il notebook di analisi:

```bash
jupyter notebook notebooks/churn_analysis.ipynb
```
Puoi eseguire tutte le celle in sequenza selezionando **Kernel -> Restart & Run All**. Il notebook contiene già tutti gli output e i grafici pronti per la consultazione (elaborati durante l'ultimo addestramento).

---

## 📈 Sintesi dei Risultati Principali
1. **Segmentazione (K-Means, K=3):**
   * **Cluster 0 (Clienti Volatili):** Breve permanenza (~15 mesi), tassi di abbandono altissimi (~43.2%).
   * **Cluster 1 (Clienti High-Value):** Lunga permanenza (~58 mesi), spesa elevata (~90€/mese), abbandono basso (~15.4%).
   * **Cluster 2 (Clienti Low-Cost):** Spesa minima (~21€/mese), stabilità massima, abbandono trascurabile (~7.4%).
2. **Classificazione (Churn Prediction):**
   * **Regressione Logistica (Baseline):** Ottiene l'accuratezza bilanciata migliore (~80.6%) e un'area sotto la curva ROC (AUC) di ~0.842.
   * **Decision Tree (max_depth=5):** Ottiene ~79.4% di accuratezza. È il modello ideale da presentare all'esame per l'altissima interpretabilità delle sue regole decisionali.
   * **Impatto del Clustering:** L'inclusione dei cluster non altera significativamente le performance di modelli lineari o ad albero poiché le informazioni di partenza sono già catturate, ma offre un leggero miglioramento nella stabilità della Recall per la rete neurale MLP.
