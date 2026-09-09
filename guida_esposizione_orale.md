# 🎙️ Guida Completa per l'Esposizione Orale all'Esame di IA

Questa guida ti fornisce il **discorso pronto parola per parola** per ciascuna delle 10 slide della presentazione ([presentazione.pdf](file:///home/massimo/Scrivania/IA2026/relazione/presentazione.pdf)), corredato dalle **domande probabili del professore** e dalle **risposte vincenti** collegate alla teoria del corso (Prof. G. Fiumara / testo Russell & Norvig).

---

## 📌 Slide 1: Titolo e Copertina

### 🗣️ Cosa dire a voce:
> *"Buongiorno professore, il mio progetto riguarda la **Churn Prediction e Segmentazione Comportamentale** applicata al settore delle telecomunicazioni, basato sul dataset IBM Telco Customer Churn. L'analisi integra metodologie di apprendimento non supervisionato e supervisionato trattate durante il corso."*

### ❓ Domande probabili dell'Esaminatore:
* **P:** *"Perché unire clustering non supervisionato e classificazione supervisionata?"*
  * **R:** *"Perché il clustering K-Means raggruppa i clienti in segmenti omogenei senza conoscere l'esito del Churn, creando una mappa spaziale comportamentale. Utilizzando il cluster come nuova feature nel classificatore supervisionato, forniamo al modello una conoscenza a priori che ne migliora le capacità predittive (in particolare per modelli complessi come la Rete Neurale)."*

---

## 📌 Slide 2: Inquadramento Teorico: Epistemologia e Agente PEAS

### 🗣️ Cosa dire a voce:
> *"Iniziamo dall'inquadramento teorico del problema basato su Russell & Norvig. L'Intelligenza Artificiale si definisce su due assi: Pensare vs. Agire e Umanamente vs. Razionalmente. Il mio lavoro si colloca saldamente nel quadrante dell'**Agire Razionalmente**: l'obiettivo non è simulare i processi cognitivi umani, ma massimizzare l'utilità attesa (la fidelizzazione dei clienti) sulla base dei dati storici CRM. 
> 
> A destra vediamo la modellizzazione **PEAS** dell'agente:
> - **Performance (P)**: Massimizzare la Recall per intercettare il maggior numero di clienti a rischio ed evitare i Falsi Negativi.
> - **Environment (E)**: L'ambiente CRM Telco, caratterizzato dall'essere stocastico, parzialmente osservabile, sequenziale nel tempo e multi-dimensionale.
> - **Actuators (A)**: Campagne promozionali mirate, sconti personalizzati e supporto clienti.
> - **Sensors (S)**: Registri di fatturazione, tenure e tipologia di contratti."*

### ❓ Domande probabili dell'Esaminatore:
* **P:** *"Perché l'ambiente CRM è definito stocastico e parzialmente osservabile?"*
  * **R:** *"È **stocastico** perché la decisione finale del cliente di abbandonare non è deterministica ma incerta; è **parzialmente osservabile** perché il database del CRM registra solo i dati contrattuali e finanziari, ma non registra fattori esterni invisibili come offerte di concorrenti o motivazioni personali dell'utente."*

---

## 📌 Slide 3: Fondamenti di Machine Learning e Approccio Probabilistico

### 🗣️ Cosa dire a voce:
> *"Ho formalizzato l'apprendimento automatico seguendo la celebre definizione di Tom Mitchell del 1997, basata sulla terna T, E, P:
> - **Task (T)**: La classificazione binaria finalizzata a predire l'abbandono (Churn = 1 vs No Churn = 0).
> - **Experience (E)**: L'addestramento sui dati storici dei 7.043 clienti IBM Telco.
> - **Performance (P)**: L'ottimizzazione della Recall al posto della semplice accuratezza.
> 
> A destra giustifichiamo la necessità dell'**approccio probabilistico** (di Bayes e Laplace) rispetto alla logica formale rigida (di Boole e Frege). Il comportamento umano è intrinsecamente non deterministico; un sistema a regole rigide fallirebbe per l'esplosione combinatoria delle variabili e la parzialità delle informazioni. La probabilità $P(\text{Churn} \mid \text{Dati})$ ci permette di modellare l'incertezza esprimendo un grado di rischio continuo tra $0\%$ e $100\%$ per ciascun cliente."*

### ❓ Domande probabili dell'Esaminatore:
* **P:** *"Perché preferiamo la probabilità $P(\text{Churn} \mid \text{Dati})$ rispetto a una regola secca SE... ALLORA...?"*
  * **R:** *"Perché una probabilità continua permette all'azienda di effettuare un **ranking** (graduatoria) dei clienti a rischio, allocando le risorse promozionali prioritariamente sui clienti con la probabilità più alta (es. $> 80\%$), anziché applicare sconti a massa."*

---

## 📌 Slide 4: Preprocessing dei Dati e Sbilanciamento del Target

### 🗣️ Cosa dire a voce:
> *"Il dataset comprende 7.043 clienti e 21 variabili. Nel preprocessing abbiamo gestito tre aspetti fondamentali:
> - **Data Cleaning**: Gli 11 valori vuoti di spesa totale sono stati imputati a `0.0€` perché relativi a clienti nuovi (`tenure = 0`), salvaguardando dati utili.
> - **Normalizzazione**: I dati numerici sono stati portati sulla stessa scala per evitare che le cifre in Euro schiacciassero i mesi di permanenza nei calcoli dell'IA.
> - **One-Hot Encoding**: La conversione dei testi in numeri binari ($0$ e $1$) permette ai modelli di elaborare le categorie senza creare gerarchie fittizie.
> 
> Un elemento cruciale è poi lo **sbilanciamento del target**: solo il **26.54%** dei clienti abbandona. Un classificatore banale (*dummy classifier*) che dicesse sempre 'No Churn' otterrebbe il 73.46% di Accuracy ma una **Recall pari a zero**, risultando del tutto inutile. Identificare chi abbandona ha priorità assoluta, perché perdere un cliente costa molto di più che offrire una promozione.
> 
> A destra, i **Boxplot** mostrano visivamente due pattern chiari: chi abbandona (scatola rossa) ha una permanenza mediana di soli **10 mesi** (contro i **38 mesi** di chi resta) e paga bollette medie più alte (**80€/mese** contro **64€/mese**)."*

### ❓ Domande probabili dell'Esaminatore:
* **P:** *"Perché avete imputato 0.0€ ai 11 valori vuoti di `TotalCharges` invece di eliminare le righe o usare la media?"*
  * **R:** *"Perché quei 11 clienti avevano `tenure = 0` mesi (appena acquisiti), quindi la spesa totale accumulata era logicamente pari a 0.0€. Rimuovere le righe avrebbe causato una perdita di dati non necessaria, mentre imputare la media avrebbe introdotto una forte distorsione."*
* **P:** *"A cosa servono esattamente lo `StandardScaler` e il `drop='first'` nel One-Hot Encoding?"*
  * **R:** *"Lo **StandardScaler** porta le feature continue alla stessa scala (media 0 e varianza 1, $z = \frac{x-\mu}{\sigma}$), fondamentale per gli algoritmi basati su distanze (K-Means) e gradienti (Regressione Logistica ed MLP). Il **`drop='first'`** elimina la prima colonna ridondante di ogni variabile categoriale per evitare la trappola delle dummy variable (*multicollinearità perfetta*)."*
* **P:** *"Come è fatta la struttura del Boxplot e cosa rappresenta la linea centrale?"*
  * **R:** *"La linea nera centrale è la **Mediana** (50° percentile). La scatola racchiude il 50% centrale dei dati (dall'intervallo $Q_1$ al 25% fino a $Q_3$ al 75%, cioè l'IQR). I baffi indicano i valori minimi e massimi non anomali, mentre i pallini esterni indicano gli outlier."*

---

## 📌 Slide 5: Apprendimento Non Supervisionato: Clustering K-Means

### 🗣️ Cosa dire a voce:
> *"Per l'apprendimento non supervisionato ho utilizzato l'algoritmo **K-Means** per raggruppare spontaneamente i clienti in segmenti omogenei senza usare l'etichetta del Churn.
> 
> Per determinare il numero ottimale di cluster $K$, ho analizzato due metriche complementari:
> - **Inerzia (Curva Blu)**: Misura l'errore quadratico interno al cluster (WCSS).
> - **Silhouette Score (Curva Arancione)**: Misura quanto un punto sia simile al suo cluster rispetto agli altri.
> 
> Ho selezionato **$K=3$** come scelta ottimale: rappresenta il chiaro 'punto di gomito' dell'inerzia ed ha un valore di Silhouette elevato ($s = 0.286$). Sebbene $K=2$ avesse uno score leggermente più alto, creava una divisione triviale (solo chi ha internet e chi non ce l'ha). Per $K \ge 4$ si osserva invece un crollo della Silhouette, indice di over-clustering e sovrapposizione geometrica."*

### ❓ Domande probabili dell'Esaminatore:
* **P:** *"Cos'è la funzione obiettivo minimizzata da K-Means?"*
  * **R:** *"K-Means minimizza la WCSS (Within-Cluster Sum of Squares), ovvero la somma delle distanze euclidee al quadrato tra ogni punto e il baricentro $\mu_j$ del suo cluster: $\sum_{i=1}^{N} \min_{\mu_j} \|\mathbf{x}_i - \mathbf{\mu}_j\|^2$."*

---

## 📌 Slide 6: Segmentazione PCA e Profilo dei Segmenti

### 🗣️ Cosa dire a voce:
> *"La tabella in alto riassume i 3 profili di business emersi dal K-Means:
> - **Cluster 0 (Volatili)**: Nuovi clienti con tenure breve (13 mesi), spesa mensile alta (75€) e un tasso di churn record del **47.08%** (quasi il doppio della media aziendale!). Rappresenta il target prioritario per l'Agente IA.
> - **Cluster 1 (Fedeli High-Value)**: Utenti storici (59 mesi), spesa alta (90€) e tasso di churn contenuto (15.36%).
> - **Cluster 2 (Low-Cost)**: Clienti con servizi base (27€/mese) ed elevata stabilità (churn al 12.30%).
> 
> A destra in basso, la proiezione 2D sulle prime due componenti principali della **PCA** (Principal Component Analysis) conferma visivamente che i tre cluster identificati sono geometricamente ben separati."*

### ❓ Domande probabili dell'Esaminatore:
* **P:** *"Perché abbiamo usato la PCA qui?"*
  * **R:** *"La PCA è una tecnica di riduzione della dimensionalità che proietta i dati nello spazio a varianza massima definito dai vettori propri (eigenvectors) della matrice di covarianza, permettendoci di visualizzare in 2D un dataset ad alte dimensioni."*

---

## 📌 Slide 7: Pipeline Sperimentale Supervisionata

### 🗣️ Cosa dire a voce:
> *"Questa slide descrive la pipeline di classificazione supervisionata. A sinistra vediamo il diagramma di flusso: dai dati preprocessati applichiamo uno **split stratificato 80/20** su 7.043 clienti per mantenere l'esatto sbilanciamento del target ($26.54\%$ Churn) sia nel training set che nel test set.
> 
> Abbiamo selezionato tre famiglie di modelli rappresentative trattate nel corso:
> 1. **Regressione Logistica**: Baseline parametrica per valutare la separabilità lineare iniziale dei dati.
> 2. **Albero Decisionale CART (White-Box)**: Modello non parametrico trasparente, regolarizzato con `max_depth=5` per evitare l'overfitting ed estrarre regole aziendali esplicite.
> 3. **Rete Neurale MLP (Black-Box)**: Multi-Layer Perceptron feed-forward con attivazione ReLU negli strati nascosti e sigmoide in output, per catturare relazioni non lineari complesse."*

### ❓ Domande probabili dell'Esaminatore:
* **P:** *"Cosa significa split stratificato e perché è obbligatorio?"*
  * **R:** *"Lo split stratificato garantisce che la percentuale della classe minoritaria ($26.54\%$ Churn) rimanga esattamente identica sia nel Train Set che nel Test Set, evitando che lo split casuale crei subset distorti."*

---

## 📌 Slide 8: Confronto delle Prestazioni Sperimentali

### 🗣️ Cosa dire a voce:
> *"In questa slide analizziamo i risultati ottenuti sul Test Set ($1.409$ istanze) ponendo a confronto la configurazione Base con la configurazione arricchita dalla feature di Clustering ($K=3$).
> 
> I risultati evidenziano due conclusioni fondamentali:
> 1. **Miglior Modello Globale**: La **Regressione Logistica** registra le prestazioni migliori in assoluto (Accuracy **80.6%** e Recall **55.9%**), dimostrando che i dati contrattuali Telco possiedono un'elevata separabilità lineare.
> 2. **Impatto del Clustering sulla Rete Neurale (MLP)**: L'aggiunta della colonna di cluster fa compiere un balzo straordinario alla **Recall del modello MLP dal 42.8% al 49.2% (+6.4%)**. In questo contesto, l'apprendimento non supervisionato agisce come una forma di *Feature Engineering*, guidando l'ottimizzatore Adam nell'evitare minimi locali."*

### ❓ Domande probabili dell'Esaminatore:
* **P:** *"Perché la Regressione Logistica e CART non migliorano con il cluster, mentre l'MLP sì?"*
  * **R:** *"Perché la Regressione Logistica e CART apprendono già direttamente i pesi e i taglieri lineari sulle feature originarie di permanenza e costo; la Rete Neurale MLP, essendo un ottimizzatore non convesso, trae invece grande beneficio da un'indicazione a priori del gruppo comportamentale pre-calcolato dal K-Means."*

---

## 📌 Slide 9: Spiegabilità Business: Regole dell'Albero Decisionale (CART)

### 🗣️ Cosa dire a voce:
> *"L'Albero Decisionale CART ci fornisce il valore indispensabile della trasparenza (*White-Box*). 
> 
> Per spiegare la struttura in tre passi:
> - **Passo 1 (Gini)**: L'albero misura l'impurità dei nodi con l'Indice di Gini ($G \in [0, 0.5]$). $G=0$ indica un nodo puro al $100\%$ (tinta scura), mentre $G=0.5$ indica la massima confusione (tinta chiara).
> - **Passo 2 (Colori)**: I nodi arancioni indicano la prevalenza di clienti che RESTANO; i nodi blu indicano la prevalenza di clienti che ABBANDONANO.
> - **Passo 3 (La Regola Estratta)**: Lo split di radice individua come primo fattore la permanenza breve ($\text{tenure} \le 16.5 \text{ mesi}$). Sotto questo ramo, chi possiede un abbonamento in **Fibra Ottica** finisce nei nodi blu scuri a massimo churn (**$>70\%$ di abbandoni**). 
> 
> L'Agente CRM può utilizzare questa regola esplicita per intervenire tempestivamente con sconti e assistenza sui clienti con Fibra di recente acquisizione."*

### ❓ Domande probabili dell'Esaminatore:
* **P:** *"Cos'è la formula dell'Indice di Gini e perché si usa?"*
  * **R:** *"La formula è $G_i = 1 - \sum_{k=1}^{K} p_{i,k}^2$. Misura il grado di disordine/impurità di un nodo. CART lo sceglie per trovare lo split che massimizza il guadagno di purezza (*Information Gain*)."*

---

## 📌 Slide 10: Discussione Critica, Conclusioni & Sviluppi Futuri

### 🗣️ Cosa dire a voce:
> *"Per concludere la presentazione, ricapitoliamo 4 punti fondamentali:
> 1. **Valore dell'Integrazione Unsupervised/Supervised**: Il clustering K-Means arricchisce lo spazio delle feature offrendo una mappa pre-calcolata che aumenta la Recall dell'MLP del **+6.4%**.
> 2. **Razionalità Economica e Costo CAC**: L'Agente IA dota di priorità la Recall perché il Customer Acquisition Cost (CAC) per sostituire un cliente perso è nettamente superiore al costo di retention.
> 3. **Trio dei Modelli**: La Regressione Logistica garantisce il miglior rendimento predittivo, la Rete Neurale beneficia maggiormente del clustering, e l'Albero CART offre la spiegabilità necessaria per le regole aziendali.
> 4. **Sviluppi Futuri (Next Steps)**: Per superare i limiti emersi prevediamo l'uso di **SMOTE** (per generare campioni sintetici di Churn), modelli Ensemble come **Random Forest** (alberi in parallelo a maggioranza) e **XGBoost** (alberi in serie sequenziali), ed infine il **Cost-Sensitive Learning** per pesare la funzione di perdita direttamente in Euro (€) sui costi CAC reali.
> 
> Vi ringrazio per l'attenzione e sono a disposizione per le vostre domande!"*

---

## 📚 Glossario Tascabile dei Termini per l'Orale

| Termine | Definizione Breve da Usare all'Orale |
| :--- | :--- |
| **PEAS** | Modello formalizzante dell'agente razionale: *Performance, Environment, Actuators, Sensors*. |
| **CAC** | *Customer Acquisition Cost*: costo economico sostenuto dal marketing per trovare un nuovo cliente. |
| **Recall** | Quanti clienti in abbandono reale l'IA è riuscita a scovare (minimizza i Falsi Negativi). |
| **AUC-ROC** | Area Sotto la Curva ROC: misura la capacità del modello di separare le due classi a prescindere dalla soglia. |
| **K-Means** | Algoritmo di clustering non supervisionato basato sulla minimizzazione delle distanze dal centroide (WCSS). |
| **Silhouette Score** | Indice tra $-1$ e $+1$ che misura compattezza interna al cluster e separazione dagli altri gruppi. |
| **PCA** | *Principal Component Analysis*: riduzione della dimensionalità tramite componenti principali ortogonali a varianza massima. |
| **StandardScaler** | Trasformazione $z = (x - \mu)/\sigma$ che porta ogni feature ad avere media 0 e varianza 1. |
| **One-Hot Encoding** | Conversione di variabili categoriali in colonne binarie ($0/1$), usata con `drop='first'` per evitare la multicollinearità. |
| **Indice di Gini** | Misura di impurità di un nodo negli alberi CART ($0=$ totalmente puro, $0.5=$ 50/50). |
| **CART** | *Classification and Regression Trees*: algoritmo per la costruzione di alberi decisionali binari. |
| **MLP** | *Multi-Layer Perceptron*: rete neurale artificiale feed-forward con strati nascosti e retropropagazione dell'errore (Backpropagation). |
| **SMOTE** | *Synthetic Minority Over-sampling Technique*: algoritmo per generare campioni sintetici della classe minoritaria. |
