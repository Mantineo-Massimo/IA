# Dispense e Lezioni del Corso di Laboratorio di Intelligenza Artificiale
**Docente:** Prof. Giacomo Fiumara (`gfiumara@unime.it`)  
**Università degli Studi di Messina** — Dipartimento MIFT  
**Corso di Laurea Triennale in Informatica (L-31)** — A.A. 2025/2026  

---

## 📚 Testi di Riferimento Ufficiali
1. **Russell, S. & Norvig, P.** — *Artificial Intelligence: A Modern Approach* (4th Edition), Pearson.
2. **Géron, A.** — *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (3rd Edition), O'Reilly Media.
3. **Mitchell, T. M.** — *Machine Learning*, McGraw-Hill, 1997.

---

## 🗺️ Indice delle Lezioni e Correlazione con il Progetto

| File | Titolo Lezione | Fonte | Argomenti Principali | Applicazione nel Progetto Telco Churn |
| :--- | :--- | :--- | :--- | :--- |
| [`01-Introduzione.pdf`](01-Introduzione.pdf) | **Introduzione all'IA** | Russell & Norvig, Cap. 1 | Definizione di IA, 4 approcci, fondamenti storici, Turing test, Beneficial AI | Approcci alla Definizione di IA e motivazione dell'Agire Razionalmente |
| [`02-Agenti_intelligenti.pdf`](02-Agenti_intelligenti.pdf) | **Agenti Intelligenti** | Russell & Norvig, Cap. 2 | Agente e ambiente, PEAS, proprietà degli ambienti, tassonomia agenti | Modellizzazione PEAS dell'Agente CRM per la fidelizzazione clienti |
| [`03-Introduzione_al_ML.pdf`](03-Introduzione_al_ML.pdf) | **Machine Learning Landscape** | Géron, Cap. 1 | Definizione $\langle T, E, P \rangle$, tipologie ML, pipeline, rischi (overfitting/bias) | Formulazione del task di apprendimento, gestione sbilanciamento e cleaning |
| [`04-Classificazione.pdf`](04-Classificazione.pdf) | **Classificazione** | Géron, Cap. 3 | MNIST, confusion matrix, Precision, Recall, F1, curve PR e ROC-AUC | Selezione metriche, curve ROC e calibrazione soglia ($\tau=0.35$) su sbilanciamento |
| [`05-Decision Trees.pdf`](05-Decision Trees.pdf) | **Decision Trees** | Géron, Cap. 6 | Algoritmo CART, impurità di Gini, regolarizzazione, White-Box, cenni su PCA | Modello CART interpretabile (`max_depth=5`, feature non scalate), regole su tenure e fibra |
| [`06-Unsupervised_Learning.pdf`](06-Unsupervised_Learning.pdf) | **Unsupervised Learning** | Géron, Cap. 9 | K-Means, inerzia WCSS, Silhouette, Elbow, DBSCAN, Isolation Forest, GMM, Anomaly Detection | Segmentazione clienti ($K=3$) con K-Means e profilazione cluster |
| [`07-Introduction_to_NNs.pdf`](07-Introduction_to_NNs.pdf) | **Reti Neurali con Keras** | Géron, Cap. 10 | TLU, Perceptron, XOR, MLP, attivazioni (ReLU, Sigmoide), Backprop, Softmax+CE, Early Stopping | Rete Neurale MLP feedforward + Early Stopping per la classificazione non lineare |
| [`08-Training_DeepNeuralNetworks.pdf`](08-Training_DeepNeuralNetworks.pdf) | **Training Deep Neural Networks** | Géron, Cap. 11 | Vanishing/Exploding gradients, He/Glorot init, BatchNorm, Transfer Learning, Adam, L1/L2, Dropout | Regolarizzazione L2 ($\alpha=0.1$) e ottimizzatore Adam nell'MLP |

---

## 📖 Analisi Dettagliata per Singola Lezione

---

### Lezione 01: Introduzione all'Intelligenza Artificiale
* **File:** [`01-Introduzione.pdf`](01-Introduzione.pdf) (86 slide)
* **Riferimento:** Stuart Russell & Peter Norvig, *Artificial Intelligence: A Modern Approach*, Capitolo 1.

#### 1. I Quattro Approcci alla Definizione di IA
Russell & Norvig categorizzano le definizioni storiche di IA lungo due direttrici: **Aderenza alle prestazioni umane** vs. **Razionalità**, e **Pensiero** vs. **Comportamento**:

| Dimensione | Umanamente (Human-centric) | Razionalmente (Ideal Rationality) |
| :--- | :--- | :--- |
| **Pensiero** | **Pensare Umanamente (Thinking Humanly):** Scienze cognitive, modelli computazionali dei processi cerebrali e introspezione psicologica. | **Pensare Razionalmente (Thinking Rationally):** "Leggi del pensiero" aristoteliche, logica formale e sillogismi inconfutabili. |
| **Comportamento** | **Agire Umanamente (Acting Humanly):** Il Test di Turing (1950); un computer supera il test se un interrogatore umano non distingue le sue risposte da quelle umane. | **Agire Razionalmente (Acting Rationally):** Il paradigma dell'**Agente Razionale** che opera in modo da massimizzare il valore atteso della misura di prestazione. |

#### 2. Fondamenti Disciplinari dell'IA
* **Filosofia (428 a.C.–presente):** Teorie della mente, logica deduttiva, epistemologia, empirismo e connessione mente-corpo.
* **Matematica (800 d.C.–presente):** Logica formale (Boole, Frege), computabilità e limiti della computazione (Gödel, Turing), probabilità e teoria delle decisioni (Bayes, Fermat, Pascal).
* **Economia (1776–presente):** Teoria dell'utilità attesa (von Neumann & Morgenstern), teoria dei giochi (Nash), ricerca operativa e processi decisionali di Markov (MDP).
* **Neuroscienze (1861–presente):** Funzionamento dei neuroni biologici, potenziale d'azione e plasticità sinaptica (Hebb).
* **Psicologia e Scienze Cognitive (1879–presente):** Psicologia cognitiva, percezione, attenzione e rappresentazione della conoscenza.
* **Ingegneria Informatica (1940–presente):** Hardware, architettura di von Neumann, compilatori e calcolo parallelo.
* **Teoria del Controllo e Cibernetica (1948–presente):** Sistemi retroazionati (feedback loops), funzioni di costo stocastiche.
* **Linguistica (1957–presente):** Elaborazione del linguaggio naturale (NLP), grammatiche generative (Chomsky).

#### 3. Cenni Storici Chiave
* **1943:** McCulloch & Pitts modellano il primo neurone artificiale matematico.
* **1950:** Alan Turing pubblica *"Computing Machinery and Intelligence"* (Test di Turing, machine learning, giochi).
* **1956:** Convegno di Dartmouth; John McCarthy conia ufficialmente il termine **"Artificial Intelligence"**.
* **1969–1986:** Sistemi Basati sulla Conoscenza e Sistemi Esperti (DENDRAL, MYCIN, R1).
* **Anni '70 e fine anni '80:** Gli "Inverni dell'IA" (*AI Winters*, Rapporto Lighthill 1973 e declino dei sistemi esperti) dovuti all'eccesso di aspettative e limiti computazionali.
* **1986–presente:** Rinascita delle reti neurali con la riscoperta della Backpropagation.
* **2011/2012–presente:** Rivoluzione del Deep Learning (ImageNet 2012, GPU, disponibilità di Big Data), fino ai modelli generativi e LLM.
* **Beneficial AI:** Etica, robustezza, fairness, allineamento dell'IA ai valori umani e impatto economico/sociale.

---

### Lezione 02: Agenti Intelligenti e Ambienti
* **File:** [`02-Agenti_intelligenti.pdf`](02-Agenti_intelligenti.pdf) (64 slide)
* **Riferimento:** Stuart Russell & Peter Norvig, *Artificial Intelligence: A Modern Approach*, Capitolo 2.

#### 1. Nozione di Agente e Ciclo Percezione-Azione
* **Definizione di Agente:** Qualsiasi entità in grado di percepire il proprio ambiente attraverso **sensori** e agire su di esso mediante **attuatori**.
* **Sequenza di Percezioni ($P^*$):** La storia completa di tutto ciò che l'agente ha percepito fino all'istante corrente.
* **Funzione Agente ($f$):** Mappa matematica astratta che correla qualsiasi sequenza di percezioni a un'azione:
  $$f: P^* \to A$$
* **Programma Agente:** L'implementazione computazionale concreta che esegue la funzione agente su un'architettura fisica.

#### 2. Il Concetto di Razionalità
* **Misura di Prestazione (Performance Measure):** Criterio oggettivo esterno per valutare il successo dell'agente nel suo ambiente. Non va confusa con il "sentimento soggettivo" dell'agente.
* **Definizione di Agente Razionale:** Per ogni sequenza di percezioni possibile, l'agente seleziona un'azione che **massimizza il valore atteso della misura di prestazione**, date le evidenze fornite dalle percezioni e dalla conoscenza pregressa incorporata nell'agente.
* **Razionalità vs. Onniscienza:** L'onniscienza conosce l'esito reale delle proprie azioni; la razionalità massimizza l'esito *atteso* in condizioni di incertezza.
* **Autonomia:** Capacità dell'agente di apprendere dall'esperienza e compensare la conoscenza iniziale incompleta o errata.

#### 3. Il Modello PEAS
Per definire rigorosamente un problema di progettazione di un agente:
* **P (Performance):** Metriche di efficacia e sicurezza.
* **E (Environment):** L'ambiente operativo circostante.
* **A (Actuators):** Dispositivi o interfacce di output per compiere azioni.
* **S (Sensors):** Canali o interfacce di input per acquisire percezioni.

#### 4. Proprietà dell'Ambiente di Lavoro
1. **Completamente osservabile vs. Parzialmente osservabile:** I sensori rilevano lo stato completo dell'ambiente in ogni istante oppure solo informazioni parziali e rumorose.
2. **Singolo agente vs. Multi-agente:** L'agente opera isolato oppure interagisce con altri agenti (in modo cooperativo o competitivo).
3. **Deterministico vs. Stocastico:** Il prossimo stato dell'ambiente è interamente determinato dallo stato attuale e dall'azione dell'agente, oppure intervengono probabilità e incertezza.
4. **Episodico vs. Sequenziale:** L'esperienza è divisa in episodi atomici indipendenti oppure le azioni attuali condizionano tutte le decisioni future.
5. **Statico vs. Dinamico:** L'ambiente non muta mentre l'agente sta deliberando l'azione, oppure evolve nel tempo.
6. **Discreto vs. Continuo:** Il tempo, gli stati, le percezioni e le azioni sono numerabili/finiti o continui.
7. **Noto vs. Sconosciuto:** L'agente conosce a priori le leggi fisiche/probabilistiche dell'ambiente o deve apprenderle tramite esplorazione.

#### 5. Tassonomia delle Architetture di Agenti
* **Agenti Reattivi Semplici (Simple Reflex Agents):** Selezionano le azioni basandosi unicamente sulla percezione corrente mediante regole condizione-azione (`SE... ALLORA...`). Falliscono in ambienti parzialmente osservabili.
* **Agenti Reattivi Basati su Modello (Model-based Reflex Agents):** Mantengono uno stato interno (*internal state*) che riflette gli aspetti non osservati del mondo e la dinamica temporale dell'ambiente.
* **Agenti con Obiettivi (Goal-based Agents):** Integrano la conoscenza dello stato con una descrizione esplicita degli obiettivi da raggiungere (ricerca e pianificazione).
* **Agenti con Funzione di Utilità (Utility-based Agents):** Valutano i trade-off tra obiettivi conflittuali quantificando la preferenza tramite una funzione di utilità scalare.
* **Agenti che Apprendono (Learning Agents):** Suddivisi in quattro componenti concettuali:
  * **Elemento di Apprendimento (Learning Element):** Genera miglioramenti al comportamento.
  * **Elemento Esecutivo (Performance Element):** Seleziona le azioni esterne.
  * **Critico (Critic):** Confronta il risultato con un criterio di qualità e invia feedback.
  * **Generatore di Problemi (Problem Generator):** Suggerisce azioni esplorative innovative.

---

### Lezione 03: Introduzione al Machine Learning
* **File:** [`03-Introduzione_al_ML.pdf`](03-Introduzione_al_ML.pdf) (87 slide)
* **Riferimento:** Aurélien Géron, *Hands-On Machine Learning*, Capitolo 1.

#### 1. Definizioni Cardine di Machine Learning
* **Arthur Samuel (1959):** *"Il campo di studio che conferisce ai computer la capacità di apprendere senza essere esplicitamente programmati."*
* **Tom Mitchell (1997) — Definizione Operativa:**
  > *"Un programma per calcolatore apprende dall'esperienza $E$ con riferimento a una determinata classe di compiti $T$ e a una misura di prestazione $P$, se le sue prestazioni nel compito $T$, misurate da $P$, migliorano con l'esperienza $E$."*
  $$\langle T, E, P \rangle$$

#### 2. Programmazione Tradizionale vs. Machine Learning
* **Approccio Tradizionale:** Regole scritte a mano da esperti umani + Dati $\to$ Risposte / Output.  
  *Criticità:* se il problema è complesso (es. riconoscimento volti o filtro spam), le regole diventano milioni, fragili e impossibili da manutenere.
* **Approccio ML:** Dati + Risposte note $\to$ Algoritmo di Apprendimento $\to$ Regole / Modello matematico predittivo.

#### 3. Tassonomia dei Sistemi di Machine Learning
* **Supervisione Umana:**
  * **Supervisionato (Supervised):** Il dataset di addestramento contiene sia le feature $X$ sia le etichette target $y$ (es. Regressione Lineare, Regressione Logistica, Decision Trees, SVM, Neural Networks).
  * **Non Supervisionato (Unsupervised):** Dati privi di etichetta (es. K-Means, DBSCAN, PCA, Autoencoder).
  * **Semi-supervisionato (Semisupervised):** Pochi dati etichettati uniti a una grande mole di dati non etichettati.
  * **Auto-supervisionato (Self-Supervised):** Genera automaticamente etichette a partire da dati non etichettati (es. ricostruzione di parti mascherate di immagini o testo), spesso come pre-addestramento per *Transfer Learning*.
  * **Apprendimento per Rinforzo (Reinforcement Learning):** Un agente seleziona azioni in un ambiente, riceve ricompense o penalità (reward/penalty) e apprende la politica ottima ($\pi^*$).
* **Capacità di Aggiornamento Continuo:**
  * **Batch Learning (Offline):** Il sistema viene addestrato su tutti i dati disponibili e poi rilasciato in produzione senza aggiornarsi dinamicamente.
  * **Online Learning (Incrementale):** Il sistema riceve flussi di dati sequenziali (*data streams*) o mini-batch e aggiorna i propri pesi in tempo reale (adatto a sistemi out-of-core).
* **Generalizzazione:**
  * **Instance-based Learning:** Memorizza i campioni e classifica le nuove istanze calcolando la similarità (es. k-NN).
  * **Model-based Learning:** Costruisce un modello matematico con parametri ottimizzati minimizzando una funzione di costo.

#### 4. Principali Sfide e Fallimenti del Machine Learning
* **Dati Insufficienti:** Gli algoritmi complessi richiedono ordini di grandezza maggiori di dati per convergere.
* **Dati non Rappresentativi e Sampling Bias:** Campioni non casuali che alterano la distribuzione reale (famoso caso del sondaggio presidenziale *Literary Digest* del 1936).
* **Bassa Qualità dei Dati:** Presenza di rumore, valori mancanti (*missing values*) ed outlier anomali.
* **Feature Irrilevanti:** Necessità di *Feature Selection*, *Feature Extraction* (PCA) e creazione di nuove feature (*Feature Engineering*).
* **Underfitting (Alto Bias):** Il modello è troppo semplice per catturare la struttura intrinseca dei dati (es. retta su dati quadratici).  
  *Soluzioni:* Aumentare la complessità del modello, fornire feature migliori, ridurre la regolarizzazione.
* **Overfitting (Alta Varianza):** Il modello memorizza il rumore del dataset di addestramento e non generalizza sui nuovi dati.  
  *Soluzioni:* Semplificare il modello, raccogliere più dati, rimuovere rumore, applicare regolarizzazione.

#### 5. Validazione e Split dei Dati
* **Train Set vs. Test Set:** Solitamente 80% train e 20% test.
* **Validation Set:** Utilizzato per confrontare modelli e sintonizzare gli iperparametri senza sporcare il Test Set (*Data Snooping Bias*).
* **K-Fold Cross-Validation:** Suddivisione del training set in $K$ fold per valutare la stabilità delle performance.

---

### Lezione 04: Classificazione e Valutazione dei Modelli
* **File:** [`04-Classificazione.pdf`](04-Classificazione.pdf) (70 slide)
* **Riferimento:** Aurélien Géron, *Hands-On Machine Learning*, Capitolo 3.

#### 1. Il Dataset MNIST e la Classificazione Binaria
* **MNIST:** 70.000 immagini 28x28 pixel di cifre manoscritte (0–9), con 784 feature numeriche (intensità di grigio [0, 255]).
* **Classificatore Binario:** Esempio "Riconoscitore del numero 5": target binario $y \in \{0, 1\}$ ("5" vs "non-5").
* **SGDClassifier:** Discesa del gradiente stocastica, efficiente per dataset di grandi dimensioni.

#### 2. Limiti dell'Accuratezza su Classi Sbilanciate
Se solo il 10% delle immagini contiene la cifra "5", un classificatore banale (*dummy*) che predice sempre "non-5" raggiunge il **90% di accuratezza pur essendo del tutto cieco alla classe positiva**.

#### 3. Matrice di Confusione e Metriche Derivate

| | Predetto Negativo ($0$) | Predetto Positivo ($1$) |
| :--- | :---: | :---: |
| **Reale Negativo ($0$)** | **TN** (True Negative) | **FP** (False Positive - Errore Tipo I) |
| **Reale Positivo ($1$)** | **FN** (False Negative - Errore Tipo II) | **TP** (True Positive) |

* **Precision (Precisione):** Accuratezza delle predizioni positive.
  $$\text{Precision} = \frac{TP}{TP + FP}$$
* **Recall / Sensitivity / TPR (Sensibilità / Tasso Veri Positivi):** Frazione di positivi reali individuati correttamente dal modello.
  $$\text{Recall} = \frac{TP}{TP + FN}$$
* **Specificity / TNR (Specificità / Tasso Veri Negativi):**
  $$\text{Specificity} = \frac{TN}{TN + FP}$$
* **F1-Score (Media Armonica):** Dà peso elevato ai valori bassi; è alta solo se sia Precision sia Recall sono bilanciate.
  $$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{TP}{TP + \frac{FN + FP}{2}}$$

#### 4. Il Trade-off Precision/Recall e le Curve Diagnostiche
* **Decision Threshold:** Variando la soglia di confidenza del classificatore, all'aumentare della Precision la Recall inevitabilmente decresce.
* **Curva ROC (Receiver Operating Characteristic):** Grafico che traccia il True Positive Rate (TPR / Recall) in funzione del False Positive Rate (FPR):
  $$\text{FPR} = \frac{FP}{FP + TN} = 1 - \text{Specificity}$$
* **AUC-ROC (Area Under the Curve):**
  * $0.5$: Classificatore casuale (moneta non truccata).
  * $1.0$: Classificatore perfetto.
* **Scelta tra PR Curve e ROC Curve:** Si preferisce la curva **Precision-Recall** quando la classe positiva è rara (forte sbilanciamento) o quando i Falsi Positivi sono più costosi dei Falsi Negativi; altrimenti si usa la curva **ROC**.

#### 5. Classificazione Multiclasse e Strategie
* **Algoritmi nativi multiclasse:** Logistic Regression (via Softmax), Random Forest, Naive Bayes, Reti Neurali.
* **Strategie per estendere classificatori strettamente binari (es. SVM, SGDClassifier):**
  * **One-versus-Rest (OvR / OvA):** Si addestrano $N$ classificatori binari (uno per ciascuna classe contro tutte le altre). Per 10 cifre servono 10 classificatori.
  * **One-versus-One (OvO):** Si addestra un classificatore per ogni coppia di classi. Per $N$ classi servono $\frac{N(N-1)}{2}$ classificatori (per MNIST: 45 classificatori).
* **Analisi degli Errori (Error Analysis):** Visualizzazione della matrice di confusione normalizzata per riga per individuare le classi confuse più frequentemente (es. cifre 4 e 9, o 3 e 5).

---

### Lezione 05: Decision Trees
* **File:** [`05-Decision Trees.pdf`](05-Decision Trees.pdf) (45 slide)
* **Riferimento:** Aurélien Géron, *Hands-On Machine Learning*, Capitolo 6.

#### 1. Principi di Funzionamento e Proprietà
* Modelli supervisionati non parametrici per classificazione, regressione e compiti multi-output.
* **Vantaggio Straordinario:** Non richiedono ridimensionamento delle feature (*no feature scaling*, né standardizzazione né normalizzazione).
* Creano **confini decisionali ortogonali** (iperpiani paralleli agli assi delle feature).
* Rappresentano la quintessenza dei modelli **White-Box**: ogni decisione è spiegabile e tracciabile fino alla foglia terminale.

#### 2. Misure di Impurità
* **Indice di Impurità di Gini:** Misura la probabilità che un elemento estratto a caso sia classificato erroneamente se etichettato secondo la distribuzione del nodo:
  $$G_i = 1 - \sum_{k=1}^K p_{i,k}^2$$
  * $G_i = 0.0$: Nodo perfettamente puro (tutti i campioni appartengono a una sola classe).
  * Per classificazione binaria bilanciata: $G_i = 0.5$ (massima impurità).
* **Entropia di Shannon:** Misura dell'incertezza informativa:
  $$H_i = -\sum_{k=1, p_{i,k}>0}^K p_{i,k} \log_2(p_{i,k})$$
  * Nella maggior parte dei casi Gini ed Entropia portano ad alberi molto simili; Gini è leggermente più veloce da calcolare (evita il logaritmo).

#### 3. Algoritmo CART (Classification and Regression Trees)
L'algoritmo di Scikit-Learn costruisce alberi binari eseguendo una ricerca euristica golosa (*greedy search*). Ad ogni passo cerca la coppia feature $k$ e soglia $t_k$ che minimizza la funzione di costo pesata:
$$J(k, t_k) = \frac{m_{\text{left}}}{m} G_{\text{left}} + \frac{m_{\text{right}}}{m} G_{\text{right}}$$
dove $m$ è il numero totale di istanze del nodo e $G$ è l'impurità dei due sotto-insiemi generati.

#### 4. Complessità Computazionale
* **Fase di Predizione:** $O(\log_2(m))$, indipendente dal numero di feature, estremamente rapida.
* **Fase di Addestramento:** $O(n \cdot m \log_2(m))$, dove $n$ è il numero di feature e $m$ il numero di campioni.

#### 5. Iperparametri di Regolarizzazione
Gli alberi non vincolati tendono rapidamente all'**overfitting**:
* `max_depth`: Limita la profondità massima dell'albero (iperparametro principale di regolarizzazione).
* `min_samples_split`: Numero minimo di campioni richiesti per tentare una scissione.
* `min_samples_leaf`: Numero minimo di campioni che devono risiedere in ciascuna foglia.
* `max_leaf_nodes`: Limite massimo sul numero totale di foglie.

#### 6. Limiti e Instabilità dei Decision Trees
* **Sensibilità alla Rotazione:** Poiché i tagli sono rigorosamente paralleli agli assi, una semplice rotazione delle coordinate dei dati può degradare fortemente la struttura dell'albero (mitigabile pre-elaborando i dati con la PCA).
* **Alta Varianza:** Piccole fluttuazioni nei dati di addestramento possono generare alberi completamente diversi (risolvibile impiegando ensemble di alberi: *Random Forest* e *Gradient Boosting*).

---

### Lezione 06: Apprendimento Non Supervisionato (Unsupervised Learning)
* **File:** [`06-Unsupervised_Learning.pdf`](06-Unsupervised_Learning.pdf) (83 slide)
* **Riferimento:** Aurélien Géron, *Hands-On Machine Learning*, Capitolo 9.

#### 1. Panoramica
L'apprendimento non supervisionato opera su campioni non etichettati per estrarre la struttura sottostante. Ambiti applicativi primari:
1. **Clustering:** Suddivisione di istanze simili in gruppi.
2. **Rilevamento di Anomalie e Novità (Anomaly/Novelty Detection):** Identificazione di pattern atipici e frodi.
3. **Stima di Densità (Density Estimation):** Modellazione della funzione di densità di probabilità dei dati.

#### 2. L'Algoritmo K-Means
* Algoritmo iterativo basato su centroidi proposto originariamente da Stuart Lloyd (1957).
* **Fasi dell'algoritmo:**
  1. Inizializzazione casuale di $K$ centroidi nello spazio delle feature.
  2. **Assegnazione:** Ogni punto viene assegnato al centroide più vicino (distanza euclidea).
  3. **Aggiornamento:** Ogni centroide viene ricalcolato come media algebrica di tutti i punti assegnati al cluster.
  4. Ripetizione fino a convergenza (i centroidi non si muovono più).
* **Inerzia o WCSS (Within-Cluster Sum-of-Squares):** La funzione obiettivo minimizzata da K-Means:
  $$\text{Inerzia} = \sum_{i=1}^m \min_{\mu_j \in C} \|x^{(i)} - \mu_j\|^2$$
* **K-Means++:** Variante intelligente per l'inizializzazione dei centroidi che seleziona i semi iniziali distanziati tra loro proporzionalmente al quadrato della distanza minima, riducendo drasticamente il rischio di convergenza a minimi locali subottimali.
* **Mini-Batch K-Means:** Variante che utilizza piccoli batch di campioni in memoria ad ogni iterazione, accelerando l'algoritmo di un fattore 3x–4x su dataset massivi.

#### 3. Selezione del Numero Ottimale di Cluster ($K$)
* **Metodo del Gomito (Elbow Method):** Grafico dell'Inerzia in funzione di $K$. Si seleziona il punto di flesso (*elbow*) oltre il quale la riduzione dell'inerzia diventa marginale.
* **Silhouette Score:** Coefficiente che quantifica la compattezza e la separazione dei cluster:
  $$s_i = \frac{b_i - a_i}{\max(a_i, b_i)}, \quad s_i \in [-1, +1]$$
  * $a_i$: Distanza media intra-cluster (distanza tra $i$ e gli altri punti dello stesso cluster).
  * $b_i$: Distanza media inter-cluster più vicina (distanza tra $i$ e i punti del cluster più vicino).
  * $+1$: Il punto è perfettamente integrato nel proprio cluster e lontano dagli altri.
  * $0$: Il punto si trova sul confine decisionale tra due cluster.
  * $-1$: Il punto è stato probabilmente assegnato al cluster errato.

#### 4. Algoritmo DBSCAN
* **Density-Based Spatial Clustering of Applications with Noise:** Algoritmo basato sulla densità locale dei punti definito da due parametri: $\varepsilon$ (raggio del vicinato) e `min_samples` (numero minimo di vicini).
* **Classificazione dei punti:**
  * **Core Point:** Ha almeno `min_samples` punti nel suo $\varepsilon$-intorno.
  * **Border Point:** Si trova nel vicinato di un core point ma ha meno di `min_samples`.
  * **Noise Point / Outlier:** Non appartiene ad alcun intorno denso (etichettato come $-1$).
* **Vantaggi di DBSCAN:** Identifica cluster di forma arbitraria (forme concave o ad anello) e rileva automaticamente gli outlier senza dover specificare $K$ a priori. *Nota scikit-learn:* `DBSCAN` espone `fit_predict()` ma **non** dispone di un metodo `.predict()` nativo per nuovi campioni.

#### 4.1 Isolation Forest (Anomaly Detection)
* Algoritmo non supervisionato basato su alberi casuali (*isolation trees*): le anomalie sono rare e diverse, quindi vengono isolate con un **cammino medio più corto** (*short path length*) rispetto ai punti normali.
* `predict(X)` restituisce `1` (inlier) o `-1` (outlier); iperparametri principali: `n_estimators` (default `100`), `max_samples` (default `min(256, n_samples)`), `contamination`.
* Vantaggi: evita il calcolo costoso delle distanze a coppie e scala molto bene in alta dimensionalità.

#### 5. Modelli a Mistura di Gaussiane (GMM)
* Modello probabilistico che assume che i dati siano stati generati da una combinazione convessa di $K$ distribuzioni normali multivariate con parametri sconosciuti $(\mu_k, \Sigma_k, \pi_k)$.
* Addestrato mediante l'algoritmo **Expectation-Maximization (EM)**:
  * **Passo E (Expectation):** Stima delle probabilità a posteriori che ciascun punto appartenga a ciascuna componente gaussiana (*soft clustering*).
  * **Passo M (Maximization):** Ristima dei parametri di ciascuna gaussiana pesata sulle responsabilità calcolate nel passo E.
* Selezione del modello tramite criteri teorici informativi che penalizzano la complessità: **AIC** (Akaike Information Criterion) e **BIC** (Bayesian Information Criterion).

---

### Lezione 07: Introduzione alle Reti Neurali Artificiali con Keras
* **File:** [`07-Introduction_to_NNs.pdf`](07-Introduction_to_NNs.pdf) (122 slide)
* **Riferimento:** Aurélien Géron, *Hands-On Machine Learning*, Capitolo 10.

#### 1. Dalle Reti Biologiche ai Modelli Connessionisti
* Ispirazione dai neuroni biologici (soma, dendriti, assone, sinapsi).
* **Neurone di McCulloch & Pitts (1943):** Modello logico semplificato con ingressi booleani e soglia di attivazione. Capace di calcolare funzioni booleane (AND, OR, NOT).

#### 2. Il Perceptron di Frank Rosenblatt (1957)
* Basato sulla **Threshold Logic Unit (TLU)**: calcola una somma pesata degli input e applica una funzione a gradino (Heaviside step function):
  $$z = \mathbf{w}^T \mathbf{x} + b, \quad \hat{y} = \text{step}(z)$$
* **Regola di Apprendimento del Perceptron:**
  $$w_{i,j}^{(\text{next})} = w_{i,j} + \eta \, (y_j - \hat{y}_j) \, x_i$$
  Se i dati sono linearmente separabili, il teorema di convergenza del Perceptron garantisce che l'algoritmo troverà una soluzione in tempo finito.
* **Il Limite Storico dello XOR (Minsky & Papert, 1969):** Un singolo Perceptron non può risolvere problemi non linearmente separabili come la funzione logica XOR.

#### 3. Il Multi-Layer Perceptron (MLP)
Per superare i limiti del Perceptron si introduce una struttura a strati multipli (*feedforward*):
* **Input Layer:** Distribuisce le feature di ingresso.
* **Hidden Layers:** Uno o più strati nascosti con neuroni di elaborazione intermedia.
* **Output Layer:** Produce le predizioni finali.

#### 4. Funzioni di Attivazione Non Lineari
Senza funzioni di attivazione non lineari, qualsiasi rete profonda a $L$ strati collassa matematicamente in una semplice combinazione lineare ($W_2 W_1 \mathbf{x} = W_{\text{tot}} \mathbf{x}$).
* **Sigmoide (Logistica):**
  $$\sigma(z) = \frac{1}{1 + e^{-z}}, \quad \sigma'(z) = \sigma(z)(1 - \sigma(z))$$
  Schiaccia l'output in $(0, 1)$, ideale per probabilità binarie, ma soffre di saturazione dei gradienti per $|z| \gg 0$.
* **Tangente Iperbolica (Tanh):**
  $$\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}, \quad \text{output centrato in zero } (-1, 1)$$
* **Rectified Linear Unit (ReLU):**
  $$\text{ReLU}(z) = \max(0, z)$$
  Lo standard de facto per gli strati nascosti: veloce da calcolare, evita la saturazione per valori positivi.

#### 5. Algoritmo di Backpropagation (Rumelhart, Hinton, Williams, 1986)
Metodo per addestrare reti neurali combinando la discesa del gradiente con la **regola della catena** per la derivazione di funzioni composte:
1. **Forward Pass:** I dati attraversano la rete strato dopo strato calcolando le attivazioni fino all'output e determinando l'errore tramite la loss function $\mathcal{L}$.
2. **Reverse Pass:** L'errore viene propagato all'indietro calcolando i gradienti parziali della loss rispetto a ciascun peso:
   $$\frac{\partial \mathcal{L}}{\partial w_{ij}} = \frac{\partial \mathcal{L}}{\partial z_j} \frac{\partial z_j}{\partial w_{ij}}$$
3. **Aggiornamento dei Pesi:**
   $$\mathbf{w} \leftarrow \mathbf{w} - \eta \, \nabla_{\mathbf{w}} \mathcal{L}$$

#### 6. Architetture MLP Tipiche e Framework Keras
* **Regressione:** Neurone di output lineare, loss MSE o MAE.
* **Classificazione Binaria:** 1 neurone di output con attivazione sigmoide, loss **Binary Cross-Entropy**:
  $$\mathcal{L} = -\frac{1}{m} \sum_{i=1}^m \left[ y^{(i)} \log(\hat{y}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)}) \right]$$
* **Classificazione Multiclasse:** $K$ neuroni di output con funzione **Softmax**, loss **Categorical Cross-Entropy**:
  $$\text{Softmax}(z)_i = \frac{e^{z_i}}{\sum_{j=1}^K e^{z_j}}, \qquad \frac{\partial \mathcal{L}}{\partial z_i} = \hat{y}_i - y_i$$
  *(Gradiente elegante: predizione meno target; per stabilità numerica si sottrae $z_{\max}$ ai logit prima dell'esponenziale.)*
* **Overfitting ed Early Stopping:** Monitorando le curve sul validation set, l'**Early Stopping** arresta l'addestramento prima che la rete vada in overfitting.
* **Keras & Scikit-Learn API:** `MLPClassifier` / `MLPRegressor` in scikit-learn (che supportano `predict_proba()` ed `early_stopping=True`); `Sequential` / `Functional` API in Keras con `compile(optimizer, loss, metrics)` e `fit(X_train, y_train, epochs, validation_data)`.

---

### Lezione 08: Addestramento di Reti Neurali Profonde (Deep Neural Networks)
* **File:** [`08-Training_DeepNeuralNetworks.pdf`](08-Training_DeepNeuralNetworks.pdf) (81 slide)
* **Riferimento:** Aurélien Géron, *Hands-On Machine Learning*, Capitolo 11.

#### 1. Il Problema dei Gradienti Instabili
Nelle reti molto profonde, durante il backward pass il gradiente calcolato con la catena delle derivate viene moltiplicato per i pesi strato dopo strato:
* **Vanishing Gradients (Gradienti Svanenti):** I gradienti si rimpiccioliscono progressivamente verso i primi strati; i pesi iniziali non si aggiornano e l'apprendimento stagna. (Tipico con attivazioni sigmoidee centrate su valori saturi).
* **Exploding Gradients (Gradienti Esplodenti):** I gradienti crescono esponenzialmente, portando a valori numerici `NaN` e divergenza dell'algoritmo.

#### 2. Tecniche di Inizializzazione dei Pesi
Per garantire che il flusso del segnale e della varianza rimanga stabile sia nel forward sia nel backward pass:
* **Glorot / Xavier Initialization (per Sigmoide, Tanh, Softmax):**
  $$\text{Varianza: } \sigma^2 = \frac{2}{\text{fan}_{\text{in}} + \text{fan}_{\text{out}}}$$
* **He / Kaiming Initialization (per ReLU e varianti):**
  $$\text{Varianza: } \sigma^2 = \frac{2}{\text{fan}_{\text{in}}}$$
* **LeCun Initialization (per SELU):**
  $$\text{Varianza: } \sigma^2 = \frac{1}{\text{fan}_{\text{in}}}$$

#### 3. Funzioni di Attivazione Avanzate
* **Problema della "Dying ReLU":** Neuroni che generano output 0 per tutti i campioni del dataset a causa di gradienti nulli permanenti.
* **Leaky ReLU:** Aggiunge una pendenza minima $\alpha$ per $z < 0$:
  $$\text{LeakyReLU}(z) = \max(\alpha z, z), \quad \alpha \approx 0.01 \text{ o } 0.2$$
* **ELU (Exponential Linear Unit):** Curvatura graduale per valori negativi, riducendo l'offset della media:
  $$\text{ELU}(z) = \begin{cases} \alpha(e^z - 1) & \text{se } z < 0 \\ z & \text{se } z \ge 0 \end{cases}$$
* **SELU (Scaled ELU):** Garantisce l'auto-normalizzazione della rete profonda ($\mu \approx 0, \sigma \approx 1$) se combinata con inizializzazione LeCun e ingressi standardizzati.
* **GELU, Swish, Mish:** Attivazioni non lineari lisce utilizzate nei modelli moderni (Transformer e LLM).

#### 4. Batch Normalization (Ioffe & Szegedy, 2015)
Operazione inserita tipicamente prima o dopo la funzione di attivazione:
1. Valuta media $\mu_B$ e varianza $\sigma_B^2$ sul mini-batch corrente.
2. Centra e scala le attivazioni:
   $$\hat{x}^{(i)} = \frac{x^{(i)} - \mu_B}{\sqrt{\sigma_B^2 + \varepsilon}}$$
3. Applica due parametri apprendibili per ripristinare la varianza ottimale:
   $$z^{(i)} = \gamma \, \hat{x}^{(i)} + \beta$$
* **Benefici:** Risolve i gradienti instabili, rende l'addestramento molto meno dipendente dall'inizializzazione dei pesi, consente learning rate più alti e agisce come debole regolarizzatore.

#### 5. Gradient Clipping
Soluzione comune nelle reti ricorrenti (RNN): limita il gradiente a una soglia massima prestabilita per prevenire l'esplosione numerica:
$$\text{if } \|\mathbf{g}\| > c \implies \mathbf{g} \leftarrow c \frac{\mathbf{g}}{\|\mathbf{g}\|}$$

#### 5.1 Transfer Learning e Unsupervised Pretraining
* **Transfer Learning:** Riutilizza gli strati inferiori di un modello già addestrato su un task simile (che estraggono feature generali), congelandoli inizialmente (`layer.trainable = False`) e riaddestrando/raffinando (*fine-tuning*) gli strati superiori con learning rate ridotto.
* **Unsupervised / Self-Supervised Pretraining:** Quando i dati etichettati scarseggiano, si addestra prima un modello non supervisionato (es. Autoencoder o task ausiliario auto-supervisionato) per apprendere rappresentazioni utili, per poi fare *fine-tuning* supervisionato.

#### 6. Ottimizzatori Avanzati (Faster Optimizers)
Rispetto alla discesa del gradiente standard (SGD):
* **Momentum Optimization:** Simula una sfera con massa inerziale che scende lungo un pendio, accelerando lungo le direzioni consistenti:
  $$\mathbf{m} \leftarrow \beta \mathbf{m} - \eta \nabla_\theta \mathcal{L}, \quad \theta \leftarrow \theta + \mathbf{m}$$
* **NAG (Nesterov Accelerated Gradient):** Calcola il gradiente non nella posizione attuale ma leggermente più avanti nella direzione del momento (*lookahead*).
* **RMSProp:** Adatta il tasso di apprendimento per ogni parametro dividendo per la media mobile esponenziale dei quadrati dei gradienti, smorzando le oscillazioni nelle valli ripide.
* **Adam (Adaptive Moment Estimation):** Combina i principi di Momentum e RMSProp; calcola le medie mobili esponenziali sia del gradiente ($1^\circ$ momento) sia del suo quadrato ($2^\circ$ momento), con correzione del bias iniziale. Lo standard industriale più robusto.

#### 7. Learning Rate Scheduling
* Il learning rate costante rischia di divergere se troppo alto o di impiegare troppo tempo a convergere se troppo basso.
* **Power Scheduling, Exponential Scheduling, Piecewise Constant** e la tecnica **1Cycle Scheduling** (aumento lineare fino a un picco e successivo decadimento esponenziale).

#### 8. Tecniche di Regolarizzazione per Evitare l'Overfitting
* **Regolarizzazione $L_1$ (Lasso):** Aggiunge $\alpha \sum |w_i|$ alla loss, inducendo pesi sparsi (*feature selection* implicita).
* **Regolarizzazione $L_2$ (Ridge / Weight Decay):** Aggiunge $\frac{\lambda}{2} \sum w_i^2$ alla loss, forzando i pesi a rimanere piccoli e distribuiti.
* **Dropout (Srivastava et al., 2014):** Ad ogni passo di training spegne casualmente ogni neurone con probabilità $p$ (solitamente tra $0.1$ e $0.5$). Costringe la rete a non affidarsi a singoli neuroni e simula un ensemble di architetture; **Monte Carlo (MC) Dropout** mantiene attivo il dropout in inferenza su più passaggi per stimare l'incertezza predittiva.
* **Max-Norm Regularization:** Impone un limite superiore $\|\mathbf{w}\|_2 \le r$ alla norma dei pesi entranti in ciascun neurone, stabilizzando l'addestramento.
* **Early Stopping (introdotto nella Lezione 07):** Interrompe l'addestramento non appena le prestazioni sul validation set smettono di migliorare per un numero prefissato di epoche (*patience*).

---

## 🎯 Domande Tipiche d'Esame sulle Lezioni

### 1. Fondamenti & Agenti (Lezioni 01 & 02)
* **D:** *Qual è la differenza fondamentale tra razionalità e onniscienza?*  
  **R:** L'onniscienza implica la conoscenza a priori di tutti gli eventi futuri e degli esiti effettivi di ogni azione. La razionalità massimizza invece l'utilità attesa in base alle sole percezioni passate, alla conoscenza del mondo e all'incertezza intrinseca dell'ambiente.
* **D:** *Come si descrive un ambiente PEAS per un sistema di Churn Telco?*  
  **R:** $P$ = massimizzazione del valore netto di fidelizzazione (riduzione del churn bilanciata rispetto al costo delle promozioni e al CAC — distinta dalla metrica predittiva $P$ di Mitchell); $E$ = ecosistema della clientela Telco interagente via CRM (parzialmente osservabile, stocastico, dinamico, multi-agente; sequenziale nelle campagne di retention, episodico nella singola predizione); $A$ = offerte promozionali mirate, sconti e supporto tecnico prioritario; $S$ = log di fatturazione, contratti, *tenure* e servizi attivi.

### 2. Machine Learning & Classificazione (Lezioni 03 & 04)
* **D:** *Perché l'accuratezza non è affidabile con dataset sbilanciati?*  
  **R:** Perché un modello dummy che predice costantemente la classe maggioritaria può ottenere un'accuratezza altissima (es. 90% su target con 10% di positivi) avendo però Recall = 0%, risultando totalmente inutile nel riconoscere i casi d'interesse critico.
* **D:** *Qual è la relazione tra la curva ROC e la curva Precision-Recall?*  
  **R:** La curva ROC valuta TPR vs FPR ed è indipendente dalla proporzione delle classi. La curva Precision-Recall è invece sensibilissima allo sbilanciamento e va preferita quando la classe positiva è rara o i Falsi Positivi comportano costi elevati.

### 3. Modelli ad Albero & Clustering (Lezioni 05 & 06)
* **D:** *Perché i Decision Tree non necessitano di standardizzazione delle feature?*  
  **R:** Perché l'algoritmo CART valuta le soglie di taglio su una singola variabile alla volta in modo isolato; l'ordinamento monotono dei valori non varia applicando trasformazioni di scala.
* **D:** *Come si sceglie il valore ottimo di $K$ nel K-Means?*  
  **R:** Si combinano il Metodo del Gomito (identificando il flesso dell'Inerzia WCSS) e l'analisi del Silhouette Score medio e dei grafici di silhouette per verificare compattezza e separazione dei gruppi.

### 4. Reti Neurali & Deep Learning (Lezioni 07 & 08)
* **D:** *Perché la funzione di attivazione ReLU ha soppiantato la Sigmoide negli strati nascosti?*  
  **R:** Perché per valori positivi la derivata della ReLU è costante e pari a 1, eliminando il fenomeno dei gradienti svanenti (*vanishing gradients*) e consentendo l'addestramento efficace di reti con decine o centinaia di strati.
* **D:** *A cosa serve la Batch Normalization?*  
  **R:** Normalizza le attivazioni all'interno di ciascun mini-batch a media zero e varianza unitaria, stabilizzando la distribuzione degli input strato per strato (*riduce l'internal covariate shift*), accelerando la convergenza e agendo da blando regolarizzatore.
