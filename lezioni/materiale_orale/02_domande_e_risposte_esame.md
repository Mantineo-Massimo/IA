# Foglio 2 — Banca Completa di Domande e Risposte d'Esame

**Progetto:** Churn Prediction e Segmentazione Comportamentale (IBM Telco Customer Churn)  
**Studente:** Massimo Mantineo (Matricola: 541924)  
**Corso:** Laboratorio di Intelligenza Artificiale — Prof. Giacomo Fiumara (Lezioni 01–08)

---

## Indice delle Categorie

- **Sezione A:** Fondamenti di IA & Agenti Intelligenti (Lezioni 01–02 — Slide 2)
- **Sezione B:** Machine Learning, Dataset & Preprocessing (Lezione 03 — Slide 3–4)
- **Sezione C:** Classificazione, Sbilanciamento & Soglia Decisionale (Lezione 04 — Slide 4 & 8)
- **Sezione D:** Apprendimento Non Supervisionato: K-Means & PCA (Lezione 06 — Slide 5–6)
- **Sezione E:** Decision Trees CART & Interpretabilità White-Box (Lezione 05 — Slide 7 & 9)
- **Sezione F:** Regressione Logistica & Reti Neurali MLP (Lezioni 07–08 — Slide 7–8)
- **Sezione G:** Analisi Critica, Domande sul Codice, Etica & Sviluppi Futuri (Slide 8 & 10)

---

## Sezione A — Fondamenti di IA & Agenti Intelligenti (Lezioni 01–02)

### A1. Quali sono i quattro approcci alla definizione di IA secondo Russell & Norvig e perché avete scelto "Agire Razionalmente"?
**Risposta:** Russell e Norvig organizzano le definizioni di IA lungo due dimensioni: *Pensiero* vs *Comportamento* e *Umanamente* vs *Razionalmente*. I quattro approcci sono:
1. **Pensare Umanamente** (scienze cognitive / modellazione cognitiva);
2. **Pensare Razionalmente** (leggi del pensiero, logica formale);
3. **Agire Umanamente** (approccio comportamentale del Test di Turing);
4. **Agire Razionalmente** (approccio dell'agente razionale).

Abbiamo scelto **Agire Razionalmente** perché in un sistema aziendale di *churn management* non interessa imitare il processo cognitivo di un operatore umano né applicare sillogismi rigidi in un dominio incerto, ma selezionare azioni (offerte, supporto) che **massimizzino il valore atteso della misura di prestazione $P$** date le percezioni disponibili.

---

### A2. Qual è la differenza tra razionalità e onniscienza?
**Risposta:** Un agente **onnisciente** conosce in anticipo l'esito effettivo e certo di ogni propria azione — cosa impossibile nel mondo reale. Un agente **razionale**, invece, massimizza la prestazione *attesa* sulla base delle sole percezioni raccolte finora e della conoscenza pregressa: se un cliente con probabilità di churn del 10% decide comunque di andarsene per un'offerta esterna imprevedibile, l'agente resta comunque razionale se la sua decisione era ottimale rispetto alle informazioni disponibili al momento della scelta.

---

### A3. A quale architettura di agente appartiene il vostro sistema CRM?
**Risposta:** È un **agente basato su utilità con apprendimento** (*utility-based learning agent*):
- è **basato su utilità** perché deve gestire obiettivi conflittuali — ridurre gli abbandoni ma senza erogare sconti inutili a chi sarebbe rimasto comunque — quantificando il compromesso rispetto al CAC tramite una funzione di utilità;
- è **con apprendimento** perché non conosce a priori le leggi dell'ambiente clienti e aggiorna le proprie decisioni a partire dall'esperienza storica (i 4 moduli concettuali di R&N: *elemento esecutivo* che fa la predizione/azione, *elemento di apprendimento* che addestra i modelli, *critico* che valuta le metriche/esiti di retention e *generatore di problemi* che può testare nuove campagne esplorative).

---

### A4. Descriva il modello PEAS del progetto e motivi le proprietà dell'ambiente $E$.
**Risposta:**
- **P (Performance):** valore netto di fidelizzazione (riduzione del churn bilanciata rispetto al costo delle promozioni e al CAC).
- **E (Environment):** ecosistema operativo della clientela Telco via CRM. È:
  - *parzialmente osservabile*: i log non osservano lo stato mentale del cliente né le offerte dei competitor;
  - *nondeterministico / stocastico*: a parità di profilo e offerta, un cliente può restare o abbandonare con una certa probabilità;
  - *multi-agente*: coesistono clienti e operatori concorrenti (ambiente parzialmente competitivo);
  - *continuo*: importi economici e tempo variano su scale continue;
  - *sconosciuto*: la distribuzione $P(\text{Churn}=1 \mid \mathbf{x})$ non è nota a priori ma appresa dai dati;
  - *sequenziale e dinamico* nel ciclo reale di CRM (un'offerta oggi modifica le scelte future e il mercato evolve mentre decidiamo), ma **episodico e statico** nella singola predizione *batch* sullo snapshot del dataset (ogni cliente viene valutato indipendentemente su una fotografia fissa).
- **A (Actuators):** invio di offerte personalizzate, sconti mirati, attivazione di supporto tecnico prioritario.
- **S (Sensors):** database e log CRM (`tenure`, `MonthlyCharges`, `TotalCharges`, tipo di contratto, servizi attivi, metodo di pagamento).

---

### A5. Qual è la differenza tra la $P$ del PEAS e la $P$ di Tom Mitchell $\langle T, E, P \rangle$?
**Risposta:** È una distinzione di livello architetturale:
- nel **PEAS** (Russell & Norvig), $P$ è la **misura di prestazione esterna nell'ambiente**: valuta il beneficio economico reale prodotto dalle azioni dell'agente (clienti trattenuti rispetto ai costi delle offerte e al CAC);
- nella definizione $\langle T, E, P \rangle$ di **Mitchell**, $P$ misura la **qualità predittiva del modello di Machine Learning** sul proprio task statistico ($T$), quindi nel nostro progetto *Recall*, *Precision*, *F1-Score* e *AUC-ROC*.

---

## Sezione B — Machine Learning, Dataset & Preprocessing (Lezione 03)

### B1. Come classifica il vostro sistema secondo la tassonomia del Machine Learning di Géron (Lezione 03)?
**Risposta:**
- **Per supervisione:** è un sistema **ibrido**, che combina apprendimento **non supervisionato** (K-Means + PCA) e **supervisionato** (classificazione binaria con LogReg, CART, MLP).
- **Per aggiornamento:** è **Batch (Offline)**, perché i modelli vengono addestrati sull'intero Training Set storico e poi valutati/rilasciati senza aggiornamento incrementale istantaneo (in produzione si ri-addestra periodicamente).
- **Per generalizzazione:** è **Model-Based**, perché costruisce modelli matematici con parametri o regole appresi dai dati, invece di confrontare ogni nuovo cliente con tutti gli esempi memorizzati (come farebbe un *Instance-Based* tipo k-NN).

---

### B2. Perché parlate di approccio "probabilistico discriminativo" e non bayesiano/generativo?
**Risposta:** Un modello **generativo** (come *Naive Bayes* o *Gaussian Mixture Models*) modella la distribuzione congiunta $P(\mathbf{x}, y)$ stimando la verosimiglianza $P(\mathbf{x} \mid y)$ e la probabilità a priori $P(y)$, per poi risalire a $P(y \mid \mathbf{x})$ col teorema di Bayes. I nostri tre classificatori (**Regressione Logistica**, **CART**, **MLP**) sono invece **probabilistici discriminativi**: apprendono *direttamente* il confine decisionale e la stima condizionata $P(\text{Churn}=1 \mid \mathbf{x}) \in [0,1]$, senza fare assunzioni forti (spesso violate su dati Telco correlati) come l'indipendenza condizionata delle feature.

---

### B3. Perché avete imputato a `0.0` gli 11 valori vuoti di `TotalCharges` invece di eliminarli o usare la mediana?
**Risposta:** Perché ispezionando le 11 righe con spazio bianco (`" "`) in `TotalCharges` abbiamo verificato programmaticamente (`blank_rows['tenure'].unique()`) che hanno **tutte `tenure == 0`**. Sono quindi clienti appena acquisiti che non hanno ancora completato il primo ciclo mensile di fatturazione: la loro spesa totale cumulata è oggettivamente `0.0 €`. Eliminarle ci farebbe perdere informazioni sui neo-clienti (fascia molto sensibile), mentre imputare media (`~2.299 €`) o mediana inserirebbe un valore completamente assurdo e distorto per un cliente con 0 mesi di anzianità.

---

### B4. Cos'è il Data Leakage e come garantite che nel vostro notebook sia esattamente zero?
**Risposta:** Il *data leakage* (contaminazione dei dati) si verifica quando informazioni statistiche del Test Set — come media, deviazione standard, centroidi dei cluster o soglie ottimali — filtrano nella fase di addestramento, producendo metriche artificialmente ottimistiche che non si confermano in produzione.
Nel nostro notebook lo preveniamo rigorosamente in 4 punti:
1. eseguiamo `train_test_split(..., test_size=0.20, stratify=y, random_state=42)` **prima** di qualunque trasformazione parametrica;
2. `StandardScaler` e `OneHotEncoder` usano `.fit_transform()` **solo su `X_train_raw`** e `.transform()` su `X_test_raw`;
3. `KMeans` e `PCA` vengono fittati **solo su `X_train_scaled`**, e al Test Set si applica solo `kmeans_chosen.predict(X_test_scaled)`;
4. la soglia $\tau = 0.35$ viene verificata con `cross_val_predict(..., cv=5)` **solo sul Training Set**, mai scegliendola guardando il Test Set.

---

### B5. Perché avete usato uno split stratificato (`stratify=y`)?
**Risposta:** Perché il dataset è sbilanciato (`26,54%` di Churn). Uno split puramente casuale potrebbe generare per fluttuazione campionaria un Training Set con una percentuale di positivi diversa da quella del Test Set, distorcendo sia l'addestramento sia la valutazione. Con `stratify=y` entrambi i set mantengono esattamente il `26,54%` di Churn (`1.495 / 5.634` nel Train e `374 / 1.409` nel Test).

---

### B6. Differenza tra Standardizzazione (Z-Score) e Normalizzazione (Min-Max): perché avete scelto `StandardScaler`?
**Risposta:**
- **Normalizzazione Min-Max:** scala i valori in $[0, 1]$ tramite $\frac{x - x_{\min}}{x_{\max} - x_{\min}}$; è molto sensibile agli outlier, perché un singolo valore estremo schiaccia tutti gli altri in un intervallo strettissimo.
- **Standardizzazione Z-Score (`StandardScaler`):** centra a media $\mu = 0$ e scala a deviazione standard $\sigma = 1$ tramite $z = \frac{x - \mu}{\sigma}$. È più robusta agli outlier ed è lo standard raccomandato per algoritmi basati su distanze euclidee (K-Means), varianza (PCA) e discesa del gradiente con regolarizzazione L2 (Regressione Logistica e MLP).

---

### B7. Perché avete impostato `OneHotEncoder(drop='first')`? Ci sono controindicazioni?
**Risposta:** Con $M$ categorie, codificarle con $M$ colonne binarie genera una somma sempre uguale a 1 (multicollinearità perfetta, nota come **trappola delle variabili dummy**), che rende la matrice delle feature linearmente dipendente e può creare problemi nei modelli lineari come la Regressione Logistica (anche se con L2 il problema è mitigato). Con `drop='first'` si generano $M-1$ colonne binarie (passando da 16 categoriali a 27 colonne dummy, per **30 feature totali**).
**Controindicazione (nota critica matura):** in algoritmi basati su distanze euclidee come K-Means, eliminare una categoria di riferimento introduce una lieve asimmetria geometrica tra la categoria eliminata (vettore di soli `0`, distanza $\sqrt{1}=1$ da ogni altra categoria) e due categorie non di riferimento (distanza $\sqrt{2}$ tra loro). È un limite che riconosciamo esplicitamente nella relazione.

---

## Sezione C — Classificazione, Sbilanciamento & Soglia Decisionale (Lezione 04)

### C1. Perché l'Accuracy non è sufficiente sul dataset Telco?
**Risposta:** Perché con il `73,46%` di clienti rimasti e il `26,54%` di abbandoni, un classificatore banale (*dummy*) che rispondesse sempre `0` ("Rimasto") otterrebbe un'**Accuracy del 73,46%**, apparentemente dignitosa, ma una **Recall dello 0%** (non intercetterebbe neppure un cliente in fuga). Su classi sbilanciate occorre guardare *Precision*, *Recall*, *F1-Score* e *AUC-ROC*.

---

### C2. Quali sono le formule di Precision, Recall e F1-Score, e perché F1 usa la media armonica?
**Risposta:** Dalla Matrice di Confusione (`TP`, `TN`, `FP`, `FN`):
- $\text{Precision} = \frac{TP}{TP + FP}$: tra tutti i clienti segnalati come a rischio, quanti stavano davvero per abbandonare (affidabilità degli allarmi).
- $\text{Recall (TPR / Sensibilità)} = \frac{TP}{TP + FN}$: tra tutti i clienti che abbandonano davvero, quanti siamo riusciti a intercettare.
- $F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$: è la **media armonica** tra Precision e Recall. A differenza della media aritmetica, la media armonica penalizza fortemente i valori estremi: un modello con Precision 100% e Recall 1% avrebbe media aritmetica 50,5%, ma F1 vicino al 2%. Quindi F1 è alta solo quando *entrambe* le metriche sono buone.

---

### C3. Perché nel vostro dominio aziendale la Recall è più critica della Precision?
**Risposta:** Per l'**asimmetria economica dei costi**:
- un **Falso Negativo (FN)** è un cliente che sta per andarsene e non viene individuato $\rightarrow$ l'azienda lo perde e deve sostenere l'intero **CAC** (marketing, acquisizione, attivazione) per rimpiazzarlo, oltre a perdere i ricavi futuri;
- un **Falso Positivo (FP)** è un cliente fedele a cui viene proposta per prudenza un'offerta o un controllo di assistenza $\rightarrow$ comporta un costo promozionale molto inferiore al CAC.
Per questo un agente razionale accetta più Falsi Positivi pur di abbattere drasticamente i Falsi Negativi (aumentando la Recall).

---

### C4. Cos'è la Curva ROC e cosa misura esattamente l'AUC-ROC?
**Risposta:** La **Curva ROC** (*Receiver Operating Characteristic*) traccia il **True Positive Rate** ($\text{TPR} = \text{Recall} = \frac{TP}{TP+FN}$) sull'asse verticale in funzione del **False Positive Rate** ($\text{FPR} = \frac{FP}{FP+TN} = 1 - \text{Specificity}$) sull'asse orizzontale, al variare di tutte le possibili soglie decisionali $\tau \in [0, 1]$.
L'**AUC-ROC** (*Area Under the Curve*) ha un'interpretazione probabilistica molto intuitiva: è la **probabilità che il classificatore assegni uno score di rischio più alto a un cliente in churn scelto a caso rispetto a un cliente rimasto scelto a caso**. Vale `0,5` per un classificatore casuale (diagonale) e `1,0` per uno perfetto; i nostri `0,842–0,843` indicano quindi una forte capacità di *ranking*.

---

### C5. Quando si preferisce la Curva Precision-Recall (PR) rispetto alla Curva ROC?
**Risposta:** Come visto nella Lezione 04 (Géron Cap. 3), la Curva ROC può apparire eccessivamente ottimistica quando la classe positiva è **molto rara** (es. 1% di frodi), perché un gran numero di `TN` mantiene comunque basso il denominatore di $\text{FPR} = \frac{FP}{FP+TN}$. La **Curva PR** non usa i `TN`, quindi va preferita quando la classe positiva è rarissima o quando i Falsi Positivi sono molto costosi. Nel nostro caso (`26,54%` positivi, sbilanciamento moderato) la ROC-AUC resta molto informativa, ma l'abbiamo sempre affiancata a Precision, Recall e F1-Score espliciti.

---

### C6. Come avete calibrato la soglia $\tau = 0.35$ e qual è la differenza rispetto a `class_weight='balanced'`?
**Risposta:**
- **Calibrazione di soglia ($\tau = 0.35$):** non cambia l'addestramento né i pesi del modello (quindi l'AUC-ROC resta identica), ma sposta il punto operativo sulle probabilità predette (`y_prob >= 0.35`). L'abbiamo verificata tramite *5-fold cross-validation* (`cross_val_predict`) **solo sul Training Set** (`Recall CV = 70,4%`, `F1 = 0,627`), e sul Test Set porta la Recall da `55,9%` a **`70,3%`** (riducendo i `FN` da `165` a `111`) aumentando persino l'F1 da `0,605` a **`0,614`**.
- **`class_weight='balanced'`:** agisce invece *in fase di addestramento* nella funzione di costo (Log-Loss), moltiplicando gli errori sulla classe minoritaria (`Churn=1`) per un peso inversamente proporzionale alla sua frequenza ($\frac{N}{2 \cdot N_1} \approx 1.88$, contro $\frac{N}{2 \cdot N_0} \approx 0.68$ per `Churn=0`). Sul Test Set spinge la Recall fino al **`78,3%`** (`293` intercettati, solo `81` persi, `F1 = 0,614`), riducendo la Precision al `50,4%`.

---

## Sezione D — Apprendimento Non Supervisionato: K-Means & PCA (Lezione 06)

### D1. Come funziona l'algoritmo K-Means e cos'è K-Means++?
**Risposta:** L'algoritmo di Lloyd (K-Means) minimizza l'Inerzia (**WCSS**, *Within-Cluster Sum-of-Squares*):
$$\text{WCSS} = \sum_{j=1}^{K} \sum_{\mathbf{x}_i \in C_j} \|\mathbf{x}_i - \boldsymbol{\mu}_j\|^2$$
Opera iterativamente in 3 passi: (1) inizializza $K$ centroidi; (2) **assegnazione**, ogni istanza viene assegnata al centroide più vicino secondo la distanza euclidea; (3) **aggiornamento**, ogni centroide viene ricalcolato come media dei punti del proprio cluster, fino a convergenza (nel nostro caso 23 iterazioni).
Poiché l'inizializzazione casuale può bloccarsi in minimi locali scadenti, `scikit-learn` usa di default **K-Means++** (che sceglie i centroidi iniziali distanti tra loro con probabilità proporzionale al quadrato della distanza dai centroidi già estratti) ripetuto `n_init = 10` volte, conservando la partizione con Inerzia minima.

---

### D2. Perché per scegliere $K$ non basta minimizzare l'Inerzia? E cos'è esattamente il Silhouette Score?
**Risposta:** Perché l'Inerzia WCSS **decresce monotonicamente** all'aumentare di $K$ (fino a diventare `0` quando $K = N$, cioè ogni cliente è un cluster a sé). Bisogna quindi cercare un gomito (*Elbow Method*) e affiancarlo al **Silhouette Score**:
$$s(i) = \frac{b(i) - a(i)}{\max\big(a(i),\, b(i)\big)} \in [-1, +1]$$
dove $a(i)$ è la distanza media del punto $i$ dagli altri punti del *proprio* cluster (**coesione**) e $b(i)$ è la distanza media dal *cluster più vicino* diverso dal proprio (**separazione**). Vicino a `+1` indica cluster compatti e ben separati, `0` indica punti sul confine, valori negativi indicano assegnazioni errate.

---

### D3. Un Silhouette Score di `0,282` (a $K=3$) non è un po' basso? E perché nel codice usate `sample_size=2000`?
**Risposta:** È una domanda molto acuta. In 2 dimensioni con nuvole sferiche artificiali ci si aspetta una Silhouette $> 0,6$, ma qui operiamo in uno **spazio a 30 dimensioni misto**, composto da 3 variabili continue Z-Score e **27 variabili binarie one-hot** (`0/1`), dove i clienti reali formano un continuum comportamentale con zone di transizione e subiscono la *curse of dimensionality* delle distanze euclidee. In questo contesto `0,282` indica una separazione moderata ma strutturalmente reale (confermata sia dal netto crollo della Silhouette per $K \ge 4$ sia dalla forte differenziazione di churn `43,4%` vs `15,9%` vs `7,2%` stabile anche sul Test Set).
Quanto a `sample_size=2000` (con `random_state=42`), il calcolo esatto della matrice delle distanze a coppie costa $O(N^2)$: su 5.634 punti sono ~15,9 milioni di coppie per ogni $K$; campionare 2.000 istanze rende il calcolo molto più rapido mantenendo una stima statistica affidabile e riproducibile.

---

### D4. Perché avete scelto K-Means invece di DBSCAN o Gaussian Mixture Models (GMM)?
**Risposta:** Per tre motivi metodologici:
1. **Assegnazione out-of-sample senza leakage:** K-Means espone il metodo `.predict()` che assegna i clienti del Test Set ai centroidi appresi sul Training Set. **DBSCAN** è un algoritmo transduttivo basato su densità locale e in `scikit-learn` non dispone di `.predict()` per nuovi campioni; inoltre su spazi 30D con molte feature binarie la scelta del raggio $\varepsilon$ diventa molto instabile.
2. **Interpretabilità CRM:** i centroidi di K-Means corrispondono direttamente a profili medi di cliente facilmente comunicabili al marketing.
3. **Scalabilità lineare** $O(N \cdot K \cdot d \cdot \text{iter})$, laddove un GMM full-covariance su 30 feature richiederebbe la stima di matrici di covarianza $30 \times 30$ su variabili per lo più binarie (poco aderenti all'ipotesi di normalità multivariata).

---

### D5. Come funziona la PCA e che ruolo svolge nel vostro progetto?
**Risposta:** La **PCA** (*Principal Component Analysis*) riduce la dimensionalità proiettando i dati centrati $\mathbf{X}_c = \mathbf{X}_{\text{train}} - \bar{\mathbf{X}}_{\text{train}}$ lungo le direzioni ortogonali di massima varianza, ossia gli autovettori $\mathbf{v}_k$ della matrice di covarianza campionaria $\boldsymbol{\Sigma} = \frac{1}{N-1}\mathbf{X}_c^T\mathbf{X}_c$ associati agli autovalori maggiori $\lambda_k$.
Nel nostro progetto **NON** usiamo la PCA per addestrare i classificatori (perché perderemmo il 42% della varianza e l'interpretabilità delle singole feature), ma **esclusivamente per visualizzare in 2D i cluster K-Means** appresi nello spazio 30D. Le prime due componenti spiegano il **57,87%** della varianza:
- **$PC_1$ (39,37%):** Spesa e Volume Servizi (`MonthlyCharges +0.501`, `TotalCharges +0.491`, `tenure +0.338`, vs `InternetService_No -0.166`);
- **$PC_2$ (18,50%):** Anzianità e Stabilità Contrattuale (`tenure +0.612`, `TotalCharges +0.338`, `Contract_Two year +0.221`, vs `MonthlyCharges -0.277` e `Fiber optic -0.163`).

---

### D6. Nel codice riordinate i cluster per tasso di churn decrescente usando `y_train`: non è data leakage?
**Risposta:** **No**, per due ragioni rigorose:
1. l'algoritmo K-Means è addestrato (`fit_predict`) **esclusivamente su `X_train_scaled`**, senza mai vedere `y_train`; quindi le partizioni e i centroidi sono al 100% non supervisionati;
2. il riordino (`cluster_remap`) è una semplice **rinominazione biunivoca degli ID** (`{2->0, 1->1, 0->2}`) calcolata *solo sul Training Set* e poi applicata in modo fisso al Test Set. Non altera in alcun modo i gruppi, né le distanze, né usa mai `y_test`: serve unicamente a garantire che il Cluster `0` sia sempre i *Volatili* in grafici e tabelle indipendentemente dall'ordine casuale dei centroidi.

---

## Sezione E — Decision Trees CART & Interpretabilità White-Box (Lezione 05)

### E1. Come costruisce l'albero l'algoritmo CART e qual è la formula dell'Indice di Gini?
**Risposta:** L'algoritmo **CART** (*Classification and Regression Trees*) costruisce alberi binari mediante ricerca euristica *greedy* top-down: ad ogni nodo sceglie la coppia `(feature k, soglia t_k)` che minimizza l'impurità pesata dei due figli:
$$J(k, t_k) = \frac{m_{\text{left}}}{m} G_{\text{left}} + \frac{m_{\text{right}}}{m} G_{\text{right}}$$
L'**Indice di Gini** del nodo $i$ è:
$$G_i = 1 - \sum_{k=1}^{K} p_{i,k}^2$$
Nel nostro caso binario ($K=2$), $G_i = 1 - (p_{i,0}^2 + p_{i,1}^2) \in [0, 0.5]$: vale `0` se il nodo è puro al 100%, e raggiunge il massimo `0,5` quando le due classi sono al 50%/50%. Rispetto all'Entropia di Shannon ($H_i = -\sum p_{i,k} \log_2 p_{i,k}$), produce alberi quasi identici ma è leggermente più veloce perché evita i logaritmi.

---

### E2. Perché avete addestrato il Decision Tree su feature numeriche NON scalate?
**Risposta:** Perché gli alberi CART valutano soglie ortogonali su una singola variabile per volta (`x_k <= t_k`), quindi sono **invarianti a qualsiasi trasformazione monotona crescente** come la standardizzazione Z-Score (Lezione 05). Le metriche predittive rimangono identiche al decimale (`79,42%` Acc, `54,01%` Recall, `0,8267` AUC), ma lasciando `tenure`, `MonthlyCharges` e `TotalCharges` in `passthrough` otteniamo regole *White-Box* direttamente leggibili in **mesi** (`tenure <= 16.5`) ed **Euro** (`TotalCharges <= 120.0`) invece che in deviazioni standard.

---

### E3. Qual è la complessità computazionale di un Decision Tree e perché avete fissato `max_depth=5`?
**Risposta:**
- **Training:** $O(n \cdot m \log_2 m)$, dove $n$ è il numero di feature (`30`) e $m$ il numero di istanze (`5.634`);
- **Predizione:** $O(\log_2 m)$ (nel nostro caso al più 5 confronti dalla radice alla foglia!), indipendente da $n$, quindi rapidissima.

Abbiamo vincolato `max_depth = 5` (ottenendo 32 foglie) come **iperparametro di regolarizzazione**: un albero non vincolato è un modello non parametrico con altissimi gradi di libertà che continuerebbe a suddividere fino a creare foglie pure da 1–2 clienti, andando in pesante **overfitting** (alta varianza) sul Training Set.

---

### E4. [DOMANDA TRABOCCHETTO SU SLIDE 9] Nel nodo blu scuro di Slide 9 (`165 su 193`, 85,5% churn), questi clienti sono PRIMA o DOPO lo split su `TechSupport_Yes <= 0.5`?
**Risposta:** *(Risposta da 30 e lode!)* Ottima osservazione professore: quel nodo al **livello 3** rappresenta già tutti i clienti che soddisfano il percorso dalla radice fino a lì — cioè **`tenure <= 16.5` mesi**, **Fibra Ottica** (`InternetService_Fiber optic > 0.5`) e **`TotalCharges <= 120 €`** — e sono esattamente **193 clienti di cui 165 in churn (85,5%, $G=0.248$)**. L'etichetta `TechSupport_Yes <= 0.5` stampata in cima a quel riquadro è la **condizione di split usata dal nodo per generare i figli del livello 4** (non mostrati nel grafico perché abbiamo visualizzato i primi 3 livelli con `plot_tree(..., max_depth=3)` per leggibilità). Se scendiamo al livello 4 del modello addestrato (`max_depth=5`), i clienti *senza* supporto tecnico (`TechSupport_Yes <= 0.5`) sono `185` e il loro churn sale ulteriormente all'**87,6%** (`162/185`, $G=0.218$), mentre gli `8` clienti con supporto tecnico scendono al `37,5%`!

---

### E5. Quali sono i limiti strutturali di un singolo Decision Tree?
**Risposta:** Due limiti principali (Lezione 05):
1. **Confini decisionali ortogonali agli assi** (a scalino), che rendono l'albero sensibile alle rotazioni dei dati;
2. **Alta varianza (instabilità):** piccole variazioni nel Training Set possono cambiare lo split di radice e stravolgere l'intero albero. Per superare questo limite negli sviluppi futuri proponiamo **Random Forest** (bagging + random subspace) e **XGBoost** (gradient boosting).

---

## Sezione F — Regressione Logistica & Reti Neurali MLP (Lezioni 07–08)

### F1. Come funziona la Regressione Logistica e quali feature risultano più decisive nel vostro modello?
**Risposta:** La Regressione Logistica calcola una combinazione lineare $z = \mathbf{w}^T\mathbf{x} + b$ e la trasforma in probabilità tramite la funzione **sigmoide**:
$$P(Y=1 \mid \mathbf{x}) = \sigma(z) = \frac{1}{1 + e^{-z}}$$
I pesi vengono stimati minimizzando la **Binary Cross-Entropy (Log-Loss)** con regolarizzazione L2 di default (`C = 1.0`, ottimizzatore `lbfgs`): essendo una funzione di costo **convessa**, garantisce la convergenza all'ottimo globale (nel nostro caso in 40 iterazioni).
Esaminando i pesi $\mathbf{w}$ appresi sul Training Set standardizzato:
- **fattori che aumentano di più il churn:** `InternetService_Fiber optic` ($+1,181$), `TotalCharges` ($+0,528$), `PaymentMethod_Electronic check` ($+0,393$);
- **fattori più protettivi (riducono il churn):** `Contract_Two year` ($-1,314$), `tenure` ($-1,257$), `Contract_One year` ($-0,684$).
Questi coefficienti concordano perfettamente con le regole del Decision Tree!

---

### F2. Perché il Perceptron non basta e servono strati nascosti con attivazioni non lineari nell'MLP?
**Risposta:** Un singolo Perceptron (TLU di Rosenblatt, 1957) può separare solo classi **linearmente separabili** e fallisce su problemi semplici come la funzione logica **XOR** (Minsky & Papert, 1969). Impilare più strati in un **MLP** (*Multi-Layer Perceptron*) risolve il limite solo se si inseriscono **funzioni di attivazione non lineari**: senza di esse, una composizione di trasformazioni lineari $W_3(W_2(W_1\mathbf{x}))$ collasserebbe matematicamente in un'unica matrice lineare $W_{\text{tot}}\mathbf{x}$, equivalente a un modello lineare singolo.

---

### F3. Perché nell'MLP usiamo ReLU negli strati nascosti e Sigmoide nello strato di output?
**Risposta:**
- **Strati nascosti — ReLU ($\max(0, z)$):** evita il problema dei **gradienti svanenti (*vanishing gradients*)** tipico della sigmoide (che satura per $|z|$ grande con derivata vicina a 0), ha derivata costante pari a `1` per $z > 0$ ed è computazionalmente leggerissima (Lezione 08).
- **Strato di output — Sigmoide (`out_activation_='logistic'`):** essendo una classificazione binaria con loss *Binary Cross-Entropy*, serve comprimere il logit finale in una probabilità valida $\hat{y} \in (0, 1)$.

---

### F4. Come viene addestrato l'MLP (Backpropagation + Adam) e perché avete inserito L2 ed Early Stopping?
**Risposta:** L'MLP apprende con **Backpropagation**: nel *forward pass* calcola le predizioni e la Binary Cross-Entropy; nel *backward pass* applica la **regola della catena** per propagare i gradienti della loss rispetto a ogni peso; l'ottimizzatore **Adam** (*Adaptive Moment Estimation*) combina *Momentum* (1° momento dei gradienti) e *RMSProp* (2° momento / learning rate adattivo per parametro) per convergere rapidamente e in modo stabile.
Su dati tabellari di 5.634 istanze, una rete `(64, 32)` ha **4.097 parametri** (contro appena **31** della Regressione Logistica!) e quindi tende fortemente all'**overfitting**. Applicando le tecniche delle Lezioni 07 e 08:
- **regolarizzazione L2 (`alpha=0.1`):** aggiunge $\frac{\alpha}{2}\|\mathbf{W}\|_2^2$ alla loss, mantenendo i pesi piccoli e distribuiti (*weight decay*);
- **Early Stopping (`early_stopping=True`):** riserva automaticamente il `10%` del Training Set come validation interna e ferma l'addestramento se lo score di validazione non migliora per `10` epoche consecutive (`n_iter_no_change=10`), ripristinando i pesi migliori. Grazie a questo, la nostra MLP si ferma a **17 epoche** (Base) e **19 epoche** (+Cluster) invece di girare per 500 epoche, portando l'AUC da ~`0,78` (in overfitting senza early stopping) fino a **`0,8429`**.

---

## Sezione G — Analisi Critica, Codice, Etica & Sviluppi Futuri (Slide 8 & 10)

### G1. Perché l'aggiunta del cluster K-Means aiuta leggermente l'MLP, lascia invariato il CART e peggiora leggermente la Regressione Logistica?
**Risposta:** È il risultato critico più istruttivo del progetto:
- **Regressione Logistica (`AUC 0,8422 -> 0,8420`, `Recall 55,9% -> 55,1%`):** i cluster K-Means sono una funzione (non lineare a regioni di Voronoi) delle *stesse* 30 variabili di input già presenti nel modello; aggiungere le colonne `Cluster_1` e `Cluster_2` introduce ridondanza/collinearità e una lieve perdita di calibrazione lineare.
- **Decision Tree (`79,42%` Acc, `54,01%` Recall invariati):** fino a `max_depth=5`, gli split ottimali su `tenure`, `InternetService_Fiber optic` e `TotalCharges` riducono l'impurità di Gini più di quanto farebbe uno split sull'etichetta discreta del cluster, quindi l'albero non sceglie mai i cluster nei suoi tagli.
- **MLP (`Acc 80,0% -> 80,3%`, `F1 0,579 -> 0,585`, `AUC 0,8424 -> 0,8429`):** la rete neurale riesce a sfruttare il segnale aggregato di appartenenza al macro-segmento combinandolo in modo non lineare nei suoi strati nascosti, ottenendo un miglioramento piccolo ma coerente su tutte e cinque le metriche.
In conclusione, il valore primario del clustering resta **descrittivo e strategico** (profilazione CRM).

---

### G2. Che cos'è il Rasoio di Occam e perché qui favorisce la Regressione Logistica?
**Risposta:** Il **Rasoio di Occam** (principio di parsimonia) afferma che, a parità di capacità esplicativa o predittiva, si deve preferire il modello più semplice. Nel nostro caso Regressione Logistica (`31` parametri, problema convesso, coefficienti interpretabili, `AUC = 0,8422`, `F1 = 0,6049`) ed MLP (`4.225` parametri con cluster, non convessa, black-box, `AUC = 0,8429`, `F1 = 0,5847`) sono praticamente indistinguibili nella separazione globale: la complessità della rete neurale non è giustificata da un salto prestazionale sufficiente, quindi in produzione si sceglie la **Regressione Logistica** per il ranking, affiancata da **CART** per la spiegabilità al business.

---

### G3. [DOMANDE SUL CODICE] Perché nel notebook usate `clone(base_model)`, `predict_proba()[:, 1]` e `random_state=42`?
**Risposta:**
- **`sklearn.base.clone(base_model)` (Cella 19):** crea una copia pulita e *non ancora addestrata* del modello con gli stessi iperparametri, garantendo che il modello addestrato su Base (`m_orig`) e quello addestrato su `+Cluster` (`m_clust`) siano istanze totalmente indipendenti e non condividano pesi o stati interni.
- **`predict_proba(X)[:, 1]` (Cella 19):** `predict()` restituisce solo etichette binarie rigide `0/1` alla soglia `0,50`, mentre la curva ROC e l'AUC-ROC richiedono lo **score continuo di probabilità della classe positiva** (`colonna 1` della matrice `[P(y=0), P(y=1)]`) per valutare tutti i possibili valori di soglia.
- **`random_state=42` (Cella 2):** fissa il seme del generatore pseudocasuale per split stratificato, inizializzazione K-Means++, campionamento Silhouette, PCA, CART e pesi iniziali dell'MLP, garantendo il **100% di riproducibilità deterministica** (requisito esplicito del corso).

---

### G4. Perché nella conclusione citate GDPR e *fairness*? Quali rischi etici esistono in un modello di Churn?
**Risposta:** Il dataset contiene attributi demografici sensibili come **`SeniorCitizen`** (età avanzata) e **`gender`**. Un uso acritico in produzione solleva due rischi:
1. **Fairness / Discriminazione:** se il modello associasse probabilità diverse a gruppi protetti, l'azienda potrebbe erogare sconti migliori a certe fasce demografiche svantaggiandone altre (es. clienti anziani meno propensi a cambiare operatore puniti con tariffe più alte). Occorre quindi effettuare *fairness audit* (o rimuovere gli attributi protetti valutando anche eventuali proxy).
2. **GDPR:** impone minimizzazione dei dati, pseudonimizzazione (`customerID` rimosso) e trasparenza sulle decisioni automatizzate con impatto significativo sul cliente (art. 22 GDPR) — ragione in più per apprezzare l'interpretabilità *White-Box* dell'albero CART e dei pesi della Regressione Logistica.

---

### G5. Cosa intende negli sviluppi futuri per "Soft Clustering con `transform()`", "SMOTE in CV" e "Cost-Sensitive Learning"?
**Risposta:**
1. **Soft Clustering (`kmeans.transform(X)`):** invece di passare ai classificatori un'etichetta binaria rigida (che perde la posizione fine del cliente dentro il cluster), `kmeans.transform()` restituisce le **3 distanze euclidee continue** da ciascun centroide, fornendo ai modelli informazione geometrica molto più ricca e continua.
2. **SMOTE in Cross-Validation:** *Synthetic Minority Over-sampling Technique* genera esempi sintetici della classe minoritaria (`Churn=1`) interpolando tra vicini prossimi. Va applicato **rigorosamente solo all'interno del training fold** (tramite `imblearn.pipeline.Pipeline`), mai prima dello split, altrimenti i duplicati sintetici contaminerebbero il validation/test set causando grave data leakage.
3. **Cost-Sensitive Learning:** invece di usare una soglia fissa o pesi simmetrici, si definisce una matrice di costo esplicita in Euro — costo reale di un `FN` (CAC + margine perso) vs costo reale di un `FP` (valore dello sconto/chiamata) — e si sceglie la soglia $\tau^*$ che **massimizza direttamente il profitto atteso in Euro**, realizzando fino in fondo la funzione di utilità dell'agente razionale.
