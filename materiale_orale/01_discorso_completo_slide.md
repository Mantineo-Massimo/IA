# Foglio 1 — Discorso Completo Parola per Parola (Slide 1 → Slide 10)

**Progetto:** Churn Prediction e Segmentazione Comportamentale (Dataset IBM Telco Customer Churn)  
**Studente:** Massimo Mantineo (Matricola: 541924)  
**Corso:** Laboratorio di Intelligenza Artificiale — Prof. Giacomo Fiumara (A.A. 2025/2026)  
**Slide di riferimento:** [`relazione/presentazione.pdf`](../relazione/presentazione.pdf) (10 slide)

---

## Tabella di Marcia e Consigli di Esposizione

- **Durata complessiva consigliata:** **10 – 12 minuti** (ritmo calmo, chiaro, senza correre sui numeri).
- **Regola d'oro:** Non leggere le slide parola per parola come un elenco puntato: usa le slide come *appoggio visivo* e **indica fisicamente i grafici e le tabelle** nei momenti segnati con `[Indica ...]`.
- **Respiro e ritmo:** Fai una piccola pausa (1–2 secondi) a ogni cambio di slide per dare al professore il tempo di guardare la nuova schermata.

| Slide | Titolo | Tempo Consigliato | Messaggio Chiave da Lasciare al Professore |
| :---: | :--- | :---: | :--- |
| **1** | Titolo e Copertina | 0:30 | Presentazione chiara del progetto e dei due pilastri (non supervisionato + supervisionato). |
| **2** | Definizione di IA e Agente PEAS | 1:15 | Agire Razionalmente (no onniscienza), agente utility-based con apprendimento e proprietà PEAS. |
| **3** | Task ML e Approccio Probabilistico | 1:10 | Batch/Model-Based, Mitchell $\langle T,E,P \rangle$ vs PEAS, probabilità discriminativa vs regole fisse. |
| **4** | Dataset, Preprocessing e Sbilanciamento | 1:15 | Zero data leakage, pulizia `TotalCharges`, 30 feature, trappola dell'Accuracy e boxplot. |
| **5** | Clustering K-Means e Scelta di $K$ | 1:15 | Perché K-Means (`.predict()`), Elbow + Silhouette $\rightarrow$ $K=3$ ($s=0.282$). |
| **6** | Segmentazione PCA e Profili | 1:10 | 3 segmenti (Volatili 43,4% confermato sul Test) e significato di $PC_1$ e $PC_2$ (57,9% var.). |
| **7** | Pipeline Supervisionata e Modelli | 1:10 | Pipeline pulita 30D vs 32D, LogReg, CART non scalato, MLP con L2 + Early Stopping. |
| **8** | Confronto Prestazioni e Soglia | 1:30 | Risultati veri sul Test Set, Rasoio di Occam, impatto reale del cluster, soglia $\tau=0.35$/`balanced`. |
| **9** | Spiegabilità Business: Albero CART | 1:15 | Gini in $[0, 0.5]$, lettura White-Box in mesi/Euro, regola critica Fibra Ottica (85,5% churn). |
| **10** | Discussione Critica e Conclusioni | 1:10 | Sintesi matura: limiti, GDPR/fairness, ensemble, soft clustering, SMOTE e Cost-Sensitive. |

---

## Slide 1 — Copertina (Titolo)

> *[Guarda il professore, tono sicuro e naturale]*
>
> «Buongiorno professore, sono Massimo Mantineo e oggi presento il mio progetto di Laboratorio di Intelligenza Artificiale, intitolato **"Churn Prediction e Segmentazione Comportamentale"**, applicato al settore delle telecomunicazioni sul dataset pubblico **IBM Telco Customer Churn**.
>
> L'obiettivo del lavoro è progettare una pipeline completa e rigorosa che integri le due anime del Machine Learning affrontate durante il corso: da un lato l'**apprendimento non supervisionato**, per scoprire segmenti comportamentali latenti nella base clienti, e dall'altro l'**apprendimento supervisionato**, per stimare la probabilità di abbandono — il cosiddetto *churn* — e verificare criticamente se e quanto l'informazione di clustering migliori davvero le prestazioni predittive.»

**Transizione verso la Slide 2:**
> «Prima di entrare nei dati e nei modelli, vorrei inquadrare il problema dal punto di vista teorico dell'Intelligenza Artificiale, partendo dai fondamenti di Russell e Norvig visti nelle prime due lezioni.»

---

## Slide 2 — Inquadramento Teorico: Definizione di IA e Agente PEAS

> *[Indica prima lo schema a 4 quadranti a sinistra, poi il modello PEAS e il ciclo in basso a destra]*
>
> «Secondo l'impostazione di Russell e Norvig, le definizioni storiche di Intelligenza Artificiale si distribuiscono lungo due assi: **Pensiero contro Comportamento** e **aderenza al modello Umano contro Razionalità ideale**. Ne nascono quattro approcci: pensare umanamente, cioè la modellazione cognitiva; pensare razionalmente, basato sulle leggi formali della logica; agire umanamente, legato al Test di Turing; e infine **agire razionalmente**, evidenziato in basso a destra nello schema.
>
> Il nostro progetto si colloca proprio nel quadrante dell'**Agire Razionalmente**. Non vogliamo imitare il ragionamento di un operatore umano, bensì costruire un **agente con apprendimento basato su utilità** (*utility-based learning agent*): un sistema capace di scegliere azioni che massimizzino il valore atteso della misura di prestazione $P$ in condizioni di incertezza. Qui è fondamentale distinguere **razionalità** e **onniscienza**: l'agente non può prevedere con certezza assoluta il futuro di ogni cliente, ma agisce in modo razionale utilizzando una funzione di utilità per bilanciare obiettivi tra loro conflittuali, ossia evitare l'abbandono contenendo al tempo stesso il costo delle offerte promozionali.
>
> *[Sposta l'attenzione sulla colonna destra e sul diagramma del ciclo PEAS]*
>
> Formalizzando il sistema CRM tramite il modello **PEAS**:
> - la **Performance ($P$)** è un criterio oggettivo esterno definito sull'ambiente aziendale: massimizzare il valore netto di fidelizzazione, riducendo il churn senza sprecare budget rispetto al **CAC**, cioè il *Customer Acquisition Cost*, il costo di acquisizione di un nuovo cliente;
> - l'**Environment ($E$)** è l'ecosistema operativo della clientela Telco. Applicando la tassonomia di Russell e Norvig, è un ambiente **parzialmente osservabile** — perché i nostri log non vedono, ad esempio, le offerte ricevute dalla concorrenza o l'insoddisfazione inespressa del cliente — **nondeterministico**, e quindi modellato **stocasticamente** tramite probabilità, **multi-agente** (per la presenza di clienti e operatori concorrenti), **continuo** su variabili come tempo e importi di spesa, e **sconosciuto** a priori nelle sue leggi, motivo per cui lo apprendiamo dai dati. È inoltre importante distinguere il piano operativo da quello sperimentale: nel ciclo reale delle campagne di retention l'ambiente è **sequenziale e dinamico**, perché una promozione oggi influenza il comportamento futuro del cliente; nella singola predizione *batch* su uno snapshot del dataset, invece, il problema viene trattato come **episodico e statico**;
> - gli **Actuators ($A$)** sono le leve di intervento del CRM: offerte mirate, sconti personalizzati e attivazione prioritaria del supporto tecnico;
> - i **Sensors ($S$)** sono i dati estratti dai log aziendali: fatturazione, durata contrattuale o *tenure*, tipologia di contratto e servizi attivi.
>
> Come riassume lo schema in basso a destra, i sensori alimentano l'agente razionale — che combina i modelli di Machine Learning con la logica di utilità — e gli attuatori agiscono sull'ambiente, il cui esito viene poi valutato esternamente tramite $P$.»

**Transizione verso la Slide 3:**
> «A questo punto la domanda naturale è: come fa concretamente l'agente a tradurre le percezioni dei sensori in decisioni efficaci sugli attuatori? Lo fa apprendendo dai dati storici attraverso due task di Machine Learning integrati.»

---

## Slide 3 — Formulazione dei Task ML e Approccio Probabilistico

> *[Indica il banner in alto, poi la colonna sinistra e infine la colonna destra]*
>
> «Passando dal PEAS al Machine Learning della Lezione 03, progettiamo un sistema di tipo **Batch**, cioè addestrato offline sullo storico disponibile, e **Model-Based**, perché costruisce modelli matematici predittivi anziché limitarsi a memorizzare le istanze. Il progetto integra due task complementari:
>
> Il primo è il **Task Supervisionato**, che formalizziamo con la definizione classica $\langle T, E, P \rangle$ di Tom Mitchell:
> - il **Task ($T$)** è una classificazione binaria probabilistica sulla variabile target `Churn`;
> - l'**Experience ($E$)** è costituita dalle **5.634 istanze etichettate** del Training Set, estratte dai 7.043 clienti complessivi del dataset;
> - la **Performance ($P$)** è misurata tramite metriche adeguate allo sbilanciamento delle classi: **Recall**, **Precision**, **F1-Score** e **AUC-ROC**.
>
> Il secondo è il **Task Non Supervisionato**, di natura esplorativa: applichiamo il clustering **K-Means** sulle **30 feature** del cliente escludendo completamente l'etichetta `Churn`, valutando la qualità dei gruppi con **Inerzia WCSS** e **Silhouette Score** e ispezionandoli in 2D mediante **PCA**, per poi verificare sperimentalmente se aggiungere l'etichetta di cluster come feature aiuti i classificatori supervisionati.
>
> *[Indica la colonna di destra]*
>
> Perché ricorriamo al Machine Learning anziché alla programmazione tradizionale? Scrivere a mano regole booleane `if-then` su 30 variabili interagenti porta a sistemi fragili, difficili da mantenere al variare delle tariffe di mercato e incapaci di graduare il rischio. Adottiamo invece un approccio **probabilistico discriminativo**: modelli come la Regressione Logistica, l'albero CART e la rete neurale MLP stimano direttamente la probabilità a posteriori $P(\text{Churn}=1 \mid \mathbf{x}) \in [0, 1]$. Disporre di uno score continuo invece di un semplice sì o no ci consente di fare **ranking** dei clienti per urgenza e, soprattutto, di tarare la soglia decisionale in funzione del CAC.
>
> *[Indica il riquadro in basso]*
>
> Sottolineo infine una distinzione teorica importante: la **$P$ del PEAS** misura l'impatto economico esterno nell'ambiente aziendale, mentre la **$P$ di Mitchell** misura la qualità predittiva del modello di Machine Learning sul proprio task.»

**Transizione verso la Slide 4:**
> «Definito il quadro metodologico, passiamo ora ai dati reali: vediamo la struttura del dataset IBM Telco, le scelte di pulizia, la prevenzione del *data leakage* e le prime evidenze esplorative.»

---

## Slide 4 — Dataset Telco: Preprocessing e Sbilanciamento del Target

> *[Indica prima il banner in alto, poi la colonna sinistra e infine i due boxplot a destra]*
>
> «Lavoriamo sul dataset **IBM Telco Customer Churn**, composto da **7.043 clienti** e **21 colonne** originali: 19 feature utili — di cui 3 numeriche (`tenure`, `MonthlyCharges`, `TotalCharges`) e 16 categoriali — più la colonna `customerID`, che rimuoviamo subito perché identificativo univoco privo di potere generalizzante, e la variabile target `Churn`, codificata in binario `1` per abbandono e `0` per permanenza.
>
> Nella fase di preparazione abbiamo applicato quattro principi di rigore:
> 1. **Data Cleaning:** ispezionando i tipi di dato, `TotalCharges` risultava importata come stringa a causa di **11 righe contenenti uno spazio vuoto**. Verificando quei record abbiamo constatato che appartengono tutti a clienti appena acquisiti con **`tenure = 0`**, che non hanno ancora maturato la prima fattura. Imputare quei valori a **0,0 €** e convertire la colonna in `float64` è quindi la scelta semanticamente esatta: non perdiamo campioni e non introduciamo distorsioni come accadrebbe imputando la media o la mediana.
> 2. **Zero Data Leakage:** per evitare qualsiasi contaminazione tra addestramento e test, eseguiamo lo **split stratificato 80/20** — ottenendo 5.634 istanze di Training e 1.409 di Test, entrambe con esattamente il 26,54% di Churn — **prima** di qualsiasi trasformazione parametrica, quindi prima dello scaling, dell'encoding, del clustering e della PCA.
> 3. **Standardizzazione e Encoding:** addestriamo `StandardScaler` (trasformazione Z-Score a media 0 e deviazione standard 1) sulle 3 variabili numeriche e `OneHotEncoder` con **`drop='first'`** sulle 16 categoriali esclusivamente sul Training Set, evitando la trappola delle variabili *dummy* e arrivando a **30 feature** finali.
> 4. **Sbilanciamento del target:** solo il **26,54%** dei clienti abbandona, contro il 73,46% che resta. Come visto nella Lezione 04, questo rende l'Accuracy da sola fuorviante: un classificatore *dummy* che predicesse sempre "Rimasto" otterrebbe comunque il **73,46% di Accuracy**, ma con **Recall uguale a 0%**, fallendo completamente l'obiettivo aziendale.
>
> *[Indica i due boxplot a destra]*
>
> Dando uno sguardo all'analisi esplorativa nei boxplot a destra — dove il verde rappresenta i clienti rimasti, il rosso chi abbandona e la linea orizzontale nera indica la mediana — emergono subito due segnali fortissimi: chi abbandona ha una **permanenza mediana di soli 10 mesi**, contro **38 mesi** per chi resta, e paga una **spesa mensile mediana di 79,65 €**, contro **64,43 €** dei clienti fedeli. Il rischio di churn si concentra quindi chiaramente sui **nuovi clienti ad alta spesa mensile**.»

**Transizione verso la Slide 5:**
> «Sulle 30 feature standardizzate del solo Training Set, passiamo ora alla prima grande fase sperimentale: l'apprendimento non supervisionato tramite clustering K-Means.»

---

## Slide 5 — Apprendimento Non Supervisionato: Clustering K-Means

> *[Indica prima i punti a sinistra, poi il grafico a doppio asse a destra]*
>
> «Il nostro obiettivo nella fase non supervisionata è segmentare i **5.634 clienti del Training Set**, nello spazio standardizzato a 30 feature, **senza mai utilizzare l'etichetta `Churn`**, così da scoprire raggruppamenti puramente comportamentali e contrattuali.
>
> Tra gli algoritmi della Lezione 06 abbiamo scelto **K-Means** con inizializzazione *K-Means++* invece di DBSCAN per tre ragioni precise: scalabilità lineare, produzione di centroidi medi immediatamente interpretabili in termini di profilo cliente per il CRM, e soprattutto la disponibilità in *scikit-learn* del metodo induttivo **`.predict()`**, che ci permette di assegnare i clienti del Test Set ai centroidi appresi sul Training Set **senza ri-addestrare e senza indurre data leakage** — cosa che DBSCAN non supporta nativamente.
>
> *[Indica le due curve nel grafico a destra]*
>
> Per scegliere il numero ottimale di cluster abbiamo esplorato $K$ da 2 a 6 combinando due metriche complementari riportate nel grafico:
> - la **curva blu** sull'asse sinistro è l'**Inerzia WCSS**, cioè la somma delle distanze quadratiche dei punti dal proprio centroide: diminuisce sempre al crescere di $K$, per cui cerchiamo il punto di gomito (*elbow*) oltre il quale il guadagno diventa marginale;
> - la **curva arancione tratteggiata** sull'asse destro è il **Silhouette Score**, che varia in $[-1, +1]$ bilanciando coesione intra-cluster $a_i$ e separazione dal cluster vicino $b_i$.
>
> Confrontando i valori:
> - per **$K=2$** la Silhouette è leggermente più alta (`0,293`), ma ispezionando i gruppi si ottiene una divisione banale basata solo sulla presenza o assenza del servizio internet;
> - per **$K=3$** osserviamo esattamente il **gomito netto dell'Inerzia**, che scende da `33.485` a `24.642`, mentre la Silhouette rimane praticamente stabile a **`0,282`**. Va precisato che in uno spazio misto a 30 dimensioni, ricco di variabili binarie one-hot, un Silhouette Score intorno a `0,28` indica una separazione moderata con fisiologiche zone di confine, del tutto coerente per isolare macro-segmenti aziendali;
> - da **$K \ge 4$** in poi, invece, la Silhouette crolla nettamente da `0,230` fino a `0,185` e l'Inerzia si appiattisce, segno di frammentazione artificiale (*over-clustering*).
>
> La scelta più solida dal punto di vista sia geometrico sia di business è quindi **$K = 3$**.»

**Transizione verso la Slide 6:**
> «Vediamo ora che volto hanno questi tre segmenti dal punto di vista aziendale e come si dispongono geometricamente proiettati in due dimensioni tramite PCA.»

---

## Slide 6 — Segmentazione PCA e Profilo dei Segmenti

> *[Indica prima la tabella in alto riga per riga, poi lo scatter plot PCA in basso a destra]*
>
> «Dopo aver addestrato K-Means con $K=3$ sul Training Set, abbiamo riordinato deterministicamente gli identificativi dei cluster per tasso di churn decrescente così da renderne immediata la lettura. Ricordo che l'etichetta `Churn` è stata usata *solo a posteriori* per profilare i gruppi, mai per costruirli.
>
> I tre segmenti emersi sono nettamente distinti:
> - il **Cluster 0**, che chiamiamo **Volatili**, raccoglie 2.557 clienti con anzianità breve (media **15,2 mesi**), spesa mensile medio-alta (**67,87 €**) e un tasso di abbandono critico del **43,4%**; aspetto fondamentale, applicando `.predict()` sul Test Set questo stesso segmento mantiene un churn praticamente identico del **43,3%**, dimostrando che la segmentazione generalizza in modo molto stabile;
> - il **Cluster 1**, i **Fedeli High-Value**, conta 1.863 clienti storici con una permanenza media di **57,7 mesi** e la spesa mensile più alta (**89,44 €**, oltre 5.100 € di spesa cumulata), ma con un tasso di churn contenuto al **15,9%**;
> - infine il **Cluster 2**, i **Low-Cost**, comprende 1.214 clienti con anzianità intermedia (**30,2 mesi**) e spesa molto ridotta (**21,12 €/mese** — tipicamente sola telefonia base senza internet), caratterizzati dal tasso di abbandono minimo, appena il **7,2%**.
>
> *[Indica il grafico PCA in basso a destra]*
>
> Per visualizzare questi gruppi in due dimensioni abbiamo applicato la **Principal Component Analysis (PCA)** sui dati di training centrati a 30 dimensioni. Le prime due componenti catturano complessivamente il **57,9% della varianza totale** — un risultato molto buono per un dataset a 30 feature. Analizzando i pesi (*loadings*) degli autovettori:
> - l'asse orizzontale **$PC_1$** spiega il **39,4%** della varianza e rappresenta **Spesa e Volume dei Servizi** — si vede infatti molto chiaramente la nuvola verde dei clienti *Low-Cost* completamente separata sulla sinistra, rispetto ai clienti con internet e servizi avanzati verso destra;
> - l'asse verticale **$PC_2$** spiega il **18,5%** della varianza e coglie l'**Anzianità e Stabilità Contrattuale**, separando i *Volatili* blu, concentrati in basso, dai *Fedeli High-Value* arancioni verso l'alto a destra.
>
> Sottolineo che la PCA è stata impiegata qui esclusivamente a scopo diagnostico e visivo, non per comprimere gli input dei classificatori.»

**Transizione verso la Slide 7:**
> «Ottenuti i tre segmenti, possiamo ora rispondere alla domanda sperimentale centrale del progetto: come si comportano diverse famiglie di classificatori supervisionati e che effetto ha aggiungere il cluster come feature?»

---

## Slide 7 — Pipeline Sperimentale Supervisionata

> *[Indica prima il diagramma di flusso a sinistra, poi i tre modelli a destra]*
>
> «Questa slide riassume la nostra pipeline sperimentale supervisionata, costruita per essere totalmente priva di *data leakage*. Come mostra lo schema a sinistra, si parte sempre dallo **split stratificato 80/20** (5.634 istanze di Training e 1.409 di Test). Solo sul Training Set vengono fittati preprocessing e K-Means, generando due configurazioni parallele da confrontare a parità esatta di condizioni:
> - la configurazione **Base (`Orig`)**, con le **30 feature** originali preprocessate;
> - la configurazione **Arricchita (`+Clust`)**, con **32 feature**, ottenuta concatenando la codifica One-Hot con `drop='first'` del cluster assegnato (`Cluster_1` e `Cluster_2`, prendendo il Cluster 0 come riferimento).
>
> *[Indica la colonna di destra]*
>
> Su entrambe le configurazioni cloniamo e confrontiamo tre classificatori probabilistici discriminativi di complessità crescente:
> 1. la **Regressione Logistica**, nostra **baseline lineare** convessa basata su funzione sigmoide e minimizzazione della *Log-Loss* (*Binary Cross-Entropy*), utile per misurare quanta parte del problema sia già catturabile da un iperpiano lineare;
> 2. l'**Albero Decisionale CART** regolarizzato a **`max_depth = 5`**, nostro modello **White-Box**. Qui abbiamo fatto una scelta metodologica mirata basata sulla Lezione 05: lo abbiamo addestrato su una pipeline dedicata con le **feature numeriche non scalate**. Poiché gli alberi eseguono tagli su una singola variabile per volta e sono invarianti alle trasformazioni monotone, le prestazioni predittive restano identiche, ma in cambio otteniamo regole decisionali immediatamente leggibili in mesi ed Euro;
> 3. infine la **Rete Neurale MLP** (*Multi-Layer Perceptron*) con due strati nascosti da **64 e 32 neuroni**, attivazione **ReLU**, output sigmoideo e ottimizzatore **Adam**. Seguendo le Lezioni 07 e 08, per evitare l'overfitting tipico delle reti neurali su dati tabellari di medie dimensioni, abbiamo introdotto **regolarizzazione L2 esplicita ($\alpha = 0.1$)** ed **early stopping** su un 10% di validazione interna del Training Set.»

**Transizione verso la Slide 8:**
> «Passiamo ora ai numeri: vediamo come si sono comportati questi tre modelli sul Test Set e quale impatto ha avuto l'aggiunta del cluster.»

---

## Slide 8 — Confronto delle Prestazioni Sperimentali

> *[Indica il grafico a barre a sinistra, poi la tabella in alto a destra e infine i due box in basso]*
>
> «Qui presentiamo i risultati reali sul **Test Set** di 1.409 clienti (di cui 374 in abbandono), confrontando la configurazione Base in blu e quella `+Cluster` in rosso, valutati inizialmente alla soglia standard $\tau = 0.50$. Nel grafico a sinistra vediamo Recall e AUC-ROC; nella tabella a destra riportiamo Recall, F1-Score e AUC-ROC.
>
> Emerge un quadro sperimentale molto interessante e tutt'altro che scontato:
> - La **Regressione Logistica** nella configurazione Base raggiunge le migliori prestazioni complessive alla soglia standard: **80,6% di Accuracy**, **55,9% di Recall**, **F1-Score di 0,605** e **AUC-ROC di 0,842**. Aggiungendo la feature di cluster, subisce invece una leggera flessione (Recall da 55,9% a 55,1%, F1 da 0,605 a 0,598): il motivo è chiaro, i cluster K-Means sono costruiti a partire dalle stesse variabili (`tenure`, costi, servizi), quindi in un modello lineare introducono lieve ridondanza e collinearità.
> - Il **Decision Tree CART** rimane **perfettamente invariato** tra Base e `+Cluster` (54,0% Recall, 0,582 F1, 0,827 AUC), perché gli split ottimali su `tenure` e Fibra Ottica già dominano l'informazione discreta del cluster.
> - La **Rete Neurale MLP**, grazie alla regolarizzazione L2 e all'*early stopping* — che ferma l'addestramento intorno alla 17ª–19ª epoca anziché andare in overfitting sulle 500 epoche — raggiunge un'AUC-ROC di `0,842` in configurazione Base ed è l'**unico modello che beneficia dell'aggiunta del cluster** su tutte le metriche: Accuracy da 80,0% a 80,3%, Recall da 51,9% a 52,1%, F1 da 0,579 a 0,585, toccando la **migliore AUC-ROC assoluta pari a 0,843**.
>
> *[Indica il Box 1 e poi il Box 2 in basso a destra]*
>
> Da questi risultati traiamo due lezioni fondamentali:
> 1. **Rasoio di Occam:** Regressione Logistica (`0,842` AUC) e MLP (`0,843` AUC) sono sostanzialmente equivalenti nella capacità globale di ordinare un cliente a rischio sopra un cliente fedele. Ciò indica che il fenomeno ha una forte struttura lineare/additiva: per il principio di parsimonia, la **Regressione Logistica** rimane il modello di riferimento più semplice, stabile e manutenibile, mentre il valore principale del clustering K-Means si conferma di tipo **descrittivo e strategico**.
> 2. **Trade-off di Soglia (Lezione 04):** con la soglia standard $\tau = 0.50$, anche il miglior modello intercetta solo il 55,9% dei clienti in fuga, perdendone 165 su 374. Poiché perdere un cliente comporta un costo CAC elevato, abbiamo calibrato la soglia decisionale in *5-fold cross-validation* sul solo Training Set — quindi ancora una volta senza data leakage: portando la soglia a **$\tau = 0.35$** la Recall della Logistica sale al **70,3%** e addirittura l'F1 migliora a **0,614**; utilizzando invece **`class_weight='balanced'`**, la Recall sale fino al **78,3%** (intercettando 293 clienti a rischio su 374), con un compromesso molto favorevole rispetto al CAC.»

**Transizione verso la Slide 9:**
> «Se la Regressione Logistica è il nostro motore preferito per il ranking probabilistico, il Decision Tree CART ci offre qualcosa di insostituibile per il dialogo con il management aziendale: la piena spiegabilità *White-Box*. Vediamolo nel dettaglio.»

---

## Slide 9 — Spiegabilità Business: Regole dell'Albero Decisionale (CART)

> *[Indica prima i box 1 e 2 in basso per spiegare come si legge l'albero, poi segui col dito il percorso critico dalla radice fino al nodo blu scuro]*
>
> «In questa slide osserviamo i primi tre livelli dell'albero decisionale CART addestrato sui **5.634 clienti del Training Set non scalato**.
>
> Per leggerlo correttamente, ricordiamo come funziona l'**Indice di Gini** nella classificazione binaria ($K=2$): $G_i = 1 - \sum_k p_{i,k}^2$, che varia tra **0** — nodo puro al 100%, rappresentato da un colore saturo — e **0,5** — massima incertezza 50/50, rappresentato da un colore chiaro. I nodi **arancioni** indicano prevalenza di clienti rimasti, i nodi **blu** prevalenza di abbandoni; il vettore `value = [Rimasti, Abbandoni]` nella radice riporta esattamente `4.139` rimasti e `1.495` abbandoni, con `gini = 0,39`.
>
> Poiché i dati numerici non sono stati scalati, possiamo leggere direttamente soglie reali e azionabili:
> - il **primo split alla radice** è **`tenure <= 16.5` mesi**: l'algoritmo identifica immediatamente i primi 16 mesi e mezzo di contratto come la fascia temporale in cui si concentra il rischio. Chi supera i 16,5 mesi (ramo destro) ha una prevalenza nettissima di permanenza;
> - se invece seguiamo il ramo sinistro (`tenure <= 16.5`, vero) e verifichiamo se il cliente ha un abbonamento in **Fibra Ottica** (`InternetService_Fiber optic > 0.5`, quindi il ramo destro al secondo livello), il nodo diventa subito blu: su 868 clienti in questa fascia, **593 abbandonano**, pari al **68,3% di churn**;
> - scendendo ancora nel sotto-ramo sinistro con spesa cumulata **`TotalCharges <= 120 €`** — cioè clienti in Fibra Ottica nei primissimi uno o due mesi dall'attivazione — arriviamo al nodo blu scuro in basso: **165 abbandoni su 193 clienti**, pari all'**85,5% di churn** con un indice di Gini molto basso (`0,248`). E quel nodo esegue poi l'ulteriore taglio su **`TechSupport_Yes <= 0.5`**, confermando che al livello successivo l'assenza di supporto tecnico aggrava ulteriormente il tasso di abbandono (portandolo all'87,6%).
>
> Dal punto di vista operativo, questa singola regola *White-Box* offre al CRM un'indicazione chiarissima: concentrare programmi proattivi di *onboarding* e supporto tecnico prioritario sui neo-abbonati in Fibra Ottica fin dalle prime settimane.»

**Transizione verso la Slide 10:**
> «Avviandomi verso la chiusura, tiriamo le fila dell'intero percorso discutendo criticamente risultati, limiti e sviluppi futuri.»

---

## Slide 10 — Discussione Critica, Conclusioni & Sviluppi Futuri

> *[Indica il banner in alto e scorri in ordine i quattro riquadri]*
>
> «In sintesi, ogni tecnica utilizzata nel progetto trova un ruolo ben preciso all'interno dell'agente CRM: **K-Means** per la segmentazione strategica, **Regressione Logistica** per il ranking predittivo, **Albero CART** per la spiegabilità al management e la **calibrazione di soglia** per allineare le decisioni alla razionalità economica.
>
> Ricapitolo i quattro punti conclusivi:
> 1. **Ruolo reale del clustering:** K-Means dimostra un grande valore **descrittivo e strategico**, isolando il segmento dei *Volatili* con il 43,4% di churn in modo coerente fra Training e Test Set; come *feature engineering* per i modelli supervisionati porta invece solo un lieve incremento sull'MLP (AUC `0,843`), risultando ridondante per modelli lineari e ad albero.
> 2. **Razionalità economica e soglia CAC:** l'asimmetria dei costi tra perdita del cliente e costo dell'offerta giustifica lo spostamento della soglia da `0,50` a `0,35` o l'uso di `class_weight='balanced'`, elevando la Recall dal **55,9% fino al 78,3%**.
> 3. **Confronto modelli e limiti:** la Regressione Logistica resta il miglior compromesso secondo il **Rasoio di Occam**. Tra i limiti metodologici da riconoscere ci sono l'assunzione di cluster sferici tipica del K-Means, la natura statica del dataset — privo di serie storiche di utilizzo o ticket di assistenza — e, in prospettiva di messa in produzione, la necessaria conformità al **GDPR** (trasparenza delle decisioni automatizzate e minimizzazione dei dati) e controlli di **fairness** per evitare discriminazioni su attributi sensibili come `SeniorCitizen` o `gender`.
> 4. **Sviluppi futuri:** le estensioni più promettenti sono l'introduzione di modelli **ensemble** come *Random Forest* e *XGBoost* per catturare interazioni non lineari e ridurre la varianza del singolo albero; il passaggio dal clustering discreto al **Soft Clustering**, fornendo le distanze continue dai centroidi tramite `kmeans.transform()`; e infine l'impiego di **SMOTE** (rigorosamente dentro i fold di validazione) e di tecniche di **Cost-Sensitive Learning** con una funzione di costo pesata direttamente in Euro sul CAC reale dell'operatore.
>
> Grazie per l'attenzione, resto a disposizione per qualsiasi domanda o approfondimento sul progetto e sul codice.»

---

## Promemoria Rapido dei Numeri Chiave da Tenere in Testa

- **Dataset:** `7.043` righe, `21` colonne (`19` feature utili + `customerID` rimosso + `Churn`), `11` righe `TotalCharges` vuote con `tenure = 0` $\rightarrow$ `0.0 €`.
- **Split 80/20 stratificato:** `5.634` Train (`4.139` rimasti, `1.495` churn) | `1.409` Test (`1.035` rimasti, `374` churn) | tasso churn **26,54%**.
- **Feature:** `3` numeriche + `27` one-hot (`drop='first'`) = **30 feature Base** (+ `2` one-hot cluster = **32 feature +Clust**).
- **K-Means ($K=3$):** Inerzia `33.485 -> 24.642`, Silhouette **`0,282`** (`0,2817`).
  - Cluster 0 (Volatili): `2.557` train, `15,2` mesi, `67,87 €`, **43,4%** churn (`43,3%` Test).
  - Cluster 1 (High-Value): `1.863` train, `57,7` mesi, `89,44 €`, **15,9%** churn (`13,3%` Test).
  - Cluster 2 (Low-Cost): `1.214` train, `30,2` mesi, `21,12 €`, **7,2%** churn (`8,0%` Test).
- **PCA 2D:** `PC1 = 39,4%` (Spesa/Servizi), `PC2 = 18,5%` (Anzianità/Contratti), Totale = **57,9%** (`57,87%`).
- **Prestazioni Test Set ($\tau = 0.50$):**
  - LogReg Base: Acc **80,6%**, Prec **65,9%**, Rec **55,9%**, F1 **0,605**, AUC **0,842** (`TP=209, FN=165, FP=108, TN=927`).
  - CART Base = +Clust: Acc **79,4%**, Prec **63,1%**, Rec **54,0%**, F1 **0,582**, AUC **0,827** (`TP=202, FN=172, FP=118, TN=917`).
  - MLP +Clust: Acc **80,3%** (da 80,0%), Prec **66,6%** (da 65,5%), Rec **52,1%** (da 51,9%), F1 **0,585** (da 0,579), AUC **0,843** (da 0,842).
- **Calibrazione Soglia LogReg:** $\tau=0.35$ $\rightarrow$ Rec **70,3%**, F1 **0,614** (`TP=263, FN=111`); `class_weight='balanced'` $\rightarrow$ Rec **78,3%**, F1 **0,614** (`TP=293, FN=81`).
- **Regola CART:** `tenure <= 16.5` $\rightarrow$ Fibra Ottica (`68,3%`, `593/868`) $\rightarrow$ `TotalCharges <= 120 €` (**85,5%**, `165/193`, $G=0.248$) $\rightarrow$ split successivo `TechSupport_Yes <= 0.5` (al livello 4: `87,6%`, `162/185`).
