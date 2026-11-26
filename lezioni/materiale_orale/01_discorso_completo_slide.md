# Foglio 1 — Discorso Completo Semplificato (Slide 1 → Slide 10)

**Stile:** Italiano naturale da esposizione orale — **zero formule recitate a voce**, frasi brevi e scorrevoli, **100% tecnicamente corretto**.  
**Progetto:** Churn Prediction e Segmentazione Comportamentale (Dataset IBM Telco Customer Churn)  
**Studente:** Massimo Mantineo (Matricola: 541924)  
**Slide di riferimento:** [`relazione/presentazione.pdf`](../relazione/presentazione.pdf) (10 slide)

---

## Tabella di Marcia (Durata Totale: ~9 – 10 minuti)

- **Regola d'oro:** Le formule sono già scritte sulle slide — **non leggerle mai a voce**. Spiega a parole normali *cosa significano*: dimostri di averle capite davvero e parli in modo molto più sicuro.
- **Ritmo:** Fai una breve pausa (1–2 secondi) a ogni cambio slide e indica grafici e tabelle nei punti segnati con `[Indica ...]`.

| Slide | Titolo | Tempo | Cosa devi far capire al professore |
| :---: | :--- | :---: | :--- |
| **1** | Copertina | 0:25 | Presentazione chiara e i due obiettivi (raggruppare clienti + prevedere abbandoni). |
| **2** | IA & Agente PEAS | 1:05 | Agire Razionalmente (no onniscienza), scelta conveniente sotto incertezza, i 4 punti PEAS. |
| **3** | Task ML & Probabilità | 0:45 | Due compiti offline (supervisionato + K-Means/PCA), probabilità da 0 a 1 vs regole fisse, 2 *P*. |
| **4** | Dataset & Preprocessing | 1:00 | Pulizia 11 vuoti a 0 €, split 80/20 prima di tutto (zero leakage), sbilanciamento e boxplot. |
| **5** | Clustering K-Means | 1:00 | Perché K-Means (`.predict()`), curva blu (vicinanza al centro) + arancione (compattezza) → 3 gruppi. |
| **6** | PCA & Profili | 1:00 | I 3 segmenti (Volatili 43,4% confermato sul test) e cosa mostrano i 2 assi della PCA. |
| **7** | Pipeline Supervisionata | 0:50 | Base (30 var.) vs +Cluster (32 var.), LogReg, CART non scalato, MLP frenata contro overfitting. |
| **8** | Confronto Prestazioni | 1:15 | LogReg vince (Rasoio di Occam), +Cluster aiuta solo un po' l'MLP, soglia 35% / balanced fino al 78%. |
| **9** | Spiegabilità CART | 1:00 | Gini (0 puro, 0,5 misto), regola chiara: <16,5 mesi + Fibra + <120 € (85,5%, fino a 87,6% senza supporto). |
| **10** | Conclusioni & Sviluppi | 0:50 | Sintesi in 4 punti: ruolo clustering, CAC, limiti/GDPR, ensemble + soft clustering + SMOTE. |

---

## Slide 1 — Titolo e Copertina

> *[Guarda il professore, tono calmo e sicuro]*
>
> «Buongiorno professore, sono Massimo Mantineo e presento il mio progetto di Laboratorio di Intelligenza Artificiale: **"Churn Prediction e Segmentazione Comportamentale"**, svolto sul dataset pubblico **IBM Telco Customer Churn**.
>
> L'obiettivo del lavoro è unire le due tecniche principali viste a lezione: da una parte l'**apprendimento non supervisionato**, per dividere i clienti in gruppi simili in base al loro comportamento; dall'altra l'**apprendimento supervisionato**, per prevedere chi sta per abbandonare l'operatore — il cosiddetto *churn* — e verificare se conoscere il gruppo di appartenenza migliora davvero le previsioni.»

**Passaggio alla Slide 2:**
> «Prima di passare ai dati, inquadriamo brevemente il problema dal punto di vista teorico dell'Intelligenza Artificiale.»

---

## Slide 2 — Inquadramento Teorico: Definizione di IA e Agente PEAS

> *[Indica prima lo schema a 4 riquadri a sinistra, poi i 4 punti PEAS e il disegnino a destra]*
>
> «Come abbiamo visto con Russell e Norvig, ci sono quattro modi storici di definire l'Intelligenza Artificiale, combinando *pensiero* o *comportamento* con *modello umano* o *razionalità*. Noi scegliamo l'**Agire Razionalmente**, evidenziato in basso a destra.
>
> Questo significa che non vogliamo imitare il ragionamento di una persona, ma costruire un **agente con apprendimento basato su utilità**: un sistema che impara dai dati e sceglie l'azione più conveniente in condizioni di incertezza. Qui è importante distinguere **razionalità** da **onniscienza**: l'agente non può sapere con certezza il futuro di ogni cliente, ma sceglie l'azione migliore con le informazioni che ha, bilanciando il vantaggio di trattenere un cliente con il costo delle promozioni.
>
> *[Indica i 4 punti del PEAS sulla destra]*
>
> Descrivendolo con il modello **PEAS**:
> - **Performance (P):** è l'obiettivo aziendale esterno, cioè ridurre gli abbandoni tenendo i costi delle offerte ben al di sotto del **CAC**, il costo per acquisire un cliente nuovo;
> - **Environment (E):** è l'insieme dei clienti Telco. È un ambiente **parzialmente osservabile**, perché dai log non vediamo per esempio le offerte della concorrenza; è **non deterministico**, quindi lo gestiamo con le probabilità; è **multi-agente**, **continuo** su mesi e importi, e **sconosciuto** a priori finché non lo impariamo dai dati. Inoltre, nelle campagne reali è **dinamico e sequenziale** perché un'offerta oggi cambia le scelte future, mentre nella singola previsione sul nostro dataset è **statico ed episodico**;
> - **Actuators (A):** sono le azioni del CRM, cioè offerte mirate, sconti e supporto tecnico prioritario;
> - **Sensors (S):** sono i dati raccolti dal CRM su fatturazione, mesi di permanenza, contratti e servizi attivi.»

**Passaggio alla Slide 3:**
> «Vediamo ora come l'agente impara a collegare i dati dei sensori alle decisioni.»

---

## Slide 3 — Formulazione dei Task ML e Approccio Probabilistico

> *[Indica prima la colonna sinistra, poi quella destra e infine la riga in basso]*
>
> «In questa slide traduciamo l'agente in due compiti di Machine Learning, addestrati offline sui dati storici (quindi un sistema **Batch** e **Model-Based**):
>
> - Prima il compito **supervisionato**, formalizzato secondo Mitchell: usiamo **5.634 clienti** di cui conosciamo già l'esito per insegnare al modello a prevedere chi abbandonerà, e lo valutiamo con metriche adatte ai dati sbilanciati come **Recall, Precision, F1 e AUC-ROC**.
> - Poi il compito **non supervisionato**: usiamo **K-Means** per raggruppare i clienti simili senza mai guardare chi ha abbandonato, li visualizziamo in 2D con la **PCA**, e controlliamo se sapere il gruppo aiuta i classificatori.
>
> *[Indica la colonna di destra]*
>
> Sulla destra spiego perché usiamo il Machine Learning invece di regole scritte a mano: con 30 caratteristiche del cliente, delle regole fisse *if-then* diventano fragili, non si adattano ai cambi di mercato e danno solo un sì o un no secco. I nostri modelli invece stimano per ogni cliente una **probabilità di abbandono da 0 a 1**, così possiamo ordinarli dal più a rischio al meno a rischio e decidere da quale soglia in su intervenire rispetto al costo di acquisizione.
>
> *[Indica il riquadro in basso]*
>
> Chiudo con una precisazione: la *P* del PEAS guarda al risultato economico in azienda, mentre la *P* di Mitchell qui guarda solo a quanto prevede bene il modello.»

**Passaggio alla Slide 4:**
> «Passiamo ora ai dati veri e a come li abbiamo preparati.»

---

## Slide 4 — Dataset Telco: Preprocessing e Sbilanciamento del Target

> *[Indica prima i 4 punti a sinistra, poi i due boxplot a destra]*
>
> «Il dataset IBM Telco contiene **7.043 clienti** e **21 colonne**: togliamo subito il codice identificativo `customerID`, restano 19 caratteristiche utili (3 numeriche e 16 categoriali) più la colonna target `Churn`.
>
> Per preparare i dati in modo pulito abbiamo seguito quattro passaggi:
> 1. **Pulizia dati:** nella colonna della spesa totale (`TotalCharges`) c'erano **11 valori vuoti**. Controllandoli, abbiamo visto che erano tutti clienti appena arrivati, con **0 mesi di permanenza**, quindi non avevano ancora la prima bolletta: li abbiamo messi coerentemente a **0 euro**, senza buttare dati né falsarli con la media.
> 2. **Zero Data Leakage:** per non contaminare la valutazione, abbiamo diviso subito i dati in **80% training e 20% test** — 5.634 clienti per addestrare e 1.409 per testare, con la stessa quota di abbandoni in entrambi — **prima** di fare qualsiasi calcolo di scaling o clustering.
> 3. **Standardizzazione e codifica:** solo sul training portiamo le 3 variabili numeriche sulla stessa scala con la standardizzazione (così gli importi in migliaia di euro non sovrastano i mesi di permanenza) e trasformiamo le categorie in colonne binarie con One-Hot Encoding (`drop='first'`), arrivando a **30 variabili**.
> 4. **Sbilanciamento:** solo il **26,54%** dei clienti abbandona. Questo rende l'Accuracy da sola ingannevole: un modello pigro che risponde sempre "resta" avrebbe comunque il **73,46% di Accuracy**, ma **Recall allo 0%**, cioè non salverebbe nessun cliente.
>
> *[Indica i due boxplot a destra]*
>
> Guardando i grafici a destra — verde chi resta, rosso chi abbandona, linea nera la mediana — il segnale è chiarissimo: chi abbandona ha una permanenza mediana di soli **10 mesi contro 38**, e una spesa mensile più alta, quasi **80 euro contro 64**. Quindi il rischio si concentra sui **clienti nuovi che pagano tanto**.»

**Passaggio alla Slide 5:**
> «Sulle 30 variabili del solo training set, passiamo ora alla segmentazione con K-Means.»

---

## Slide 5 — Apprendimento Non Supervisionato: Clustering K-Means

> *[Indica i punti a sinistra e poi le due curve blu e arancione nel grafico a destra]*
>
> «Qui vogliamo dividere i **5.634 clienti del training set** in gruppi omogenei **senza mai usare l'etichetta Churn**, basandoci solo sul loro profilo.
>
> Abbiamo scelto **K-Means** invece di DBSCAN per tre motivi pratici: è veloce, produce profili medi facili da capire per il marketing, e soprattutto ha il metodo **`.predict()`** che ci permette di assegnare i clienti nuovi del test set ai gruppi già trovati **senza ri-addestrare e senza data leakage** — cosa che DBSCAN non fa nativamente.
>
> *[Indica il grafico a destra]*
>
> Per scegliere quanti gruppi creare, abbiamo provato da 2 a 6 cluster guardando due curve insieme:
> - la **linea blu** (Inerzia) misura quanto i clienti sono vicini al centro del proprio gruppo: scende sempre aggiungendo gruppi, quindi cerchiamo il punto in cui piega, il cosiddetto "gomito";
> - la **linea arancione** (Silhouette) misura se i gruppi sono compatti al loro interno e ben separati tra loro.
>
> Vediamo il confronto: con **2 gruppi** la divisione è banale, separa solo chi ha internet da chi non ce l'ha; con **3 gruppi** abbiamo il gomito netto della linea blu e la Silhouette resta praticamente invariata a **0,282** — un valore moderato ma del tutto normale quando si lavora con 30 variabili miste; da **4 gruppi in su** la Silhouette crolla nettamente perché i gruppi iniziano a frammentarsi. Per questo scegliamo **K = 3**.»

**Passaggio alla Slide 6:**
> «Vediamo subito che caratteristiche hanno questi tre gruppi e come appaiono nel grafico.»

---

## Slide 6 — Segmentazione PCA e Profilo dei Segmenti

> *[Indica prima le tre righe della tabella in alto, poi le tre nuvole colorate nel grafico a destra]*
>
> «Solo dopo aver costruito i 3 gruppi sul training set abbiamo guardato il loro tasso di abbandono per capire chi c'è dentro, ottenendo tre profili aziendali molto netti:
> - **Cluster 0 — i Volatili:** 2.557 clienti con anzianità breve (circa **15 mesi**), spesa medio-alta (**68 euro al mese**) e un tasso di abbandono critico del **43,4%**. Aspetto molto importante: applicando il modello al test set, questo segmento mantiene quasi lo stesso abbandono (**43,3%**), segno che i gruppi sono stabili;
> - **Cluster 1 — i Fedeli High-Value:** 1.863 clienti storici (quasi **58 mesi**), spesa alta (**89 euro al mese**), ma churn contenuto al **15,9%**;
> - **Cluster 2 — i Low-Cost:** 1.214 clienti con spesa molto bassa (**21 euro al mese**, servizi base senza internet) che quasi non abbandonano (**7,2%**).
>
> *[Indica il grafico PCA in basso a destra]*
>
> Per vedere questi gruppi in due dimensioni abbiamo usato la **PCA**, che riassume quasi il **58%** dell'informazione delle 30 variabili originali: sull'asse orizzontale ($PC_1$, 39,4%) si distribuiscono **spesa e servizi** — infatti i Low-Cost verdi sono isolati a sinistra — mentre sull'asse verticale ($PC_2$, 18,5%) conta l'**anzianità del contratto**, che separa in basso i Volatili blu dai Fedeli arancioni in alto a destra. Preciso che la PCA l'abbiamo usata solo per visualizzare i gruppi, non per addestrare i modelli.»

**Passaggio alla Slide 7:**
> «Adesso vediamo come abbiamo impostato i modelli supervisionati per prevedere il Churn.»

---

## Slide 7 — Pipeline Sperimentale Supervisionata

> *[Indica prima lo schema a sinistra dall'alto in basso, poi i 3 modelli a destra]*
>
> «Lo schema a sinistra riassume la nostra pipeline senza *data leakage*: prima dividiamo 80% training e 20% test, poi solo sul training facciamo preprocessing e K-Means, ottenendo due versioni parallele dei dati da confrontare: la versione **Base** con 30 variabili e la versione **+Cluster** con 32 variabili (aggiungendo il gruppo di appartenenza).
>
> *[Indica la colonna di destra]*
>
> Su entrambe le versioni confrontiamo tre modelli di complessità crescente:
> 1. **Regressione Logistica:** il nostro modello lineare di riferimento, semplice e veloce, utile per capire quanto il problema sia già separabile in modo lineare.
> 2. **Albero Decisionale CART** con profondità massima 5: il nostro modello trasparente (*White-Box*). L'abbiamo addestrato apposta sui dati numerici **non scalati**, perché gli alberi non risentono della scala dei numeri e in questo modo possiamo leggere le regole direttamente in mesi ed euro.
> 3. **Rete Neurale MLP** con due strati nascosti da **64 e 32 neuroni**, attivazione ReLU, uscita sigmoidea e ottimizzatore Adam. Per evitare l'overfitting tipico delle reti su dati tabellari, come visto nelle Lezioni 7 e 8, abbiamo aggiunto due freni: la **regolarizzazione L2** e l'**early stopping**, che ferma l'addestramento appena le prestazioni su una quota interna di validazione smettono di migliorare.»

**Passaggio alla Slide 8:**
> «Vediamo ora i numeri ottenuti sul test set.»

---

## Slide 8 — Confronto delle Prestazioni Sperimentali

> *[Indica il grafico a barre a sinistra, poi la tabella e i due riquadri a destra]*
>
> «Qui confrontiamo i tre modelli sui **1.409 clienti del test set** (di cui 374 in abbandono): le barre blu sono il modello Base, quelle rosse il modello con il Cluster, usando inizialmente la soglia standard del 50%. Ricordo che l'**AUC-ROC** misura quanto il modello è bravo a mettere un cliente a rischio sopra un cliente fedele.
>
> I risultati ci dicono tre cose molto interessanti:
> - La **Regressione Logistica** base è la migliore su Accuracy (**80,6%**), Recall (**55,9%**) e F1 (**0,605**), con AUC **0,842**. Aggiungendo il cluster cala leggermente (Recall 55,1%): è normale, perché i cluster sono ricavati dalle stesse variabili che la Logistica usa già, quindi creano una leggera ridondanza.
> - L'**Albero CART** resta **identico** con o senza cluster (Recall 54%, AUC 0,827), perché i tagli diretti su mesi e Fibra Ottica sono già più informativi dell'etichetta di gruppo.
> - La **Rete Neurale MLP**, grazie a regolarizzazione L2 ed *early stopping* — che la ferma intorno alla 17ª–19ª epoca evitando l'overfitting — raggiunge la Logistica ed è l'**unico modello che migliora leggermente con il cluster** su tutte le metriche, arrivando alla migliore AUC (**0,843**).
>
> *[Indica i due box in basso a destra]*
>
> Da qui traiamo due conclusioni pratiche:
> 1. **Rasoio di Occam:** visto che Regressione Logistica e Rete Neurale hanno praticamente la stessa capacità di distinguere i clienti (`0,842` contro `0,843`), conviene scegliere la **Regressione Logistica**, che è molto più semplice e facile da gestire; il valore principale del clustering resta quindi quello di descrivere i segmenti.
> 2. **Calibrazione della soglia:** con la soglia al 50% la Logistica intercetta solo il 55,9% dei clienti in fuga. Siccome perdere un cliente costa caro (CAC), abbiamo provato a spostare la soglia al **35%** — verificandola in cross-validation solo sul training — oppure a bilanciare il peso delle classi (`class_weight='balanced'`): così la Recall sale al **70,3%** e fino al **78,3%**, migliorando persino l'F1 fino a **0,614**.»

**Passaggio alla Slide 9:**
> «Ora vediamo invece il punto di forza dell'albero CART: la sua spiegabilità immediata per l'azienda.»

---

## Slide 9 — Spiegabilità Business: Regole dell'Albero Decisionale (CART)

> *[Indica prima i box 1 e 2 in basso, poi segui col dito il percorso dalla cima fino al riquadro blu scuro]*
>
> «In questa slide vediamo i primi tre livelli dell'albero decisionale addestrato sui **5.634 clienti del training non scalato**.
>
> Per leggerlo bastano due regole: l'**indice di Gini** misura quanto un gruppo è misto — **0** vuol dire puro al 100% con colore intenso, **0,5** vuol dire metà e metà con colore chiaro; i riquadri **arancioni** indicano prevalenza di clienti rimasti, quelli **blu** prevalenza di abbandoni.
>
> Seguendo il percorso dalla cima leggiamo subito una regola chiarissima per il business:
> - il primo taglio è **permanenza fino a 16 mesi e mezzo**: chi è cliente da più tempo (a destra) tende quasi sempre a restare;
> - tra i clienti più recenti (a sinistra), chi ha la **Fibra Ottica** finisce subito in un nodo blu con il **68,3% di abbandoni** (593 su 868);
> - scendendo ancora su chi ha speso finora **meno di 120 euro in totale** — cioè i neo-abbonati in Fibra nei primissimi uno o due mesi — arriviamo al nodo blu scuro con l'**85,5% di abbandoni** (165 su 193); e applicando il taglio successivo visibile su quel nodo, se manca anche il **supporto tecnico**, si sale addirittura all'**87,6%**.
>
> Per l'azienda l'indicazione è immediata: offrire accoglienza e supporto tecnico prioritario ai nuovi clienti in Fibra fin dalle prime settimane.»

**Passaggio alla Slide 10:**
> «Concludo riassumendo i risultati principali, i limiti del lavoro e i possibili sviluppi futuri.»

---

## Slide 10 — Discussione Critica, Conclusioni & Sviluppi Futuri

> *[Indica il banner in alto e scorri i 4 riquadri]*
>
> «Per riassumere, nel nostro sistema ogni strumento ha un compito chiaro: **K-Means** per capire i segmenti di clienti, **Regressione Logistica** per ordinare chi è più a rischio, **Albero CART** per spiegare le cause al management e **calibrazione della soglia** per proteggere il margine aziendale.
>
> I quattro punti finali sono:
> 1. **Ruolo del clustering:** è utilissimo soprattutto a livello strategico, perché isola il segmento dei *Volatili* con il 43,4% di churn; come variabile extra nei modelli porta invece solo un piccolo miglioramento alla rete neurale.
> 2. **Soglia e costi (CAC):** siccome perdere un cliente costa molto più di un'offerta promozionale, tarare la soglia permette di salire dal 55,9% fino al **78,3% di Recall**.
> 3. **Scelta del modello e limiti:** la Regressione Logistica vince per semplicità (Rasoio di Occam). Come limiti dobbiamo tenere conto che K-Means assume gruppi di forma sferica, il dataset è una fotografia statica senza lo storico dei reclami e, in produzione reale, servono conformità al **GDPR** e controlli di **fairness** per evitare discriminazioni su età o genere.
> 4. **Sviluppi futuri:** si potrebbero provare modelli *ensemble* come **Random Forest e XGBoost**, passare ai modelli le distanze continue dai centroidi invece del semplice numero di cluster (**Soft Clustering** con `transform()`), e introdurre **SMOTE** e funzioni di costo espresse direttamente in euro sul CAC reale.
>
> Grazie per l'attenzione, sono a disposizione per eventuali domande.»

---

## Promemoria Lampo dei Numeri (solo quelli essenziali!)

- **Dataset:** `7.043` clienti (`26,5%` abbandoni), `11` vuoti in `TotalCharges` messi a `0 €`, split 80/20 (`5.634` train / `1.409` test), **30 variabili** (32 con cluster).
- **Boxplot:** chi abbandona ha mediana **10 mesi** (vs 38) e **~80 €** al mese (vs 64 €).
- **K-Means ($K=3$, Silhouette `0,282`):** Volatili **43,4%** (Test 43,3%), High-Value **15,9%**, Low-Cost **7,2%**; PCA spiega il **57,9%**.
- **Modelli (Test Set):** LogReg Base (**80,6%** Acc, **55,9%** Rec, **0,605** F1, **0,842** AUC) | CART (**79,4%** Acc, **54,0%** Rec, **0,827** AUC) | MLP +Cluster (**80,3%** Acc, **52,1%** Rec, **0,585** F1, **0,843** AUC).
- **Soglia LogReg:** al 35% → **70,3%** Recall (`0,614` F1); `balanced` → **78,3%** Recall (`0,614` F1).
- **Albero CART:** `<= 16,5` mesi → Fibra (**68,3%**) → `<= 120 €` (**85,5%**, 165/193) → senza `TechSupport` (**87,6%**, 162/185).
