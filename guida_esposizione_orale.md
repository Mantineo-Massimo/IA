# 🎙️ Guida per l'Esposizione Orale all'Esame di IA

Questa guida ti fornisce **cosa dire voce per voce** per ciascuna delle 10 slide della presentazione, secondo lo **stile illustrativo e discorsivo** del docente (Prof. G. Fiumara).

---

## 📌 Slide 1: Titolo della Presentazione
> *"Buongiorno professore, il mio progetto riguarda la segmentazione comportamentale e la predizione dell'abbandono (churn) dei clienti nel settore Telco, basato sul dataset IBM Telco Customer Churn. L'analisi integra metodologie non supervisionate e supervisionate trattate durante il corso."*

---

## 📌 Slide 2: Modellizzazione dell'Agente Intelligente (PEAS)
> *"Ho modellato l'azienda Telco proprio come un Agente Intelligente Razionale:*
> - **Performance Measure**: L'agente vuole ridurre il tasso di churn e fidelizzare i clienti ad alto valore, ottimizzando metriche robuste come F1-Score e AUC-ROC.
> - **Environment**: L'ambiente è la piattaforma CRM (*Customer Relationship Management*, ovvero il database gestionale aziendale contenente i contratti, le bollette e i servizi attivi dei clienti). Trattasi di un ambiente stocastico e parzialmente osservabile.
> - **Actuators**: Le azioni dell'agente consistono in promozioni mirate, sconti sul rinnovo e interventi prioritari di retention.
> - **Sensors**: I dati di fatturazione, il tipo di contratto e i mesi di permanenza (*tenure*)."*

---

## 📌 Slide 3: Il Dataset IBM Telco & Preprocessing
> *"Il dataset comprende 7.043 clienti con 21 variabili.*
> *Durante la fase di Data Cleaning ho riscontrato 11 valori vuoti nella spesa totale per i clienti con tenure pari a zero (nuovi abbonati), imputandoli correttamente a 0.0.*
> *Un aspetto fondamentale analizzato è lo **sbilanciamento del target**: solo il **26.54%** dei clienti abbandona. Un classificatore banale che predicesse sempre la classe maggioritaria avrebbe un'accuratezza del 73.4% ma una **Recall pari a zero**, risultando del tutto inutile per il business.*
> *Per il preprocessing ho applicato uno **StandardScaler** sui dati numerici e l'**One-Hot Encoding** su quelli categoriali per evitare la multicollinearità."*

---

## 📌 Slide 4: Clustering K-Means & Selezione di K
> *"Per l'apprendimento non supervisionato ho utilizzato l'algoritmo **K-Means** per raggruppare i clienti in segmenti omogenei senza etichette.*
> *Per determinare il numero di cluster $K$, ho analizzato due metriche:*
> 1. **Il Metodo del Gomito**: come si vede nel grafico, l'inerzia crolla rapidamente fino a $K=3$.
> 2. **Il Silhouette Score**: che misura la separazione tra i gruppi. $K=2$ ha uno score leggermente più alto ($0.295$) ma separa solo chi ha il servizio internet da chi non ce l'ha. **$K=3$ (score $0.286$) è la scelta ottimale** perché crea 3 gruppi commerciali con significato di business reale."*

---

## 📌 Slide 5: Visualizzazione 2D dei Cluster (PCA)
> *"Come mostrato nel grafico principale della slide, ho applicato la **Principal Component Analysis (PCA)** per ridurre lo spazio delle feature a due componenti principali ($PC_1$ e $PC_2$).*
> *La proiezione 2D conferma visivamente che i 3 cluster individuati da K-Means occupano regioni dello spazio ben definite e ben separate tra loro, validando la bontà della partizione prima di passare alla classificazione."*

---

## 📌 Slide 6: Profilo dei 3 Segmenti & Impatto Commerciale
> *"Dalla profilazione dei 3 cluster K-Means emergono tre tipologie di clienti chiare e azionabili per il business:*
> - **Cluster 0 (Nuovi / Volatili)**: Presenta un tasso di abbandono record del **47.08%** (quasi il doppio della media). Sono clienti di recente acquisizione (permanenza media 13 mesi) con spesa mensile alta (75€/mese). È il **target prioritario** per l'agente IA!
> - **Cluster 1 (Fedeli High-Value)**: Clienti storici (permanenza 58 mesi) con alta spesa (90€/mese) e basso abbandono (15.36%).
> - **Cluster 2 (Low-Cost)**: Utenti stabili con offerta economica base (26€/mese) e churn minimo (12.30%)."*

---

## 📌 Slide 7: Apprendimento Supervisionato: I Tre Classificatori
> *"Per la parte di classificazione supervisionata ho utilizzato uno **Stratified Train-Test Split** (80% Train, 20% Test) mantenendo esattamente la percentuale del 26.54% di churn.*
> *Ho messo a confronto tre modelli del corso:*
> 1. **Regressione Logistica**: baseline lineare probabilistica basata su funzione Sigmoide.
> 2. **Decision Tree (CART)**: modello ad albero regolarizzato a `max_depth=5` con l'Indice di Gini per prevenire l'overfitting.
> 3. **Multi-Layer Perceptron (MLP)**: rete neurale deep a 2 strati nascosti (64 e 32 neuroni) con attivazione ReLU, ottimizzatore Adam e regolarizzazione L2."*

---

## 📌 Slide 8: Confronto delle Prestazioni Predittive
> *"Professore, la tabella a tutta pagina sintetizza i risultati dei tre modelli a confronto, sia sul dataset base che sul dataset arricchito con la feature di clustering:*
> 1. **La Regressione Logistica** è il miglior modello in assoluto per capacità predittiva generale (Accuracy **80.6%** e AUC-ROC **0.842**).
> 2. **L'Impatto del Clustering**: mentre sui modelli lineari e ad albero l'informazione di cluster è ridondante, sulla **Rete Neurale MLP** l'inserimento del cluster fa balzare la **Recall dal 42.8% al 49.2% (+6.4%)** e l'F1-Score al **51.8% (+4.2%)**, permettendo alla rete di scoprire molti più clienti a rischio abbandono."*

---

## 📌 Slide 9: Spiegabilità Business: Regole del Decision Tree
> *"Il Decision Tree è un **modello White-Box** indispensabile per spiegare le regole di business.*
> *Come mostrato nell'albero visualizzato a destra:*
> - **Regola n. 1 (Split di Radice)**: la presenza di un contratto mensile senza vincoli (*Contract_Month-to-month*) costituisce la causa primaria di abbandono.
> - **Regole Secondarie**: nei contratti mensili, i fattori che scatenano il churn sono la mancanza di supporto tecnico (*TechSupport_No*) e i costi elevati della connessione in Fibra Ottica."*

---

## 📌 Slide 10: Discussione Critica, Conclusioni & Sviluppi Futuri
> *"Per concludere:*
> - **Feature Engineering Non Supervisionato**: la variabile di cluster ha agito come informazione sintetica che ha aiutato la rete neurale ad uscire da minimi locali durante l'addestramento Adam.
> - **Priorità di Business**: nel settore Telco la **Recall** è la metrica più critica, poiché perdere un cliente ha un costo di gran lunga superiore al costo di un'offerta di retention.
> - **Sviluppi Futuri**: testare tecniche di bilanciamento target come **SMOTE**, modelli Ensemble come **Random Forest** e **XGBoost**, o algoritmi di Cost-Sensitive Learning."*
