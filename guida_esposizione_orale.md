# 🎙️ Guida per l'Esposizione Orale all'Esame di IA (Versione 10 Slide)

Questa guida ti fornisce **cosa dire voce per voce** per ciascuna delle 10 slide della presentazione, secondo le regole del corso d'esame (prof. G. Fiumara) per studente singolo.

---

## 📌 Slide 1: Titolo della Presentazione
> *"Buongiorno professore, il mio progetto riguarda la segmentazione comportamentale e la predizione dell'abbandono (churn) dei clienti nel settore Telco, basato sul dataset IBM Telco Customer Churn. L'analisi integra metodologie non supervisionate e supervisionate trattate durante il corso."*

---

## 📌 Slide 2: Inquadramento Teorico: Epistemologia e Agente PEAS
> *"Iniziamo dall'inquadramento teorico del problema basato su Russell & Norvig. L'Intelligenza Artificiale viene definita su due assi: Pensare vs. Agire e Umanamente vs. Razionalmente. Il mio lavoro si colloca saldamente nel quadrante dell'**Agire Razionalmente**: l'obiettivo non è simulare i processi cognitivi umani, ma massimizzare l'utilità attesa (la fidelizzazione) sulla base dei dati storici CRM. 
> A destra vediamo la modellizzazione PEAS dell'agente:
> - **Performance**: Massimizzare la Recall per ridurre l'abbandono aziendale ed evitare falsi negativi.
> - **Environment**: L'ambiente CRM Telco, caratterizzato dall'essere stocastico, parzialmente osservabile, sequenziale nel tempo e multi-dimensionale.
> - **Actuators**: Campagne promozionali mirate, sconti personalizzati e supporto.
> - **Sensors**: Registri di fatturazione, tenure e tipologia di contratti."*

---

## 📌 Slide 3: Fondamenti di Machine Learning e Approccio Probabilistico
> *"Ho formalizzato l'apprendimento automatico seguendo la celebre definizione di Tom Mitchell del 1997, basata sulla terna T, E, P:
> - **Task (T)**: La classificazione binaria finalizzata a predire il Churn.
> - **Experience (E)**: L'addestramento sui dati storici dei 7.043 clienti IBM Telco.
> - **Performance (P)**: Ottimizzazione di metriche robuste (Recall, AUC-ROC) al posto della semplice accuratezza.
> A destra giustifichiamo la necessità dell'approccio probabilistico (di Bayes e Laplace) rispetto alla logica formale rigida (di Boole e Frege). Il comportamento dei clienti è intrinsecamente non deterministico; un sistema a regole rigide fallirebbe per l'esplosione combinatoria delle variabili e la parzialità delle informazioni. La probabilità ci permette di modellare l'incertezza esprimendo gradi di credenza continui."*

---

## 📌 Slide 4: Preprocessing dei Dati e Sbilanciamento del Target
> *"Il dataset contiene 7.043 clienti e 21 variabili. Nel data cleaning ho individuato 11 record con valori vuoti in `TotalCharges` corrispondenti a clienti appena acquisiti con `tenure=0`, che ho imputato a `0.0`. Ho applicato uno `StandardScaler` sulle variabili continue e l'encode `One-Hot Encoding` con l'opzione `drop='first'` su quelle categoriali per prevenire la multicollinearità.
> Un aspetto critico è lo **sbilanciamento del target**: solo il **26.54%** dei clienti abbandona. Un classificatore banale (*dummy classifier*) otterrebbe il 73.46% di Accuracy ma una **Recall pari a zero**, risultando inutile. Inoltre, in ambito business, il costo di un Falso Negativo (cliente perso non identificato) è notevolmente superiore al costo di un Falso Positivo (promo offerta a cliente fedele). Pertanto, la Recall assume priorità assoluta."*

---

## 📌 Slide 5: Apprendimento Non Supervisionato: Clustering K-Means
> *"Per l'apprendimento non supervisionato ho utilizzato l'algoritmo **K-Means** per raggruppare i clienti in segmenti omogenei senza utilizzare le etichette. Per la scelta di $K$, ho combinato il Metodo del Gomito (Inerzia) e lo Silhouette Score. Anche se $K=2$ ha uno score geometrico leggermente superiore, ho scelto **$K=3$** perché $K=2$ creava solo una distinzione banale (chi ha internet e chi non ce l'ha). $K=3$ rappresenta il punto di gomito ottimale ed ha un significato commerciale concreto. Valori superiori di K portano invece a un netto calo della silhouette e a un over-clustering geometricamente confuso."*

---

## 📌 Slide 6: Segmentazione PCA e Profilo dei Segmenti
> *"In alto vediamo la profilazione dei 3 cluster K-Means ottenuti:
> - **Cluster 0 (Volatili)**: Ha una tenure media brevissima (13 mesi), spesa mensile alta (75€) e un tasso di churn del **47.08%** (quasi il doppio della media). Rappresenta il target prioritario per l'agente IA!
> - **Cluster 1 (Fedeli High-Value)**: Clienti storici ad alta spesa e basso abbandono (15.36%).
> - **Cluster 2 (Low-Cost)**: Utenti stabili con offerta economica base e churn minimo (12.30%).
> A destra in basso, la proiezione 2D sulle prime due componenti principali della **PCA** valida visivamente la netta separazione geometrica dei tre gruppi prima di passare alla classificazione."*

---

## 📌 Slide 7: Pipeline Sperimentale Supervisionata
> *"Questo slide mostra la pipeline di classificazione supervisionata. A sinistra vediamo il flowchart del processo: partendo dai dati preprocessati, applichiamo uno **split stratificato 80/20** su 7.043 clienti per mantenere l'esatto sbilanciamento del target nel training e nel test set.
> Addestriamo poi in parallelo tre modelli per confrontare approcci diversi: uno lineare probabilistico (la Regressione Logistica), uno White-Box trasparente (l'Albero CART) e uno Black-Box profondo (la Rete Neurale MLP)."*

---

## 📌 Slide 8: Confronto delle Prestazioni e Curve ROC
> *"La tabella a sinistra riassume le metriche dei tre modelli, sia sul dataset base che sul dataset arricchito con la feature di clustering. 
> La **Regressione Logistica** è il miglior modello per capacità predittiva generale (Accuracy **80.6%** e AUC-ROC **0.842**).
> Un risultato estremamente interessante riguarda la **Rete Neurale MLP**: l'integrazione del cluster fa compiere un balzo alla **Recall dal 42.8% al 49.2% (+6.4%)** e all'F1-Score al **51.8%**. Graficamente a destra, le curve ROC mostrano chiaramente come la curva rossa della rete MLP si sposti verso l'alto a destra, indicando che la feature di clustering agisce come una forma di *feature engineering non supervisionato* che aiuta l'ottimizzatore Adam ad evitare minimi locali."*

---

## 📌 Slide 9: Spiegabilità Business: Regole dell'Albero Decisionale (CART)
> *"L'albero decisionale ci offre il valore fondamentale dell'interpretabilità (*White-Box*), basandosi sull'indice di impurità di Gini riportato a sinistra. 
> Lo split di radice mostra che il primo fattore spartiacque è la permanenza breve (**Tenure $\le$ 16.5 mesi**). Sotto questo ramo, la combinazione critica che porta ai nodi blu ad altissimo churn (>70% abbandoni) è la presenza di un abbonamento in **Fibra Ottica**, isolando la clientela di recente attivazione a massimo rischio e permettendo all'Agente IA di intervenire con promozioni e supporto mirato."*

---

## 📌 Slide 10: Discussione Critica, Conclusioni & Sviluppi Futuri
> *"Per concludere la presentazione, ricapitoliamo 4 punti fondamentali:
> 1. **Perché il Clustering Aiuta**: L'integrazione del clustering ha fornito una mappa spaziale pre-calcolata che ha aiutato la Rete Neurale (MLP), aumentando la Recall del **+6.4%**.
> 2. **Impatto sul Business**: L'Agente CRM prioritizza la Recall perché il costo di acquisizione di un nuovo cliente (CAC) è molto più alto del costo di retention.
> 3. **Ruolo dei 3 Modelli**: La Regressione Logistica offre il miglior equilibrio globale; la Rete Neurale trae la massima spinta dal clustering; l'Albero CART offre la trasparenza (White-Box) necessaria per spiegare le regole.
> 4. **Sviluppi Futuri**: Per il futuro prevediamo il bilanciamento sintetico con **SMOTE**, modelli Ensemble come **XGBoost / Random Forest**, e il **Cost-Sensitive Learning** per pesare i modelli direttamente sui costi reali in Euro (€)."*
