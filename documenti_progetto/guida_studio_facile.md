# Guida Semplificata all'Esame
## Progetto: Segmentazione Clienti e Churn Prediction
*Questa guida è pensata per spiegare i concetti chiave dell'esame in modo visuale, logico e semplice, usando esempi pratici ed evitando formule a memoria.*

---

## 💡 Il Progetto in 3 Righe (Il "Pitch" iniziale)
Se il professore ti chiede: *"Di cosa tratta il suo progetto?"* rispondi così:
> "Ho analizzato i dati di un operatore telefonico per fare due cose:
> 1. Raggruppare i clienti in base a come spendono e a che contratti hanno (**Clustering con $k$-Means**).
> 2. Prevedere quali clienti stanno per abbandonare la compagnia (**Classificazione** con Regressione Logistica, Albero di Decisione e Rete Neurale), scoprendo che le informazioni dei gruppi (cluster) aiutano la rete neurale a fare predizioni migliori."

---

## 🛠️ Preprocessing (La preparazione dei dati)

### ❓ Cos'è l'imputazione con la Mediana?
* **Cos'è**: Quando mancano dei dati (buchi vuoti), li riempiamo con il valore centrale (la mediana) per non lasciare celle vuote che bloccherebbero i calcoli.
* **Perché la Mediana e non la Media?**
  * La mediana è **robusta contro i valori estremi (outlier)**.
  * *Esempio pratico*: Se 4 persone guadagnano 1.500€ e arriva Bill Gates che guadagna 10 milioni, lo stipendio *medio* della stanza sale a 2 milioni di euro (un dato falso per tutti gli altri!). Lo stipendio *mediano* rimane 1.500€ (molto più onesto e reale).
* **Nel progetto**: C'erano 11 record senza *TotalCharges* (nuovi clienti). Invece della mediana, abbiamo inserito `0.0` perché per logica un nuovo cliente (permanenza = 0 mesi) non ha ancora pagato nulla.

### ❓ Cos'è il One-Hot Encoding?
* **Cos'è**: I modelli matematici leggono solo numeri, non parole. Il One-Hot Encoding trasforma le parole (es. Tipo di Contratto: *Mensile*, *Annuale*, *Biennale*) in colonne di `0` e `1` (come delle bandierine).
* **Perché escludiamo la prima colonna (`drop_first=True`)?**
  * Per evitare la multicollinearità (informazione doppia). Se sappiamo che non è un contratto *Annuale* (0) e non è *Biennale* (0), per esclusione dev'essere *Mensile*. Tenere tutte e tre le colonne confonderebbe il modello.
* **Come verificare se è corretto?**
  * Controlliamo la tabella: le nuove colonne devono contenere solo `0` o `1`, e la somma delle colonne collegate per ogni riga deve essere al massimo 1.

### ❓ Cos'è lo Standard Scaling (Standardizzazione)?
* **Cos'è**: Riporta tutte le variabili numeriche ad avere media = 0 e deviazione standard = 1.
* **Esempio pratico**: Se confrontiamo l'altezza in metri (es. `1.8`) e il peso in grammi (es. `80000`), il modello penserà che il peso è immensamente più importante solo perché il numero è gigantesco. Lo scaling mette tutti sulla stessa linea di partenza.
* **Cos'è la deviazione standard?**
  * Misura quanto i dati sono "sparsi" o "vicini" rispetto al valore medio.

### ⚠️ Cos'è il Data Leakage (La trappola da evitare)?
* **Cos'è**: È il "trapelamento" di informazioni dal futuro (il Test Set) nel passato (il Training Set).
* **Perché lo split Train/Test si fa PRIMA dello scaling?**
  * Se calcolassimo la media sull'intero dataset prima di dividerlo, il Train Set conoscerebbe già informazioni che appartengono al Test Set (che dovrebbe essere invisibile). È come far fare ad uno studente un test dopo che ha già sbirciato le soluzioni: prenderà un voto alto, ma non avrà imparato davvero.

---

## 🤖 Gli Algoritmi (Spiegati in modo semplice)

### 1. Clustering ($k$-Means) - *Non Supervisionato*
* **Come funziona**: Raggruppa i dati simili tra loro. Mette dei punti nello spazio (i centroidi), calcola le distanze dei clienti e li associa al gruppo più vicino. Sposta i centroidi finché i gruppi non si muovono più.
* **Come abbiamo scelto $k=2$ (il numero di gruppi)?**
  * Usando due metriche: l'**Inertia** (più è bassa, più i gruppi sono compatti) e il **Silhouette Score** (più è vicino a 1, più i gruppi sono separati e non sovrapposti). $k=2$ era il valore migliore.
* **Chi sono i 2 Cluster?**
  * **Cluster 0**: Clienti fedeli, spendono poco ($\approx$ 21\$ al mese), contratti lunghi.
  * **Cluster 1**: Clienti a rischio, spendono molto ($\approx$ 76\$ al mese), contratti mensili.

### 2. Regressione Logistica - *Supervisionato*
* **Cos'è**: La nostra baseline (il modello di riferimento semplice).
* **Funzione Sigmoide**: Prende il risultato del calcolo e lo schiaccia in un intervallo tra `0` e `1`. Questo valore viene letto come una probabilità (es. "c'è l'80% di probabilità che questo cliente abbandoni").

### 3. Decision Tree (Albero di Decisione)
* **Cos'è**: Funziona come un gioco a quiz di "Sì o No" (es. *Il cliente ha un contratto mensile? Sì $\rightarrow$ Ha una spesa alta? No $\rightarrow$ Allora non abbandona*).
* **Perché `max_depth=5`?**
  * Se l'albero cresce troppo diventa troppo specifico e fa overfitting. Limitare la profondità lo costringe a trovare regole generali e semplici.

### 4. MLP (Rete Neurale)
* **Come si addestra?**
  * **Forward Propagation**: I dati entrano, passano per i nodi (neuroni) che applicano dei pesi, usano una funzione non lineare (**ReLU** nei layer intermedi) e producono un output.
  * **Backpropagation**: Il modello calcola l'errore commesso e torna indietro per aggiustare i pesi dei nodi per sbagliare meno la volta successiva (usando l'ottimizzatore Adam).
* **Perché la Sigmoide in output?**
  * Perché per decidere tra due classi (0 o 1) abbiamo bisogno che l'ultimo neurone sputi una probabilità compresa tra 0 e 1.

---

## 📈 Risultati ed Overfitting

### ❓ Che differenza c'è tra Overfitting e Underfitting?
* **Overfitting**: Il modello impara a memoria i dati di addestramento (Train). Ha ottime prestazioni sul Train, ma pessime sul Test (non sa generalizzare).
* **Underfitting**: Il modello è troppo semplice e non impara nulla (prestazioni scadenti sia su Train che su Test).

### ❓ Come abbiamo visto l'Overfitting nei risultati?
* Guardando la tabella delle metriche:
  * **Logistic Regression e Decision Tree**: Hanno prestazioni stabili sia su Train che su Test ($\approx$ 80% accuracy). **Nessun overfitting**.
  * **Rete MLP**: Ha un'accuratezza del **91% sul Train** ma scende al **74% sul Test**. Questo è un chiaro sintomo di **forte overfitting** (la rete è troppo complessa per questo dataset).

### ❓ Qual è stato l'impatto del Clustering sul modello MLP?
* Non ha risolto l'overfitting, ma ha aiutato moltissimo la **Recall** (il richiamo) sul Test Set, facendola passare **dal 42% al 58% (+15%)**.
* Significa che grazie alla colonna del cluster, la rete neurale è diventata molto più brava a trovare i clienti reali che stanno per abbandonare, che è l'obiettivo principale per l'azienda.
