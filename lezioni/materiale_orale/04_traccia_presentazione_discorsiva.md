# Foglio 4 — Esposizione Orale Completa con Ogni Concetto e «Come si fa»

**Slide abbinate:** [`relazione/presentazione_discorsiva.pdf`](../relazione/presentazione_discorsiva.pdf) (10 slide, zero codice sullo schermo, 100% allineate alle dispense del corso)  
**Come usarla:** Non hai limiti di tempo: esponi con calma seguendo le 10 slide. Ogni termine che nomini ha già dentro sia **cosa significa** sia **come si calcola o funziona**, senza termini fuori dispensa (`SMOTE`, `GDPR`, nome `Rasoio di Occam` rimossi).

---

## Slide 1 — Copertina
> «Buongiorno professore, sono Massimo Mantineo e presento il mio progetto di Laboratorio di Intelligenza Artificiale sulla **previsione dell'abbandono dei clienti** — il cosiddetto churn — nel settore delle telecomunicazioni, sul dataset pubblico IBM Telco da 7.043 clienti.
>
> Come indicato in copertina, unisco le due tecniche principali viste a lezione: l'**apprendimento non supervisionato**, cioè dividere i clienti in gruppi simili senza sapere chi abbandona, e l'**apprendimento supervisionato**, cioè usare esempi passati con esito noto per prevedere chi sta per andarsene e verificare se conoscere il gruppo aiuta le previsioni.»

**Passaggio → Slide 2:** «Prima di guardare i dati, vediamo perché nasce questo problema e come lo inquadriamo come agente intelligente.»

---

## Slide 2 — Il Problema e l'Agente Razionale (PEAS)
> «Nel settore telefonico acquisire un cliente nuovo costa molto più che trattenere un cliente attuale. Tra i **quattro approcci di Russell e Norvig** — cioè pensare come umani, agire come umani (Test di Turing), pensare razionalmente con la logica e **agire razionalmente** — scegliamo proprio quest'ultimo: invece di imitare la mente umana, progettiamo un **agente razionale** che sceglie l'azione più conveniente in condizioni di incertezza. È fondamentale distinguere **razionalità** da **onniscienza**: l'agente non prevede con certezza il futuro, ma impara dai dati storici e bilancia i clienti salvati con il costo delle offerte.
>
> Lo descriviamo con il modello **PEAS**:
> - **Obiettivo (P — Performance):** ridurre gli abbandoni spendendo meno del **CAC**, cioè il costo per acquisire un cliente nuovo.
> - **Ambiente (E — Environment):** clientela Telco, **parzialmente osservabile** (non vediamo la concorrenza), **incerto** (lavoriamo con probabilità) e che **cambia nel tempo** nelle campagne reali, mentre nella singola previsione resta fisso.
> - **Azioni (A — Actuators):** offerte mirate, sconti e supporto tecnico prioritario.
> - **Dati in ingresso (S — Sensors):** i dati del gestionale clienti (**CRM**) su durata del contratto, bollette e servizi attivi.»

**Passaggio → Slide 3:** «Adesso vediamo come fa concretamente l'agente a imparare dai dati storici.»

---

## Slide 3 — Come Impara l'Agente: Due Compiti di Machine Learning
> «L'agente impara **offline** — cioè viene addestrato una volta sull'intero blocco di dati storici, non cliente per cliente in tempo reale — attraverso due compiti di Machine Learning:
> - Il **compito supervisionato** segue la definizione di **Mitchell (T, E, P)**: un algoritmo impara dall'**Esperienza (E)** dei **5.634 clienti passati** con esito noto se migliora sul **Compito (T)** di prevedere chi abbandonerà rispetto alla misura di **Prestazione (P)**, valutata con metriche per dati sbilanciati (**Recall, Precision, F1 e AUC-ROC**).
> - Il **compito non supervisionato** raggruppa i clienti simili con **K-Means** senza guardare gli abbandoni, li visualizza in 2D con la **PCA** e verifica se il gruppo aiuta le previsioni.
>
> Perché il Machine Learning e non regole scritte a mano? Con **30 caratteristiche** le regole fisse sono fragili, non si adattano al mercato e rispondono solo **sì o no**. I nostri modelli invece stimano una **probabilità da 0 a 1**, così ordiniamo i clienti per rischio e scegliamo da quale soglia intervenire. Infine, nel **PEAS** misuriamo il guadagno economico reale in azienda, mentre nel **Machine Learning (Mitchell)** quanto prevede bene il modello.»

**Passaggio → Slide 4:** «Passiamo ora ai dati reali del dataset, a tutto il preprocessing svolto e ai primi segnali che emergono dal Boxplot.»

---

## Slide 4 — I Dati, il Preprocessing e il Boxplot
> «Lavoriamo sul dataset IBM Telco, con **7.043 clienti** e il **26,5%** di abbandoni. Prima dei modelli svolgiamo cinque passaggi:
> 1. **Pulizia:** togliamo il codice identificativo del cliente (non aiuta a generalizzare) e mettiamo a **0 euro** 11 spese totali vuote perché erano clienti appena arrivati a 0 mesi.
> 2. **Split stratificato 80/20 (Zero Data Leakage):** dividiamo **prima** di qualsiasi calcolo in **80% training** (5.634 clienti per far studiare i modelli) e **20% test** (1.409 clienti tenuti nascosti per l'esame finale); è **stratificato** perché mantiene esattamente lo stesso 26,5% di abbandoni in entrambi, ed è fatto subito per evitare il **data leakage**, cioè non far filtrare informazioni del test nel training.
> 3. **Standardizzazione:** le 3 variabili numeriche hanno unità diversissime — i mesi arrivano a 72, la spesa totale supera 8.000 euro. Siccome il computer non distingue mesi ed euro, un numero grande schiaccerebbe uno piccolo nei calcoli di distanza o peso. **Come si calcola?** A ogni valore **togliamo la media e dividiamo per la deviazione standard** (che misura quanto i dati sono sparsi attorno alla media), calcolate **solo sul training**. Così portiamo tutto sullo stesso metro: **0** = nella media, **positivo** = sopra la media, **negativo** = sotto la media.
> 4. **One-Hot Encoding:** i modelli leggono solo numeri, non parole; trasformiamo le **16 variabili testuali** in **27 colonne 0 e 1**. Per esempio, se il tipo di internet ha 3 casi (Fibra, DSL, Nessuno), bastano 2 colonne 0/1 perché se sono entrambe 0 significa automaticamente il terzo caso: togliere una categoria ridondante per variabile evita problemi nei modelli lineari (3 + 27 = **30 variabili finali**).
> 5. **Sbilanciamento:** con il 26,5% di abbandoni, dire sempre "resta" darebbe 73,5% di accuratezza ma **Recall 0%**.
>
> A destra usiamo il **Boxplot** per confrontare Rimasti (verde) e Abbandoni (rosso) senza farsi falsare dagli outlier: la **linea nera** è la **mediana** — si calcola ordinando tutti i clienti dal valore più piccolo al più grande e prendendo quello esattamente nel mezzo (o la media dei due centrali se sono pari); la **scatola** va dal 25° al 75° percentile (i quartili Q1 e Q3, cioè il 50% centrale dei clienti); i **baffi** arrivano fino a 1,5 volte la lunghezza della scatola e i **pallini** oltre i baffi sono i casi anomali (outlier). Chi abbandona resta poco (**10 mesi** vs 38) e paga di più (**80 euro** vs 64).»

**Passaggio → Slide 5:** «Una volta preparati i dati in modo pulito, partiamo dalla prima fase: dividere i clienti di training in gruppi simili con K-Means.»

---

## Slide 5 — Dividere i Clienti in Gruppi con K-Means
> «Sul solo training di 5.634 clienti eseguiamo il **clustering**, cioè raggruppiamo i clienti simili **senza mai guardare gli abbandoni**.
>
> **Come funziona K-Means?** Sceglie `K` centri iniziali, assegna ogni cliente al centro più vicino in linea retta, poi ricalcola ogni centro come la **media** dei suoi clienti e ripete questi passi finché i centri non si fermano (ripetendo da 10 partenze per trovare la soluzione migliore). Scegliamo **K-Means** invece di **DBSCAN** per due motivi: crea **profili medi** chiari per il marketing, e impara i centri solo sul training per poi assegnare ogni **cliente nuovo del test** al centro più vicino senza ricalcolare tutto da capo (cosa che DBSCAN non permette nativamente).
>
> Nel grafico a destra leggiamo due curve:
> - la **linea blu (Inerzia)** si calcola misurando la distanza di ogni cliente dal centro del suo gruppo, elevandola al **quadrato** e sommando tutto — scende sempre, quindi cerchiamo il **gomito**;
> - la **linea arancione (Silhouette)** varia da **-1 a +1** e per ogni cliente confronta quanto è vicino ai compagni del proprio gruppo rispetto al gruppo vicino più prossimo (differenza tra le due distanze divisa per il loro massimo).
>
> Confrontando i riquadri: **K=2** è troppo semplice (separa solo chi ha internet da chi no); **K=3 (scelto)** è il gomito netto con buona compattezza (Silhouette **0,282**, normale con 30 variabili miste); da **K=4** in su i gruppi si spezzettano troppo e la Silhouette crolla.»

**Passaggio → Slide 6:** «Scelti i 3 gruppi, vediamo chi c'è dentro ciascun gruppo e come li visualizziamo nella mappa 2D.»

---

## Slide 6 — I Tre Tipi di Clienti Scoperti e la Mappa PCA
> «Solo dopo aver formato i 3 gruppi abbiamo guardato il loro tasso di abbandono. I colori delle schede a sinistra corrispondono ai puntini del grafico:
> - **Gruppo 0 (Blu — Volatili):** 2.557 clienti recenti (**15 mesi**), spesa media (**68 euro**) e **43,4%** di abbandoni (confermato al **43,3%** sul test, segno che i gruppi sono stabili);
> - **Gruppo 1 (Arancione — Fedeli High-Value):** 1.863 clienti storici (**58 mesi**), spesa alta (**89 euro**) e solo **15,9%** di abbandoni (13,3% sul test);
> - **Gruppo 2 (Verde — Low-Cost):** 1.214 clienti con servizi base senza internet (**21 euro**) e abbandoni minimi (**7,2%**, 8,0% sul test).
>
> A destra usiamo la **PCA** (Analisi delle Componenti Principali) **solo per disegnare il grafico**: siccome non possiamo disegnare 30 variabili su uno schermo piatto, la PCA le riassume in **2 assi**. **Come sceglie questi 2 assi?** Cerca prima la direzione nello spazio lungo cui i clienti **variano di più** (massima varianza, **PC1 = 39%**), legata a spesa e servizi (i verdi Low-Cost a sinistra), e poi la seconda direzione con più variabilità rimasta **perpendicolare** alla prima (**PC2 = 19%**), legata all'anzianità (i blu Volatili in basso, gli arancioni Fedeli in alto a destra). In totale tiene il **58%** dell'informazione, mentre i modelli supervisionati usano tutte le 30 variabili.»

**Passaggio → Slide 7:** «Capiti i tre profili di clienti, passiamo alla seconda fase: i tre modelli supervisionati per prevedere chi sta per abbandonare.»

---

## Slide 7 — Come Prevediamo l'Abbandono: I Tre Modelli
> «Dopo lo split 80/20, senza data leakage, confrontiamo **3 modelli** sia sui dati **Base (30 variabili)** sia aggiungendo 2 colonne 0/1 per il **Gruppo K-Means (32 variabili)**:
> 1. **Regressione Logistica:** ragiona come una pagella a punti — dà un **peso** a ogni dato del cliente (es. Fibra alza il rischio, tanti mesi lo abbassano), li somma e passa il totale dentro una curva a "S" chiamata **sigmoide** che schiaccia qualunque punteggio in una **probabilità da 0% a 100%**. Semplice, veloce e stabile: è il nostro punto di riferimento.
> 2. **Albero CART:** funziona come una **catena di domande sì/no**. **Come sceglie quale domanda fare per prima?** A ogni nodo prova tutte le domande possibili e sceglie quella che **abbassa di più l'indice di Gini**, creando due sottogruppi più puri. Limitiamo la profondità a **5 livelli** così non impara a memoria i singoli clienti, e lo addestriamo **senza standardizzazione** perché guarda una caratteristica alla volta: così leggiamo subito **mesi ed euro veri**.
> 3. **Rete Neurale MLP:** lavora su **due livelli collegati**: un primo strato di **64 neuroni** incrocia più indizi insieme e un secondo strato di **32 neuroni** li raffina fino alla probabilità finale, cogliendo legami complessi. Usa **ReLU** (un interruttore che lascia passare solo i segnali positivi utili, evitando che la rete si blocchi) e **Adam** (che a ogni errore aggiusta i pesi in modo rapido e intelligente). Contro l'**overfitting** — cioè imparare a memoria il training andando male sul test — usiamo la **penalità L2** (tiene i pesi piccoli ed equilibrati) e l'**early stopping** (ferma lo studio appena smette di migliorare su una piccola verifica interna).
>
> Lo **scopo** è capire chi prevede meglio sui clienti nuovi e se il gruppo K-Means porta un vero vantaggio.»

**Passaggio → Slide 8:** «Vediamo adesso i risultati ottenuti mettendo questi tre modelli alla prova sui 1.409 clienti di test.»

---

## Slide 8 — I Risultati sul Test Set e la Soglia di Allarme
> «A sinistra vediamo i risultati sui **1.409 clienti di test** (blu = Base, rosso = con Gruppo K-Means). Tutte le metriche nascono dai 4 conti della **Matrice di Confusione**: abbandoni beccati (`TP`), abbandoni persi (`FN`), falsi allarmi (`FP`) e rimasti riconosciuti (`TN`). **Come si calcolano?**
> - **Recall** (Grafico 1) = abbandoni beccati diviso tutti gli abbandoni veri;
> - **Precision** = abbandoni beccati diviso tutti i clienti segnalati;
> - **F1** = media armonica tra Precision e Recall (`2 × Precision × Recall / (Precision + Recall)`: resta alta solo se entrambe sono buone);
> - **Accuracy** = risposte giuste diviso tutti i clienti;
> - **AUC-ROC** (Grafico 2) si ottiene facendo scorrere la soglia da 0 a 1, disegnando Recall contro falsi allarmi e prendendo l'**area sotto la curva** (da 0,5 = caso a 1 = perfetto).
>
> A destra i 3 risultati:
> 1. **Vince la Regressione Logistica:** miglior Recall (**55,9%**), Accuracy **80,6%**, F1 **0,605** e AUC **0,842**, uguale alla Rete Neurale (**0,843**). **A parità di risultato scegliamo il modello più semplice**: ha pochissimi parametri, è più stabile e rischia meno overfitting.
> 2. **Il Gruppo K-Means (barre rosse)** aiuta di pochissimo solo la Rete Neurale, cala un filo sulla Logistica perché ripete informazioni già presenti, e lascia identico l'Albero.
> 3. **Come intercettare più clienti in fuga:** con soglia 50% ne troviamo poco più di metà. Abbassiamo l'allarme al **35%**, verificandolo prima in **5-fold cross-validation sul solo training** — cioè dividiamo il training in 5 fette uguali, a turno 4 studiano e 1 verifica, ripetendo 5 volte senza mai toccare il test — oppure usiamo **pesi bilanciati**, dicendo al modello che sbagliare un abbandono pesa circa **3 volte di più** che sbagliare un cliente che resta. Così la Logistica sale al **70%–78% di Recall** (F1 **0,614**), proteggendo dal CAC.»

**Passaggio → Slide 9:** «Se la Logistica vince sulle previsioni, l'Albero Decisionale è fondamentale perché ci spiega chiaramente il perché i clienti se ne vanno: guardiamolo nella prossima slide.»

---

## Slide 9 — Cosa Ci Insegna l'Albero Decisionale (Regole Chiare)
> «L'albero ci mostra regole chiare per l'azienda sui 5.634 clienti di training non scalati. I riquadri **arancioni** indicano prevalenza di rimasti, quelli **blu** prevalenza di abbandoni. L'**indice di Gini** misura quanto un gruppo è misto e si calcola così: si prende la proporzione di rimasti e quella di abbandoni nel riquadro, si elevano entrambe al quadrato, si sommano e si sottrae il risultato da 1. Con **2 classi** varia quindi da **0** (puro al 100%, perché 1 meno 1 fa 0, colore **scuro**) a **0,5** (metà e metà, perché 1 meno 0,25 meno 0,25 fa 0,5, colore **chiaro**).
>
> Seguendo passo passo le frecce blu:
> 1. cliente recente (**meno di 16,5 mesi**) + **Fibra Ottica** → abbandona già il **68,3%** (593 su 868);
> 2. in più ha speso **meno di 120 euro** (primi 1–2 mesi) → riquadro blu scuro con l'**85,5%** di abbandoni (165 su 193);
> 3. al passo successivo, senza **supporto tecnico**, si sale all'**87,6%**.
>
> L'**azione aziendale** è chiara: offrire supporto tecnico gratuito ai nuovi clienti in Fibra fin dai primi mesi.»

**Passaggio → Slide 10:** «Avviandomi alla chiusura, tiriamo le somme con i punti chiave, i limiti e gli sviluppi futuri.»

---

## Slide 10 — Conclusioni, Limiti e Sviluppi Futuri
> «In sintesi, ogni strumento ha un ruolo preciso: **K-Means** per capire i profili, **Regressione Logistica** per le previsioni, **Albero CART** per spiegare il perché e **soglia calibrata** per proteggere i costi. Portiamo a casa che il clustering isola i Volatili (**43,4%**), la Logistica vince per semplicità e solidità, e tarare la soglia sul CAC porta la Recall fino al **78,3%**.
>
> Come **limiti** abbiamo che K-Means forma gruppi "a sfera" (perché usa la distanza in linea retta dal centro), i dati sono statici senza lo storico dei reclami, e in produzione bisogna garantire **privacy**, **trasparenza** sulle decisioni e **fairness** (evitare discriminazioni su età o genere). Come **sviluppi futuri**: modelli ensemble come **Random Forest** (tanti alberi indipendenti in parallelo che votano a maggioranza) e **XGBoost** (alberi costruiti uno dopo l'altro dove ciascuno corregge gli errori del precedente), **soft clustering** (usare le distanze continue dai centri invece del solo numero di gruppo), analisi con **curva Precision-Recall** (più sensibile su dati sbilanciati) e costi pesati direttamente in euro sul CAC reale. Grazie per l'attenzione!»

---

## Mini-Glossario Lampo (se il prof ti interrompe e chiede «Cos'è / Come si calcola...?»)
- **4 Approcci IA / Mitchell:** Pensare/Agire come umani, Pensare/Agire razionalmente (scelto). Mitchell: impara da Esperienza (E) sul Compito (T) se migliora la Prestazione (P).
- **Split stratificato e Data Leakage:** divide 80% training e 20% test mantenendo lo stesso 26,5% di abbandoni **prima** di ogni calcolo, così nessun dato del test entra nel training.
- **Standardizzazione:** a ogni numero si toglie la media e si divide per la deviazione standard (solo sul training): `0` = media, `+` sopra, `-` sotto.
- **One-Hot Encoding:** trasforma le 16 variabili testuali in 27 colonne 0/1 togliendo 1 categoria ridondante per variabile (es. 3 tipi → 2 colonne).
- **Boxplot e Mediana:** mediana = dati in ordine, valore centrale (o media dei 2 centrali se pari); scatola = 25°–75° percentile (50% centrale); baffi = 1,5 volte la scatola; pallini = outlier.
- **K-Means, Inerzia e Silhouette:** K-Means assegna al centro più vicino e aggiorna i centri come media. Inerzia = somma distanze al quadrato dal centro. Silhouette (`-1` a `+1`) = `(distanza altro gruppo − distanza proprio gruppo) / massimo tra le due`.
- **PCA:** sceglie i 2 assi con massima varianza (PC1 39%, PC2 19% perpendicolare) per disegnare in 2D tenendo il 58% dell'informazione.
- **Sigmoide, ReLU, Adam, L2, Early Stopping:** Sigmoide trasforma la somma in probabilità 0–1; ReLU lascia passare solo segnali positivi; Adam aggiorna i pesi; L2 tiene i pesi equilibrati; Early Stopping ferma lo studio appena non migliora (anti-overfitting).
- **Metriche e AUC-ROC:** Recall = `TP/(TP+FN)`; Precision = `TP/(TP+FP)`; F1 = media armonica tra Precision e Recall; Accuracy = `(TP+TN)/tutti`; AUC-ROC = area sotto la curva Recall vs falsi allarmi (`0,5` caso – `1` perfetto).
- **5-fold Cross-Validation e Pesi Bilanciati:** 5-fold divide il training in 5 fette (4 studiano, 1 verifica a turno, mai il test); pesi bilanciati = sbagliare un abbandono pesa ~3 volte di più.
- **Indice di Gini:** `1 − (quota rimasti² + quota abbandoni²)`; con 2 classi va da `0` (puro) a `0,5` (metà e metà).
- **Etica e Modelli Futuri:** Privacy, trasparenza e fairness (no discriminazioni su età/genere); Random Forest = alberi in parallelo a maggioranza; XGBoost = alberi in sequenza che correggono gli errori precedenti.
