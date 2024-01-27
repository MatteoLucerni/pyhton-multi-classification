La classe LogisticRegeression riconosce in automatico se bisogna utilizzare la tecnica OvA/OvR

Una nuova metrica interessante per la classificazione multiclasse è la matrice di confusione, ti permette di verificare le predizioni del modllo classe per classe.
L'output appunto sarà una matrice le cui colonne rappresentano le classi predette e le righe le classi correte

Per esempio avere un numero 2 alla colonna 5 riga 7 significa che 5 immagini rappresentanti un 7 sono state classificate come 5 sbagliando

Scikit learn offre una classe deicata alla tecnica One versus all / One versus Rest

OneVsRestClassifier(Classifier())

All'interno della classe bisogna specificare il classifier che si vuole utilizzare, nel nostro caso LogisticRegression()

Questo porterà allo stesso risultato di prima
