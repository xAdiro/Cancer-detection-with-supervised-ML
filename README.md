# Cancer detection with supervised ML
Używając popularnych metod uczenia maszynowego (SVM, Random Forest, Regresja logistyczna, kNN, Naiwny Bayes), jak i metody hybrydowej o nie opartej, stworzyliśmy modele do klasyfikacji zdjęć zmian skórynch, jako rakowe lub nie.

Ze zdjęć z zestawu dataset HAM10000 wydobyliśmy 44 cechy, które nastepnie przekazaliśmy do modeli. Skrypty do wykonania tej czynności znajdują się w folderze `generowanie_cech`, a plik `main.py` służy do wygenerowania tych cech dla zdjęć. Skrypt wykorzystuje wiele rdzeni procesora, korzystając z biblioteki `multiprocessing`, aby przyspieszyć ten proces.

Dokładny opis wraz z kodem można znaleźc w pliku `main.ipynb` lub `main.html`
