import numpy as np
import skimage.io
from skimage.color import rgb2lab
from sklearn.cluster import KMeans
from skimage.util import img_as_ubyte


# klasa do wyliczenia centroid klastrów pikseli (a*,b*)
# obraz, jako argument do konstruktora, zmien_obraz, aby zmienić obraz
# obraz = ścieżka do pliku/numpy array
class KlastryKSrednich:
    def __init__(self, obraz) -> None:
        if type(obraz) == str:
            self.obr = rgb2lab(skimage.io.imread(obraz), "D65", "2")
        else:
            self.obr = rgb2lab(obraz, "D65", "2")
        self.__obl_klastry()

    def zmien_obraz(self, obraz):
        if type(obraz) == str:
            self.obr = rgb2lab(skimage.io.imread(obraz), "D65", "2")
        else:
            self.obr = rgb2lab(obraz, "D65", "2")
        self.__obl_klastry()

    # trenuje modele k-means
    # wywoływane w konstruktorze/zmien_obraz
    def __obl_klastry(self):
        a = np.array([[x] for x in self.obr[:, :, 1].flatten()])
        b = np.array([[x] for x in self.obr[:, :, 2].flatten()])
        self.k_srednich_a = KMeans(n_clusters=3).fit(a)
        self.k_srednich_b = KMeans(n_clusters=3).fit(b)

    # zwraca centroid każdego klastara ([[a*1],[a*2],[a*3]],[[b*1],[b*2],[b*3]])
    def centroidy(self):
        return (self.k_srednich_a.cluster_centers_, self.k_srednich_b.cluster_centers_)

    # zapisuje wizualizacje klastrów
    # sciezka_wynik = ścieżka do pliku
    def zapisz_klaster_obraz_a(self, sciezka_wynik):
        kl = self.k_srednich_a.labels_
        kl = np.reshape(kl, (self.obr.shape[0], self.obr.shape[1])) * 127
        skimage.io.imsave(sciezka_wynik, img_as_ubyte(kl), check_contrast=False)

    # zapisuje wizualizacje klastrów
    # sciezka_wynik = ścieżka do pliku
    def zapisz_klaster_obraz_b(self, sciezka_wynik):
        kl = self.k_srednich_b.labels_
        kl = np.reshape(kl, (self.obr.shape[0], self.obr.shape[1])) * 127
        skimage.io.imsave(sciezka_wynik, img_as_ubyte(kl), check_contrast=False)
