import math
import numpy as np
import skimage.io
import skimage.color
from skimage.measure.entropy import shannon_entropy


# klasa do generacji cech histogramów
# obraz i maska jako argumenty do konstruktora, zmien_obraz aby zmienić obraz/maskę
# obraz = ścieżka do pliku/numpy array
# maska = maska (numpy bool array uzyskany z Thresholding)
class CechyHistogram:
    def __init__(self, obraz, maska) -> None:
        if type(obraz) == str:
            self.obr = skimage.io.imread(obraz)
        else:
            self.obr = obraz
        self.maska = maska
        self.__obl_hist()

    def zmien_obraz(self, obraz, maska):
        if type(obraz) == str:
            self.obr = skimage.io.imread(obraz)
        else:
            self.obr = obraz
        self.maska = maska
        self.__obl_hist()

    # wylicza histogramy, wywoływane w konstruktorze/zmien_obraz
    def __obl_hist(self):
        r = self.obr[:, :, 0]  # czerwony
        g = self.obr[:, :, 1]  # zielony
        b = self.obr[:, :, 2]  # niebieski
        self.hist_r, bin_brzegi_r = np.histogram(r[self.maska], bins=256, range=(0, 256))
        self.hist_g, bin_brzegi_g = np.histogram(g[self.maska], bins=256, range=(0, 256))
        self.hist_b, bin_brzegi_b = np.histogram(b[self.maska], bins=256, range=(0, 256))
        self.mids_r = bin_brzegi_r[:-1]
        self.mids_g = bin_brzegi_g[:-1]
        self.mids_b = bin_brzegi_b[:-1]

        hsv_obr = skimage.color.rgb2hsv(self.obr)
        h = hsv_obr[:, :, 0]  # odcień
        s = hsv_obr[:, :, 1]  # nasycenie
        v = hsv_obr[:, :, 2]  # wartość
        self.hist_h, bin_brzegi_h = np.histogram(h[self.maska], bins=100, range=(0, 1))
        self.hist_s, bin_brzegi_s = np.histogram(s[self.maska], bins=100, range=(0, 1))
        self.hist_v, bin_brzegi_v = np.histogram(v[self.maska], bins=100, range=(0, 1))
        self.mids_h = bin_brzegi_h[:-1]
        self.mids_s = 0.5*(bin_brzegi_s[1:] + bin_brzegi_s[:-1])
        self.mids_v = 0.5*(bin_brzegi_v[1:] + bin_brzegi_v[:-1])

        lab_obr = skimage.color.rgb2lab(self.obr, "D65", "2")
        l = lab_obr[:, :, 0]  # luminacja
        a = lab_obr[:, :, 1]  # tienta
        b = lab_obr[:, :, 2]  # temperatura
        self.hist_l, bin_brzegi_l = np.histogram(l[self.maska], bins=100, range=(0, 100))
        self.hist_a, bin_brzegi_a = np.histogram(a[self.maska], bins=256, range=(-128, 128))
        self.hist_b, bin_brzegi_b = np.histogram(b[self.maska], bins=256, range=(-128, 128))
        self.mids_l = 0.5*(bin_brzegi_l[1:] + bin_brzegi_l[:-1])
        self.mids_a = 0.5*(bin_brzegi_a[1:] + bin_brzegi_a[:-1])
        self.mids_b = 0.5*(bin_brzegi_b[1:] + bin_brzegi_b[:-1])

    # zwraca średnią histogramów na każdym kanale [R,G,B]
    def hist_srednia_rgb(self):
        wynik = []
        wynik.append(np.average(self.mids_r, weights=self.hist_r))
        wynik.append(np.average(self.mids_g, weights=self.hist_g))
        wynik.append(np.average(self.mids_b, weights=self.hist_b))
        return wynik

    # zwraca średnią histogramów na każdym kanale [H,S,V]
    def hist_srednia_hsv(self):
        wynik = []
        wynik.append(np.average(self.mids_h, weights=self.hist_h))
        wynik.append(np.average(self.mids_s, weights=self.hist_s))
        wynik.append(np.average(self.mids_v, weights=self.hist_v))
        return wynik

    # zwraca średnią histogramów na każdym kanale [L,a*,b*]
    def hist_srednia_lab(self):
        wynik = []
        wynik.append(np.average(self.mids_l, weights=self.hist_l))
        wynik.append(np.average(self.mids_a, weights=self.hist_a))
        wynik.append(np.average(self.mids_b, weights=self.hist_b))
        return wynik

    # zwraca wariancję histogramów na każdym kanale [R,G,B]
    def hist_var_rgb(self):
        wynik = []
        srednia = self.hist_srednia_rgb()
        wynik.append(np.average((self.mids_r - srednia[0])**2, weights=self.hist_r))
        wynik.append(np.average((self.mids_g - srednia[1])**2, weights=self.hist_g))
        wynik.append(np.average((self.mids_b - srednia[2])**2, weights=self.hist_b))
        return wynik

    # zwraca wariancję histogramów na każdym kanale [H,S,V]
    def hist_var_hsv(self):
        wynik = []
        srednia = self.hist_srednia_hsv()
        wynik.append(np.average((self.mids_h - srednia[0])**2, weights=self.hist_h))
        wynik.append(np.average((self.mids_s - srednia[1])**2, weights=self.hist_s))
        wynik.append(np.average((self.mids_v - srednia[2])**2, weights=self.hist_v))
        return wynik

    # zwraca wariancję histogramów na każdym kanale [L,a*,b*]
    def hist_var_lab(self):
        wynik = []
        srednia = self.hist_srednia_lab()
        wynik.append(np.average((self.mids_l - srednia[0])**2, weights=self.hist_l))
        wynik.append(np.average((self.mids_a - srednia[1])**2, weights=self.hist_a))
        wynik.append(np.average((self.mids_b - srednia[2])**2, weights=self.hist_b))
        return wynik

    # zwraca skośność histogramu na każdym kanale [R,G,B]
    def hist_skos_rgb(self):
        wynik = []
        srednia = self.hist_srednia_rgb()
        wynik.append((np.sum((self.hist_r-srednia[0])**3)/self.hist_r.size)/math.sqrt((np.sum((self.hist_r-srednia[0])**2)/self.hist_r.size)**3))
        wynik.append((np.sum((self.hist_g-srednia[1])**3)/self.hist_g.size)/math.sqrt((np.sum((self.hist_g-srednia[1])**2)/self.hist_g.size)**3))
        wynik.append((np.sum((self.hist_b-srednia[2])**3)/self.hist_b.size)/math.sqrt((np.sum((self.hist_b-srednia[2])**2)/self.hist_b.size)**3))
        return wynik

    # zwraca skośność histogramu na każdym kanale [H,S,V]
    def hist_skos_hsv(self):
        wynik = []
        srednia = self.hist_srednia_hsv()
        wynik.append((np.sum((self.hist_h-srednia[0])**3)/self.hist_h.size)/math.sqrt((np.sum((self.hist_h-srednia[0])**2)/self.hist_h.size)**3))
        wynik.append((np.sum((self.hist_s-srednia[1])**3)/self.hist_s.size)/math.sqrt((np.sum((self.hist_s-srednia[1])**2)/self.hist_s.size)**3))
        wynik.append((np.sum((self.hist_v-srednia[2])**3)/self.hist_v.size)/math.sqrt((np.sum((self.hist_v-srednia[2])**2)/self.hist_v.size)**3))
        return wynik

    # zwraca skośność histogramu na każdym kanale [L,a*,b*]
    def hist_skos_lab(self):
        wynik = []
        srednia = self.hist_srednia_lab()
        wynik.append((np.sum((self.hist_l-srednia[0])**3)/self.hist_l.size)/math.sqrt((np.sum((self.hist_l-srednia[0])**2)/self.hist_l.size)**3))
        wynik.append((np.sum((self.hist_a-srednia[1])**3)/self.hist_a.size)/math.sqrt((np.sum((self.hist_a-srednia[1])**2)/self.hist_a.size)**3))
        wynik.append((np.sum((self.hist_b-srednia[2])**3)/self.hist_b.size)/math.sqrt((np.sum((self.hist_b-srednia[2])**2)/self.hist_b.size)**3))
        return wynik

    # zwraca kurtozę histogramu na każdym kanale [R,G,B]
    def hist_kurt_rgb(self):
        wynik = []
        srednia = self.hist_srednia_rgb()
        wynik.append((np.sum((self.hist_r-srednia[0])**4)/self.hist_r.size)/math.sqrt((np.sum((self.hist_r-srednia[0])**2)/self.hist_r.size)**4)-3)
        wynik.append((np.sum((self.hist_g-srednia[1])**4)/self.hist_g.size)/math.sqrt((np.sum((self.hist_g-srednia[1])**2)/self.hist_g.size)**4)-3)
        wynik.append((np.sum((self.hist_b-srednia[2])**4)/self.hist_b.size)/math.sqrt((np.sum((self.hist_b-srednia[2])**2)/self.hist_b.size)**4)-3)
        return wynik

    # zwraca kurtozę histogramu na każdym kanale [H,S,V]
    def hist_kurt_hsv(self):
        wynik = []
        srednia = self.hist_srednia_hsv()
        wynik.append((np.sum((self.hist_h-srednia[0])**4)/self.hist_h.size)/math.sqrt((np.sum((self.hist_h-srednia[0])**2)/self.hist_h.size)**4)-3)
        wynik.append((np.sum((self.hist_s-srednia[1])**4)/self.hist_s.size)/math.sqrt((np.sum((self.hist_s-srednia[1])**2)/self.hist_s.size)**4)-3)
        wynik.append((np.sum((self.hist_v-srednia[2])**4)/self.hist_v.size)/math.sqrt((np.sum((self.hist_v-srednia[2])**2)/self.hist_v.size)**4)-3)
        return wynik

    # zwraca kurtozę histogramu na każdym kanale [L,a*,b*]
    def hist_kurt_lab_lb(self):
        wynik = []
        srednia = self.hist_srednia_lab()
        wynik.append((np.sum((self.hist_l-srednia[0])**4)/self.hist_l.size)/math.sqrt((np.sum((self.hist_l-srednia[0])**2)/self.hist_l.size)**4)-3)
        wynik.append((np.sum((self.hist_b-srednia[2])**4)/self.hist_b.size)/math.sqrt((np.sum((self.hist_b-srednia[2])**2)/self.hist_b.size)**4)-3)
        return wynik

    # zwraca entropie obrazu w RGB (float)
    def entropia_rgb(self):
        wynik = []
        wynik.append(shannon_entropy(self.obr[:, :, 0]))
        wynik.append(shannon_entropy(self.obr[:, :, 1]))
        wynik.append(shannon_entropy(self.obr[:, :, 2]))
        return wynik

    # zwraca entropie obrazu w HSV (float)
    def entropia_hsv(self):
        hsv_obr = skimage.color.rgb2hsv(self.obr)
        wynik = []
        wynik.append(shannon_entropy(hsv_obr[:, :, 0]))
        wynik.append(shannon_entropy(hsv_obr[:, :, 1]))
        wynik.append(shannon_entropy(hsv_obr[:, :, 2]))
        return wynik

    # zwraca entropie obrazu w La*b* (float)
    def entropia_lab_l(self):
        lab_obr = skimage.color.rgb2lab(self.obr, "D65", "2")
        wynik = shannon_entropy(lab_obr[:, :, 0])
        return wynik
