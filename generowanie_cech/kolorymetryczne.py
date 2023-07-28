import numpy as np
import skimage.io
import skimage.color


# klasa do generacji cech kolorymetrycznych
# obraz i maska jako argumenty do konstruktora, zmien_obraz aby zmienić obraz/maskę
# obraz = ścieżka do pliku/numpy array
# maska = maska (numpy bool array from Thresholding)
class CechyKolorymetryczne:
    def __init__(self, obraz, maska) -> None:
        if type(obraz) == str:
            self.obraz = skimage.io.imread(obraz)
        else:
            self.obraz = obraz
        self.maska = maska

    def zmien_obraz(self, obraz, maska):
        if type(obraz) == str:
            self.obraz = skimage.io.imread(obraz)
        else:
            self.obraz = obraz
        self.maska = maska

    # zwraca średnią pikseli na każdym kanale [R,G,B]
    def srednia_rgb(self):

        wynik = []
        for channel in range(0, 3):
            wartosc = 0.0
            licznik = 0
            for i in range(0, self.obraz.shape[0]):
                for j in range(0, self.obraz.shape[1]):
                    if self.maska[i, j]:
                        wartosc += self.obraz[i, j, channel]
                        licznik += 1
            wynik.append(wartosc/licznik)
        return wynik

    # zwraca średnią pikseli na każdym kanale [H,S,V]
    def srednia_hsv(self):
        hsv_obr = skimage.color.rgb2hsv(self.obraz)
        wynik = []
        for kanal in range(0, 3):
            wartosc = 0.0
            licznik = 0
            for i in range(0, hsv_obr.shape[0]):
                for j in range(0, hsv_obr.shape[1]):
                    if self.maska[i, j]:
                        wartosc += hsv_obr[i, j, kanal]
                        licznik += 1
            wynik.append(wartosc/licznik)
        return wynik

    # zwraca średnią pikseli na każdym kanale [L,a*,b*]
    def srednia_lab(self):
        lab_obr = skimage.color.rgb2lab(self.obraz, "D65", "2")
        wynik = []
        for channel in range(0, 3):
            wartosc = 0.0
            licznik = 0
            for i in range(0, lab_obr.shape[0]):
                for j in range(0, lab_obr.shape[1]):
                    if self.maska[i][j]:
                        wartosc += lab_obr[i, j, channel]
                        licznik += 1
            wynik.append(wartosc/licznik)
        return wynik

    # zwraca odchylenie standardowe na każdym kanale [R,G,B]
    def std_rgb(self):
        wynik = []
        r = self.obraz[:, :, 0]  # czerwony
        g = self.obraz[:, :, 1]  # zielony
        b = self.obraz[:, :, 2]  # niebieski

        wynik.append(np.std(r[self.maska]))
        wynik.append(np.std(g[self.maska]))
        wynik.append(np.std(b[self.maska]))
        return wynik

    # zwraca odchylenie standardowe na każdym kanale [H,S,V]
    def std_hsv(self):
        hsv_obr = skimage.color.rgb2hsv(self.obraz)
        wynik = []
        h = hsv_obr[:, :, 0]  # odcień
        s = hsv_obr[:, :, 1]  # nasycenie
        v = hsv_obr[:, :, 2]  # wartość

        wynik.append(np.std(h[self.maska]))
        wynik.append(np.std(s[self.maska]))
        wynik.append(np.std(v[self.maska]))
        return wynik

    # zwraca odchylenie standardowe na każdym kanale [L,a*,b*]
    def std_lab(self):
        lab_obr = skimage.color.rgb2lab(self.obraz, "D65", "2")
        wynik = []
        l = lab_obr[:, :, 0]  # luminacja
        a = lab_obr[:, :, 1]  # tienta
        b = lab_obr[:, :, 2]  # temperatura

        wynik.append(np.std(l[self.maska]))
        wynik.append(np.std(a[self.maska]))
        wynik.append(np.std(b[self.maska]))
        return wynik
