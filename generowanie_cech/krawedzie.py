from skimage.color import rgb2gray
from skimage.feature import canny
import skimage.io
import numpy as np


# klasa do wykrycia krawędzi, obliczenia maski krawędzi i cech kolorymetrycznych krawędzi
# arg do konstruktora to obraz w formie numpy array/scieżka do pliku
# zmien_obraz(), aby zmienić aktualnie analizowany obraz
class Krawedzie:
    def __init__(self, obraz) -> None:
        if type(obraz) == str:
            self.obr_kolor = skimage.io.imread(obraz)
            obr = rgb2gray(self.obr_kolor)
        else:
            self.obr_kolor = obraz
            obr = rgb2gray(self.obr_kolor)
        self.maska_krawedzi = canny(obr, sigma=1)

    def zmien_obraz(self, obraz):
        if type(obraz) == str:
            self.obr_kolor = skimage.io.imread(obraz)
            obr = rgb2gray(self.obr_kolor)
        else:
            self.obr_kolor = obraz
            obr = rgb2gray(self.obr_kolor)
        self.maska_krawedzi = canny(obr, sigma=1)

    # zwraca maskę krawędzi (2d numpy bool array)
    def maska(self):
        return self.maska_krawedzi

    # zwraca gęstość krawędzi (float)
    def gestosc_krawedzi(self):
        return np.count_nonzero(self.maska_krawedzi) / (self.maska_krawedzi.shape[0] * self.maska_krawedzi.shape[1])

    # zwraca średni kolor krawędzi dla każdego kanału [czerwony,zielony,niebieski]
    def sredni_kolor_krawedzi_rgb(self):
        r = self.obr_kolor[:, :, 0]  # czerwony
        g = self.obr_kolor[:, :, 1]  # zielony
        b = self.obr_kolor[:, :, 2]  # niebieski
        return [r[self.maska_krawedzi].mean(), g[self.maska_krawedzi].mean(), b[self.maska_krawedzi].mean()]

    # zwraca odchylenie standardowe koloru krawędzi dla każdego kanału [czerwony,zielony,niebieski]
    def std_kolor_krawedzi_rgb(self):
        r = self.obr_kolor[:, :, 0]  # czerwony
        g = self.obr_kolor[:, :, 1]  # zielony
        b = self.obr_kolor[:, :, 2]  # niebieski
        return [r[self.maska_krawedzi].std(), g[self.maska_krawedzi].std(), b[self.maska_krawedzi].std()]
