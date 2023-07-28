from typing import List
from typing import Tuple

import skimage.io
from skimage.filters import threshold_otsu
from skimage.color import rgb2gray
from skimage.util import img_as_ubyte


# klasa do generacji maski używając OTSU
# konwertuje obraz do skali szarości (grayscale), znajduje próg (threshold(float)) i generuje maske

class ProgowanieOTSU:
    def __init__(self) -> None:
        pass

    # zwraca maske w postaci numpy array
    # obraz = scieżka do pliku/numpy array
    def maska(self, obraz):
        skala_szarosci = rgb2gray(obraz)

        prog = threshold_otsu(skala_szarosci)
        wynik = skala_szarosci <= prog
        return wynik

    # generuje maske i zapisuje do png (true=255, false=0)
    # obraz = scieżka do pliku/numpy array
    # sciezka_wynik = scieżka do zapisu
    def zapisz_maske(self, obraz, sciezka_wynik):
        wynik = self.maska(obraz)
        skimage.io.imsave(sciezka_wynik, img_as_ubyte(wynik))
        return wynik


def wybierz_co_trzeba(maska: List[List[bool]]):

    szerokosc = len(maska[0])
    wysoksoc = len(maska)
    srodek = (int(szerokosc/2), int(wysoksoc/2))

    nowa_maska = [[False] * szerokosc for i in range(wysoksoc)]

    nowa_maska[srodek[0]][srodek[1]] = True

    while do_wywolania := szukaj_sasiadow(nowa_maska, maska, srodek[0], srodek[1]):
        for xy in do_wywolania:
            do_wywolania.remove(xy)
            szukaj_sasiadow(nowa_maska, maska, xy[0], xy[1])

    return nowa_maska


def szukaj_sasiadow(nowa_maska, stara_maska, x, y, do_wywolania=[]) -> List[Tuple[int, int]]:
    MAX_ODLEGLOSC = 10
    promien = int(MAX_ODLEGLOSC / 2)

    for i in range(y-promien, y+promien):
        for j in range(x-promien, x+promien):
            if i < 0 or j < 0 or i >= len(stara_maska) or j >= len(stara_maska[0]): continue

            if stara_maska[i][j] and not nowa_maska[i][j]:
                nowa_maska[i][j] = True
                do_wywolania.append([j, i])

    return do_wywolania

