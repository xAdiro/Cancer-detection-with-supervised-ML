from skimage.feature import shape_index
from skimage.color import rgb2gray
import skimage.io


# Klasa do obliczania cech krzywizny
# Jeśli potraktujemy intensywność piksela jako wysokość, to obraz bedzie strukturą 3D, której krzywizna
# ma pewne górki, zagłębienia itp., które można znaleźć licząc hesjan dla obrazu. Jako cech można użyć
# ilości/gęstości różnego rodzaju struktur
# do konstruktora obraz=(numpy array lub ścieżka do pliku) oraz maska=(niewymagana, zamaskowane piksele będą
# traktowane jako kolor czarny)
# zmien_obraz, aby zmienić analizowany obraz
class CechyKrzywizny:
    def __init__(self, obraz, maska=None) -> None:
        if type(obraz) == str:
            obr = rgb2gray(skimage.io.imread(obraz))
        else:
            obr = rgb2gray(obraz)
        if maska is not None:
            obr = obr * maska
        self.il_pikseli = obr.shape[0] * obr.shape[1]
        self.__oblicz(obr)

    def zmien_obraz(self, obraz, maska=None):
        if type(obraz) == str:
            obr = rgb2gray(skimage.io.imread(obraz))
        else:
            obr = rgb2gray(obraz)
        if maska is not None:
            obr = obr * maska
        self.il_pikseli = obr.shape[0] * obr.shape[1]
        self.__oblicz(obr)

    def __oblicz(self, obraz):
        s = shape_index(obraz, sigma=0.1).ravel()
        self.index = {
            "spherical_cup": 0,
            "through": 0,
            "rut": 0,
            "saddle_rut": 0,
            "saddle": 0,
            "saddle_ridge": 0,
            "ridge": 0,
            "dome": 0,
            "spherical_cap": 0
        }
        for x in s:
            if -1 <= x < -0.875:
                self.index["spherical_cup"] += 1
            if -0.875 <= x < -0.625:
                self.index["through"] += 1
            if -0.625 <= x < -0.375:
                self.index["rut"] += 1
            if -0.375 <= x < -0.125:
                self.index["saddle_rut"] += 1
            if -0.125 <= x < 0.125:
                self.index["saddle"] += 1
            if 0.125 <= x < 0.375:
                self.index["saddle_ridge"] += 1
            if 0.375 <= x < 0.625:
                self.index["ridge"] += 1
            if 0.625 <= x < 0.875:
                self.index["dome"] += 1
            if 0.875 <= x < 1:
                self.index["spherical_cap"] += 1

    # funkcje krzywizna_nazwa() zwracają (ilość, gęstość) wystąpień danej struktury
    def krzywizna_spherical_cup(self):
        return self.index["spherical_cup"], self.index["spherical_cup"] / self.il_pikseli

    def krzywizna_through(self):
        return self.index["through"], self.index["through"] / self.il_pikseli

    def krzywizna_rut(self):
        return self.index["rut"], self.index["rut"] / self.il_pikseli

    def krzywizna_saddle_rut(self):
        return self.index["saddle_rut"], self.index["saddle_rut"] / self.il_pikseli

    def krzywizna_saddle(self):
        return self.index["saddle"], self.index["saddle"] / self.il_pikseli

    def krzywizna_saddle_ridge(self):
        return self.index["saddle_ridge"], self.index["saddle_ridge"] / self.il_pikseli

    def krzywizna_ridge(self):
        return self.index["ridge"], self.index["ridge"] / self.il_pikseli

    def krzywizna_dome(self):
        return self.index["dome"], self.index["dome"] / self.il_pikseli

    def krzywizna_spherical_cap(self):
        return self.index["spherical_cap"], self.index["spherical_cap"] / self.il_pikseli
