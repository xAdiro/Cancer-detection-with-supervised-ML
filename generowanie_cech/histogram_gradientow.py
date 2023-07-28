import skimage.io
import numpy as np
from skimage.feature import hog
from skimage.color import rgb2gray


# klasa tworząca histogram zorientowanych gradientów (Histogram of oriented gradients) dla obrazu
# histogram ten to masywny, jednowymiarowy wektor cech (rozmiar kilku tysięcy), więc można spróbować
# skompresować te cechy wyciągając z nich średnią/odchylenie standardowe (trudno powiedzieć czy będą one zawierać wartościowe informacje)
# klasa jako argument konstruktora bierze obraz (numpy array)
# zmien_obraz(), aby zmienić analizowane zdjęcie
class CechyHOG:
    def __init__(self, obraz) -> None:
        if type(obraz) == str:
            obr = rgb2gray(skimage.io.imread(obraz))
        else:
            obr = rgb2gray(obraz)
        cechy_wszystkie = hog(obr, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(1, 1), feature_vector=False)
        # dla kazdego bloku znajdź najbardziej wpływową orientacje
        cechy_zredukowane = np.zeros((cechy_wszystkie.shape[0], cechy_wszystkie.shape[1]))
        for i in range(cechy_wszystkie.shape[0]):
            for j in range(cechy_wszystkie.shape[1]):
                cechy_zredukowane[i, j] = np.argmax(cechy_wszystkie[i, j, 0, 0, :]) / 8
        self.cechy = cechy_zredukowane.ravel()

    def zmien_obraz(self, obraz):
        if type(obraz) == str:
            obr = rgb2gray(skimage.io.imread(obraz))
        else:
            obr = rgb2gray(obraz)
        cechy_wszystkie = hog(obr, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(1, 1), feature_vector=False)
        # dla kazdego bloku znajdź najbardziej wpływową orientacje
        cechy_zredukowane = np.zeros((cechy_wszystkie.shape[0], cechy_wszystkie.shape[1]))
        for i in range(cechy_wszystkie.shape[0]):
            for j in range(cechy_wszystkie.shape[1]):
                cechy_zredukowane[i, j] = np.argmax(cechy_wszystkie[i, j, 0, 0, :]) / 8
        self.cechy = cechy_zredukowane.ravel()

    # zwraca średnią gradientów (float)
    def srednia(self):
        return self.cechy.mean()

    # zwraca odchylenie standardowe gradientów (float)
    def std(self):
        return self.cechy.std()

    # zwraca wektor cech [float,float,...]
    def wektor_cech(self):
        return self.cechy
