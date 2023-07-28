from skimage.feature import graycomatrix, graycoprops
from skimage.color import rgb2gray
from skimage.util import img_as_ubyte
import skimage.io


# klasa do analizy tekstury obrazu, cechy macierzy glcm
# obraz(numpy array) jako argument do konstruktora
# zmien_obraz, aby zmienić aktualnie analizowany obraz
class CechyGLCM:
    def __init__(self, obraz) -> None:
        if type(obraz) == str:
            obr = img_as_ubyte(rgb2gray(skimage.io.imread(obraz)))
        else:
            obr = img_as_ubyte(rgb2gray(obraz))
        self.glcm = graycomatrix(obr, distances=[5], angles=[0], levels=256, symmetric=True, normed=True)

    def zmien_obraz(self, obraz):
        if type(obraz) == str:
            obr = img_as_ubyte(rgb2gray(skimage.io.imread(obraz)))
        else:
            obr = img_as_ubyte(rgb2gray(obraz))
        self.glcm = graycomatrix(obr, distances=[5], angles=[0], levels=256, symmetric=True, normed=True)

    # zwraca kontrast glcm (float)
    def kontrast(self):
        return graycoprops(self.glcm, 'contrast')[0][0]

    # zwraca odmienność glcm (float)
    def odmiennosc(self):
        return graycoprops(self.glcm, 'dissimilarity')[0][0]

    # zwraca homogeniczność glcm (float)
    def homogenicznosc(self):
        return graycoprops(self.glcm, 'homogeneity')[0][0]

    # zwraca energię glcm (float)
    def energia(self):
        return graycoprops(self.glcm, 'energy')[0][0]

    # zwraca korelację glcm (float)
    def korelacja(self):
        return graycoprops(self.glcm, 'correlation')[0][0]
    
