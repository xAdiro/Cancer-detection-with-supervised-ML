import multiprocessing

import skimage.io
import numpy as np
from math import isnan
from csv import writer
from skimage.color import rgb2gray

from multiprocessing import Process

import progowanie
from klasteryzacja import KlastryKSrednich
from kolorymetryczne import CechyKolorymetryczne
from macierz_GLCM import CechyGLCM
from histogram import CechyHistogram
from histogram_gradientow import CechyHOG
from lokalna_krzywizna import CechyKrzywizny
from krawedzie import Krawedzie


def generowanie_cech(sciezka_do_pliku):
    print('0. Generacja cech')
    print('ROI: Segmentacja regionow zainteresowania (maska)')

    obraz_0 = skimage.io.imread(sciezka_do_pliku)

    if len(obraz_0.shape) > 2:
        obraz = obraz_0[:, :, :3]
    else:
        obraz = np.ndarray(shape=(obraz_0.shape[0], obraz_0.shape[1], 3))
        for i in range(0, obraz_0.shape[0]):
            for j in range(0, obraz_0.shape[1]):
                wartosc = obraz_0[i][j]
                obraz[i][j][0] = wartosc
                obraz[i][j][1] = wartosc
                obraz[i][j][2] = wartosc

    obraz_maska = progowanie.ProgowanieOTSU()
    maska = progowanie.wybierz_co_trzeba(obraz_maska.maska(obraz))

    cechy = []

    print('Cechy: Klasteryzacja')

    obraz_klasteryzacja = KlastryKSrednich(obraz)
    centroidy = obraz_klasteryzacja.centroidy()

    print('Cechy: Kolorymetryczne')

    print(obraz)
    print(maska)

    obraz_kolorymetryczne = CechyKolorymetryczne(obraz, maska)
    # kolor_srednia_lab = obraz_kolorymetryczne.srednia_lab()
    kolor_std_lab = obraz_kolorymetryczne.std_lab()

    print('Cechy: Macierz GLCM')

    obraz_glcm = CechyGLCM(obraz)
    glcm_kontrast = obraz_glcm.kontrast()
    glcm_odmiennosc = obraz_glcm.odmiennosc()
    glcm_homogenicznosc = obraz_glcm.homogenicznosc()
    glcm_energia = obraz_glcm.energia()
    glcm_korelacja = obraz_glcm.korelacja()

    print('Cechy: Histogram')

    obraz_histogram = CechyHistogram(obraz, maska)
    hist_srednia_lab = obraz_histogram.hist_srednia_lab()
    hist_wariancja_lab = obraz_histogram.hist_var_lab()
    hist_skosnosc_lab = obraz_histogram.hist_skos_lab()
    hist_kurtoza_lab_lb = obraz_histogram.hist_kurt_lab_lb()
    hist_entropia_lab_l = obraz_histogram.entropia_lab_l()

    print('Cechy: Histogram hog')

    obraz_hog = CechyHOG(obraz)
    hog_srednia = obraz_hog.srednia()
    hog_std = obraz_hog.std()

    print('Cechy: Krawedzie')

    obraz_krawedzie = Krawedzie(obraz)
    krawedzie_gestosc = obraz_krawedzie.gestosc_krawedzi()
    krawedzie_srednia_rgb = obraz_krawedzie.sredni_kolor_krawedzi_rgb()
    krawedzie_std_rgb = obraz_krawedzie.std_kolor_krawedzi_rgb()

    print('Cechy: Lokalna krzywizna')

    obraz_krzywizna = CechyKrzywizny(obraz, maska)
    krzywizna_spherical_cup = obraz_krzywizna.krzywizna_spherical_cup()
    krzywizna_through = obraz_krzywizna.krzywizna_through()
    krzywizna_rut = obraz_krzywizna.krzywizna_rut()
    krzywizna_saddle_rut = obraz_krzywizna.krzywizna_saddle_rut()
    krzywizna_saddle = obraz_krzywizna.krzywizna_saddle()
    krzywizna_saddle_ridge = obraz_krzywizna.krzywizna_saddle_ridge()
    krzywizna_ridge = obraz_krzywizna.krzywizna_ridge()
    krzywizna_dome = obraz_krzywizna.krzywizna_dome()
    krzywizna_spherical_cap = obraz_krzywizna.krzywizna_spherical_cap()

    # Popraw NaN na 0.0

    for i in range(0, len(cechy)):
        if isnan(cechy[i]):
            cechy[i] = 0.0

    # Dodaj cechy do wektora

    for i in range(0, 2):
        for j in range(0, 3):
            cechy.append(centroidy[i][j][0])

    for i in range(0, 3):
        cechy.append(kolor_std_lab[i])

    cechy.append(glcm_kontrast)
    cechy.append(glcm_odmiennosc)
    cechy.append(glcm_homogenicznosc)
    cechy.append(glcm_energia)
    cechy.append(glcm_korelacja)

    for i in range(0, 3):
        cechy.append(hist_srednia_lab[i])
    for i in range(0, 3):
        cechy.append(hist_wariancja_lab[i])
    for i in range(0, 3):
        cechy.append(hist_skosnosc_lab[i])
    for i in range(0, 2):
        cechy.append(hist_kurtoza_lab_lb[i])
    cechy.append(hist_entropia_lab_l)

    cechy.append(hog_srednia)
    cechy.append(hog_std)

    cechy.append(krawedzie_gestosc)
    for i in range(0, 3):
        cechy.append(krawedzie_srednia_rgb[i])
    for i in range(0, 3):
        cechy.append(krawedzie_std_rgb[i])

    cechy.append(krzywizna_spherical_cup[1])
    cechy.append(krzywizna_through[1])
    cechy.append(krzywizna_rut[1])
    cechy.append(krzywizna_saddle_rut[1])
    cechy.append(krzywizna_saddle[1])
    cechy.append(krzywizna_saddle_ridge[1])
    cechy.append(krzywizna_ridge[1])
    cechy.append(krzywizna_dome[1])
    cechy.append(krzywizna_spherical_cap[1])

    print('Cechy wygenerowane')

    return cechy

def funkcja_procesu(plik_metadane, index_pocz):
    with open(f'cechy{index_pocz}.csv', 'w', newline='') as fcsv:
        w = writer(fcsv, delimiter=';')
        n = 0

        for i in range(index_pocz, len(plik_metadane), 10):
            line = plik_metadane[i]

            if line[2] == 'vasc':
                continue

            nazwa = line[1]
            czyRak = 0 if line[2] in NIE_RAKOW else 1

            sciezka_do_pliku = f'HAM10000/{nazwa}.jpg'
            cechy = generowanie_cech(sciezka_do_pliku)
            cechy.insert(0, czyRak)

            w.writerow(cechy)

            n += 1
            print(f'Proces {index_pocz}: {(n / (len(plik_metadane) / 10.0)) * 100} %')


NIE_RAKOW = ['bkl', 'df', 'nv']

def main():
    with open('HAM10000_metadata', 'r') as ftxt:
        # line = ftxt.readline()
        lines = ftxt.readlines()

        for i in range(len(lines)):
            lines[i] = lines[i].split(',')

    procesy = []

    for i in range(10):
        proces = Process(target=funkcja_procesu, args=(lines, i+1))
        proces.start()
        procesy.append(proces)

    for proces in procesy:
        proces.join()

    # while True:
    #     line = ftxt.readline()
    #     index = [0, 0, 0]
    #
    #     if not line:
    #         break
    #
    #     # print(line)
    #     index[0] = line.find(',')
    #     line = line.replace(',', '.', 1)
    #     index[1] = line.find(',')
    #     line = line.replace(',', '.', 1)
    #     index[2] = line.find(',')
    #
    #     sciezka_do_pliku = 'C:/Users/mariu/Desktop/TestyFunkcji/HAM10000/' + line[index[0]+1:index[1]] + '.jpg'
    #     typ = line[index[1]+1:index[2]]
    #
    #     cechy = generowanie_cech(sciezka_do_pliku)
    #     czyRak = 0 if typ == 'bkl' or typ == 'df' or typ == 'nv' else 1
    #     cechy.insert(0, czyRak)
    #     # print(cechy)
    #
    #     writer.writerow(cechy)


if __name__ == '__main__':
    main()












# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", type=str, required=True, help = "path to input image")
# args = vars(ap.parse_args())
#
# image = cv2.imread(args["image"])
# cv2.imshow("Image", image)
#
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# blurred = cv2.GaussianBlur(gray, (7, 7), 0)
#
# (T, threshInv) = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
# cv2.imshow("OTSU Thresholding", threshInv)
# cv2.waitKey(0)

# obraz_0 = skimage.io.imread('unknown.png')
#
# if len(obraz_0.shape) > 2:
#     obraz = obraz_0[:, :, :3]
# else:
#     obraz = np.ndarray(shape=(obraz_0.shape[0], obraz_0.shape[1], 3))
#     for i in range(0, obraz_0.shape[0]):
#         for j in range(0, obraz_0.shape[1]):
#             wartosc = obraz_0[i][j]
#             obraz[i][j][0] = wartosc
#             obraz[i][j][1] = wartosc
#             obraz[i][j][2] = wartosc
#
# obraz_maska = ProgowanieOTSU()
# maska = obraz_maska.maska(obraz, 'wynik2.png')
#
# srodek = [int(maska.shape(0) / 2), int(maska.shape(1) / 2)]
# min_wymiar = 0 if srodek[0] < srodek[1] else 1
# procent = 0.8


