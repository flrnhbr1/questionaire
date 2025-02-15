import csv
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_kmo
import pandas as pd




def read_data_ffz_mlq_relevant(filename, q_ffz, q_mlq):
    # initializing the titles and rows list
    fields = []
    rows = []

    # reading csv file
    with open(filename, 'r') as csvfile:
        # creating a csv reader object
        csvreader = csv.reader(csvfile)

        # extracting field names through first row
        fields = next(csvreader)

        # extracting each data row one by one
        for row in csvreader:
            rows.append(row)

        # get total number of rows
        # print("Total no. of rows: %d" % (csvreader.line_num))

    data_table_ffz = []
    data_table_mlq = []

    # iterate over datasets
    for row in rows:
        # only consider finished datasets
        if row[fields.index('FINISHED')] == '1' and '-9' not in row:

            # only consider relevant jobs
            a104 = int(row[fields.index('A104')])
            if a104 == 1 or a104 == 2 or a104 == 3 or a104 == 5 or a104 == 6:
                # for ffz
                temp_array = []
                # iterate over questions
                for f in q_ffz:
                    temp_array.append(int(row[fields.index(f)]))
                data_table_ffz.append(temp_array)

                # for mlq
                temp_array = []
                # iterate over questions
                for m in q_mlq:
                    temp_array.append(int(row[fields.index(m)]))
                data_table_mlq.append(temp_array)

    # data tables are structured as following:
    # [[Person1_Question1, Person1_Question2, ...], [Person1_Question1, Person1_Question2, ...]]
    return data_table_ffz, data_table_mlq


if __name__ == '__main__':

    data_filename = './data_test430616_2025-01-19_21-03.csv'

    questions_ffz = ['A201', 'A203', 'A204', 'A205', 'A206', 'A207', 'A208', 'A210', 'A211', 'A214', 'A215',
                     'A216', 'A217']

    questions_mlq = ['A303', 'A306', 'A307', 'A308', 'A310', 'A311', 'A312', 'A315', 'A316', 'A318', 'A319', 'A321',
                     'A322', 'A325', 'A326', 'A327', 'A328', 'A329', 'A330', 'A331', 'A334', 'A335', 'A336', 'A337',
                     'A339', 'A341', 'A347', 'A351']

    # Lade nur die ausgewählten Spalten
    data = pd.read_csv(data_filename, usecols=questions_mlq)

    # Überprüfe die ersten Zeilen der Daten
    print(data.head())
    print(data)

    # Filtere Zeilen, bei denen eine der Spalten NaN oder '-9' enthält
    data_filtered = data[~data.apply(lambda row: row.isna().any() or (row == -9).any(), axis=1)]

    # Zeige das gefilterte DataFrame
    print(data_filtered)

    # Schritt 2: Überprüfe, ob die Daten geeignet sind (KMO und Bartlett-Test)
    kmo_all, kmo_model = calculate_kmo(data_filtered)
    print(f"KMO-Wert: {kmo_model}")

    # Schritt 3: Führe die explorative Faktorenanalyse (EFA) durch
    # Bestimme zuerst, wie viele Faktoren extrahiert werden sollen (z. B. 3 für die 3 Führungsstile)
    fa = FactorAnalyzer(n_factors=3, rotation=None)  # 3 Faktoren für 3 Führungsstile
    fa.fit(data_filtered)

    # Schaue dir die Eigenwerte an (um zu entscheiden, wie viele Faktoren extrahiert werden)
    eigenwerte = fa.get_eigenvalues()
    print(f"Eigenwerte: {eigenwerte}")

    # Scree-Plot zur Bestimmung der Anzahl der Faktoren
    plt.plot(range(1, len(eigenwerte) + 1), eigenwerte, marker='o')
    plt.title('Scree Plot')
    plt.xlabel('Anzahl der Faktoren')
    plt.ylabel('Eigenwert')
    plt.show()

    # Schritt 4: Führe die Varimax-Rotation durch (für besser interpretierbare Faktoren)
    fa_rot = FactorAnalyzer(n_factors=3, rotation='varimax')
    fa_rot.fit(data_filtered)

    # Zeige die rotierte Faktorladungenmatrix an
    faktor_ladungen_rotated = fa_rot.loadings_
    print("Rotierte Faktorladungen:")
    print(faktor_ladungen_rotated)

    # Schritt 5: Visualisiere die rotierte Faktorladungenmatrix
    plt.matshow(faktor_ladungen_rotated, cmap='coolwarm')
    plt.title('Rotierte Faktorladungen')
    plt.colorbar()
    plt.show()

    # Schritt 6: Interpretiere die Faktorladungen (welches Item lädt auf welchem Faktor)
    for i, faktor in enumerate(faktor_ladungen_rotated.T):
        print(f"Faktor {i + 1}:")
        for j, ladung in enumerate(faktor):
            if ladung > 0.4:  # Zeige nur Items mit hoher Ladung (>0.4)
                print(f"  - {data_filtered.columns[j]}: {ladung}")

    # Schritt 7: Verwende die extrahierten Faktoren in einer Regressionsanalyse
    # Beispiel: Berechne die Faktorscores für jeden Führungsstil (transformational, transactional, laissez-faire)
    factorscores = fa_rot.transform(data_filtered)

    # Berechne die Faktorscores für den "transformationalen Führungsstil" (z. B. Faktor 1)
    transformational_score = factorscores[:, 0]  # Faktor 1 für transformationalen Führungsstil
