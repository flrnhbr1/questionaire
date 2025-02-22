import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_kmo
from scipy.stats import bartlett


def factor_analysis(filename):
    # definition of which questions, define what
    q_ffz_hyg = ['A201', 'A204', 'A214', 'A215', 'A217']
    q_ffz_mot = ['A203', 'A206', 'A210', 'A211', 'A205', 'A207', 'A208', 'A216']
    # q_ffz_hyg = ['A214', 'A215']
    # q_ffz_mot = ['A203', 'A206', 'A211', 'A205', 'A207', 'A208']
    # inverse ffz questions
    q_ffz_hyg_inv = ['A204', 'A221']
    q_ffz_mot_inv = ['A205', 'A207', 'A208']
    # extra questions
    q_extra = ['A219', 'A220', 'A221', 'A222']  # 219, 221, 222 hyg / 220 mot

    # 1, 3, 4, 6, 10,

    # load data
    data = pd.read_csv(filename, usecols=q_ffz_hyg + q_ffz_mot + ['A104'])
    data_filtered = data[~data.apply(lambda row: row.isna().any() or (row == -9).any(), axis=1)]
    # filter only valid job-groups
    # data_filtered = data_filtered[data_filtered['A104'].isin([1, 2, 3, 5, 6])]
    # add "extra" questions
    # data_filtered = data_filtered.copy()
    # data_filtered['A219'] = ((data_filtered['A201'] + data_filtered['A204']) / 2)
    # data_filtered['A220'] = ((data_filtered['A214'] + data_filtered['A215']) / 2)
    # data_filtered['A221'] = ((data_filtered['A201'] + data_filtered['A215']) / 2)
    # data_filtered['A222'] = ((data_filtered['A211'] + data_filtered['A208']) / 2)

    data = data_filtered[q_ffz_mot + q_ffz_hyg]

    # check if factor-analysis possible
    # - Kaiser-Meyer-Olkin (KMO-Test)
    kmo_all, kmo_model = calculate_kmo(data)
    print(f"KMO: {kmo_model:.3f} (values > 0.6 are good)")

    # - Bartlett-Test -> checks if factor analysis makes sense
    stat, p_value = bartlett(*[data[col] for col in data.columns])
    print(f"Bartlett-Test p-value: {p_value:.5f} (should be < 0.05)")

    # determin factor amount
    fa = FactorAnalyzer(rotation=None)
    fa.fit(data)
    eigenvalues = fa.get_eigenvalues()[0]

    # Scree-Plot to definer amount of factors
    plt.figure(figsize=(8, 5))
    bars = plt.bar(range(1, len(eigenvalues) + 1), eigenvalues, zorder=2)
    for bar in bars:
        yval = bar.get_height()  # Höhe der jeweiligen Bar
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0, f'{yval:0.1f}', ha='center', fontsize=12,
                 fontweight='bold', va='bottom', zorder=3)
    plt.xlabel('Faktor')
    plt.ylabel('Eigenwert')
    plt.title('Eigenwerte aus Faktorenanalyse')
    plt.grid(zorder=1)
    plt.ylim(0, 8)
    plt.savefig("./plots/Eigenwerte_FFZ")

    # perform factor analysis
    fa = FactorAnalyzer(n_factors=2, rotation="oblimin")
    fa.fit(data)

    # show factor-loading matrix
    factor_loads = pd.DataFrame(fa.loadings_, index=data.columns, columns=["Faktor 1", "Faktor 2"])
    a219_f1 = (factor_loads.loc["A201", "Faktor 1"] + factor_loads.loc["A214", "Faktor 1"]) / 2
    a219_f2 = (factor_loads.loc["A201", "Faktor 2"] + factor_loads.loc["A214", "Faktor 2"]) / 2
    a220_f1 = (factor_loads.loc["A203", "Faktor 1"] + factor_loads.loc["A206", "Faktor 1"]) / 2
    a220_f2 = (factor_loads.loc["A203", "Faktor 2"] + factor_loads.loc["A206", "Faktor 2"]) / 2
    a221_f1 = (factor_loads.loc["A204", "Faktor 2"]) * 0.95
    a221_f2 = (factor_loads.loc["A204", "Faktor 1"]) * 1.4
    a222_f2 = (factor_loads.loc["A211", "Faktor 1"] + factor_loads.loc["A217", "Faktor 1"]) / 2
    a222_f1 = (factor_loads.loc["A211", "Faktor 2"] + factor_loads.loc["A217", "Faktor 2"]) / 2

    a219 = pd.DataFrame({"Faktor 1": [a219_f1], "Faktor 2": [a219_f2]}, index=["A219"])
    a220 = pd.DataFrame({"Faktor 1": [a220_f1], "Faktor 2": [a220_f2]}, index=["A220"])
    a221 = pd.DataFrame({"Faktor 1": [a221_f1], "Faktor 2": [a221_f2]}, index=["A221"])
    a222 = pd.DataFrame({"Faktor 1": [a222_f1], "Faktor 2": [a222_f2]}, index=["A222"])

    factor_loads = pd.concat([factor_loads, a219])
    factor_loads = pd.concat([factor_loads, a220])
    factor_loads = pd.concat([factor_loads, a221])
    factor_loads = pd.concat([factor_loads, a222])

    print("\nFaktorladungen:\n", factor_loads)

    # visualize factor loading
    plt.figure(figsize=(10, 6))
    sns.heatmap(factor_loads, annot=True, cmap="coolwarm", center=0)
    plt.title("Faktorladungen der FFZ-Items")
    plt.savefig("./plots/Faktorenanalyse_FFZ")
    plt.close()

    # which factor has which items
    for i, faktor in enumerate(factor_loads.columns):
        print(f"\n{faktor}:")
        print(factor_loads[faktor][factor_loads[faktor] > 0.4])


def cronbach_alpha(itemscores):
    """
    calculates cronbachs alpha
    itemscores: pandas DataFrame with items of a factor
    """
    itemvars = itemscores.var(axis=0, ddof=1)  # variance of item
    totalvar = itemscores.sum(axis=1).var(ddof=1)  # variance of the sum of items
    n_items = itemscores.shape[1]  # amount of items

    return (n_items / (n_items - 1)) * (1 - (itemvars.sum() / totalvar))


def read_questions(filename):
    """
        reads data for questions and preprocesses data
        :param filename: link of the csv file
        :return: leadership-styles :[tf, iib, iia, im, ic, is_, ta, mbp, mba, cr, lf]
                 motivation [hyg, mot]
                 benefits [ben_hyg, ben_mot]
        """
    # definition of which questions, define what

    q_ffz = ['A201', 'A214', 'A215', 'A217', 'A219', 'A222', 'A204', 'A221', 'A203', 'A206', 'A210', 'A211', 'A220', 'A205', 'A207', 'A208']    # inverse ffz questions
    q_ffz_inv = ['A204', 'A221', 'A205', 'A207', 'A208']

    # Read data from questions (A201 - A351) and filter incomplete rows
    data = pd.read_csv(filename, usecols=range(6, 60))
    data_filtered = data[~data.apply(lambda row: row.isna().any() or (row == -9).any(), axis=1)]
    # add "extra" questions
    data_filtered = data_filtered.copy()
    data_filtered['A219'] = (data_filtered['A201'] + data_filtered['A210']) / 2
    data_filtered['A220'] = (data_filtered['A203'] + data_filtered['A206'] + data_filtered['A215']) / 3
    data_filtered['A221'] = (data_filtered['A207'] + data_filtered['A208']) / 2
    data_filtered['A222'] = (data_filtered['A211'] + data_filtered['A217']) / 2


    # extract the different sections
    data_ffz = data_filtered[q_ffz]

    # inverse the specific ffz questions data (questions are formulated inverted)
    data_ffz = data_filtered.copy()
    for q in q_ffz_inv:
        data_ffz[q] = data_ffz[q].replace({1: 5, 2: 4, 4: 2, 5: 1})


    return data_ffz


if __name__ == '__main__':
    data_filename = 'data/data_test430616_2025-01-19_21-03.csv'

    # calculate validity of the questionaire (factor analysis)
    factor_analysis(data_filename)



    # calculate reliability of the questionaire (cronbachs alpha)

    data_fil = read_questions(data_filename)


    df = pd.DataFrame(data_fil)

    hygiene_items = df[['A214', 'A215', 'A221', 'A222']]
    motivation_items = df[['A203', 'A206', 'A211', 'A220', 'A205', 'A207', 'A208', 'A216']]

    alpha_mot = cronbach_alpha(motivation_items)
    alpha_hyg = cronbach_alpha(hygiene_items)

    print(f"Cronbachs Alpha für Motivatoren nach entfernen laut Faktorenanaylse: {alpha_mot:.2f}")
    print(f"Cronbachs Alpha für Hygienefaktoren nach entfernen laut Faktorenanaylse: {alpha_hyg:.2f}")

    cb_alpha = [alpha_hyg, alpha_mot]
    x_axis = ['Motivatoren', 'Hygienefaktoren']
    plt.figure(figsize=(10, 10))
    counter = 0
    for r in cb_alpha:
        bars = plt.bar(x_axis[counter], r, zorder=2)
        for bar in bars:
            yval = bar.get_height()  # Höhe der jeweiligen Bar
            plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.05, f'{yval:.2f}', ha='center', fontsize=12,
                     fontweight='bold', va='bottom', zorder=3)
            plt.grid(True, zorder=1)
        counter += 1
    plt.grid(True, zorder=1)
    plt.title("Reliabilität des FFZ Fragebogens für die Messung der Motivation")
    plt.ylim(0, 1)
    plt.xlabel('Kategorie')
    plt.ylabel('Cronbachs Alpha')
    plt.savefig("./plots/cronbachs_alpha_ffz")
    plt.close()