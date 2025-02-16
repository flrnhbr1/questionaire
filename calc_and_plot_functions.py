import csv
import statsmodels.api as sm
import statsmodels.api as sm_cor
import numpy as np
import matplotlib.pyplot as plt


# calculation functions
def calc_regression(x, y):
    # calc regression
    x = sm.add_constant(x)
    model = sm.OLS(y, x).fit()
    y_pred = model.predict(x)

    return x, model, y_pred


# plotting functions
def plot_demographics_pie(geschlecht, alter, betriebszugehoerigkeit, beruf, var):
    x_gender = [var[2][0], var[2][1], var[2][2], var[2][3]]
    x_age = [var[2][5], var[2][6], var[2][7], var[2][8], var[2][9], var[2][10], var[2][11]]
    x_years = [var[2][13], var[2][14], var[2][15], var[2][16], var[2][17], var[2][18]]
    x_job = [var[2][20], var[2][21], var[2][22], var[2][23], var[2][24], var[2][25], var[2][26], var[2][27],
             var[2][28]]

    gesamt_anzahl = sum(geschlecht)
    geschlecht = np.round((np.array(geschlecht)/gesamt_anzahl)*100, 2)
    alter = np.round((np.array(alter)/gesamt_anzahl)*100, 2)
    betriebszugehoerigkeit = np.round((np.array(betriebszugehoerigkeit)/gesamt_anzahl)*100, 2)
    beruf = np.round((np.array(beruf)/gesamt_anzahl)*100, 2)

    plt.figure(figsize=(20, 10))
    plt.pie(geschlecht, startangle=90)
    labels = [f'{l}:  {s:0.1f}%' for l, s in zip(x_gender, geschlecht)]
    plt.legend(labels)
    plt.title('Geschlecht (n=' + str(gesamt_anzahl) + ')')
    plt.savefig("./plots/_demographics_gender")
    plt.close()

    plt.figure(figsize=(20, 10))
    plt.pie(alter, startangle=90)
    labels = [f'{l}: {s:0.1f}%' for l, s in zip(x_age, alter)]
    plt.legend(labels)
    plt.title('Alter (n=' + str(gesamt_anzahl) + ')')
    plt.savefig("./plots/_demographics_age")
    plt.close()

    plt.figure(figsize=(20, 10))
    plt.pie(betriebszugehoerigkeit, startangle=90)
    labels = [f'{l}: {s:0.1f}%' for l, s in zip(x_years, betriebszugehoerigkeit)]
    plt.legend(labels)
    plt.title('Betriebszugehörigkeit (n=' + str(gesamt_anzahl) + ')')
    plt.savefig("./plots/_demographics_years_in_company")
    plt.close()

    plt.figure(figsize=(20, 10))
    plt.pie(beruf, startangle=90)
    labels = [f'{l}: {s:0.1f}%' for l, s in zip(x_job, beruf)]
    plt.legend(labels)
    plt.title('Beruf (n=' + str(gesamt_anzahl) + ')')
    plt.savefig("./plots/_demographics_job")
    plt.close()


def plot_validity_count(filename):
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

    not_finished = wrong_job = valid = 0

    # iterate over datasets
    for row in rows:
        # is valid?
        if row[fields.index('FINISHED')] == '1' and '-9' not in row:

            # is correct jobb?
            a104 = int(row[fields.index('A104')])
            if a104 == 1 or a104 == 2 or a104 == 3 or a104 == 5 or a104 == 6:
                valid += 1
            else:
                wrong_job += 1

        else:
            not_finished += 1

    values = np.array([not_finished, wrong_job, valid])
    percentages = (values/np.sum(values))*100
    types = ['Unvollständig ausgefüllt', 'Nicht inkludierte Berufsgruppe', 'Gültig']
    plt.figure(figsize=(20, 10))
    plt.pie(values, autopct=lambda p: f'{int(p/100*sum(values))}', startangle=90)
    labels = [f'{l}:  {s:0.1f}%' for l, s in zip(types, percentages)]
    plt.legend(labels)
    plt.title('Fragebogen-Gültigkeit (n=' + str(not_finished+valid+wrong_job) + ')')
    # plt.show()
    plt.savefig("./plots/_validity")
    plt.close()



def plot_regression(x_plt, y_plt, y_pred_plt, name, col, y_label):

    # Plot
    fig = plt.figure(figsize=(10, 10))
    plt.scatter(x_plt[0], y_plt, color=col, label="Datenpunkte", zorder=2)
    plt.plot(x_plt[0], y_pred_plt, color='black', label="Regressionslinie", zorder=2, linewidth=4)
    plt.xlabel("Warscheinlichkeit für " + name + " Führungsstil [%]")
    plt.ylabel(y_label)
    plt.legend()
    plt.grid(True, zorder=1)

    # plt.show()
    plt.savefig("./plots/regression_" + str(name) + "_" + str(y_label))
    plt.close()


def plot_regression_coeff_leadership_styles(models, name, x_axis_vector, cols):
    R_squared_adj = []
    plt.figure(figsize=(10, 10))
    for i in range(len(models)):
        model = models[i]
        # get coefficients
        beta1 = model.params[0] * 100

        # get confidence intervals for beta 1
        conf_intervals = model.conf_int(0.05).iloc[1]

        # limits of confidence intervalls
        lower_bound_beta1 = conf_intervals[0] * 100
        upper_bound_beta1 = conf_intervals[1] * 100

        # to shift the text
        # bias_text_hor = 0.15
        # if i == len(x_axis_vector) - 1:
        #     bias_text_hor -= 0.3
        bias_text_hor = 0.0
        bias_text_ver_reg = 4
        bias_text_ver_conf = 1
        plt.errorbar(x=x_axis_vector[i], y=beta1, yerr=[[beta1 - lower_bound_beta1], [upper_bound_beta1 - beta1]],
                     fmt='o', color=cols[i], ecolor=cols[i], elinewidth=2, capsize=5)
        # name regression coefficient
        plt.text(i + bias_text_hor, beta1+bias_text_ver_reg, f"{beta1:.2f}" + "%", ha='center', va='center',
                 fontsize=10, color='black', zorder=3, bbox=dict(facecolor='white', edgecolor=cols[i], boxstyle='round,pad=0.3'))
        # name confidence intervall limits
        plt.text(i + bias_text_hor, lower_bound_beta1-bias_text_ver_conf, f"{lower_bound_beta1:.2f}" + "%", ha='center', va='top',
                 fontsize=8, color=cols[i])
        plt.text(i + bias_text_hor, upper_bound_beta1+bias_text_ver_conf , f"{upper_bound_beta1:.2f}" + "%", ha='center', va='bottom',
                 fontsize=8, color=cols[i])

        # for model quality
        R_squared_adj.append(np.round(model.rsquared_adj * 100, 2))

    plt.ylim(-100, 130)
    plt.title('Regressionskoeffizienten mit 95% Konfidenzintervallen')
    plt.xlabel('Führungsstil')
    plt.ylabel('Einfluss auf die Motivation [%]')
    plt.grid(True)
    # plt.xticks(rotation=90)
    plt.savefig("./plots/conf_intervals_beta1_" + str(name) + ".png")
    plt.close()

    # plot model quality
    plt.figure(figsize=(10, 10))
    counter = 0
    for r in R_squared_adj:
        bars = plt.bar(x_axis_vector[counter], r, zorder=2, color=cols[counter])
        for bar in bars:
            yval = bar.get_height()  # Höhe der jeweiligen Bar
            plt.text(bar.get_x() + bar.get_width() / 2, yval + 1, f'{yval}' + '%', ha='center', fontsize=12,
                     fontweight='bold', va='bottom', zorder=3)
            plt.grid(True, zorder=1)
        counter += 1
    plt.grid(True, zorder=1)
    plt.title("Modellqualität (Adjusted R²)")
    plt.ylim(0, 100)
    plt.xlabel('Führungsstil')
    plt.ylabel('Adjusted R² [%]')
    plt.savefig("./plots/R_squared_" + str(name) + ".png")
    plt.close()


def plot_regression_coeff_benefits(models, name, x_axis_vector, cols):
    R_squared_adj = []
    plt.figure(figsize=(10, 10))
    for i in range(len(models)):
        model = models[i]
        # get coefficients
        beta1 = model.params[0] * 100

        # get confidence intervals for beta 1
        conf_intervals = model.conf_int(0.05).iloc[1]

        # limits of confidence intervalls
        lower_bound_beta1 = conf_intervals[0] * 100
        upper_bound_beta1 = conf_intervals[1] * 100

        # to shift the text
        # bias_text_hor = 0.15
        # if i == len(x_axis_vector) - 1:
        #     bias_text_hor -= 0.3
        bias_text_hor = 0.0
        bias_text_ver_reg = 4
        bias_text_ver_conf = 1
        plt.errorbar(x=x_axis_vector[i], y=beta1, yerr=[[beta1 - lower_bound_beta1], [upper_bound_beta1 - beta1]],
                     fmt='o', color=cols[i], ecolor=cols[i], elinewidth=2, capsize=5)
        # name regression coefficient
        plt.text(i + bias_text_hor, beta1+bias_text_ver_reg, f"{beta1:.2f}" + "%", ha='center', va='center',
                 fontsize=10, color='black', zorder=3, bbox=dict(facecolor='white', edgecolor=cols[i], boxstyle='round,pad=0.3'))
        # name confidence intervall limits
        plt.text(i + bias_text_hor, lower_bound_beta1-bias_text_ver_conf, f"{lower_bound_beta1:.2f}" + "%", ha='center', va='top',
                 fontsize=8, color=cols[i])
        plt.text(i + bias_text_hor, upper_bound_beta1+bias_text_ver_conf , f"{upper_bound_beta1:.2f}" + "%", ha='center', va='bottom',
                 fontsize=8, color=cols[i])

        # for model quality
        R_squared_adj.append(np.round(model.rsquared_adj * 100, 2))

    plt.ylim(-100, 130)
    plt.title('Regressionskoeffizienten mit 95% Konfidenzintervallen')
    plt.xlabel('Führungsstil')
    plt.ylabel('Einfluss auf die Anreize [%]')
    plt.grid(True)
    # plt.xticks(rotation=90)
    plt.savefig("./plots/conf_intervals_beta1_" + str(name) + ".png")
    plt.close()

    # plot model quality
    plt.figure(figsize=(10, 10))
    counter = 0
    for r in R_squared_adj:
        bars = plt.bar(x_axis_vector[counter], r, zorder=2, color=cols[counter])
        for bar in bars:
            yval = bar.get_height()  # Höhe der jeweiligen Bar
            plt.text(bar.get_x() + bar.get_width() / 2, yval + 1, f'{yval}' + '%', ha='center', fontsize=12,
                     fontweight='bold', va='bottom', zorder=3)
            plt.grid(True, zorder=1)
        counter += 1
    plt.grid(True, zorder=1)
    plt.title("Modellqualität (Adjusted R²)")
    plt.ylim(0, 100)
    plt.xlabel('Führungsstil')
    plt.ylabel('Adjusted R² [%]')
    plt.savefig("./plots/R_squared_" + str(name) + ".png")
    plt.close()



# #def plot_regression_coefficients_benefits(model_hyg, model_mot):
#     x_axis_vector = ['Hygienefaktoren', 'Motivatoren']
#     name = 'Anreize'
#     R_squared_adj = []
#     plt.figure(figsize=(10, 10))
#     counter = 0
#     for i in range(2):
#         model = model_hyg
#         # workaround for iteration over models
#         if i == 0:
#             model = model_hyg
#         if i == 1:
#             model = model_mot
#
#         # get coefficients
#         beta1 = model.params[0] * 100
#
#         # get confidence intervals for beta 1
#         conf_intervals = model.conf_int(0.05).iloc[1]
#
#         # limits of confidence intervalls
#         lower_bound_beta1 = conf_intervals[0] * 100
#         upper_bound_beta1 = conf_intervals[1] * 100
#
#         # to shift the text
#         bias_text = 0.15
#         if counter == len(x_axis_vector) - 1:
#             bias_text -= 0.3
#
#         plt.errorbar(x=x_axis_vector[counter], y=beta1, yerr=[[beta1 - lower_bound_beta1], [upper_bound_beta1 - beta1]],
#                      fmt='o', color='b', ecolor='r', elinewidth=2, capsize=5)
#         # name regression coefficient
#         plt.text(counter + bias_text, beta1, f"{beta1:.2f}" + "%", ha='center', va='center', fontsize=10, color='blue')
#         # name confidence intervall limits
#         plt.text(counter + bias_text, lower_bound_beta1, f"{lower_bound_beta1:.2f}" + "%", ha='center', va='top',
#                  fontsize=8, color='red')
#         plt.text(counter + bias_text, upper_bound_beta1, f"{upper_bound_beta1:.2f}" + "%", ha='center', va='bottom',
#                  fontsize=8, color='red')
#
#         # for model quality
#         R_squared_adj.append(np.round(model.rsquared_adj * 100, 2))
#         counter += 1
#
#
#     plt.ylim(-100, 100)
#     plt.title('Regressionskoeffizienten mit 95% Konfidenzintervallen')
#     plt.xlabel('Unabhängige Variable')
#     plt.ylabel('Koeffizient / Einfluss auf die Zufriedenheit/Motivation [%]')
#     plt.grid(True)
#     # plt.xticks(rotation=90)
#     plt.savefig("./plots/conf_intervals_beta1_" + str(name) + ".png")
#     plt.close()
#
#     # plot model quality
#     plt.figure(figsize=(10, 10))
#     counter = 0
#     for r in R_squared_adj:
#         bars = plt.bar(x_axis_vector[counter], r, zorder=2)
#         for bar in bars:
#             yval = bar.get_height()  # Höhe der jeweiligen Bar
#             plt.text(bar.get_x() + bar.get_width() / 2, yval + 1, f'{yval}' + '%', ha='center', fontsize=12,
#                      fontweight='bold',
#                      va='bottom', zorder=3)
#             plt.grid(True, zorder=1)
#         counter += 1
#     plt.grid(True, zorder=1)
#     plt.title("Modellqualität (Adjusted R²)")
#     plt.ylim(0, 100)
#     plt.xlabel('Unabhängige Variable')
#     plt.ylabel('[%]')
#     plt.savefig("./plots/R_squared_" + str(name) + ".png")
#     plt.close()
