import read_functions as rd
import calc_and_plot_functions as clc_plt

if __name__ == '__main__':
    data_filename = 'data/data_test430616_2025-01-19_21-03.csv'
    var_filename = 'data/values_test430616_2025-01-04_17-54.csv'

    variables = rd.read_variable_values_csv(var_filename)
    prob_leadership_styles, prob_motivation, benefit_amount = rd.read_data_from_questions(data_filename)

    # analyze relevant demographics of data
    gender, age, years, job = rd.read_demographic_data_relevant(data_filename)
    clc_plt.plot_demographics_pie(gender, age, years, job, variables)
    clc_plt.plot_validity_count(data_filename)

    # calculate and plot regressions

    # for all leadership styles vs. motivation
    # prepare data-carriers
    x = [None] * 11
    models = [None] * 11
    y_pred = [None] * 11
    names = ['TF', 'TF_IIB', 'TF_IIA', 'TF_IM', 'TF_IC', 'TF_IS', 'TA', 'TA_MBP', 'TA_MBA', 'TA_CR', 'LF']
    colors = ['blue', 'cornflowerblue', 'cornflowerblue', 'cornflowerblue', 'cornflowerblue', 'cornflowerblue', 'green',
              'limegreen', 'limegreen', 'limegreen', 'red']

    for i in range(len(x)):
        x[i], models[i], y_pred[i] = clc_plt.calc_regression(prob_leadership_styles[i], prob_motivation[1])
        clc_plt.plot_regression(x[i], prob_motivation[1], y_pred[i], names[i], colors[i],
                                "Mitarbeiter*innenmotivation [%]")

    # plot regressions coefficients and confidence intervals
    clc_plt.plot_regression_coeff_leadership_motivation(models, 'Führungsstile', names, colors)

    clc_plt.plot_correlation_leadership_benefits(prob_leadership_styles, benefit_amount, names)
