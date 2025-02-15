import read_functions as rd
import calc_and_plot_functions as clc_plt


if __name__ == '__main__':
    data_filename = 'data/data_test430616_2025-01-19_21-03.csv'
    var_filename = 'data/values_test430616_2025-01-04_17-54.csv'

    variables = rd.read_variable_values_csv(var_filename)
    prob_leadership_styles, prob_motivation, prob_benefits = rd.read_data_from_questions(data_filename)

    # hyg_per_person
    # mot_per_person, prob_tf_per_person, prob_ta_per_person, prob_lf_per_person, benef_hyg, benef_mot, \
    #     prob_mbp_per_person, prob_mba_per_person, prob_cr_per_person = rd.read_data_from_questions(data_filename)


    # analyze relevant demographics of data
    gender, age, years, job = rd.read_demographic_data_relevant(data_filename)
    clc_plt.plot_demographics_pie(gender, age, years, job, variables)
    clc_plt.plot_validity_count(data_filename)

    # calculate and plot regressions

    # transformational
    x = [None] * 11
    models = [None] * 11
    y_pred = [None] * 11
    names = ['TF', 'TF_IIB', 'TF_IIA', 'TF_IM', 'TF_IC', 'TF_IS', 'TA', 'TA_MBP', 'TA_MBA', 'TA_CR', 'LF']
    colors = ['blue', 'cornflowerblue', 'cornflowerblue', 'cornflowerblue', 'cornflowerblue', 'cornflowerblue', 'green',
              'limegreen', 'limegreen', 'limegreen', 'red']

    # for all leadership styles
    for i in range(len(x)):
        x[i], models[i], y_pred[i] = clc_plt.calc_regression(prob_leadership_styles[i], prob_motivation[1])
        clc_plt.plot_regression_leadership_style(x[i], prob_motivation[1], y_pred[i], names[i], colors[i])


    # for all benefits
    x_ben_hyg, model_ben_hyg, y_pred_ben_hyg = clc_plt.calc_regression(prob_benefits[0], prob_motivation[0])
    clc_plt.plot_regression_benefits(x_ben_hyg, prob_motivation[0], y_pred_ben_hyg, 'Hygienefaktoren', 'gold')

    x_ben_mot, model_ben_mot, y_pred_ben_mot = clc_plt.calc_regression(prob_benefits[1], prob_motivation[1])
    clc_plt.plot_regression_benefits(x_ben_mot, prob_motivation[1], y_pred_ben_mot, 'Motivatoren', 'brown')

    # plot regressions coefficients and confidence intervals
    clc_plt.plot_regression_coefficients_leadership_styles(models, 'Führungsstile', names, colors)

    #
    # clc_plt.plot_regression_coefficients_benefits(model_ben_hyg, model_ben_mot)
    # print(model_tf.summary())
    # print(model_ta.summary())
    # print(model_lf.summary())
