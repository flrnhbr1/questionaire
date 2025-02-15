import csv
import pandas as pd


def read_data_from_questions(filename):
    """
        reads data for questions and preprocesses data
        :param filename: link of the csv file
        :return: leadership-styles :[tf, iib, iia, im, ic, is_, ta, mbp, mba, cr, lf]
                 motivation [hyg, mot]
                 benefits [ben_hyg]
        """
    # definition of which questions, define what
    q_ffz_hyg = ['A201', 'A214', 'A215', 'A217', 'A219', 'A222', 'A204', 'A221']
    q_ffz_mot = ['A203', 'A206', 'A210', 'A211', 'A220', 'A205', 'A207', 'A208']
    # inverse ffz questions
    q_ffz_hyg_inv = ['A204', 'A221']
    q_ffz_mot_inv = ['A205', 'A207', 'A208']
    # questions transformational
    q_mlq_tf = ['A306', 'A308', 'A310', 'A315', 'A318', 'A319', 'A321', 'A326', 'A329', 'A330', 'A331', 'A334', 'A336',
                'A341', 'A347']
    q_mlq_iib = ['A306', 'A308', 'A334', 'A341']
    q_mlq_iia = ['A310', 'A318', 'A321']
    q_mlq_im = ['A315', 'A326', 'A336']
    q_mlq_ic = ['A319', 'A329', 'A331']
    q_mlq_is = ['A330', 'A347']
    # questions transactional
    q_mlq_ta = ['A303', 'A311', 'A312', 'A316', 'A322', 'A327', 'A335'] #303, 312, evtl 322,
    q_mlq_mbp = ['A303', 'A312']
    q_mlq_mba = ['A322', 'A327']
    q_mlq_cr = ['A311', 'A316', 'A335']
    # questions laissez faire
    q_mlq_lf = ['A307', 'A328']
    # questions benefits
    q_ben_hyg = ['A218_01', 'A218_02', 'A218_04', 'A218_05', 'A218_08']
    q_ben_mot = ['A218_03', 'A218_06', 'A218_07']

    # Read data from questions (A201 - A351) and filter incomplete rows
    data = pd.read_csv(filename, usecols=range(6, 60))
    data_filtered = data[~data.apply(lambda row: row.isna().any() or (row == -9).any(), axis=1)]

    # add "extra" questions
    data_filtered = data_filtered.copy()
    data_filtered['A219'] = (data_filtered['A201'] + data_filtered['A210']) / 2
    data_filtered['A220'] = (data_filtered['A203'] + data_filtered['A206'] + data_filtered['A215']) / 2
    data_filtered['A221'] = (data_filtered['A207'] + data_filtered['A208']) / 2
    data_filtered['A222'] = (data_filtered['A211'] + data_filtered['A217']) / 2

    # extract the different sections
    data_ffz_hyg = data_filtered[q_ffz_hyg]
    data_ffz_mot = data_filtered[q_ffz_mot]

    data_mlq_tf = data_filtered[q_mlq_tf]
    data_mlq_iib = data_filtered[q_mlq_iib]
    data_mlq_iia = data_filtered[q_mlq_iia]
    data_mlq_im = data_filtered[q_mlq_im]
    data_mlq_ic = data_filtered[q_mlq_ic]
    data_mlq_is = data_filtered[q_mlq_is]


    data_mlq_ta = data_filtered[q_mlq_ta]
    data_mlq_mbp = data_filtered[q_mlq_mbp]
    data_mlq_mba = data_filtered[q_mlq_mba]
    data_mlq_cr = data_filtered[q_mlq_cr]

    data_mlq_lf = data_filtered[q_mlq_lf]

    data_ben_hyg = data_filtered[q_ben_hyg]
    data_ben_mot = data_filtered[q_ben_mot]


    # preprocess benefit data (1 = no, 2 = yes)
    data_ben_hyg = data_ben_hyg.copy()
    for q in q_ben_hyg:
        data_ben_hyg[q] = data_ben_hyg[q].replace({1: 0, 2: 1})

    data_ben_mot = data_ben_mot.copy()
    for q in data_ben_mot:
        data_ben_mot[q] = data_ben_mot[q].replace({1: 0, 2: 1})


    # inverse lf data (questions are formulated inverted)
    data_mlq_lf = data_mlq_lf.copy()
    data_mlq_lf['A307'] = data_mlq_lf['A307'].replace({1: 5, 2: 4, 4: 2, 5: 1})
    data_mlq_lf['A328'] = data_mlq_lf['A328'].replace({1: 5, 2: 4, 4: 2, 5: 1})

    # inverse the specific ffz questions data (questions are formulated inverted)
    data_ffz_hyg = data_ffz_hyg.copy()
    for q in q_ffz_hyg_inv:
        data_ffz_hyg[q] = data_ffz_hyg[q].replace({1: 5, 2: 4, 4: 2, 5: 1})

    data_ffz_mot = data_ffz_mot.copy()
    for q in q_ffz_mot_inv:
        data_ffz_mot[q] = data_ffz_mot[q].replace({1: 5, 2: 4, 4: 2, 5: 1})


    # calc per person and normalize (0-5)
    hyg = data_ffz_hyg.sum(axis=1) / len(q_ffz_hyg)
    mot = data_ffz_mot.sum(axis=1) / len(q_ffz_mot)

    tf = data_mlq_tf.sum(axis=1) / len(q_mlq_tf)
    iib = data_mlq_iib.sum(axis=1) / len(q_mlq_iib)
    iia = data_mlq_iia.sum(axis=1) / len(q_mlq_iia)
    im = data_mlq_im.sum(axis=1) / len(q_mlq_im)
    ic = data_mlq_ic.sum(axis=1) / len(q_mlq_ic)
    is_ = data_mlq_is.sum(axis=1) / len(q_mlq_is)

    ta = data_mlq_ta.sum(axis=1) / len(q_mlq_ta)
    mbp = data_mlq_mbp.sum(axis=1) / len(q_mlq_mbp)
    mba = data_mlq_mba.sum(axis=1) / len(q_mlq_mba)
    cr = data_mlq_cr.sum(axis=1) / len(q_mlq_cr)

    lf = data_mlq_lf.sum(axis=1) / len(q_mlq_lf)

    ben_hyg = data_ben_hyg.sum(axis=1)
    ben_mot = data_ben_mot.sum(axis=1)

    return [tf, iib, iia, im, ic, is_, ta, mbp, mba, cr, lf], [hyg, mot], [ben_hyg, ben_mot]


def read_variable_values_csv(filename):
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

    # columns
    var = []
    response = []
    meaning = []

    # iterate over datasets
    for row in rows:
        var.append(row[fields.index('VAR')])
        response.append(row[fields.index('RESPONSE')])
        meaning.append(row[fields.index('MEANING')])
    variables_table = [var, response, meaning]
    return variables_table


def read_demographic_data_relevant(filename):
    # initializing the titles and rows list
    fields = []
    rows = []
    considered_data_count = 0

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

    # printing the field names
    # print('Field names are:' + ', '.join(field for field in fields))

    # variables
    gender = [0, 0, 0, 0]
    age = [0, 0, 0, 0, 0, 0, 0]
    years = [0, 0, 0, 0, 0, 0]
    job = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    # iterate over datasets
    for row in rows:
        # only consider finished datasets
        if row[fields.index('FINISHED')] == '1' and '-9' not in row:

            # only consider relevant jobs
            a104 = int(row[fields.index('A104')])
            if a104 == 1 or a104 == 2 or a104 == 3 or a104 == 5 or a104 == 6:

                # increment counter
                considered_data_count += 1

                # A101 Geschlecht
                i = int(row[fields.index('A101')])
                gender[i - 1] += 1

                # A102 Alter
                i = int(row[fields.index('A102')])
                age[i - 1] += 1

                # A103 Zugehörigkeit
                i = int(row[fields.index('A103')])
                years[i - 1] += 1

                # A104 Beruf
                i = int(row[fields.index('A104')])
                job[i - 1] += 1

    # delete empty object
    job = [job[0], job[1], job[2], job[4], job[5]]
    return gender, age, years, job


