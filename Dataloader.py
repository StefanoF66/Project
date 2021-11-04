import pandas as pd


def biotocsv(filename,output_name):
    '''
    this function convert a bio file to a csv file
    '''
    with open(filename) as f:
        data_list = list(line.rstrip('\n') for line in f)
        list_of_words = []
        list_of_tags = []
        list_of_sentence_numbers = []
        count = 0

        for i in range(len(data_list) - 1):
            if data_list[i] == '':
                count = count + 1
            else:
                list_of_sentence_numbers.append(count)

        for i in range(len(data_list)-1):
            if data_list[i] == '':
                continue
            elif data_list[i] == '"':
                dataframe = data_list[i].split()
                list_of_words.append('\"'.join(dataframe[:-1]))
                list_of_tags.append(dataframe[-1])
            else:
                dataframe = data_list[i].split()
                list_of_words.append(' '.join(dataframe[:-1]))
                list_of_tags.append(dataframe[-1])

        output = pd.DataFrame(list(zip(list_of_words, list_of_tags, list_of_sentence_numbers)),
                      columns=['Word', 'Tag', 'Sentence #'])
        output.to_csv(output_name, index=False)


def loading_data(csv_test,csv_val,csv_train):
    """
    A function that takes 3 csv files in input and returns 3 dataframes
    inputs:
    :param csv_test,csv_val,csv_train: 3 csv files
    :return: df_test, df_val, df_train dataframes
    """
    df_test = pd.read_csv(csv_test, encoding="utf-8", sep="\t", doublequote=False, quoting=3).fillna(method="ffill")
    df_val = pd.read_csv(csv_val, encoding="utf-8", sep="\t", doublequote=False, quoting=3).fillna(method="ffill")
    df_train = pd.read_csv(csv_train, encoding="utf-8", sep="\t", doublequote=False, quoting=3).fillna(method="ffill")
    return df_test, df_val, df_train

def dictionary_builder(df_test, df_val, df_train):
    """
    A function that takes 3 dataframes and build the dictionary of their tags
    inputs:
    :param df_test, df_val, df_train: 3 dataframes
    :return:
    tag_list, list of tags with PAD
    """
    tags_list = list(set(df_test["Tag"].values).union(set(df_val["Tag"].values), set(df_train["Tag"].values)))
    tags_list.append("PAD")
    tags_enum = {t: i for i, t in enumerate(tags_list)}
    return tags_list, tags_enum
