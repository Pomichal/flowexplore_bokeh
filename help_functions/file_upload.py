from io import StringIO, BytesIO
import base64
import pandas as pd
from os.path import join, dirname

population_colors = pd.read_csv(join(dirname(__file__), '../data/colors.csv'))  # TODO add more colors


def extract_file(file_source, typ='txt'):
    filename = file_source.data['file_name'][0]
    raw_contents = file_source.data['file_contents'][0]

    # remove the prefix that JS adds
    prefix, b64_contents = raw_contents.split(",", 1)
    file_contents = base64.b64decode(b64_contents)
    file_io = StringIO(bytes.decode(file_contents)) if typ == 'txt' else BytesIO(file_contents)
    return file_io, filename


def file_callback_tree(file_source, dropdown_value, viz_df, tree, main_source):  # TODO file check

    file_io, filename = extract_file(file_source)
    df = pd.read_csv(file_io)
    if dropdown_value == 'coordinates':
        tree['coordinates'] = df
        viz_df['x'] = tree['coordinates'].iloc[:, 1].values
        viz_df['y'] = tree['coordinates'].iloc[:, 2].values
        viz_df['populationID'] = -1
        main_source.data = viz_df.to_dict(orient='list')

    elif dropdown_value == 'edges':
        tree['edges'] = df

    return viz_df, tree


def file_callback_populations(file_source, viz_df, main_source):  # TODO file check

    file_io, filename = extract_file(file_source)

    text = list(iter(file_io.getvalue().splitlines()))
    viz_df['populationID'] = -1
    pops = pd.DataFrame()
    for line in text:
        if line != "":
            split_line = line.split(":")
            pop_name = split_line[0]

            pops = pops.append({'population_name': pop_name,
                                'color': population_colors.loc[len(pops), 'color_name']},
                               ignore_index=True)
            indices = [int(a) for a in split_line[1].split(",")]

            viz_df.loc[indices, 'populationID'] = len(pops) - 1
            patches = {
                'populationID': [(i, len(pops) - 1) for i in indices]
            }
            main_source.patch(patches)

    return viz_df, pops, main_source


def file_callback_pat(file_source):  # TODO file check, upload population data

    file_io, filename = extract_file(file_source)
    df = pd.read_csv(file_io)
    name = filename.split(".")[0]
    if 'Unnamed: 0' in df.columns:  # TODO drop all Unnamed
        df.drop(columns=['Unnamed: 0'], inplace=True)
    return df, name


def file_callback_clinical(file_source):  # TODO file check

    file_io, filename = extract_file(file_source, typ='excel')

    clinical_data = pd.read_excel(file_io, header=[0, 1, 2])

    return clinical_data

