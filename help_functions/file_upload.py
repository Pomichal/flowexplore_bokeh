from io import StringIO, BytesIO
import base64
import pandas as pd
from os.path import join, dirname

population_colors = pd.read_csv(join(dirname(__file__), '../data/colors.csv'))  # TODO add more colors


def extract_file(file_source):
    filename = file_source.data['file_name'][0]
    raw_contents = file_source.data['file_contents'][0]

    # remove the prefix that JS adds
    prefix, b64_contents = raw_contents.split(",", 1)
    file_contents = base64.b64decode(b64_contents)
    file_io = StringIO(bytes.decode(file_contents))
    return file_io, filename


def file_callback_tree(file_source, dropdown_value, df_viz, tree, source):  # TODO file check

    file_io, filename = extract_file(file_source)
    df = pd.read_csv(file_io)
    if dropdown_value == 'coordinates':
        tree['coordinates'] = df
        df_viz['x'] = tree['coordinates'].iloc[:, 1].values
        df_viz['y'] = tree['coordinates'].iloc[:, 2].values
        df_viz['populationID'] = -1
        source.data = df_viz.to_dict(orient='list')

    elif dropdown_value == 'edges':
        tree['edges'] = df

    return df_viz, tree


def file_callback_populations(file_source, df_viz, source):  # TODO file check

    file_io, filename = extract_file(file_source)

    text = list(iter(file_io.getvalue().splitlines()))
    df_viz['populationID'] = -1
    populations = pd.DataFrame()
    for line in text:
        if line != "":
            split_line = line.split(":")
            pop_name = split_line[0]

            populations = populations.append({'population_name': pop_name,
                                              'color': population_colors.loc[len(populations), 'color_name']},
                                             ignore_index=True)
            indices = [int(a) for a in split_line[1].split(",")]

            df_viz.loc[indices, 'populationID'] = len(populations) - 1
            patches = {
                'populationID': [(i, len(populations) - 1) for i in indices]
            }
            source.patch(patches)

    return df_viz, populations, source
