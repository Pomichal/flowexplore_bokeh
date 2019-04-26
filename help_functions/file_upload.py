from io import StringIO, BytesIO
import base64
import pandas as pd
from bokeh.models import ColumnDataSource


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
    else:
        print("something went wrong, unknown dropdown value")  # TODO error message?
    return df_viz, tree
