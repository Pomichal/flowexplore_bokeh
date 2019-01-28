import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap


def prepare_data(df_patient, df_coordinates):
    """
    creates a dataframe with patient data and [x,y] coordinates for each cluster.

    :param df_patient: dataframe with patients data for each cluster
    :param df_coordinates: dataframe with coordinates for clusters
    """
    df_patient['x'] = df_coordinates['layout.V1']
    df_patient['y'] = df_coordinates['layout.V2']
    data = df_patient.drop('Unnamed: 0', axis=1)

    # rename attributes example
    # data.rename(index=str, columns={"csv.pbm_038.count": "count"}, inplace=True)

    # columns = sorted(data.columns)
    # discrete = [x for x in columns if data[x].dtype == object]
    # continuous = [x for x in columns if x not in discrete]

    return data  # , columns  # , discrete, continuous


def create_color_map():

    c_dict = {'red': ((0.0, 0.0, 0.0),
                     (0.25, 0.0, 0.0),
                     (0.5, 0.8, 1.0),
                     (0.75, 1.0, 1.0),
                     (1.0, 0.4, 1.0)),

             'green': ((0.0, 0.0, 0.0),
                       (0.25, 0.0, 0.0),
                       (0.5, 0.9, 0.9),
                       (0.75, 0.0, 0.0),
                       (1.0, 0.0, 0.0)),

             'blue': ((0.0, 0.0, 0.4),
                      (0.25, 1.0, 1.0),
                      (0.5, 1.0, 0.8),
                      (0.75, 0.0, 0.0),
                      (1.0, 0.0, 0.0))
             }

    blue_red1 = LinearSegmentedColormap('BlueRed1', c_dict)

    colors = []
    for c in range(0, 256):
        rgb = blue_red1(c)[:3]  # will return rgba, we take only first 3 so we get rgb
        colors.append(mpl.colors.rgb2hex(rgb))

    return colors


def scale(old_value, old_min, old_max, new_min=10, new_max=40):
    """
    Scale data into range new_min, new_max
    :param old_value:
    :param old_min:
    :param old_max:
    :param new_min:
    :param new_max:
    :return: scaled value
    """
    old_range = (old_max - old_min)
    new_range = (new_max - new_min)
    return (((old_value - old_min) * new_range) / old_range) + new_min
