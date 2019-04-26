from os.path import join, dirname
import pandas as pd

population_colors = pd.read_csv(join(dirname(__file__), '../data/colors.csv'))  # TODO add more colors


def create_bubble(source, pops, b_name, viz_df):
    pops = pops.append({'population_name': b_name,
                        'color': population_colors.loc[len(pops), 'color_name']},
                       ignore_index=True)
    selected = source.selected.indices
    viz_df.loc[selected, 'populationID'] = len(pops) - 1
    patches = {
        'populationID': [(i, len(pops) - 1) for i in selected]
    }
    source.patch(patches)
    return viz_df, pops
