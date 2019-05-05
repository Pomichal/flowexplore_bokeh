from os.path import join, dirname
import pandas as pd

population_colors = pd.read_csv(join(dirname(__file__), '../data/colors.csv'))


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


def select_patient(viz_df, source, patient_select_value, pat_df):
    if patient_select_value != 'None':
        if viz_df.empty:        # without tree structure
            viz_df = pat_df
            viz_df['populationID'] = -1
            source.data = viz_df.to_dict(orient='list')
        else:       # with tree structure
            drop_cols = map(lambda a: a if a not in ['x', 'y', 'populationID'] else None, viz_df.columns.tolist())
            viz_df.drop(columns=filter(lambda v: v is not None, drop_cols), inplace=True)
            viz_df = viz_df.join(pat_df)
            source.data = viz_df.to_dict(orient='list')
    else:
        if 'x' not in viz_df.columns.tolist():
            viz_df = pd.DataFrame()
            source.data = {}
        else:
            drop_cols = map(lambda a: a if a not in ['x', 'y', 'populationID'] else None, viz_df.columns.tolist())
            viz_df.drop(columns=filter(lambda v: v is not None, drop_cols), inplace=True)
    return viz_df
