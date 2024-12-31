import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from itertools import combinations

stats = pd.read_csv('PlayerStats.csv')
players = stats.loc[:, ['season', 'week', 'tm_alias', 'opp_alias', 'player', 'started', 'rush_yds', 'rush_att', 'targets', 'rec', 'rec_yds', 
                        'pass_cmp', 'pass_att', 'pass_yds', 'pass_cmp_pct', 'scoring', 'pass_tds', 'pass_int', 'rush_tds', 'rec_tds', 'fumbles_lost', 
                        'two_pt_md', 'fumbles_rec_td', 'kick_ret_tds', 'punt_ret_tds']]
players = players[players['season'] > 1999]
players = players.rename(columns={'scoring': 'kick_pts'})

roster = pd.read_csv('SeasonRoster.csv')
roster = roster.loc[:, ['season', 'player', 'position', 'pfr_approximate_value']]
roster['position'] = roster['position'].replace('TE', 'WR')
roster = roster[roster['position'].isin(['RB', 'WR', 'QB', 'K'])]
players = pd.merge(players, roster, on=['season', 'player'], how='inner')
players = players[(players['started'] == 1) | (players['position'] == 'K')]

odds = pd.read_csv('games.csv')
odds['tm_alias'] = odds['home_team'].replace({'JAX': 'JAC', 'STL': 'LA', 'OAK': 'LV', 'SD': 'LAC'})
odds['opp_alias'] = odds['away_team'].replace({'JAX': 'JAC', 'STL': 'LA', 'OAK': 'LV', 'SD': 'LAC'})
odds = odds.loc[:, ['season', 'week', 'spread_line', 'tm_alias', 'opp_alias']]
odds2 = odds.rename(columns={'tm_alias': 'opp_alias', 'opp_alias': 'tm_alias'})
odds2['spread_line'] = odds2['spread_line'] * -1
odds = pd.concat([odds, odds2])
players = pd.merge(players, odds, on=['season', 'week', 'tm_alias', 'opp_alias'], how='left')

players['pass_fan'] = ((0.04 * players['pass_yds'].fillna(0)) + (4 * players['pass_tds'].fillna(0)) + (-1 * players['pass_int'].fillna(0)) +
                   (0.1 * players['rush_yds'].fillna(0)) + (6 * players['rush_tds'].fillna(0)) + (0.1 * players['rec_yds'].fillna(0)) + 
                   (6 * players['rec_tds'].fillna(0)) + (1 * players['rec'].fillna(0)) + (-1 * players['fumbles_lost'].fillna(0)) + 
                   (2 * players['two_pt_md'].fillna(0)) + (6 * players['fumbles_rec_td'].fillna(0)) + (6 * players['kick_ret_tds'].fillna(0)) + 
                   (6 * players['punt_ret_tds'].fillna(0)))
players['pass_fan'] = players['pass_fan'][players['pass_fan'] != 0]
players['rec_fan'] = players['pass_fan']
players['rush_fan'] = players['pass_fan']
players['all_tds'] = players['pass_tds'].fillna(0) + players['rush_tds'].fillna(0) + players['rec_tds'].fillna(0)

# pass_stats = ['pass_yds', 'pass_att', 'pass_cmp', 'pass_fan']
# rec_stats = ['targets', 'rec', 'rec_yds', 'rec_fan']
# rush_stats = ['rush_yds', 'rush_att', 'rush_fan']
# kick_stats = ['kick_pts']
pass_stats = ['pass_yds']
rec_stats = ['rec']
rush_stats = ['rush_att']
kick_stats = ['kick_pts']

def get_EV(players, pass_stat, rec_stat, rush_stat, kick_stat):
    players = players[~((players['position'] == 'QB') & (players[pass_stat].isna()))]
    players = players[~((players['position'] == 'WR') & (players[rec_stat].isna()))]
    players = players[~((players['position'] == 'RB') & (players[rush_stat].isna()))]
    players = players[~((players['position'] == 'K') & (players[kick_stat].isna()))]

    median_stats = players.groupby(['season', 'player']).agg({
        pass_stat: 'median',
        rec_stat: 'median',
        rush_stat: 'median',
        kick_stat: 'median'
    }).reset_index()

    players = pd.merge(players, median_stats, on=['season', 'player'], suffixes=('', '_median'))

    players['over'] = None
    for position, yard_type in [('QB', pass_stat), ('WR', rec_stat), ('RB', rush_stat), ('K', kick_stat)]:
        mask = (players['position'] == position) & (players[yard_type] > players[yard_type + '_median'])
        players.loc[mask, 'over'] = True
        mask = (players['position'] == position) & (players[yard_type] < players[yard_type + '_median'])
        players.loc[mask, 'over'] = False
        mask = (players['position'] == position) & (players[yard_type] == players[yard_type + '_median']) & (players['spread_line'] < 0)
        players.loc[mask, 'over'] = True
        mask = (players['position'] == position) & (players[yard_type] == players[yard_type + '_median']) & (players['spread_line'] > 0)
        players.loc[mask, 'over'] = False
        mask = (players['position'] == position) & (players[yard_type] == players[yard_type + '_median']) & (players['spread_line'] == 0)
        players.loc[mask, 'over'] = 'remove'

    players = players[players['over'] != 'remove']

    position_counts = players.groupby(['season', 'week', 'tm_alias', 'position']).size().unstack(fill_value=0)
    valid_games = position_counts[(position_counts['QB'] > 0) & (position_counts['WR'] > 0) & (position_counts['RB'] > 0) & (position_counts['K'] > 0)]
    players = players.set_index(['season', 'week', 'tm_alias']).loc[valid_games.index].reset_index()

    # Only keep one of each position
    players_ft = players.sort_values('pfr_approximate_value', ascending=False).drop_duplicates(['season', 'week', 'tm_alias', 'position'])

    players_ft = players_ft.loc[:, ['season', 'week', 'tm_alias', 'opp_alias', 'position', 'over']]
    QBs = players_ft[players_ft['position'] == 'QB']
    QBs = QBs.rename(columns={'over': pass_stat + '_over'})
    QBs = QBs.drop('position', axis=1)
    WRs = players_ft[players_ft['position'] == 'WR']
    WRs = WRs.rename(columns={'over': rec_stat + '_over'})
    WRs = WRs.drop('position', axis=1)
    RBs = players_ft[players_ft['position'] == 'RB']
    RBs = RBs.rename(columns={'over': rush_stat + '_over'})
    RBs = RBs.drop('position', axis=1)
    Ks = players_ft[players_ft['position'] == 'K']
    Ks = Ks.rename(columns={'over': kick_stat + '_over'})
    Ks = Ks.drop('position', axis=1)

    combined = pd.merge(QBs, WRs, on=['season', 'week', 'tm_alias', 'opp_alias'])
    combined = pd.merge(combined, RBs, on=['season', 'week', 'tm_alias', 'opp_alias'])
    combined = pd.merge(combined, Ks, on=['season', 'week', 'tm_alias', 'opp_alias'])

    opp_combined = combined.rename(columns={
        'tm_alias': 'opp_alias',
        'opp_alias': 'tm_alias',
        pass_stat + '_over': 'opp_' + pass_stat + '_over',
        rec_stat + '_over': 'opp_' + rec_stat + '_over',
        rush_stat + '_over': 'opp_' + rush_stat + '_over',
        kick_stat + '_over': 'opp_' + kick_stat + '_over'
    })

    combined = opp_combined = pd.merge(combined, opp_combined, how='inner', on=['season', 'week', 'tm_alias', 'opp_alias'])

    corr_mat = combined.loc[:, [pass_stat + '_over', rec_stat + '_over',  rush_stat + '_over', kick_stat + '_over', 
                                'opp_' + pass_stat + '_over', 'opp_' + rec_stat + '_over', 'opp_' + rush_stat + '_over', 'opp_' + kick_stat + '_over']]

    def flip_one_value(boolean_tuple, index):
        new_tuple = list(boolean_tuple)
        new_tuple[index] = not new_tuple[index]
        return tuple(new_tuple)

    def flip_two_values(boolean_tuple, indices):
        new_tuple = list(boolean_tuple)
        new_tuple[indices[0]] = not new_tuple[indices[0]]
        new_tuple[indices[1]] = not new_tuple[indices[1]]
        return tuple(new_tuple)

    def calculate_combinations_probabilities(df, cols_subset):
        # Get total number of rows
        total_rows = len(df)
        
        # Group by the combination of columns and count occurrences
        group_counts = df[list(cols_subset)].value_counts()
        
        # Calculate probabilities
        probabilities = group_counts / total_rows
        
        return probabilities

    def calc_hedge(probabilities):
        adjusted_prob = probabilities.copy()
        for bool_comb in probabilities.index:
            one_dif_prob = 0
            two_dif_prob = 0

            for i in range(len(bool_comb)):
                flipped = flip_one_value(bool_comb, i)
                if flipped in probabilities:
                    one_dif_prob += probabilities[flipped]
            
            for i, j in combinations(range(len(bool_comb)), 2):
                flipped = flip_two_values(bool_comb, (i, j))
                if flipped in probabilities:
                    two_dif_prob += probabilities[flipped]

            adjusted_prob[bool_comb] = (2 * one_dif_prob) + (0.4 * two_dif_prob)

        return adjusted_prob

    result = pd.Series(dtype=float)
    multipliers = {2: 3, 3: 5.5, 4: 10, 5: 20, 6: 25}
    for r in range(2, 9):
        # Get all combinations of r columns
        comb = combinations(corr_mat.columns, r)
        
        # Calculate probabilities for each combination
        for cols_subset in comb:
            bang_prob = calculate_combinations_probabilities(corr_mat, cols_subset)
            if r == 9 or r == 10:
                hedge_prob = calc_hedge(bang_prob)
                bang_prob *= multipliers[r]
                bang_prob += hedge_prob
            else:
                for i in range(r):
                    bang_prob *= 1.7

            combined_label = [f"{cols_subset}-{tuple(idx)}" for idx in bang_prob.index]
            new_series = pd.Series(bang_prob.values, index=combined_label)
            result = pd.concat([result, new_series])

    result = result.sort_values(ascending=False)
    return result

final_result = pd.Series(dtype=float)
for pass_stat in pass_stats:
    for rec_stat in rec_stats:
        for rush_stat in rush_stats:
            for kick_stat in kick_stats:
                final_result = pd.concat([final_result, get_EV(players, pass_stat, rec_stat, rush_stat, kick_stat)])

final_result = final_result.groupby(final_result.index).transform('mean')
final_result = final_result.sort_values(ascending=False)
print(final_result)
final_result.to_csv('parlay_EV.csv')