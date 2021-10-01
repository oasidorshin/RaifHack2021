import numpy as np
import pandas as pd

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import LabelEncoder


def preprocess_data(train, test, **kwargs):
    """
    Preprocess train and test data.

    Args:
        train: train dataframe
        test: test dataframe
        **kwargs: dictionary of keyword arguments

    Keyword Args:
        log_target: apply log1p to the target column
        remove_type_0: remove all price_type == 0 data
        cluster: add dbscan cluster feature with eps
        clean_floor_num: clear floor column and transform to numerical
        clean_region_city: clean up similar/same region and city names
        only_test_cities: only cities present in test for train data
        encode_cat: label-encode categorical columns

    Returns:
        (train, test, num_columns, cat_columns, target),
            where train and test are preprocessed dataframes,
            num_columns is list of all numerical columns (without target),
            cat_columns is list of all categorical columns (without target),
            target is target column

    Author:
        Oleg
    """

    # Defining basic columns
    cat_columns = ['region', 'city', 'realty_type', 'osm_city_nearest_name']
    num_columns = ['lat', 'lng', 'floor', 'osm_amenity_points_in_0.001',
                   'osm_amenity_points_in_0.005', 'osm_amenity_points_in_0.0075',
                   'osm_amenity_points_in_0.01', 'osm_building_points_in_0.001',
                   'osm_building_points_in_0.005', 'osm_building_points_in_0.0075',
                   'osm_building_points_in_0.01', 'osm_catering_points_in_0.001',
                   'osm_catering_points_in_0.005', 'osm_catering_points_in_0.0075',
                   'osm_catering_points_in_0.01', 'osm_city_closest_dist',
                   'osm_city_nearest_population',
                   'osm_crossing_closest_dist', 'osm_crossing_points_in_0.001',
                   'osm_crossing_points_in_0.005', 'osm_crossing_points_in_0.0075',
                   'osm_crossing_points_in_0.01', 'osm_culture_points_in_0.001',
                   'osm_culture_points_in_0.005', 'osm_culture_points_in_0.0075',
                   'osm_culture_points_in_0.01', 'osm_finance_points_in_0.001',
                   'osm_finance_points_in_0.005', 'osm_finance_points_in_0.0075',
                   'osm_finance_points_in_0.01', 'osm_healthcare_points_in_0.005',
                   'osm_healthcare_points_in_0.0075', 'osm_healthcare_points_in_0.01',
                   'osm_historic_points_in_0.005', 'osm_historic_points_in_0.0075',
                   'osm_historic_points_in_0.01', 'osm_hotels_points_in_0.005',
                   'osm_hotels_points_in_0.0075', 'osm_hotels_points_in_0.01',
                   'osm_leisure_points_in_0.005', 'osm_leisure_points_in_0.0075',
                   'osm_leisure_points_in_0.01', 'osm_offices_points_in_0.001',
                   'osm_offices_points_in_0.005', 'osm_offices_points_in_0.0075',
                   'osm_offices_points_in_0.01', 'osm_shops_points_in_0.001',
                   'osm_shops_points_in_0.005', 'osm_shops_points_in_0.0075',
                   'osm_shops_points_in_0.01', 'osm_subway_closest_dist',
                   'osm_train_stop_closest_dist', 'osm_train_stop_points_in_0.005',
                   'osm_train_stop_points_in_0.0075', 'osm_train_stop_points_in_0.01',
                   'osm_transport_stop_closest_dist', 'osm_transport_stop_points_in_0.005',
                   'osm_transport_stop_points_in_0.0075',
                   'osm_transport_stop_points_in_0.01',
                   'reform_count_of_houses_1000', 'reform_count_of_houses_500',
                   'reform_house_population_1000', 'reform_house_population_500',
                   'reform_mean_floor_count_1000', 'reform_mean_floor_count_500',
                   'reform_mean_year_building_1000', 'reform_mean_year_building_500', 'total_square']

    target = 'per_square_meter_price'

    # All additional transforms, don't forget correct ordering
    if 'log_target' in kwargs:
        train[target] = np.log1p(train[target])

    if 'cluster' in kwargs:
        train, test = cluster(train, test, kwargs.get('cluster'))
        cat_columns.append('cluster')

    if 'remove_type_0' in kwargs:
        train = train[train['price_type'] == 1]

    if 'clean_floor_num' in kwargs:
        train, test = clean_floor_num(train), clean_floor_num(test)

    if 'clean_region_city' in kwargs:
        train, test = clean_region_city(train), clean_region_city(test)

    # This needs to be last
    if 'only_test_cities' in kwargs:
        train = get_cities_in_test(train, test)

    if 'encode_categorical' in kwargs:
        # Label encoding for categorical columns
        for column in cat_columns:
            le = LabelEncoder()
            le.fit(pd.concat([train[column], test[column]]))
            train[column] = le.transform(train[column]).astype(int)
            test[column] = le.transform(test[column]).astype(int)

    for column in num_columns:
        train[column] = train[column].astype(float)
        test[column] = test[column].astype(float)

    # Type check for everything
    if 'encode_categorical' in kwargs:
        for column in cat_columns:
            assert train[column].dtype == 'int', (column, train[column].dtype)
            assert test[column].dtype == 'int', (column, test[column].dtype)

    for column in num_columns:
        assert train[column].dtype == 'float', (column, train[column].dtype)
        assert test[column].dtype == 'float', (column, test[column].dtype)

    return train, test, num_columns, cat_columns, target


def get_cities_in_test(train, test):
    """
    Author:
        Public solution
    """
    train = train[train['city'].isin(test['city'])]
    return train


def cluster(train, test, eps):
    """
    Author:
        RedPowerful
    """
    model = DBSCAN(eps=eps, min_samples=2)

    train['df_type'] = 'train'
    test['df_type'] = 'test'
    df_concat = pd.concat([train, test])
    df_concat['cluster'] = model.fit_predict(df_concat[['lat', 'lng']], sample_weight=None)

    train = df_concat[df_concat['df_type'] == 'train']
    test = df_concat[df_concat['df_type'] == 'test']
    return train, test


def clean_region_city(df):
    """
    Author:
        Public solution
    """
    new_df = df.copy()
    change_region_dict = {
        'Адыгея': 'Республика Адыгея',
        'Татарстан': 'Республика Татарстан',
        'Мордовия': 'Республика Мордовия',
        'Коми': 'Республика Коми',
        'Карелия': 'Республика Карелия',
        'Башкортостан': 'Республика Башкортостан',
        'Ханты-Мансийский АО': 'Ханты-Мансийский автономный округ - Югра',
        'Удмуртия': 'Удмуртская республика'
    }

    change_city_dict = {
        'Иркутский район, Маркова рп, Зеленый Берег мкр': 'Маркова',
        'Иркутский район, Маркова рп, Стрижи кв-л': 'Маркова',
        'город Светлый': 'Светлый',
        'Орел': 'Орёл'
    }

    def custom_func(region, city):
        if (region == 'Ленинградская область') and (city == 'Санкт-Петербург'):
            return 'Санкт-Петербург'
        if (region == 'Тюменская область') and (city == 'Нижневартовск'):
            return 'Ханты-Мансийский автономный округ - Югра'
        if (region == 'Тюменская область') and (city == 'Сургут'):
            return 'Ханты-Мансийский автономный округ - Югра'
        return region

    for err, new in zip(list(change_city_dict.keys()), list(change_city_dict.values())):
        new_df.replace(err, new, inplace=True)

    for err, new in zip(list(change_region_dict.keys()), list(change_region_dict.values())):
        new_df.replace(err, new, inplace=True)
        new_df['region'] = new_df.apply(lambda x: custom_func(x['region'], x['city']), axis=1)

    return new_df


def clean_floor_num(data):
    """
    Author:
        Public solution
    """
    data['floor'] = data['floor'].mask(data['floor'] == '-1.0', -1) \
        .mask(data['floor'] == '-2.0', -2) \
        .mask(data['floor'] == '-3.0', -3) \
        .mask(data['floor'] == 'подвал, 1', 1) \
        .mask(data['floor'] == 'подвал', -1) \
        .mask(data['floor'] == 'цоколь, 1', 1) \
        .mask(data['floor'] == '1,2,антресоль', 1) \
        .mask(data['floor'] == 'цоколь', 0) \
        .mask(data['floor'] == 'тех.этаж (6)', 6) \
        .mask(data['floor'] == 'Подвал', -1) \
        .mask(data['floor'] == 'Цоколь', 0) \
        .mask(data['floor'] == 'фактически на уровне 1 этажа', 1) \
        .mask(data['floor'] == '1,2,3', 1) \
        .mask(data['floor'] == '1, подвал', 1) \
        .mask(data['floor'] == '1,2,3,4', 1) \
        .mask(data['floor'] == '1,2', 1) \
        .mask(data['floor'] == '1,2,3,4,5', 1) \
        .mask(data['floor'] == '5, мансарда', 5) \
        .mask(data['floor'] == '1-й, подвал', 1) \
        .mask(data['floor'] == '1, подвал, антресоль', 1) \
        .mask(data['floor'] == 'мезонин', 2) \
        .mask(data['floor'] == 'подвал, 1-3', 1) \
        .mask(data['floor'] == '1 (Цокольный этаж)', 0) \
        .mask(data['floor'] == '3, Мансарда (4 эт)', 3) \
        .mask(data['floor'] == 'подвал,1', 1) \
        .mask(data['floor'] == '1, антресоль', 1) \
        .mask(data['floor'] == '1-3', 1) \
        .mask(data['floor'] == 'мансарда (4эт)', 4) \
        .mask(data['floor'] == '1, 2.', 1) \
        .mask(data['floor'] == 'подвал , 1 ', 1) \
        .mask(data['floor'] == '1, 2', 1) \
        .mask(data['floor'] == 'подвал, 1,2,3', 1) \
        .mask(data['floor'] == '1 + подвал (без отделки)', 1) \
        .mask(data['floor'] == 'мансарда', 3) \
        .mask(data['floor'] == '2,3', 2) \
        .mask(data['floor'] == '4, 5', 4) \
        .mask(data['floor'] == '1-й, 2-й', 1) \
        .mask(data['floor'] == '1 этаж, подвал', 1) \
        .mask(data['floor'] == '1, цоколь', 1) \
        .mask(data['floor'] == 'подвал, 1-7, техэтаж', 1) \
        .mask(data['floor'] == '3 (антресоль)', 3) \
        .mask(data['floor'] == '1, 2, 3', 1) \
        .mask(data['floor'] == 'Цоколь, 1,2(мансарда)', 1) \
        .mask(data['floor'] == 'подвал, 3. 4 этаж', 3) \
        .mask(data['floor'] == 'подвал, 1-4 этаж', 1) \
        .mask(data['floor'] == 'подва, 1.2 этаж', 1) \
        .mask(data['floor'] == '2, 3', 2) \
        .mask(data['floor'] == '7,8', 7) \
        .mask(data['floor'] == '1 этаж', 1) \
        .mask(data['floor'] == '1-й', 1) \
        .mask(data['floor'] == '3 этаж', 3) \
        .mask(data['floor'] == '4 этаж', 4) \
        .mask(data['floor'] == '5 этаж', 5) \
        .mask(data['floor'] == 'подвал,1,2,3,4,5', 1) \
        .mask(data['floor'] == 'подвал, цоколь, 1 этаж', 1) \
        .mask(data['floor'] == '3, мансарда', 3) \
        .mask(data['floor'] == 'цоколь, 1, 2,3,4,5,6', 1) \
        .mask(data['floor'] == ' 1, 2, Антресоль', 1) \
        .mask(data['floor'] == '3 этаж, мансарда (4 этаж)', 3) \
        .mask(data['floor'] == 'цокольный', 0) \
        .mask(data['floor'] == '1,2 ', 1) \
        .mask(data['floor'] == '3,4', 3) \
        .mask(data['floor'] == 'подвал, 1 и 4 этаж', 1) \
        .mask(data['floor'] == '5(мансарда)', 5) \
        .mask(data['floor'] == 'технический этаж,5,6', 5) \
        .mask(data['floor'] == ' 1-2, подвальный', 1) \
        .mask(data['floor'] == '1, 2, 3, мансардный', 1) \
        .mask(data['floor'] == 'подвал, 1, 2, 3', 1) \
        .mask(data['floor'] == '1,2,3, антресоль, технический этаж', 1) \
        .mask(data['floor'] == '3, 4', 3) \
        .mask(data['floor'] == '1-3 этажи, цоколь (188,4 кв.м), подвал (104 кв.м)', 1) \
        .mask(data['floor'] == '1,2,3,4, подвал', 1) \
        .mask(data['floor'] == '2-й', 2) \
        .mask(data['floor'] == '1, 2 этаж', 1) \
        .mask(data['floor'] == 'подвал, 1, 2', 1) \
        .mask(data['floor'] == '1-7', 1) \
        .mask(data['floor'] == '1 (по док-м цоколь)', 1) \
        .mask(data['floor'] == '1,2,подвал ', 1) \
        .mask(data['floor'] == 'подвал, 2', 2) \
        .mask(data['floor'] == 'подвал,1,2,3', 1) \
        .mask(data['floor'] == '1,2,3 этаж, подвал ', 1) \
        .mask(data['floor'] == '1,2,3 этаж, подвал', 1) \
        .mask(data['floor'] == '2, 3, 4, тех.этаж', 2) \
        .mask(data['floor'] == 'цокольный, 1,2', 1) \
        .mask(data['floor'] == 'Техническое подполье', -1) \
        .mask(data['floor'] == '1.2', 1) \
        .astype(float)
    return data
