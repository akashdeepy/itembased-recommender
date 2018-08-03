import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix

# display results to 3 decimal points, not in scientific notation
pd.set_option('display.float_format', lambda x: '%.3f' % x)

user_data = pd.read_table('C:/Users/Viyatra/Desktop/videodata/productlist.tsv',
                          header = None, nrows = 2e7,
                          names = ['users', 'productid', 'packagename', 'totalvisits'],
                          usecols = ['users', 'packagename', 'totalvisits'])
user_profiles = pd.read_table('C:/Users/Viyatra/Desktop/videodata/userlist.tsv',
                          header = None,
                          names = ['users', 'gender', 'age', 'country', 'signup'],
                          usecols = ['users', 'country'])

user_data.head()
user_profiles.head()

if user_data['packagename'].isnull().sum() > 0:
    user_data = user_data.dropna(axis = 0, subset = ['packagename'])

    totalpackagevisit = (user_data.
        groupby(by=['packagename'])['totalvisits'].
        sum().
        reset_index().
        rename(columns={'totalvisits': 'total_packagevisits'})
    [['packagename', 'total_packagevisits']]
        )
    totalpackagevisit.head()

    user_data_with_packagevisits = user_data.merge(totalpackagevisit, left_on='packagename', right_on='packagename', how='left')

    user_data_with_packagevisits.head()

    print (totalpackagevisit['total_packagevisits'].describe())

print (totalpackagevisit['total_packagevisits'].quantile(np.arange(.9, 1, .01)))

popularity_threshold = 40000
user_data_popular_package = user_data_with_packagevisits.query('total_packagevisits >= @popularity_threshold')
user_data_popular_package.head()

combined = user_data_popular_package.merge(user_profiles, left_on = 'users', right_on = 'users', how = 'left')
usa_data = combined.query('country == \'United States\'')
usa_data.head()


if not usa_data[usa_data.duplicated(['users', 'packagename'])].empty:
    initial_rows = usa_data.shape[0]

    print ('Initial dataframe shape {0}'.format(usa_data.shape))
    usa_data = usa_data.drop_duplicates(['users', 'packagename'])
    current_rows = usa_data.shape[0]
    print ('New dataframe shape {0}'.format(usa_data.shape))
    print ('Removed {0} rows'.format(initial_rows - current_rows))



# implementing the Nearest neighbour#



wide_package_data = usa_data.pivot(index = 'packagename', columns = 'users', values = 'totalvisits').fillna(0)
wide_package_data_sparse = csr_matrix(wide_package_data.values)

from sklearn.neighbors import NearestNeighbors

model_knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
model_knn.fit(wide_package_data_sparse)

#Making Recommendation#

query_index = np.random.choice(wide_package_data.shape[0])
distances, indices = model_knn.kneighbors(wide_package_data.iloc[query_index, :].reshape(1, -1), n_neighbors = 6)

for i in range(0, len(distances.flatten())):
    if i == 0:
        print ('Recommendations for {0}:\n'.format(wide_package_data.index[query_index]))
    else:
        print ('{0}: {1}, with distance of {2}:'.format(i, wide_package_data.index[indices.flatten()[i]], distances.flatten()[i]))


