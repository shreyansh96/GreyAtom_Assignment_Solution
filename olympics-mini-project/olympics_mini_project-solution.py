
## Olympics_Assignment_Solution
## Author: Shreyansh Agarwal
## Date: 16th Jan 18
import tkinter
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.preprocessing import scale
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score,silhouette_samples



def load_data():
    '''
    function for loading olympics.csv and do basic preprocessing
    '''
    # reading csv and skipping 1st row
    data=pd.read_csv('data/olympics.csv', skiprows=1)
    #renaming the columns names 
    data=data.rename(columns=lambda x: re.sub('^01.!..|^01.!','Gold',x))
    data=data.rename(columns=lambda x: re.sub('^02.!..|^02.!','Silver',x))
    data=data.rename(columns=lambda x: re.sub('^03.!..|^03.!','Bronze',x))
    #splitting country name and country code
    data=data.drop(data.index[-1]) # dropping row totals
    country_name=[]
    country_code=[]
    for i in data.iloc[:,0].values:
        i=i.split('\xa0')
        country_name.append(i[0])
        try:
            code=i[1]
        except:
            continue
        country_code.append(code.split()[0][1:4]) # removing parenthesis
    df=pd.DataFrame(country_code)
    df=df.rename(columns={0:'country_code'})
    del data['Unnamed: 0']
    data=pd.concat([df, data], axis=1)
    data.index=country_name
    del df
    return data


def first_country(data):
    '''
    This function returns the stats for the first country in the dataset
    '''
    return data.iloc[0,:]


def gold_medal(data): 
    '''
    This function returns the name of the team which has won highest number of combined gold medals. 
    '''
    return data.iloc[:,2].argmax()


def biggest_difference_in_gold_medal(data):
    '''
    This function returns the name of the team which has the biggest difference 
    between their summer and winter gold medal counts.
    '''
    return (data.iloc[:,2]-data.iloc[:,7]).abs().argmax()

def get_points(data):
    '''
    This function returns a pandas series object containing the weighted values where
    each gold medal counts for 3 points,
    each silver medal counts for 2 points and
    each bronze medal counts for 1 point.
    '''
    return data.iloc[:,12]*3+data.iloc[:,13]*2+data.iloc[:,14]


def k_means(data):
    '''
    This function performs elbow and silhouette analysis and 
    returns the optimal value of k and the coordinates of the cluster centers.
    '''
    points=pd.DataFrame(get_points(data))
    points=points.rename(columns={0:'points'})
    Data=pd.concat([data['# Games'],points],axis=1)
    scaled_data=scale(Data)
    ## Elbow method for finding range of optimal k values
    cluster_range = range( 1, 21 )
    cluster_SSE = []

    for num_clusters in cluster_range:
        clusters = KMeans( num_clusters )
        clusters.fit( scaled_data )
        cluster_SSE.append( clusters.inertia_ )
    plt.figure(figsize=(12,6))
    plt.plot( cluster_range,cluster_SSE, marker = "*" )
    plt.xlabel('Number of clusters', fontweight='bold')
    plt.ylabel('SSE', fontweight='bold')
    plt.title('Elbow analysis for finding optimal k', fontweight='bold')
    plt.xticks(cluster_range)
    plt.show()
    
    ## Silhouette analysis 
    
    result_dict={}
    for n_clusters in range(2,6):
        kmeans = KMeans(n_clusters=n_clusters, random_state=28)
        y_kmeans=kmeans.fit_predict(scaled_data)
        sample_silhouette_coeff = silhouette_samples(scaled_data, y_kmeans)
        silhouette_avg=silhouette_score(scaled_data,y_kmeans)

        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(20, 8)
        ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(scaled_data) + (n_clusters + 1) * 10])

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_coeff = sample_silhouette_coeff[y_kmeans == i]

            ith_cluster_silhouette_coeff.sort()

            size_cluster_i = ith_cluster_silhouette_coeff.shape[0]
            y_upper = y_lower + size_cluster_i
            color = cm.spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),0, ith_cluster_silhouette_coeff,facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")


        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
        colors = cm.spectral(y_kmeans.astype(float) / n_clusters)
        ax2.scatter(scaled_data[:, 0], scaled_data[:, 1], marker='.', s=30, lw=0, alpha=0.8,
                  c=colors)

        # Labeling the clusters
        centers = kmeans.cluster_centers_
        # Draw yellow coloured circles at cluster centers
        ax2.scatter(centers[:, 0], centers[:, 1],
                  marker='o', c="yellow", alpha=1, s=200)

        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1, s=50)

            ax2.set_title("The visualization of the clustered data.")
            ax2.set_xlabel("Number of games")
            ax2.set_ylabel("Number of points")

            plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                    "with n_clusters = %d" % n_clusters),
                   fontsize=14, fontweight='bold')

        plt.show()
        print ('The silhouette score is : ',silhouette_avg)
        result_dict.update({n_clusters:silhouette_avg})
    optimal_k=4 # from elbow and silhouette analysis
    kmeans=KMeans(n_clusters=optimal_k,random_state=28)
    kmeans.fit_predict(scaled_data)
    return optimal_k,kmeans.cluster_centers_


data=load_data()
print(first_country(data))
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print(gold_medal(data))
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print(biggest_difference_in_gold_medal(data))
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print(get_points(data))
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

optimal_k, cluster_centers=k_means(data)

print('The optimal value of k is : ',optimal_k)
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('The coordinates of cluster centers are : \n',cluster_centers)
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')






