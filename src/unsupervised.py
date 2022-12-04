import matplotlib.pyplot as plt
import pandas as pd
import mlflow
# import mlflow.sklearn
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def silhoutte_score_test(dataframe, max_clusters=10):
    """ Silhoutte score to find best number of clusters """
    df_report = pd.DataFrame(columns=['clusters', 'score'])
    sil_score_max = -1  # this is the minimum possible score

    for n_clusters in range(2, max_clusters):
        kmeans_obj = Kmeans_Clustering(n_clusters=n_clusters)
        model = kmeans_obj.get_model()
        labels = model.fit_predict(dataframe)
        sil_score = silhouette_score(dataframe, labels)
        # print("The average silhouette score for %i clusters is %0.2f" %(n_clusters, sil_score))
        df_sil_score = pd.DataFrame({'clusters': [n_clusters], 'score': [sil_score]})
        df_report = df_report.append(df_sil_score)
        if sil_score > sil_score_max:
            sil_score_max = sil_score
            best_n_clusters = n_clusters
    
    return best_n_clusters, df_report.sort_values(by='score', ascending=False)


class Kmeans_Clustering(object):
    """ https://www.statology.org/k-means-clustering-in-python/ """

    def __init__(self, n_clusters=10, _init="random", n_init=10):
        self.random_state = 42
        self.n_clusters = n_clusters
        self._init = _init
        self.n_init = n_init
        self.algo = KMeans(n_clusters=self.n_clusters, init=self._init, n_init=self.n_init, random_state=self.random_state)

    def get_model(self):
        return self.algo

    def set_model(self, n_clusters=10, init="random", n_init=10):
        self.algo = KMeans(n_clusters=n_clusters, init=init, n_init=n_init, random_state=self.random_state)

    def get_nr_clusters(self):
        return self.n_clusters

    def set_nr_clusters(self, clusters):
        self.n_clusters = clusters

    def train(self, scaled_dataframe):
        """ dataset already scaled """
        self.get_model().fit(scaled_dataframe)

    def predict(self, data):
        return self.get_model().predict(data)

    def cluster_sse(self, scaled_dataframe):
        """ Create list to hold SSE values for each k """
        sse = []
        for k in range(1, self.n_clusters + 1):
            self.set_model(n_clusters=k)
            self.train(scaled_dataframe)
            sse.append(self.get_model().inertia_)
        return sse

    def plot_sse(self, sse):
        """ Visualize results """
        plt.plot(range(1, self.n_clusters + 1), sse)
        plt.xticks(range(1, self.n_clusters + 1))
        plt.xlabel("Number of Clusters")
        plt.ylabel("SSE")
        plt.show()

    def labels(self):
        """ View cluster assignments for each observation """
        return self.get_model().labels_
        
    def train_with_mlflow_registry(self, X_train, run_name="Kmeans Model Training", model_name="kmeans_model"):
        # Useful for multiple runs (only doing one run in this sample notebook)
        with mlflow.start_run(run_name=run_name) as run:
            self.train(X_train)

            # Log parameter, metrics, and model to MLflow
            n_clusters = self.get_nr_clusters()
            mlflow.log_param("n_clusters", n_clusters)

            mlflow.sklearn.log_model(self.get_model(), model_name)

        return run
