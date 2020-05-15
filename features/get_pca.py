from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.externals import joblib
from pathlib import Path
from dynaconf import settings
import os

def get_pca_vectors(model_name, vectors, k, load_pca):
    """generates k PCA vectors 
    
    Arguments:
        vectors {np.array} -- 2D numpy array of features
        k {int} -- no of features
    
    Returns:
        np.array -- 2D numpy array of features
    """

    directory = Path(settings.path_for(settings.FILES.MODELS))
    directory = str(directory)
    
    k = min(k, vectors.shape[1])
    if load_pca:
        std_scaler = joblib.load(os.path.join(directory, model_name + '_std_scaler.pkl'))
        pca = joblib.load(os.path.join(directory, model_name + '_pca.pkl'))
    else:
        std_scaler = StandardScaler()
        pca = PCA(n_components=k)

    scaled_values = std_scaler.fit_transform(vectors)
    pca_vectors = pca.fit_transform(scaled_values)
    print("Total variance accounted for: ", sum(pca.explained_variance_ratio_))
    if not load_pca:
        joblib.dump(pca, model_name + '_pca.pkl')
        joblib.dump(std_scaler, model_name + '_std_scaler.pkl')

    return pca_vectors

