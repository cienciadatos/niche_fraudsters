# Tratamiento de datos
# ==============================================================================
import numpy as np
import pandas as pd

# Gráficos
# ==============================================================================
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Preprocesado
# ==============================================================================
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Métricas
# ==============================================================================
from sklearn.metrics import average_precision_score, f1_score, balanced_accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report

# Configuración warnings
# ==============================================================================
import warnings
warnings.filterwarnings('once')
warnings.simplefilter('ignore', (DeprecationWarning))

### FUNCIONES AUXILIARES PARA EL TRAIN
#########################################

def generate_data(DATA):
    df = pd.read_csv(DATA, sep=",")
    df = df.set_index('id')
    return df

def limpiar_txt_items(df):
    """Elimina espacios, comas, caracteres raros de las variables itemX con el
    fin de unificar valores y eliminar duplicados.
    """
    df.iloc[:,1:25] = df.iloc[:,1:25].replace(r'[^0-9a-zA-Z ]', '', 
                        regex=True).replace(r'\s+',' ',regex=True)
    return df

def preprocess(df, size=0.7):
    X = df.drop('fraud_flag',axis=1)
    Y = df['fraud_flag']
    X_train, X_test, y_train, y_test = train_test_split(X,Y,
                                          train_size=size,
                                          random_state = 123)
    return X_train, X_test, y_train, y_test

def extract_x(df):
    X = df.drop('fraud_flag',axis=1)
    Y = df['fraud_flag']
    return X, Y

def make_pipeline(X_train,X_test):
    preprocessor = ColumnTransformer(
                       [('scale', StandardScaler(), X_train.columns.to_list())],
                            remainder='passthrough')
    X_train_prep = preprocessor.fit_transform(X_train)
    X_test_prep  = preprocessor.transform(X_test)
    X_train_prep = pd.DataFrame(X_train_prep, 
                                columns=X_train.columns,
                                index=X_train.index)
    X_test_prep = pd.DataFrame(X_test_prep, 
                                columns=X_test.columns,
                                index=X_test.index)
    return X_train_prep, X_test_prep

def prepare_train(DATA, size=0.7):
    df = generate_data(DATA)
    X_train, X_test, y_train, y_test = preprocess(df,size)
    X_train, X_test = make_pipeline(X_train,X_test)
    return X_train, X_test, y_train, y_test

def make_pipeline_x(X):
    preprocessor = ColumnTransformer(
                       [('scale', StandardScaler(), X.columns.to_list())],
                            remainder='passthrough')
    X_prep = preprocessor.fit_transform(X)
    X_prep = pd.DataFrame(X_prep, 
                            columns=X.columns,
                            index=X.index)
    return X_prep

def prepare_train_x(DATA):
    df = generate_data(DATA)
    X, y = extract_x(df)
    X = make_pipeline_x(X)
    return X, y

def print_metrics(Y_test,Y_pred):
    """Imprime métricas básicas de la clasificación.
    """
    metricas = classification_report(Y_test,Y_pred,zero_division=0)
    print(metricas)

def print_conf_matrix(Y_test,Y_pred,labels=[0,1]):
    """Plotea la matriz de confusión de la clasificación."""
    cm = confusion_matrix(Y_test, Y_pred, labels=labels)
    # Definir nuevo cmap personalizado
    colors = ['#00DBDB', '#00E0E0', '#00E5E5', '#00EBEB', '#00F0F0', 
              '#00F5F5', '#00FBFB', '#FFFFFF']
    colors.reverse()
    cmap = ListedColormap(colors)
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, 
                                  display_labels=['0','1'])
    
    # Pasar cmap personalizado a la función plot()
    disp.plot(cmap=cmap, ax=plt.subplots()[1])
    
    # Establecer el color de texto en negro
    for text in disp.ax_.texts:
        text.set_color('#000000')
    
    plt.title('Matriz de Confusión')
    plt.show();
    
def calculate_pr_auc_xgb(y_test,y_pred_prob,clf,version):
    prob_pos = y_pred_prob[:, 1]
    pr_auc = average_precision_score(y_test, prob_pos,average="weighted")
    version = version.split(": ")
    pr_auc = [(version[0],version[1]),('PR AUC',pr_auc)]
    model_summary = pd.DataFrame(pr_auc, columns=['param','selected'])
    print(model_summary)
    return model_summary    

def pr_auc_score(y_true, y_pred_proba, y_pred, X_test, modelo, version):
    '''Return the area under the Precision-Recall curve.
    Args:
        - y_true (pd.DataFrame): Dataframe with a unique identifier for 
        each observation (first column) and the ground truth observations (second column).
        - y_pred_proba (pd.DataFrame): Dataframe with a unique identifier 
        for each observation (first column) and the predicted probabilities 
        estimates for the minority class (second column).
    Returns:float'''
    f1_micro = f1_score(y_true, y_pred, average="micro")
    bal_acc  = balanced_accuracy_score(y_true, y_pred)
    y_true = pd.DataFrame(y_true).reset_index()
    y_pred_proba = pd.DataFrame(y_pred_proba, index=X_test.index).reset_index()
    y_true_sorted = y_true.sort_values('id')
    y_pred_proba_sorted = y_pred_proba.sort_values('id')
    y_true_sorted = y_true_sorted.reset_index(drop=True)
    y_pred_proba_sorted = y_pred_proba_sorted.reset_index(drop=True)
    pr_auc_score = average_precision_score(np.ravel(y_true_sorted.iloc[:, 1]), 
                                           np.ravel(y_pred_proba_sorted.iloc[:, 1]))
    data = [('PR AUC',pr_auc_score),
            ('F1 Micro', f1_micro),
            ('Balanced Accuracy', bal_acc)]
    model_summary = pd.DataFrame(data, columns=['param','value'])
    print(modelo.upper(), version)
    print(model_summary)
    return model_summary

def prepare_test(DATA_TRAIN, DATA_TEST):
    df_train = generate_data(DATA_TRAIN)
    df_test = pd.read_csv(DATA_TEST)
    df_test = df_test.set_index('id')
    X_train = df_train.drop('fraud_flag',axis=1)
    y_train = df_train['fraud_flag']
    X_test  = df_test
    X_train, X_test = make_pipeline(X_train, X_test)
    return X_train, X_test, y_train

def generate_output(y_pred_prob, X_test, nombre_version):
    prob_positiva = y_pred_prob[:,1]
    output = pd.DataFrame({
            "index"      : [x for x in range(len(X_test))],
            "ID"         : X_test.index,
            "fraud_flag" : prob_positiva
    })
    print(output.head())
    output.to_csv(nombre_version + '.csv',sep=",", index=False)