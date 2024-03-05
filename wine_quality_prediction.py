import os
import warnings 

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet  #ElasticNet is a linear regression model trained with L1 and L2 prior as regularizer.
from urllib.parse import urlparse #parse a URL into six components, returning a 6-item named tuple
import mlflow   #MLflow is an open source platform for the complete machine learning lifecycle
from mlflow.models.signature import infer_signature #infer_signature is used to infer the signature of a model
import mlflow.sklearn 

import logging
import sys

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)
#logger is used to log the output of the program
#logging.WARN is used to log the warning messages
#logging.getLogger(__name__) is used to get the logger object
#logging.basicConfig() is used to configure the logging module, it is used to set the threshold level of the logger



def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

#The eval_metrics function is used to evaluate the model

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)
#np.random.seed() is used to generate random numbers
    

    #Read the wine-quality csv file from the URL
    csv_url = (
        "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv"
    )

    try:
        data = pd.read_csv(csv_url, sep=";")  #read the csv file,sep=";" is used to separate the columns   
    except Exception as e:
        logger.exception(
            "Unable to download training & test CSV, check your internet connection. Error: %s", e
        )
        #logger.exception() is used to log the exception message

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)

    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1) #axis=1 means drop column, axis=0 means drop row
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5 #sys.argv is a list in Python, which contains the command-line arguments passed to the script
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5 #sys.argv[0] is the name of the script itself


    with mlflow.start_run():
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42) #ElasticNet model  is trained with L1 and L2 prior as regularizer
        lr.fit(train_x, train_y)

        predicted_qualities = lr.predict(test_x) #predict the test_x, and store the result in predicted_qualities

        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        mlflow.log_param("alpha", alpha) #parametr alpha
        mlflow.log_param("l1_ratio", l1_ratio) #parametr l1_ratio
       
        mlflow.log_metric("rmse", rmse) #metric rmse
        mlflow.log_metric("r2", r2) #metric r2
        mlflow.log_metric("mae", mae) #metric mae

        ''' predictions= lr.predict(test_x)
        signature= infer_signature(train_x,predictions)'''

        remote_server_uri = "https://dagshub.com/supunlakshan100/mlflow-wine-quality-prediction.mlflow"
        mlflow.set_tracking_uri(remote_server_uri)
        
        

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        #model registry does not work with file store
        if tracking_url_type_store != "file": 
            #register the model
            #there are other ways to use the model registry, which depends on the use case
            #please refer to the doc for more information
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow

            mlflow.sklearn.log_model(lr, "model", registered_model_name="ElasticnetWineModel")
            
        else:
            mlflow.sklearn.log_model(lr, "model")

