'''
@author Alex Hakesley 16011419

Main program.

Creates three endpoints which can be accessed via HTTP.
Running the file curently opens a local debug server (localhost:5000),
to which requests may be sent. 

See README-16011419.md

'''

from flask import Flask, request
from flask_cors import CORS, cross_origin
from flask_marshmallow import exceptions
from flask_jwt_extended import JWTManager, jwt_required, create_access_token
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from projectSchema import PredictionDataSchema, TrainObjectDataSchema, AuthenticationSchema
import error_handlers
from feature_engineering import engineer_features
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import os
import json
import datetime as dt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import joblib

app = Flask(__name__)
app.register_blueprint(error_handlers.app_bp)
app.config['CORS_HEADERS'] = 'Content-Type'
app.config['SECRET_KEY'] = "08DdyCZlw4QBXFWmEW9rjsk4xbpgOeFE"
app.config['SQLALCHEMY_DATABASE_URI'] = "sqlite:///" + os.path.abspath('.') + '/back/db/user.sqlite'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
cors = CORS(app)
jwt = JWTManager(app)

db = SQLAlchemy(app)
db.Model.metadata.reflect(db.engine)

class User(db.Model):
    __table__ = db.Model.metadata.tables['users']

    def verify_password(self, password):
        return check_password_hash(self.password, password)

    def hash_password(self, password):
        self.password = generate_password_hash(password)

@jwt.unauthorized_loader
def unauthorized_callback(callback):
    return error_handlers.handle_unauthorized("Bearer token not found in Authorization headers")

@jwt.invalid_token_loader
def invalid_token_callback(callback):
    return error_handlers.handle_unauthorized("Invalid access token")

@jwt.expired_token_loader
def expired_token_callback(headers, payload):
    return error_handlers.handle_unauthorized("Access token has expired.")

# Authenticate endpoint.
@app.route("/authenticate", methods=["POST"])
def authenticate():
    schema = AuthenticationSchema(many=True)
    try:
        data = schema.loads(request.data)
        email = data[0]['email']
        password = data[0]['password']

        # validate user credentials against database
        user = User.query.filter_by(email=email).first()
        if not user or not user.verify_password(password):
            return error_handlers.handle_unauthorized("Invalid 'email' or 'password' parameter")

        # return access token
        access_token = create_access_token(identity=email, expires_delta=dt.timedelta(minutes=60))
        response = {
            "access_token": access_token
        }
        return response, 200
    except exceptions.ValidationError as e:
        return error_handlers.handle_bad_request(e.messages)
    except json.JSONDecodeError as e:
        return ({"error_help": "invalid json"}), 400

# Predict endpoint
@app.route("/predict", methods=['POST'])
@jwt_required()
@cross_origin()
def predict():
    schema = PredictionDataSchema(many=True)
    try:
        data = schema.loads(request.data)
        df = pd.DataFrame.from_dict(data)
        input = engineer_features(df)

        # immediately return 0 as prediction if 'open' is false
        if input.iloc[0]['open'] == 0:
            prediction = str(0)
        else:
            model = tf.keras.models.load_model(r'back/src/model/revenue_predictor.h5')
            scaler = joblib.load(r'back/src/model/scaler.joblib.dat')

            # scale inputs as model has been trained on scaled inputs.
            input_scaled = scaler.transform(input.values)
            input = pd.DataFrame(input_scaled, index=input.index, columns=input.columns);
            prediction = model.predict(input)[0][0]
            # round up and if prediction is less than 0 (which sometimes happen), round up to 0.
            prediction = str(max(0, np.round(prediction, 0)))
        
        response = {
            'prediction': prediction,
        }
        return response, 200
    except exceptions.ValidationError as e:
        return error_handlers.handle_bad_request(e.messages)
    except json.JSONDecodeError as e:
        return ({"error_help": "invalid json"}), 400

# Train endpoint
@app.route("/train", methods=['PUT'])
@jwt_required()
@cross_origin()
def train():
    schema = TrainObjectDataSchema(many=True)
    try:
        data = schema.loads(request.data)
        df = pd.DataFrame.from_dict(data)
        df = engineer_features(df)

        # partition training and validation data sets
        train = df[:len(df.index)-7]
        test = df[len(df.index)-7:]

        # split partitions into features and labels
        x_train = train.drop('revenue', axis=1)
        y_train = train['revenue']
        x_test = test.drop('revenue', axis=1)
        y_test = test['revenue']

        # build scaler   
        scaler = MinMaxScaler()
        x_train_scaled = scaler.fit_transform(x_train.values)
        x_test_scaled = scaler.transform(x_test.values)

        x_train = pd.DataFrame(x_train_scaled, index=x_train.index, columns=x_train.columns);
        x_test = pd.DataFrame(x_test_scaled, index=x_test.index, columns=x_test.columns);

        # build model
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(4, activation='relu'))
        model.add(tf.keras.layers.Dense(8, activation='relu'))
        model.add(tf.keras.layers.Dense(16, activation='relu'))
        model.add(tf.keras.layers.Dense(32, activation='relu'))
        model.add(tf.keras.layers.Dense(64, activation='relu'))
        model.add(tf.keras.layers.Dense(128, activation='relu'))
        model.add(tf.keras.layers.Dense(1))
        model.compile(
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.RootMeanSquaredError()],
            optimizer=tf.keras.optimizers.Adam())

        # fit model on train set
        model.fit(
            x_train, 
            y_train, 
            validation_data=(x_test, y_test), 
            epochs=250,
            verbose=0,
            shuffle=False,
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_root_mean_squared_error', mode='min', patience=50, restore_best_weights=True)])

        # return test performance to indicate training complete
        preds = model.predict(x_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        model.save(r'back/src/model/revenue_predictor.h5')
        joblib.dump(scaler, r'back/src/model/scaler.joblib.dat')
        return ({
            "rmse": rmse
        }), 200
    except exceptions.ValidationError as e:
        return error_handlers.handle_bad_request(e.messages)
    except json.JSONDecodeError as e:
        return ({"error_help": "invalid json"}), 400

if __name__ == "__main__":    
    # add a test user to the db
    email = "test@example.com"
    password = "MyPassword123"
    role = 1
    user = User(email = email, role = role)
    user.hash_password(password)
    if not User.query.filter_by(email = email).first():
        db.session.add(user)
        db.session.commit()

    app.run(debug=True)