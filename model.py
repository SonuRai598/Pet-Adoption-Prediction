import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess_data(file_path):
    
    df = pd.read_csv(file_path)

    # Handle missing and duplicate data 
    print("Missing Values:\n", df.isnull().sum()[df.isnull().sum() > 0])
    print("Duplicate Values:", df.duplicated().sum())

    # Group by PetType and get unique breeds and colors for each type
    breed_by_pet_type = df.groupby('PetType')['Breed'].unique()
    color_by_pet_type = df.groupby('PetType')['Color'].unique()
    
    # Printing out the unique breeds and colors for each pet type
    print("Unique breeds for each pet type:")
    print(breed_by_pet_type)
    
    print("\nUnique colors for each pet type:")
    print(color_by_pet_type)

    # Initializing LabelEncoder for categorical columns
    label_encoder = LabelEncoder()
    
    # Encode categorical columns
    df['PetTypeEncoded'] = label_encoder.fit_transform(df['PetType'])
    df['BreedEncoded'] = label_encoder.fit_transform(df['Breed'])
    df['ColorEncoded'] = label_encoder.fit_transform(df['Color'])
    df['SizeEncoded'] = label_encoder.fit_transform(df['Size'])

    # Drop original categorical columns
    df.drop(columns=['Breed', 'Color', 'PetType', 'Size', 'PetID', 'WeightKg'], inplace=True)

    # Adding columns 
    df['AgeShelterRelation'] = df['AgeMonths'] * df['TimeInShelterDays']
    df['AgeFeeInteraction'] = df['AgeMonths'] * df['AdoptionFee']

    return df


def train_model(df):
    # Selecting features and target
    X = df[['BreedEncoded', 'ColorEncoded', 'PetTypeEncoded', 'SizeEncoded', 'Vaccinated',
            'AgeMonths', 'HealthCondition', 'TimeInShelterDays', 'AdoptionFee',
            'PreviousOwner', 'AgeShelterRelation', 'AgeFeeInteraction']]
    Y = df['AdoptionLikelihood']

    # Spliting data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

    # Scaling the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Training the model
    model = LogisticRegression()
    model.fit(X_train, Y_train)

    # Saving the model, scaler
    joblib.dump(model, 'pet_adoption_prediction_model.pkl')
    joblib.dump(scaler, 'pet_adoption_scaler.pkl')

    return model, scaler


def load_model():
    # Loading the trained model and scaler
    model = joblib.load('pet_adoption_prediction_model.pkl')
    scaler = joblib.load('pet_adoption_scaler.pkl')

    return model, scaler


def predict_score(input_data):
    # Loading the model and scaler
    model, scaler = load_model()

    # Creating a DataFrame with the input data
    input_df = pd.DataFrame([input_data])

    # Initializing LabelEncoder and encode the input data
    label_encoder = LabelEncoder()
    
    # Encoding the categorical fields like PetType, Breed, Color, Size
    input_df['PetTypeEncoded'] = label_encoder.fit_transform(input_df['pet_type'])
    input_df['BreedEncoded'] = label_encoder.fit_transform(input_df['breed'])
    input_df['ColorEncoded'] = label_encoder.fit_transform(input_df['colour'])
    input_df['SizeEncoded'] = label_encoder.fit_transform(input_df['size'])

    # Add ing feature engineering columns (same as during training)
    input_df['AgeShelterRelation'] = input_df['AgeMonths'] * input_df['TimeInShelterDays']
    input_df['AgeFeeInteraction'] = input_df['AgeMonths'] * input_df['AdoptionFee']

    # Selecting and scaling the features
    features = input_df[['BreedEncoded', 'ColorEncoded', 'PetTypeEncoded', 'SizeEncoded', 'Vaccinated',
                         'AgeMonths', 'HealthCondition', 'TimeInShelterDays', 'AdoptionFee',
                         'PreviousOwner', 'AgeShelterRelation', 'AgeFeeInteraction']]
    features_scaled = scaler.transform(features)

    # Making the prediction
    prediction = model.predict(features_scaled)

    return prediction[0]

