import streamlit as st
import pandas as pd
from model import predict_score  

def main():
    
    st.sidebar.title("📌 Important Information")
    st.sidebar.markdown(
        """
        ### 📝 Key Information:
        - This model predicts the likelihood of pet adoption based on various features like pet type, breed, age, and more.
        - The model was trained using pet data and factors affecting adoption likelihood.
        
        ### 💡 Quick Tips:
        - Make sure to input the correct pet type and breed for better predictions.
        - The **predicted adoption likelihood** will help you understand the chances of an animal being adopted.

        ### 🌟 Pet Adoption Insights:
        - Adoptions tend to be higher for younger pets and those with good health.
        - Pets with an adoption fee that is too high might face lower adoption chances.

        **Thank you for using Pet Adoption Predictor!** 😊
        """
    )

    
    st.title("🐾 Pet Adoption Predictor")
    st.write("Predict the likelihood of pet adoption based on various input features.")

    
    file_path = "pet_adoption_data.csv"
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return 

    # Define pet type to breed mapping
    pet_type_to_breeds = {
        'Dog': ['Labrador', 'Poodle', 'Golden Retriever'], 
        'Cat': ['Persian', 'Siamese'],  
        'Bird': ['Parakeet'],  
        'Rabbit': ['Rabbit']  
    }
    
    # Defining input fields for the dashboard interface
    pet_type = st.selectbox("Select Pet Type:", df['PetType'].unique())
    
    # Dynamically changing the breed options based on pet type selection
    breed_options = pet_type_to_breeds.get(pet_type, [])
    breed = st.selectbox("Select Breed:", breed_options)
    
    # Dynamically show the color options based on pet type selection
    color_options = df[df['PetType'] == pet_type]['Color'].unique()
    color = st.selectbox("Select Color:", color_options)
    
    size = st.selectbox("Select Size:", df['Size'].unique())
    vaccinated = st.selectbox("Vaccinated:", [1, 0])
    age_months = st.number_input("Age (in months):", min_value=0, value=12)
    health_condition = st.selectbox("Health Condition:", df['HealthCondition'].unique())
    time_in_shelter_days = st.number_input("Time in Shelter (days):", min_value=0, value=10)
    adoption_fee = st.number_input("Adoption Fee:", min_value=0, value=100)
    previous_owner = st.selectbox("Previous Owner:", [1, 0])

    # Collect input data for prediction
    input_data = {
        'pet_type': pet_type,
        'breed': breed,
        'colour': color,
        'size': size,
        'Vaccinated': vaccinated,
        'AgeMonths': age_months,
        'HealthCondition': health_condition,
        'TimeInShelterDays': time_in_shelter_days,
        'AdoptionFee': adoption_fee,
        'PreviousOwner': previous_owner
    }

    
    st.subheader("🔮 Predict Adoption Likelihood")
    
    if st.button("Predict"):
        try:
            predicted_likelihood = predict_score(input_data)

            
            st.success(f"🎯 Predicted Adoption Likelihood: {predicted_likelihood:.2f}")
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")

   
    st.markdown(
        """
        ---
        Thank you for using the Pet Adoption Predictor! 😊
        """
    )

if __name__ == "__main__":
    main()
