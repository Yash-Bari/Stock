from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login
from django.contrib import messages

# Create your views here.
def index(request):
    return render(request, 'index.html')

from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from django.contrib import messages

def register(request):
    if request.method == 'POST':
        # get form data
        username = request.POST['username']
        email = request.POST.get('email')
        password = request.POST['password']
        confirm_password = request.POST.get('confirm_password')
        
        if password == confirm_password:
            if User.objects.filter(email=email).exists():
                messages.info(request, 'Email Already Used')
                return redirect('register/')
            elif User.objects.filter(username=username).exists():
                messages.info(request, 'Username Already Used')
                return redirect('register/')
            else:
                # create new user
                user = User.objects.create_user(username=username, email=email, password=password)
                user.save()
                messages.success(request, 'Account created successfully!')
                return render(request, 'predicter.html')
        else:
            messages.info(request, 'Password Not The Same ')
            return redirect('register/')
    else:
        return render(request, 'register.html')

    

'''def login(request):
    if request.method == 'POST':
        # get form data
        username = request.POST.get('username')
        password = request.POST.get('password')

        # authenticate user
        user = authenticate(request, username=username, password=password)
        if user==user:
            return render(request, 'predicter.html')
        if user is not None:
            # login user
            login(request, user)
            return redirect('home')
        else:
            messages.error(request, 'Invalid username or password')
            return redirect('login')
    else:
        return render(request, 'login.html')'''
    


from django.shortcuts import render
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from datetime import date, timedelta
import yfinance as yf
import pandas as pd


def predict_stock(request):
    if request.method == 'POST':
        # Get the user input
        sym= request.POST['symbol']
        symbol=sym+'.NS'

        # Set the timeframe for the stock data
        start_date = date.today() - timedelta(days=365)
        end_date = date.today()

        # Scrape the stock price data from NSE
        data = yf.download(symbol, start=start_date, end=end_date)

        # Load the data
        df = data

        # Prepare the data
        data = df.filter(['Close']).values
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)

        # Create the training dataset
        train_size = int(len(scaled_data) * 0.8)
        train_data = scaled_data[0:train_size, :]

        # Split the data into x_train and y_train
        x_train = []
        y_train = []
        for i in range(100, len(train_data)):
            x_train.append(train_data[i-100:i, 0])
            y_train.append(train_data[i, 0])

        # Convert x_train and y_train to numpy arrays
        x_train, y_train = np.array(x_train), np.array(y_train)

        # Reshape the data
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        # Build the LSTM model
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dense(50))
        model.add(Dense(1))

        # Compile the model
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Train the model
        model.fit(x_train, y_train, batch_size=1, epochs=100)

        # Prepare the test dataset
        test_data = scaled_data[train_size - 100:, :]
        x_test = []
        y_test = data[train_size:, :]
        for i in range(100, len(test_data)):
            x_test.append(test_data[i-100:i, 0])

        # Convert x_test to a numpy array
        x_test = np.array(x_test)

        # Reshape the data
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

        # Make predictions
        predictions = model.predict(x_test)
        predictions = scaler.inverse_transform(predictions)

        # Get the predicted close price
        array_2d = predictions
        sorted_array_2d = np.sort(array_2d, axis=None)
        sorted_array_2d = np.reshape(sorted_array_2d, array_2d.shape)
        predicted_close_price = predictions[-1]

        # Render the result on the webpage
        return render(request, 'result.html', {'symbol': symbol, 'predicted_close_price': predicted_close_price})

    # Render the form for user input of the stock symbol
    return render(request, 'predicter.html')


