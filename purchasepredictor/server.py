# import Flask class from the flask module
from flask import Flask, request
from flask_cors import CORS, cross_origin
import numpy as np
import pickle

# Create Flask object to run
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/')
def home():
    return "Hi, Welcome to Flask!!"

@app.route('/predict')
@cross_origin()
def predict():

	# Get values from browser
	product_id = request.args['product_id']
	qty = request.args['qty']
	supplier_id = request.args['supplier_id']
	price = request.args['price']
	tax =request.args['tax']
	
	
	testData = np.array([product_id,qty,price,supplier_id,tax]).reshape(1,5)
	#print(testData)
	class_prediced = int(svmIrisModel.predict(testData)[0])
	output = "Predicted Decision: " + str(class_prediced)
	if class_prediced == 1 :
		output = "Go ahead. Your supplier " + supplier_id + "for product " + product_id + "@Price " + price + "looks good."  
	else :
		output = "Your supplier" + supplier_id + "for product" + product_id + "@Price" + price + "is not good choice. Consider the following alternatives"
		output = output + "<br>"

		with open("stock.csv") as fp: 
			for line in fp: 
				
				cLine = line.strip().split(",")
				if cLine[0] == product_id and cLine[6]==1:
					output = output + line
				
                                
                                        
	
	return (output)
	
# Load the pre-trained and persisted SVM model
# Note: The model will be loaded only once at the start of the server
def load_model():
	global svmIrisModel
	
	svmIrisFile = open('SVMModel.pckl', 'rb')
	svmIrisModel = pickle.load(svmIrisFile)
	svmIrisFile.close()

if __name__ == "__main__":
	print("**Starting Server...")
	
	# Call function that loads Model
	load_model()
	
	# Run Server
	app.run()
