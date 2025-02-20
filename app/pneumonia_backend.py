from flask import Flask, request, jsonify 
import tensorflow as tf 
import numpy as np 
import cv2 
import os 
from werkzeug.utils import secure_filename 
from flask_cors import CORS 
 
app = Flask(__name__) 
CORS(app) 
