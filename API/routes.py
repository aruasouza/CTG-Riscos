from flask import Blueprint, request, json, Response
import json
from key import admin_pass
import models

main = Blueprint('main', __name__)

def jsonify(data,response_code):
    json_str = json.dumps(data,ensure_ascii = False)
    return Response(json_str,content_type="application/json; charset=utf-8",status = response_code)

@main.route('/ipca',methods = ['GET'])
def ipca():
    auth = request.authorization
    if not auth or (auth.username != 'deepen') or (auth.password != admin_pass):
        return jsonify({'error':'unalthorized'},400)
    return jsonify(*models.predict_ipca())

@main.route('/cambio',methods = ['GET'])
def cambio():
    auth = request.authorization
    if not auth or (auth.username != 'deepen') or (auth.password != admin_pass):
        return jsonify({'error':'unalthorized'},400)
    return jsonify(*models.predict_cambio())

@main.route('/cdi',methods = ['GET'])
def cdi():
    auth = request.authorization
    if not auth or (auth.username != 'deepen') or (auth.password != admin_pass):
        return jsonify({'error':'unalthorized'},400)
    return jsonify(*models.predict_cdi())

@main.route('/gsf',methods = ['GET'])
def gsf():
    auth = request.authorization
    if not auth or (auth.username != 'deepen') or (auth.password != admin_pass):
        return jsonify({'error':'unalthorized'},400)
    return jsonify(*models.predict_gsf())