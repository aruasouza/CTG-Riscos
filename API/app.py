from flask import Flask
from routes import main as main_blueprint
from waitress import serve

def create_app():
    app = Flask(__name__)
    app.config['JSON_AS_ASCII'] = False
    app.register_blueprint(main_blueprint)
    return app

app = create_app()

if __name__ == '__main__':
    serve(app,host='0.0.0.0', port=5000)
    # app.run(host='0.0.0.0', port=5000, debug = False)